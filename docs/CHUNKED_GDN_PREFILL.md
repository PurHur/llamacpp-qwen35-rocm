# Chunked parallel GDN prefill (CUDA/HIP)

Phase-2 MVP: parallel **chunked** prefill for `GGML_OP_GATED_DELTA_NET` on ROCm (ggml-hip → ggml-cuda), targeting Qwythos / Qwen3.5 GDN layers (`S_v=128`, scalar gate, `K=1`).

## Problem

The fused autoregressive GDN kernel loops `for (t = 0; t < n_tokens; t++)` serially inside each `(head, seq, column)` block. That is fine for decode (`n_tokens=1`) but limits prompt-processing (PP) throughput on long prefills.

## Approach

When `cparams.fused_gdn_ch` routes the graph through `build_delta_net_fused()` → `ggml_gated_delta_net()`, the CUDA backend may dispatch a **chunked parallel** path instead of the serial AR kernel:

1. **Intra-chunk** — build decay-masked `k k^T` / `k q^T` blocks and triangular solve (chunk size **C=64**).
2. **Inter-chunk** — sequential state carry across chunks (parallel over state matrix elements).
3. **Output** — per-chunk attention from state + corrected `v`.
4. **Tail** — remaining tokens `< C` handled by the existing serial AR kernel.

Reference algorithm: `build_delta_net_chunking()` in `src/models/delta-net-base.cpp` and upstream RFC [#22967](https://github.com/ggml-org/llama.cpp/issues/22967).

## Files

| File | Role |
|------|------|
| `ggml/src/ggml-cuda/gated_delta_net_chunked.cu` | Chunked kernel pipeline (scalar/HIP-safe MVP) |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | Dispatch + serial kernel (`t_offset` for tail) |
| `ggml/src/ggml-cuda/gated_delta_net.cuh` | Public chunked API |
| `ggml/src/ggml-cuda/solve_tri.cu` | Batched `ggml_cuda_solve_tri_f32()` helper |

## Dispatch gating

Chunked path is used when **all** of:

- compile-time `GGML_CUDA_GDN_CHUNKED=1` (default in fork Docker image)
- runtime `GGML_CUDA_GDN_CHUNKED` not disabled
- `n_tokens >= GGML_CUDA_GDN_CHUNK_THRESHOLD` (default **64**)
- non-KDA (`g.ne[0] == 1`)
- `K == 1` (no recurrent snapshot slots)
- `S_v == 128` (Qwythos MVP)
- fused L2-in-kernel disabled (`fuse_l2_eps == 0`)

Otherwise the serial AR kernel runs unchanged.

### A/B flags

| Flag | Default | Effect |
|------|---------|--------|
| `GGML_CUDA_GDN_CHUNKED` | `1` | `0`/`false`/`no` forces serial AR |
| `GGML_CUDA_GDN_CHUNK_THRESHOLD` | `64` | Minimum `n_tokens` for chunked path |

Compile-time opt-out: `-DGGML_CUDA_GDN_CHUNKED=0`.

## Docker fork (port 50127)

Production upstream ROCm stays on **50126**; this fork serves on **50127**.

```bash
cd /home/ai/projects/llamacpp-qwen35-rocm

# Rebuild after kernel changes
QWEN35_ROCM_FORK_CACHEBUST=$(date +%s) docker compose build

# Start fork (stop production 50126 first if GPU memory is tight / OOM)
docker compose up -d

# Health
curl -s http://127.0.0.1:50127/health
```

Environment overrides: see `docker-compose.yml` (`QWEN35_ROCM_FORK_*`).

### Chunked prefill A/B in container

```bash
# Serial AR baseline inside fork container
docker exec -e GGML_CUDA_GDN_CHUNKED=0 llamacpp-qwen35-rocm-fork ...

# Chunked (default in fork build)
docker exec llamacpp-qwen35-rocm-fork ...
```

Restart the server after changing env vars.

## Benchmarking (exclusive GPU lock)

**Only one GPU benchmark at a time** — use the flock wrapper:

```bash
# Baseline (production upstream, port 50126)
QWEN35_BENCH_URL=http://127.0.0.1:50126 ./scripts/bench-pp-tps.sh --title upstream

# Fork with chunked GDN (port 50127)
QWEN35_BENCH_URL=http://127.0.0.1:50127 ./scripts/bench-pp-tps.sh --title fork-chunked

# Fork serial A/B (restart container with GGML_CUDA_GDN_CHUNKED=0 first)
QWEN35_BENCH_URL=http://127.0.0.1:50127 ./scripts/bench-pp-tps.sh --title fork-serial
```

Do not run 50126 and 50127 benches concurrently.

## Correctness tests

```bash
# Inside llama.cpp build tree (HIP build)
./build/bin/test-backend-ops test_gated_delta_net
```

Large-`n_tokens` cases (64–1024) exercise prefill/chunked shapes.

## Limitations (MVP)

- Non-KDA, `S_v=128` only
- Scalar kernels (no WMMA/tensor-core path yet)
- No in-kernel L2 norm fusion on chunked path
- KDA + recurrent snapshot (`K>1`) remain on serial AR

Future work: RDNA3.5 WMMA, KDA chunks (C=16), lower dispatch threshold tuning, rocprof-guided fusion.
