# llamacpp-qwen35-rocm

ROCm fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) targeting **Qwen3.5 / Qwythos** Gated-DeltaNet performance on AMD gfx1151 (8060S).

## Layout

- `patches/` — git-am patches applied to upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) at `patches/UPSTREAM_REF` (~110 KiB; no model weights)
- `docker/` — ROCm llama-server image (clones upstream at build time)
- `scripts/bench-pp-tps.sh` — PP/TPS benchmark (**exclusive GPU lock**)
- `scripts/rocprof-benchmark.sh` — rocprof + benchmark (serialized)
- `bench/profile_one_at_a_time.py` — HTTP timings benchmark

## Profiling rule

**Only one GPU benchmark/profile at a time.** All scripts use `flock` on `/tmp/llamacpp-gpu-profile.lock`:

```bash
./scripts/bench-pp-tps.sh --url http://127.0.0.1:50126 --title baseline
./scripts/rocprof-benchmark.sh baseline
```

Do not run benchmarks in parallel against 50126 and 50127.

## A/B backends

| Port | Service | Image |
|------|---------|-------|
| 50126 | prompt-router production ROCm (upstream master) | `prompt-router-llama-server-rocm-qwythos` |
| 50127 | this fork (patched) | `llamacpp-qwen35-rocm` |

```bash
# Build fork
QWEN35_ROCM_FORK_CACHEBUST=$(date +%s) docker compose build

# Run fork (stop production first if GPU memory is tight)
docker compose up -d

# Compare (one at a time!)
QWEN35_BENCH_URL=http://127.0.0.1:50126 ./scripts/bench-pp-tps.sh --title upstream
QWEN35_BENCH_URL=http://127.0.0.1:50127 ./scripts/bench-pp-tps.sh --title fork
```

## Optimization tracks

1. MTP GPU-resident draft path (no D2H in hot loop)
2. Pre-GDN micro-fusion (conv + silu + l2_norm)
3. gfx1151 GDN kernel occupancy tuning
4. Parallel chunked GDN prefill kernel — see [docs/CHUNKED_GDN_PREFILL.md](docs/CHUNKED_GDN_PREFILL.md)
5. Layer megafusion (AITER-style, longer term)
