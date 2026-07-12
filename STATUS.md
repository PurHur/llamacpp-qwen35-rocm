# Fork optimization status (2026-07-12)

## Backends

| Port | Service | Image |
|------|---------|-------|
| **50126** | `prompt-router-llamacpp-rocm-qwythos-9b-v2-mtp-q8` | upstream master (production) |
| **50127** | `llamacpp-qwen35-rocm-fork` | patched fork (`llamacpp-qwen35-rocm:latest`) |

## Profiling rule

All benchmarks use `flock` on `/tmp/llamacpp-gpu-profile.lock` via:

```bash
./scripts/bench-pp-tps.sh --url http://127.0.0.1:50126 ...
./scripts/rocprof-benchmark.sh tag
```

**Never run two GPU benchmarks in parallel.**

## A/B results (serialized, same prompt)

### Short prefill (repeat=1, max_tokens=128)

| | PP | Decode |
|--|-----|--------|
| Upstream 50126 | 98.6 tok/s | 23.7 tok/s |
| Fork 50127 | 100.5 tok/s | 24.3 tok/s |

### Long prefill (repeat=4, max_tokens=128)

| | PP | Decode |
|--|-----|--------|
| Upstream 50126 | 136.4 tok/s | 27.5 tok/s |
| Fork 50127 | 139.0 tok/s | 26.6 tok/s |

Delta is modest on default bench prompts; chunked GDN + occupancy tuning need longer prompts (512–2048 tokens) to show full PP win.

## Tracks implemented

1. **MTP GPU-resident** — `handle_mtp_for_ubatch` D2D staging, reduced sync/D2H in draft loop (`docs/MTP_GPU_RESIDENT.md`)
2. **GDN occupancy + L2 fusion** — gfx1151 8-warps decode launch, fuse L2 norm into GDN kernel (`docs/GDN_TUNING.md`)
3. **Chunked GDN prefill MVP** — `gated_delta_net_chunked.cu`, env `GGML_CUDA_GDN_CHUNKED=1` (`docs/CHUNKED_GDN_PREFILL.md`)
4. **Baseline profiling** — `var/profiles/BASELINE_REPORT.md`

## Next steps

- rocprof on host (install rocm-profiler) for kernel-level breakdown
- Long-context PP bench (`--repeat 16` or prompt file 2k+ tokens)
- Layer megafusion (AITER-style) — not started
- Merge winning patches upstream or into prompt-router production image
