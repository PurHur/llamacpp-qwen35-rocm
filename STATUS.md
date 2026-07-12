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
| Upstream 50126 (prior) | 98.6 tok/s | 23.7 tok/s |
| Fork 50127 (prior) | 100.5 tok/s | 24.3 tok/s |
| **Fork 50127 (v2 — chunked enabled on gfx1151)** | **125.3 tok/s** | **29.6 tok/s** |

### Long prefill (repeat=4, max_tokens=128)

| | PP | Decode |
|--|-----|--------|
| Upstream 50126 (prior) | 136.4 tok/s | 27.5 tok/s |
| Fork 50127 (prior) | 139.0 tok/s | 26.6 tok/s |
| Upstream 50126 (v2 run) | 179.6 tok/s | 24.8 tok/s |
| **Fork 50127 (v2 — chunked enabled on gfx1151)** | **176.1 tok/s** | **25.2 tok/s** |

v2 fixes: chunked GDN was compile-disabled on RDNA3.5 (64 KiB LDS) and blocked when L2 fusion was on. Now uses CS=32, global state scratch, L2 pre-pass, threshold=16.

## Tracks implemented

1. **MTP GPU-resident** — `handle_mtp_for_ubatch` D2D staging, reduced sync/D2H in draft loop (`docs/MTP_GPU_RESIDENT.md`)
2. **GDN occupancy + L2 fusion** — gfx1151 8-warps decode launch, fuse L2 norm into GDN kernel (`docs/GDN_TUNING.md`)
3. **Chunked GDN prefill** — enabled on gfx1151 via gmem state scratch + CS=32; L2-compatible (`docs/CHUNKED_GDN_PREFILL.md`)
4. **Baseline profiling** — `var/profiles/BASELINE_REPORT.md`

## Next steps

- rocprof on host for kernel-level breakdown
- Long-context PP bench (512–2048 tokens) to stress chunked path further
- Tune CS=32 vs CS=64 on gfx1151; profile chunked vs serial on long prefill
- Layer megafusion (AITER-style) — not started
- Promote v2 fork patches to production `:50126` if stable
