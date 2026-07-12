# Fork optimization status (2026-07-12)

## Backends

| Port | Service |
|------|---------|
| **50126** | upstream master (production) |
| **50127** | patched fork (`llamacpp-qwen35-rocm:latest`) |

## Combined fork stack (v4)

Patches **0001–0006** + runtime:

- `--spec-draft-n-max 3`
- HIP graphs ON
- Chunked GDN gfx1151 (CS=32, threshold 16)
- Alpha gate fusion (ADD+SOFTPLUS+MUL)
- MMVQ RDNA3_5 → RDNA3_0 (8 warps Q8 decode)
- **GDN decode megafusion v2 ON** (shared conv+silu)

Full hot path: `docs/HOTPATH_GFX1151.md`

## Latest A/B (3 runs, short prompt)

| | PP | Decode |
|--|-----|--------|
| Upstream :50126 | 120.8 | 27.1 |
| Fork v4 (megafusion off) | 115.1 | 31.1 |
| **Fork v4 (megafusion v2 on)** | 114.3 | **33.1** |
| **Fork vs upstream** | ~same | **+22%** |

Prior peak (v3 combo, megafusion off): 38.3 decode — variance run-to-run; v4 megafuse stable ~33 tok/s.

## GitHub

https://github.com/PurHur/llamacpp-qwen35-rocm

## Next

- Beta matmul+sigmoid epilogue fusion
- MMQ RDNA3_5 prefill tuning
- Promote to production `:50126`
