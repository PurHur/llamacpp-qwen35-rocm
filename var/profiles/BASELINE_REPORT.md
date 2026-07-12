# Qwythos Production Baseline — upstream ROCm (50126)

**Date:** 2026-07-12  
**Host:** AMD 8060S gfx1151  
**Container:** `prompt-router-llamacpp-rocm-qwythos-9b-v2-mtp-q8`  
**URL:** http://127.0.0.1:50126  
**Model:** `Qwythos-9B-v2-MTP-Q8_0.gguf`  
**llama-server:** version 1 (e3546c7), upstream master image per README A/B table  

## Health check

- Container status: **Up**, port `50126→8000`
- `/health`: **HTTP 200**

## PP / TPS benchmark (`bench-pp-tps.sh`, 3 runs)

| Run | PP tok/s | Decode tok/s | Wall (s) | Draft accepted / generated |
|-----|----------|--------------|----------|----------------------------|
| 1   | 83.2     | 24.0         | 5.91     | 84 / 247                   |
| 2   | 79.6     | 20.9         | 6.74     | 77 / 295                   |
| 3   | 74.3     | 26.0         | 5.59     | 88 / 229                   |
| **Mean** | **79.0** | **23.6** | — | **249 / 771 (32.3%)** |

- PP stdev: 3.7 tok/s  
- Decode stdev: 2.1 tok/s  
- JSON: `var/profiles/upstream_baseline.json`

## rocprof (`rocprof-benchmark.sh upstream-rocm`)

**rocprof not installed** on this host — script fell back to HTTP benchmark only (2 runs):

| Run | PP tok/s | Decode tok/s | Draft accepted / generated |
|-----|----------|--------------|----------------------------|
| 1   | 89.1     | 25.2         | 86 / 240                   |
| 2   | 81.1     | 25.4         | 86 / 243                   |
| **Mean** | **85.1** | **25.3** | **172 / 483 (35.6%)** |

- Artifacts: `var/profiles/upstream-rocm_20260712_083726/bench.json`
- **No kernel trace captured** (no `trace.csv` / top-kernel summary available)

## Draft MTP accept rate

From benchmark HTTP timings (primary baseline):

- **Aggregate accept rate:** 32.3% (249 accepted / 771 generated)
- **Per-run range:** 26.1% – 35.6%
- Server slot logs during session: ~26% – 42%, mean len ~2.5–3.4 tokens

MTP draft path is **active** (`--spec-type draft-mtp --spec-draft-n-max 6` in process args).

## Fused GDN path status

| Signal | Result |
|--------|--------|
| Log lines matching `fused.*gated`, `Gated Delta`, `fgdn` | **None found** |
| `ROCm0` device in logs | **Present** — GPU backend active |
| `draft-mtp` in logs | Not logged at runtime; confirmed via **process args** |
| Fused GDN kernel strings in binary | **Not found** (strings search) |

**Conclusion:** Production container on 50126 runs **upstream master** llama.cpp (not the `llamacpp-qwen35-rocm` fork). No evidence of fused Gated-DeltaNet (FGDN) kernel paths in startup or runtime logs. GDN layers execute via standard upstream ROCm ops; fused-path optimizations are expected on the fork (50127), not this baseline.

Key log excerpt saved to `var/profiles/fused_path_log.txt`.

## Notes

- Benchmarks serialized via `/tmp/llamacpp-gpu-profile.lock` (one GPU profile at a time).
- Minor script fix applied locally: stripped CRLF from `bench-pp-tps.sh`, `rocprof-benchmark.sh`, `profile-lock.sh` (shebang `\r` was blocking execution).
