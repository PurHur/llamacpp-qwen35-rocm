# MTP Tuning A/B Report — Qwythos fork (gfx1151)

**Date:** 2026-07-12  
**Host:** AMD 8060S gfx1151  
**Container:** `llamacpp-qwen35-rocm-fork` (port 50127)  
**Model:** `Qwythos-9B-v2-MTP-Q8_0.gguf`  
**Benchmark:** `./scripts/bench-pp-tps.sh` (serialized via `profile-lock.sh`, 3 runs + 1 warmup, 128 completion tokens)  
**Fixed args:** `--jinja --ubatch-size 256 --parallel 1`

## Configuration variants

| Config | `LLAMACPP_EXTRA_ARGS` (spec-related) |
|--------|--------------------------------------|
| **mtp-baseline-n6** (default) | `--spec-type draft-mtp --spec-draft-n-max 6` |
| **mtp-nmax3** | `--spec-type draft-mtp --spec-draft-n-max 3` |
| **mtp-nmax4** | `--spec-type draft-mtp --spec-draft-n-max 4` |
| **mtp-off** | `--spec-type none` |

## Per-run results

### mtp-baseline-n6 (`--spec-draft-n-max 6`)

| Run | PP tok/s | Decode tok/s | Wall (s) | Draft accepted / generated | Accept % |
|-----|----------|--------------|----------|----------------------------|----------|
| 1   | 118.9    | 16.2         | 8.32     | 83 / 261                   | 31.8%    |
| 2   | 114.3    | 24.2         | 5.75     | 87 / 240                   | 36.3%    |
| 3   | 107.4    | 16.6         | 8.19     | 83 / 261                   | 31.8%    |
| **Mean** | **113.5** (σ 4.7) | **19.0** (σ 3.7) | — | **253 / 762** | **33.2%** |

JSON: `var/profiles/mtp-baseline-n6.json`

### mtp-nmax3 (`--spec-draft-n-max 3`)

| Run | PP tok/s | Decode tok/s | Wall (s) | Draft accepted / generated | Accept % |
|-----|----------|--------------|----------|----------------------------|----------|
| 1   | 97.8     | 29.2         | 4.91     | 78 / 145                   | 53.8%    |
| 2   | 93.1     | 31.3         | 4.65     | 80 / 138                   | 58.0%    |
| 3   | 108.9    | 30.0         | 4.79     | 78 / 145                   | 53.8%    |
| **Mean** | **100.0** (σ 6.6) | **30.2** (σ 0.8) | — | **236 / 428** | **55.1%** |

JSON: `var/profiles/mtp-nmax3.json`

### mtp-nmax4 (`--spec-draft-n-max 4`)

| Run | PP tok/s | Decode tok/s | Wall (s) | Draft accepted / generated | Accept % |
|-----|----------|--------------|----------|----------------------------|----------|
| 1   | 90.8     | 26.1         | 5.42     | 88 / 231                   | 38.1%    |
| 2   | 96.9     | 24.7         | 5.67     | 86 / 240                   | 35.8%    |
| 3   | 98.6     | 21.7         | 6.41     | 80 / 269                   | 29.7%    |
| **Mean** | **95.4** (σ 3.4) | **24.2** (σ 1.8) | — | **254 / 740** | **34.3%** |

JSON: `var/profiles/mtp-nmax4.json`

### mtp-off (`--spec-type none`)

| Run | PP tok/s | Decode tok/s | Wall (s) | Draft accepted / generated | Accept % |
|-----|----------|--------------|----------|----------------------------|----------|
| 1   | 71.7     | 15.9         | 8.68     | — / —                      | n/a      |
| 2   | 71.9     | 15.9         | 8.69     | — / —                      | n/a      |
| 3   | 71.7     | 15.9         | 8.76     | — / —                      | n/a      |
| **Mean** | **71.8** (σ 0.1) | **15.9** (σ 0.03) | — | — | n/a |

JSON: `var/profiles/mtp-off.json`

## Summary comparison

| Config | PP mean | Decode mean | Draft accept rate | vs baseline decode |
|--------|---------|-------------|---------------------|--------------------|
| **mtp-nmax3** | 100.0 | **30.2** | **55.1%** | **+59%** |
| mtp-nmax4 | 95.4 | 24.2 | 34.3% | +27% |
| mtp-baseline-n6 | **113.5** | 19.0 | 33.2% | — |
| mtp-off | 71.8 | 15.9 | n/a | −16% |

## Analysis

- **Higher `spec-draft-n-max` hurts decode throughput.** Drafting 6 tokens per step generates ~2× more draft tokens than n-max 3, but accept rate stays ~32–34%, so most extra draft work is wasted verification.
- **n-max 3 is the sweet spot.** Accept rate jumps to ~55% (fewer speculative steps, higher hit rate per step). Decode TPS reaches **30.2 tok/s** — nearly **1.6×** the default n6 config and **1.9×** speculative-off.
- **n-max 4 is a middle ground** (+27% decode vs n6) but still well below n3.
- **Disabling MTP (`--spec-type none`)** is worst for decode (15.9 tok/s) despite slightly lower PP overhead on the prompt; MTP draft path is clearly net-positive on this hardware.
- **PP prefill** is fastest at n6 (113.5 tok/s) but the n3 penalty (−12%) is acceptable given the large decode gain.

## Recommendation

**Switch production fork to `--spec-draft-n-max 3`.**

```bash
# docker-compose.yml or env override:
LLAMACPP_EXTRA_ARGS="--jinja --spec-type draft-mtp --spec-draft-n-max 3 --ubatch-size 256 --parallel 1"
```

Expected improvement: **~59% higher decode throughput** (30.2 vs 19.0 tok/s) with **higher draft accept rate** (55% vs 33%) and modest PP regression (−12%).

## Post-test state

Container restored to default **`--spec-draft-n-max 6`** after benchmarking (per Phase 2 protocol). Apply the n3 recommendation explicitly when promoting to production.
