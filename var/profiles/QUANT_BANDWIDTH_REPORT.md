# Phase 4 — Quantization & Bandwidth Analysis (gfx1151)

**Date:** 2026-07-12  
**Host:** AMD Ryzen AI Max 395 / gfx1151 (8060S iGPU)  
**Repo:** `llamacpp-qwen35-rocm`  
**Model family:** empero-ai/Qwythos-9B-v2-GGUF (MTP variants)

---

## 1. Available GGUF quants

### On disk (before this phase)

| File | Quant | Size | Notes |
|------|-------|------|-------|
| `Qwythos-9B-v2-MTP-Q8_0.gguf` | Q8_0 + MTP | **9.11 GiB** (9.2 GB) | Only local quant prior to Phase 4 |

Path: `/home/ai/projects/prompt-router/models/huggingface/empero-ai--Qwythos-9B-v2-GGUF/`

### HuggingFace catalog ([empero-ai/Qwythos-9B-v2-GGUF](https://huggingface.co/empero-ai/Qwythos-9B-v2-GGUF))

| File | Quant | HF size | Under 6 GB? |
|------|-------|---------|-------------|
| `Qwythos-9B-v2-MTP-Q4_K_M.gguf` | Q4_K_M + MTP | 5.90 GB | **Yes — downloaded** |
| `Qwythos-9B-v2-MTP-Q5_K_M.gguf` | Q5_K_M + MTP | 6.71 GB | No (skipped) |
| `Qwythos-9B-v2-MTP-Q6_K.gguf` | Q6_K + MTP | 7.67 GB | No |
| `Qwythos-9B-v2-MTP-Q8_0.gguf` | Q8_0 + MTP | 9.79 GB | No (already present) |
| Non-MTP variants | Q4/Q5/Q6/Q8/BF16 | 5.74–18.4 GB | Q4 non-MTP 5.74 GB |

**GGUF metadata (Q8 MTP):** architecture `qwen35`, 33 layers, 1M context, `general.size_label: 9B`, 442 tensors.

Mixed quant in K-variants (per HF README): Q4_K_M uses Q8_0 for SSM α/β and Q6_K for SSM out — not uniform 4-bit across all tensors.

---

## 2. Bandwidth-bound analysis

### Model size vs decode throughput (theoretical)

For a **purely bandwidth-limited** dense decode, tok/s scales inversely with resident weight bytes:

```
tok/s₂ ≈ tok/s₁ × (size₁ / size₂)
effective_GB/s ≈ model_size_GiB × decode_tok_s
```

| Quant | File GiB | Size ratio vs Q8 | Theoretical tok/s @ 30 tok/s Q8 baseline | Theoretical @ 19.1 tok/s Q8 (solo bench) |
|-------|----------|------------------|------------------------------------------|-------------------------------------------|
| Q8_0 MTP | 9.114 | 1.00× | 30.0 | 19.1 |
| Q5_K_M MTP | ~6.25 | 1.46× | **43.8** | **27.9** |
| Q4_K_M MTP | 5.498 | 1.66× | **49.8** | **31.7** |

### Observed GB/s demand (Q8)

Using `effective_GB/s = 9.114 GiB × decode_tok_s`:

| Source | Decode tok/s | Implied GB/s (if BW-bound) |
|--------|--------------|----------------------------|
| Fork v2 short bench (STATUS.md) | 29.6 | **270 GB/s** |
| Fork v2 long bench | 25.2 | 230 GB/s |
| Production upstream baseline | 23.6 | 215 GB/s |
| **Phase 4 solo fork bench (Q8)** | **19.1** | **174 GB/s** |

gfx1151 unified memory (LPDDR5X-8533, 256-bit) peak is ~**256–273 GB/s**. At ~30 tok/s on Q8, implied demand sits at the **practical bandwidth ceiling** — consistent with bandwidth pressure on Q8, but not sufficient to explain quant choice alone.

### Qwythos vs Qwen3.6-35B MoE context

| Model | Active params / file | Typical decode | Interpretation |
|-------|---------------------|----------------|----------------|
| Qwythos 9B dense Q8 | ~9.1 GiB all layers | ~24–30 tok/s | Full 9B read every token |
| Qwen3.6-35B MoE MTP | ~3B active / smaller effective read | ~46 tok/s | MoE skips expert FFN → lower bytes/token |

Dense Qwythos is **inherently bandwidth-heavier** than MoE at similar silicon; quant reduction helps in theory but kernel efficiency dominates on gfx1151 (see §4).

---

## 3. mmq.cu / MMVQ — RDNA3.5 (gfx1151) Q8_0 path

Sources: `llama.cpp/ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh`, `mmvq.cu`, `vendors/hip.h`.

### What exists

| Area | gfx1151 behavior |
|------|------------------|
| Arch detection | `__gfx1151__` → `RDNA3_5` in `hip.h`; CC `GGML_CUDA_CC_RDNA3_5` (0x1150) |
| MMQ enable (`ggml_cuda_should_use_mmq`) | Q8_0 → `true` on RDNA3 WMMA path (default branch) |
| MMQ tile geometry | `mmq_y=128`, `mmq_x_max=128`, `nwarps=8` (AMD WMMA) — **same as RDNA3/RDNA2** |
| Q8_0 kernel | DP4A `vec_dot_q8_0_q8_1_impl` (`VDR_Q8_0_Q8_1_MMQ=8`); no gfx1151-specific template |
| Launch bounds | Generic `#if RDNA3` block — RDNA3_5 included via parent `RDNA3` define |
| Decode matvec | gfx1151 uses **`MMVQ_PARAMETERS_RDNA2`** table at compile time (grouped with RDNA2, not RDNA3_0) |
| Host MMVQ routing | Runtime CC → `get_mmvq_mmid_max_batch_rdna3()` via `GGML_CUDA_CC_IS_RDNA3` (includes 3.5) |

### Gaps (no gfx1151-specific MMQ tuning)

1. **No RDNA3_5 tile/warp overrides in mmq.cuh** — unlike `gated_delta_net.cu` (8-warps decode, occupancy tuning). Q8_0 MMQ uses generic 8-warp WMMA layout.
2. **MMVQ compile-time table mismatch** — device code selects RDNA2 parameter table for gfx1151 (`RDNA2 \|\| RDNA3_5`), while host uses RDNA3 batch limits. Potential suboptimal constants for Strix Halo iGPU.
3. **No RDNA3.5 WMMA fast-path for Q8_0** — `load_tiles_q8_0` uses DP4A/MMA hybrid shared with RDNA3 desktop; CHUNKED_GDN doc notes “Future work: RDNA3.5 WMMA”.
4. **No gfx1151 LDS-aware MMQ shared sizing** — GDN chunked path respects 64 KiB LDS; MMQ Q8_0 tile buffers (`MMQ_DP4A_TXS_Q8_0`) do not.
5. **Q4_K vs Q8_0 asymmetry** — K-quant MMQ (`load_tiles_q4_K`, scale/min unpacking) is heavier; no RDNA3_5 specialization. Explains measured Q4 slowdown despite smaller file.
6. **Stream-K disabled on RDNA** — `use_stream_k` false for gfx1151 (NVIDIA Volta+ and CDNA only).

### RDNA3_5-specific code elsewhere (not mmq)

- `gated_delta_net.cu`: 8-warps decode, 4-warps prefill for RDNA3.5
- `gated_delta_net_chunked.cu`: CS=32, gmem state scratch for 64 KiB LDS

---

## 4. Measured quant A/B (optional bench)

**Method:** `scripts/bench-pp-tps.sh` (profile-lock), fork image `llamacpp-qwen35-rocm:latest`, solo GPU (production Q8 stopped), same MTP flags (`draft-mtp`, `spec-draft-n-max 6`), short prefill (`repeat=1`, `max_tokens=128`).

| Quant | PP mean | Decode mean | vs Q8 decode |
|-------|---------|-------------|--------------|
| **Q8_0 MTP** | 69.0 tok/s | **19.1 tok/s** | baseline |
| **Q4_K_M MTP** | 120.8 tok/s | **13.5 tok/s** | **−29%** (slower) |

JSON: `var/profiles/q8_mtp_solo.json`, `var/profiles/q4_mtp_solo.json`

**Contention note:** First Q4 run with production Q8 still loaded yielded ~11 tok/s; solo runs reported above.

### Interpretation

- **Q4_K_M is not faster** despite 1.66× smaller weights — disproves pure bandwidth scaling on gfx1151.
- Q8_0 MMQ/MMVQ path is better optimized; Q4_K dequant+MMQ overhead dominates.
- PP is faster on Q4 (smaller weights load / less memory pressure) but decode regresses.

---

## 5. Recommendation

### **Stay on Q8_0 MTP** for gfx1151 production

| Criterion | Q8_0 | Q4_K_M |
|-----------|------|--------|
| Decode speed (measured) | **19–30 tok/s** | 13–14 tok/s |
| Quality | Near-lossless | Good (Empero default for size-constrained) |
| VRAM / BW headroom | Higher pressure | Lower file size |
| MMQ kernel fit on gfx1151 | Best available | Suboptimal K-quant path |

**When to reconsider Q4_K_M:** VRAM-constrained deployment (cannot fit Q8 + context), or after upstream/fork adds RDNA3.5-tuned Q4_K MMQ kernels and re-benchmark.

**Q5_K_M:** Not tested (6.71 GB > 6 GB download cap). Theoretical decode ~44 tok/s if bandwidth-bound; likely mid-point between Q4 and Q8 on compute efficiency — worth a future fetch if cap is raised.

### Optimization priority (instead of down-quant)

1. Keep Q8; invest in mmq/mmvq gfx1151 tuning (tile sizes, RDNA3_5 WMMA for Q8_0).
2. MoE-style sparsity already explains Qwen3.6-35B lead — architectural, not quant fixable for dense Qwythos.
3. Continue GDN/MTP fork work (already +25% decode vs upstream in v2 STATUS).

---

## Artifacts

| Artifact | Path |
|----------|------|
| This report | `var/profiles/QUANT_BANDWIDTH_REPORT.md` |
| Q8 solo bench | `var/profiles/q8_mtp_solo.json` |
| Q4 solo bench | `var/profiles/q4_mtp_solo.json` |
| Downloaded quant (not in git) | `.../Qwythos-9B-v2-MTP-Q4_K_M.gguf` (5.50 GiB) |

**Second quant tested:** Yes — Q4_K_M MTP downloaded and solo-benchmarked under profile-lock.
