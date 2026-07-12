# Qwythos / Qwen3.5 inference hot path on AMD gfx1151 (8060S)

This document maps the **full llama.cpp execution path** for `Qwythos-9B-v2-MTP-Q8_0` on our fork, with kernel-level evidence from `rocprof --stats` and optimization status.

---

## 1. Model architecture (9B)

| Property | Value |
|----------|-------|
| Layers | 32 trunk + 1 MTP draft head |
| Attention mix | **3:1** — 24 Gated DeltaNet (GDN) + 8 full flash-attention |
| Quant | Q8_0 (bandwidth-heavy on iGPU) |
| Draft | MTP (`--spec-type draft-mtp`) |
| Hidden | 2560, GDN state S_v=128 |

Source: `llama.cpp/src/models/qwen35.cpp`, `qwen35.cpp` `load_arch_hparams`.

---

## 2. End-to-end decode flow (one generated token)

```
HTTP /v1/chat/completions
  → llama-server slot decode
  → llama_context::decode (ubatch n_tokens=1)
  → ggml_backend_sched_graph_compute_async
  → [HIP graph replay if captured] per layer subgraph
  → MTP: target verify + draft model (if speculating)
```

### Per-layer graph (main trunk)

**All layers:** RMS norm → attention block → residual → post-attn RMS → FFN → residual.

**GDN layer (24×)** — `build_layer_attn_linear()`:

| Step | GGML op(s) | CUDA kernel (typical) | Notes |
|------|------------|----------------------|-------|
| 1 | MUL_MAT wqkv | `mul_mat_vec_q` Q8_0 | **Hot** |
| 2 | MUL_MAT wqkv_gate | `mul_mat_vec_q` | z path for gated norm |
| 3 | MUL_MAT ssm_beta | `mul_mat_vec_q` | |
| 4 | UNARY sigmoid | `unary` or fused | beta |
| 5 | MUL_MAT ssm_alpha | `mul_mat_vec_q` | |
| 6 | ADD + SOFTPLUS + MUL | **`add_softplus_mul` fused** (fork) | gate |
| 7 | conv state concat | view/cpy | recurrent |
| 8 | SSM_CONV + SILU | **`ssm_conv` fused** (fork) | d_conv=4 |
| 9 | view q/k/v | noop | |
| 10 | GATED_DELTA_NET | `gated_delta_net_cuda` | 8 warps, L2 fused |
| 11 | RMS norm gated | RMS + mul | z gate |
| 12 | MUL_MAT ssm_out | `mul_mat_vec_q` | **Hot** |

**Full-attention layer (8×):** Q/K/V proj → RoPE → flash attn → WO.

**FFN (32×):** gate/up/down `mul_mat_vec_q` × 3.

### MTP overhead (draft-mtp)

- Target forward produces `h_nextn` (GPU-resident in fork)
- Draft model: 1–N extra full decodes per step (`spec-draft-n-max`)
- Accept/reject verification batch

---

## 3. Kernel profile evidence (gfx1151, rocprof)

Artifacts: `var/profiles/KERNEL_PROFILE_REPORT.md`, `scripts/rocprof-kernel-profile.sh`.

### Decode (`llama-bench -p 0 -n 128`)

| Metric | Value |
|--------|-------|
| Kernel dispatches | ~16,000 |
| GPU kernel time | ~880 ms |
| Wall time | ~14 s |
| **GPU util of wall** | **~6%** |
| Dominant kernel | `mul_mat_vec_q` (~90%) |
| GDN kernel share | ~1–2% |
| Median inter-kernel gap | ~4.4 µs |

**Conclusion:** decode is **dispatch-limited**, not FLOP-limited. Reducing launches (graphs, fusion) beats matmul micro-tuning.

### Prefill (`-p 512 -n 0`)

| Metric | Value |
|--------|-------|
| Dominant kernel | `mul_mat_q` tile 128 (~93%) |
| Fork vs upstream | Fork −21% `mul_mat_q` kernel time |
| GDN chunked | Helps long prompts when enabled |

---

## 4. Fork optimizations (validated)

| ID | Change | Decode impact | Status |
|----|--------|---------------|--------|
| O1 | MMVQ RDNA3_5 → RDNA3_0 (8 warps Q8) | **+~35%** vs prior fork | **Shipped** patch 0005 |
| O2 | MTP `spec-draft-n-max 3` | **+59%** vs n=6 | **Shipped** compose |
| O3 | HIP graphs ON (default) | +28% vs disable | **Keep default** |
| O4 | Chunked GDN gfx1151 CS=32 | +25% PP short | **Shipped** patch 0002 |
| O5 | Alpha gate ADD+SOFTPLUS+MUL fuse | modest | patch 0004 |
| O6 | GDN decode megafusion v2 (shared conv) | TBD | patch 0006, env flag |
| O7 | Q4_K_M quant | **−29%** decode | **Rejected** — stay Q8 |

### Combined stack benchmark (v3)

| | PP | Decode |
|--|-----|--------|
| Upstream :50126 | 124 | 23.2 |
| Fork :50127 | 124 | **38.3** (+65%) |

---

## 5. Optimization ideas (prioritized)

### Tier A — validated or in progress

1. **MMVQ warp config** — done (0005)
2. **MTP draft depth** — n-max 3 (done)
3. **HIP graph capture** — never disable on gfx1151
4. **GDN megafusion v2** — one conv+silu in shared mem per (head,seq), then 128 col GDN steps; replaces ssm_conv+silu+GDN (3→1 launch per GDN layer)

### Tier B — high confidence, not yet coded

5. **Beta matmul + sigmoid epilogue** — fuse `MUL_MAT(ssm_beta) → sigmoid` (reshape blocks pattern match today)
6. **WQKV + wqkv_gate batched GEMM** — single launch for two projections sharing input
7. **MMQ RDNA3_5 prefill tiles** — tune `mul_mat_q` for Strix Halo LPDDR5X
8. **Separate HIP graphs** — target decode / draft decode / verify prefill graph keys

### Tier C — research / larger refactors

9. **AITER-style full layer fusion** — conv + gate + delta in one kernel (vLLM #40711)
10. **MFMA in GDN** — S_v=128 dot products on RDNA3.5 matrix ops
11. **Backend GPU sampling** — eliminate TOP_K CPU path on ROCm
12. **vLLM/AITER serving path** — for max throughput serving vs single-user llama.cpp

---

## 6. Environment reference (fork docker-compose)

```yaml
LLAMACPP_EXTRA_ARGS: --jinja --spec-type draft-mtp --spec-draft-n-max 3 --ubatch-size 256 --parallel 1
GGML_CUDA_GDN_CHUNKED: 1
GGML_CUDA_GDN_CHUNK_THRESHOLD: 16
GGML_CUDA_GDN_DECODE_FUSED: 0   # set 1 to A/B megafusion v2
# Do NOT set GGML_CUDA_DISABLE_GRAPHS=1
HSA_OVERRIDE_GFX_VERSION: 11.5.1
```

---

## 7. Validation procedure

All GPU tests **serialized**:

```bash
./scripts/bench-pp-tps.sh --url http://127.0.0.1:50127 --title my-run
./scripts/rocprof-kernel-profile.sh fork-decode   # kernel breakdown
```

Compare fork vs upstream on same prompt, 3+ runs, report PP + decode from `timings` in HTTP response.

---

## 8. Key source files

| Area | Path |
|------|------|
| Qwen35 graph | `llama.cpp/src/models/qwen35.cpp` |
| GDN base | `llama.cpp/src/models/delta-net-base.cpp` |
| GDN CUDA | `llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu` |
| GDN chunked | `llama.cpp/ggml/src/ggml-cuda/gated_delta_net_chunked.cu` |
| GDN megafusion | `llama.cpp/ggml/src/ggml-cuda/gated_delta_net_decode_fused.cu` |
| MMVQ decode | `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` |
| Op fusion | `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_cuda_try_fuse` |
| MTP GPU | `llama.cpp/common/speculative.cpp`, `llama-context.cpp` |
| Graph capture | `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_cuda_graph_evaluate_and_capture` |

---

## 9. References

- [llama.cpp #20292](https://github.com/ggml-org/llama.cpp/issues/20292) — ROCm Qwen3.5 dispatch overhead
- [vLLM AITER GDN fusion #40711](https://github.com/vllm-project/vllm/pull/40711)
- [AMD Qwen3.5 Day-0](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html)
- Fork repo: https://github.com/PurHur/llamacpp-qwen35-rocm
