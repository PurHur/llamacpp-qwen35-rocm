# Phase 1 Kernel Profile Report — Qwythos-9B Q8_0 on gfx1151 (8060S)

**Date:** 2026-07-12  
**Host GPU:** AMD Radeon 8060S, gfx1151, 128 GB VRAM  
**Model:** `Qwythos-9B-v2-MTP-Q8_0.gguf`  
**Profiler:** `rocprof --stats` inside ROCm 7.2 containers (serialized via `scripts/profile-lock.sh`)

---

## Executive summary

| Workload | Dominant kernels | Dispatches | Kernel time | Key fork vs upstream delta |
|----------|------------------|------------|-------------|----------------------------|
| **Decode** (`-p 0 -n 128`) | `mul_mat_vec_q` fused + plain (~90%) | ~16k | ~880 ms | **Parity** — within 1% kernel time |
| **Prefill** (`-p 512 -n 0`) | `mul_mat_q` tile MM (~93%) | ~1.2k | fork 4.0 s / upstream 5.0 s | **Fork −21%** on `mul_mat_q`; upstream +17% total prefill kernel time |

**rocprof status:** **Partial success.** `rocprof` (v1) inside Docker captures kernel stats on gfx1151 despite an architecture warning and post-run segfault (exit 139). Host has no `/opt/rocm/bin/rocprof`. `rocprofv3` fails (`libdw.so.1` missing). `rocprof-benchmark.sh` HTTP fallback still applies on bare host.

**Top 5 bottlenecks (combined decode + prefill):**

1. **`mul_mat_vec_q` (Q8_0, width=1)** — decode mat-vec; ~56% + ~35% of decode kernel time (fused / non-fused variants)
2. **`mul_mat_q` (Q8_0, tile=128)** — prefill batched MM; ~93% of prefill kernel time
3. **`quantize_q8_1` / `quantize_mmq_q8_1`** — activation quant before MM; ~1.5% decode, ~0.5% prefill
4. **`k_get_rows_float`** — MoE/expert embedding gather; ~2.4% decode
5. **`gated_delta_net_cuda`** — Qwen3.5 GDN layers; ~1% decode, ~2% prefill (fork uses `<128,4,1>` chunked template)

---

## Tooling

| Tool | Location | gfx1151 result |
|------|----------|----------------|
| `rocprof --stats` | `/opt/rocm/bin/rocprof` in both containers | **Works** — emits `*.stats.csv`, `*.csv`, `*.json`; warns “v1 not supported”, app often SIGSEGV after capture |
| `rocprofv2` | container | Not tested (deprecated) |
| `rocprofv3` | container | **Fails** — `libdw.so.1` missing when launching `llama-bench` |
| `rocprof` | host `/opt/rocm/bin/` | **Absent** — only `rocm-smi` present |
| `rocprof-benchmark.sh` | host | Falls back to HTTP bench when `rocprof` not on PATH |

**Capture script added:** `scripts/rocprof-kernel-profile.sh` — stops serving container, runs one-off profiling image, restarts server.

---

## Methodology

### Decode-heavy
```bash
./scripts/profile-lock.sh -- docker run ... rocprof --stats -o decode_trace.csv \
  llama-bench -m <model> -p 0 -n 128 -r 1 --no-warmup -ngl -1 -fa on -b 2048 -t 8
```

### Prefill-heavy
```bash
# Same with -p 512 -n 0
# Fork with GGML_CUDA_GDN_CHUNKED=1 aborts (missing gdn_chunk_cumsum_kernel symbol under rocprof)
# Reported fork prefill uses GGML_CUDA_GDN_CHUNKED=0; server path validated via HTTP repeat=4
```

### HTTP prefill validation (chunked GDN on fork server)
```bash
./scripts/profile-lock.sh -- python3 bench/profile_one_at_a_time.py \
  --url http://127.0.0.1:50127 --repeat 4 --max-tokens 0
```

All GPU runs serialized through `/tmp/llamacpp-gpu-profile.lock`.

---

## Artifact paths

| Artifact | Path |
|----------|------|
| **This report** | `var/profiles/KERNEL_PROFILE_REPORT.md` |
| Fork decode kernels | `var/profiles/fork-kernel_20260712_093226/decode_top_kernels.txt` |
| Fork prefill kernels (no-chunk) | `var/profiles/fork-kernel_20260712_093226/prefill_nochunk_top_kernels.txt` |
| Fork raw traces | `var/profiles/fork-kernel_20260712_093226/decode_trace.{csv,stats.csv,json}` |
| Upstream decode | `var/profiles/upstream-kernel_20260712_093400/decode_top_kernels.txt` |
| Upstream prefill | `var/profiles/upstream-kernel_20260712_093400/prefill_top_kernels.txt` |
| Upstream raw traces | `var/profiles/upstream-kernel_20260712_093400/prefill_trace.{csv,stats.csv,json}` |
| HTTP prefill fork | `var/profiles/fork_prefill_http.json` |
| HTTP prefill upstream | `var/profiles/upstream_prefill_http.json` |
| Profile driver script | `scripts/rocprof-kernel-profile.sh` |

---

## Decode vs prefill breakdown

### Decode (`-p 0 -n 128`) — token generation hot path

| Metric | Fork (50127 build) | Upstream (50126 build) |
|--------|-------------------|--------------------------|
| Total kernel time | 880 ms | 888 ms |
| Kernel dispatches | 16,026 | 15,965 |
| Profile wall span | 2,025 ms | 2,177 ms |
| Kernel % of span | 43.5% | 40.8% |
| Gap % of span (dispatch overhead) | **56.5%** | **59.2%** |
| Median inter-dispatch gap | 4.4 µs | 4.5 µs |
| Est. end-to-end wall (9.2 tok/s bench) | ~13.9 s | ~13.9 s |
| **GPU kernel util vs full wall** | **6.4%** | **6.4%** |

**Top decode kernels (fork):**

| Rank | Kernel | Calls | Time (ms) | % |
|------|--------|-------|-----------|---|
| 1 | `mul_mat_vec_q<Q8_0,1,true>` (fused) | 1,300 | 492 | 55.9% |
| 2 | `mul_mat_vec_q<Q8_0,1,false>` | 2,620 | 305 | 34.7% |
| 3 | `k_get_rows_float` | 887 | 21 | 2.4% |
| 4 | `quantize_q8_1` | 3,921 | 13 | 1.5% |
| 5 | `rms_norm_f32<1024>` | 1,175 | 12 | 1.4% |
| 6 | `gated_delta_net_cuda<128,8,1>` | 434 | 8.6 | 1.0% |

**Interpretation:** Decode is **dispatch-bound on the CPU**, not compute-bound on the GPU. ~16k kernel launches for 128 tokens (~125 dispatches/token). The GPU is idle ~94% of end-to-end decode time — graph build, scheduling, MTP draft logic, and host sync dominate.

Fork vs upstream decode kernels are effectively identical; GDN template differs (`<128,8,1>` fork vs `<128,false,false>` upstream) but remains &lt;1% of kernel time.

---

### Prefill (`-p 512 -n 0`) — prompt processing hot path

| Metric | Fork (GDN_CHUNKED=0) | Upstream |
|--------|----------------------|----------|
| Total kernel time | 3,997 ms | 5,018 ms |
| Kernel dispatches | 1,235 | 1,283 |
| Profile wall span | 5,100 ms | 6,689 ms |
| Kernel % of span | **78.4%** | **75.0%** |
| Gap % of span | 21.6% | 25.0% |
| llama-bench PP tok/s | **124.4** | 100.1 |
| **GPU kernel util vs bench wall** | **97.1%** | **98.1%** |

**Top prefill kernels:**

| Rank | Kernel | Fork time (ms) | Upstream time (ms) | Fork % | Upstream % |
|------|--------|----------------|---------------------|--------|------------|
| 1 | `mul_mat_q<Q8_0,128,false>` | **3,707** | **4,696** | 92.8% | 93.6% |
| 2 | `gated_delta_net_cuda` | 100 | 88 | 2.5% | 1.8% |
| 3 | `mul_mat_q<Q8_0,128,true>` | 71 | 94 | 1.8% | 1.9% |
| 4 | `concat_non_cont` | 24 | 29 | 0.6% | 0.6% |
| 5 | `quantize_mmq_q8_1` | 21 | 27 | 0.5% | 0.5% |

**Interpretation:** Prefill is **compute-bound** — large `mul_mat_q` tiles saturate the GPU. Fork's primary win is a **21% faster `mul_mat_q`** pass (3.7 s vs 4.7 s), matching higher bench PP throughput.

**Chunked GDN caveat:** With `GGML_CUDA_GDN_CHUNKED=1` (production fork server), `llama-bench -p 512 -n 0` aborts:

```
Cannot find Symbol: gdn_chunk_cumsum_kernel<64>
```

rocprof capture under chunked mode records only 171 init dispatches before abort. Server HTTP prefill (chunked path active) measured **122.4 tok/s** vs upstream **133.4 tok/s** — upstream server is currently faster for long prompts, suggesting chunked GDN needs tuning or the missing symbol indicates a link/visibility bug.

---

## Dispatch overhead estimate

Derived from `BeginNs`/`EndNs` in `*_trace.csv`:

| Phase | Dispatches/token (decode) | Median gap | Gap share of GPU-active span | Dominant overhead source |
|-------|---------------------------|------------|------------------------------|--------------------------|
| Decode | ~125 | 4.4 µs | ~57% | Host graph replay, small-kernel launch storm |
| Prefill | N/A (single pass) | 4–65 µs (p90) | ~22–25% | Tile scheduling, attention flash-attn setup |

**Decode dispatch overhead (order-of-magnitude):**  
16k dispatches × ~4.4 µs median gap ≈ **70 ms** pure launch latency, but total gaps sum to **~1.1 s** within the 2 s GPU-active window — the remainder is longer host-side stalls between kernel bursts (MTP draft rounds, sync points).

**End-to-end decode:** GPU kernels account for only **~6%** of wall time (~0.88 s kernels vs ~14 s total for 128 tokens at ~9 tok/s). **Phase 2 should profile CPU/graph/MTP**, not more matmul tuning.

---

## Fork (50127) vs upstream (50126) comparison

| Scenario | Fork | Upstream | Delta |
|----------|------|----------|-------|
| Decode kernel time (128 tok) | 880 ms | 888 ms | −1% (noise) |
| Prefill kernel time (pp512, bench) | 3,997 ms | 5,018 ms | **−20%** |
| Prefill bench tok/s | 124.4 | 100.1 | **+24%** |
| HTTP prefill tok/s (repeat=4, server) | 122.4 | 133.4 | −8% (chunked GDN regression) |
| GDN kernel variant | `<128,8,1>` decode / `<128,4,1>` prefill | `<128,false,false>` | Fork templates differ |
| `l2_norm_f32` (upstream only) | absent in top-25 | 3.8 ms decode | fused into fork GDN path |

---

## Recommendations (Phase 2 priorities)

### P0 — Decode throughput (6% GPU util)
1. **Reduce kernel launch count** — graph capture / op fusion for repeated `mul_mat_vec_q` + `quantize_q8_1` chains; target fewer than 50 dispatches/token.
2. **MTP draft path audit** — 16k dispatches for 128 tokens suggests draft+verify multiplies launches; profile draft model separately.
3. **CPU-side profiling** — `perf` on `llama-server` during decode; graph allocator and scheduler likely dominate.

### P1 — Prefill throughput
4. **Investigate `mul_mat_q` tile=128 on gfx1151** — fork already 21% faster; explore tile 256 / wave-aware tuning for Strix Halo.
5. **Fix `gdn_chunk_cumsum_kernel` symbol** — chunked prefill crashes under rocprof and may hurt server PP vs upstream; verify HIP symbol export in `gated_delta_net_chunked.cu`.
6. **Flash-attn prefill** — only 0.3% of prefill time in `flash_attn_tile<256,256,8>`; not the lever.

### P2 — Tooling
7. **Add `libdw1` to Dockerfile** — unblock `rocprofv3 --kernel-trace` (successor to deprecated v1).
8. **Install `rocprofiler-sdk` on host** — optional; container-local profiling is sufficient if lock script used.
9. **Handle rocprof SIGSEGV** — use `--stats` output (valid before crash) or migrate to `rocprofv3` once lib fixed.

---

## rocprof-benchmark.sh fallback note

Previous `BASELINE_REPORT.md` run on **host** reported rocprof missing. Confirmed: host `/opt/rocm/bin/` has only `rocm-smi`. Container `llamacpp-qwen35-rocm-fork` ships full ROCm 7.2 with `rocprof`, `rocprofv2`, `rocprofv3`. **Use container-local profiling** (as done here) or `docker exec llamacpp-qwen35-rocm-fork rocprof ...` for kernel captures.

---

## Build references

| Backend | Build ID | Container |
|---------|----------|-----------|
| Fork | `c179ea7` (llama-bench) | `llamacpp-qwen35-rocm-fork` :50127 |
| Upstream | `e3546c7` (llama-bench) | `prompt-router-llamacpp-rocm-qwythos-9b-v2-mtp-q8` :50126 |
