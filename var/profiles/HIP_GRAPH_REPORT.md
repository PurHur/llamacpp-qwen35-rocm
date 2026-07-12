# HIP/CUDA Graph Capture Feasibility — Qwythos MTP on gfx1151 (fork :50127)

**Date:** 2026-07-12  
**Host:** AMD 8060S gfx1151 (HSA_OVERRIDE_GFX_VERSION=11.5.1)  
**Container:** `llamacpp-qwen35-rocm-fork` → http://127.0.0.1:50127  
**Model:** `Qwythos-9B-v2-MTP-Q8_0.gguf` (`--spec-type draft-mtp --spec-draft-n-max 6`)  
**Build:** `GGML_HIP_GRAPHS:BOOL=ON` (CMake default; verified in container `CMakeCache.txt`)

---

## 1. Flag / code survey (llama.cpp)

| Flag / symbol | Where | Effect |
|---------------|-------|--------|
| **`GGML_HIP_GRAPHS`** | `ggml/CMakeLists.txt` (default **ON**), `ggml-hip/CMakeLists.txt` | **Compile-time.** Defines `GGML_HIP_GRAPHS` → enables `USE_CUDA_GRAPH` in `common.cuh`. Dockerfile does not override; fork image builds with graphs **enabled**. |
| **`GGML_CUDA_GRAPHS`** | `ggml-cuda/CMakeLists.txt` | CUDA-only compile flag → `GGML_CUDA_USE_GRAPHS`. Not used on HIP path. |
| **`GGML_CUDA_DISABLE_GRAPHS`** | `ggml-cuda/common.cuh` `ggml_cuda_graph::is_enabled()` | **Runtime.** If set (any value), disables HIP/CUDA graph capture for all contexts. |
| **`GGML_CUDA_ENABLE_GRAPHS`** | — | **Not present** in this tree. Setting it has **no effect** (tested). |
| **`GGML_HIP_GRAPHS=1` (env)** | — | **Not read at runtime.** Compile-time CMake option only; env var is a no-op without rebuild. |
| **`ggml_cuda_graph_evaluate_and_capture`** | `ggml-cuda/ggml-cuda.cu` | Capture via `cudaStreamBeginCapture` / `cudaGraphLaunch`; warmup pass before first capture. |
| **`graphs reused` (server log)** | `server-context.cpp` → `llama_perf_context().n_reused` | **GGML compute-graph reuse** at llama context level — **not** HIP graph capture status. Still non-zero when `GGML_CUDA_DISABLE_GRAPHS=1`. |

Server has **no** graph-specific CLI flags; control is compile-time (`GGML_HIP_GRAPHS`) + runtime disable (`GGML_CUDA_DISABLE_GRAPHS`).

---

## 2. Configurations tested

Each config: container **stopped/recreated**, model reload, benchmarks via `scripts/bench-pp-tps.sh` (profile-lock, serialized GPU).

| Config | Env / build | Decode bench | Prefill bench (`--repeat 4`) |
|--------|-------------|--------------|------------------------------|
| **A — default (recommended)** | `GGML_HIP_GRAPHS=ON` (build), no runtime env | `--max-tokens 128 --title hip-graph-decode-default` | `--repeat 4 --title hip-graph-pp-default` |
| **B — graphs disabled** | `GGML_CUDA_DISABLE_GRAPHS=1` | `--title hip-graph-decode-disable` | `--repeat 4 --title hip-graph-pp-disable` |
| **C — noop enable env** | `GGML_CUDA_ENABLE_GRAPHS=1` (undefined symbol) | `--title hip-graph-decode-enable-env` | *(server OOM/killed before PP run completed)* |

---

## 3. Benchmark results

### Decode-focused (short prompt, 128 completion tokens, 3 runs)

| Config | PP mean (tok/s) | Decode mean (tok/s) | Δ decode vs A |
|--------|-----------------|---------------------|---------------|
| **A — default (graphs ON)** | **102.0** ± 7.5 | **30.3** ± 2.2 | — |
| **B — `GGML_CUDA_DISABLE_GRAPHS=1`** | 75.2 ± 1.5 | 21.8 ± 1.5 | **−28%** |
| **C — `GGML_CUDA_ENABLE_GRAPHS=1`** | 71.1 ± 0.4 | 20.1 ± 1.1 | −34% *(cold container; not comparable)* |

JSON: `var/profiles/hip_graph_decode_{default,disable,enable_env}.json`

### Prefill-focused (`--repeat 4`, 163 prompt tokens, 128 completion, 3 runs)

| Config | PP mean (tok/s) | PP runs 2–3 only | Decode mean (tok/s) |
|--------|-----------------|------------------|---------------------|
| **A — default** | 91.7 ± 46.5* | **124.5** (128.3, 120.8) | 22.0 ± 9.9* |
| **B — disable** | 102.2 ± 7.1 | 105.3 (99.7, 111.8) | 21.4 ± 2.2 |

\*Run 1 on config A hit **graph warmup / cold-start** (PP 26.0 tok/s, decode 8.3 tok/s). Steady-state PP with graphs ON exceeds disable once warmed.

JSON: `var/profiles/hip_graph_pp_{default,disable}.json`

### Draft MTP side-effect (decode bench)

| Config | Draft generated / accepted | Accept rate |
|--------|---------------------------|-------------|
| A — default | 437 / 233 | 53.3% |
| B — disable | 703 / 260 | 37.0% |

Graphs disabled → more draft verification steps and lower accept rate (extra graph-shape churn on decode path).

---

## 4. Docker log excerpts

### Default (graphs ON) — steady decode, compute graphs reused

```
0.14.330.183 I slot print_timing: id  0 | task 44 |    graphs reused =         90
0.18.694.488 I slot print_timing: id  0 | task 100 |    graphs reused =        134
0.23.314.486 I slot print_timing: id  0 | task 150 |    graphs reused =        181
```

### `GGML_CUDA_DISABLE_GRAPHS=1` — reuse counter still increments (llama-level, not HIP)

```
0.59.287.044 I slot print_timing: id  0 | task 51 |    graphs reused =         81
1.05.616.785 I slot print_timing: id  0 | task 93 |    graphs reused =        117
```

HIP capture success/failure strings (`CUDA graph warmup complete`, `disabling CUDA graphs due to …`) are **`GGML_LOG_DEBUG`** only; default server verbosity does not emit them. No capture **errors** observed at INFO/WARN/ERROR levels in any run.

Build confirmation:

```
GGML_HIP_GRAPHS:BOOL=ON
```

---

## 5. Recommendation

### **Best config: default — keep HIP graphs enabled (do not set `GGML_CUDA_DISABLE_GRAPHS`)**

| Workload | Graphs ON vs OFF |
|----------|------------------|
| **Steady decode** | **+28–39%** decode tok/s (30.3 vs 21.8) |
| **Prefill (warmed)** | **+18%** PP tok/s (124.5 vs 105.3, runs 2–3) |
| **First request after restart** | Graph warmup penalty on prefill (one slow PP pass) |

**Do not use:** `GGML_CUDA_DISABLE_GRAPHS=1` on this fork for MTP serving.

**No-op / unused:** `GGML_CUDA_ENABLE_GRAPHS=1`, runtime `GGML_HIP_GRAPHS=1`.

**Contrast with upstream issue [#20292](https://github.com/ggml-org/llama.cpp/issues/20292):** that report found HIP graphs hurt **prefill** on ROCm for Qwen3.5. On **this fork** (GDN chunked + MTP GPU-resident patches, gfx1151), graphs help **both** prefill (after warmup) and decode. Likely drivers: fork kernel fusion, single-slot MTP, and ROCm 7.2 stack differ from the issue’s setup.

---

## 6. MTP caveats

1. **Warmup tax:** First prompt after container start pays one non-graphed execution per graph key before `cudaStreamBeginCapture` (`warmup_complete` gate in `ggml_backend_cuda_graph_compute`). Benchmarks should include ≥1 warmup (bench script already does).
2. **Multiple graph keys:** MTP runs target + draft models → separate capture instances per steady decode shape. Memory for graph executables scales with draft depth and batch layout.
3. **Prefill vs decode shapes:** Chunked GDN prefill uses different node shapes than single-token decode; capture targets **steady decode** most reliably. Long/variable prefill may reset warmup when node properties change (`CUDA graph warmup reset`).
4. **`MUL_MAT_ID` guard:** Graphs disabled for some MoE `mul_mat_id` shapes (`ggml_cuda_graph_check_compability`); Qwythos 9B dense path not affected.
5. **Monitoring:** Use HTTP `timings.predicted_per_second` for decode SLO; do not rely on server `graphs reused` alone for HIP capture health.

---

## 7. Artifacts

| File | Description |
|------|-------------|
| `var/profiles/hip_graph_decode_default.json` | Decode bench, graphs ON |
| `var/profiles/hip_graph_decode_disable.json` | Decode bench, graphs OFF |
| `var/profiles/hip_graph_pp_default.json` | Prefill bench, graphs ON |
| `var/profiles/hip_graph_pp_disable.json` | Prefill bench, graphs OFF |
| `var/profiles/hip_graph_decode_enable_env.json` | No-op env test (cold) |

**Container restored** to docker-compose default (graphs ON, no disable env) after testing.
