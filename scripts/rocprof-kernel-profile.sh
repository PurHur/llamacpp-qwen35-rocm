#!/usr/bin/env bash
# Kernel-level rocprof capture for decode vs prefill via llama-bench (one GPU run at a time).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${1:-}" == "--inner" ]]; then
  shift
  TAG="${1:?}"
  BACKEND="${2:-fork}"
else
  TAG="${1:?usage: $0 <tag> [fork|upstream]}"
  BACKEND="${2:-fork}"
  exec "$ROOT/scripts/profile-lock.sh" -- "$0" --inner "$TAG" "$BACKEND"
fi

MODEL_PATH="/app/models/huggingface/empero-ai--Qwythos-9B-v2-GGUF/Qwythos-9B-v2-MTP-Q8_0.gguf"
MODELS_HOST="${QWEN35_ROCM_FORK_MODELS_PATH:-/home/ai/projects/prompt-router/models}"
OUT_DIR="$ROOT/var/profiles/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

case "$BACKEND" in
  fork)
    IMAGE="llamacpp-qwen35-rocm:latest"
    CONTAINER="llamacpp-qwen35-rocm-fork"
    HSA_OVERRIDE="${QWEN35_ROCM_FORK_HSA_OVERRIDE:-11.5.1}"
    HEALTH_URL="http://127.0.0.1:50127/health"
    ;;
  upstream)
    IMAGE="prompt-router-llama-server-rocm-qwythos:latest"
    CONTAINER="prompt-router-llamacpp-rocm-qwythos-9b-v2-mtp-q8"
    HSA_OVERRIDE="${QWEN35_ROCM_UPSTREAM_HSA_OVERRIDE:-11.5.1}"
    HEALTH_URL="http://127.0.0.1:50126/health"
    ;;
  *)
    echo "unknown backend: $BACKEND (fork|upstream)" >&2
    exit 1
    ;;
esac

run_profile() {
  local workload="$1"
  local n_prompt="$2"
  local n_gen="$3"
  local gdn_chunked="${4:-}"
  local out_base="$OUT_DIR/${workload}"

  echo "=== rocprof ${workload} backend=${BACKEND} tag=${TAG} gdn_chunked=${gdn_chunked:-default} ==="

  docker stop "$CONTAINER" >/dev/null 2>&1 || true

  local -a gdn_env=()
  if [[ -n "$gdn_chunked" ]]; then
    gdn_env=(-e "GGML_CUDA_GDN_CHUNKED=$gdn_chunked")
  else
    gdn_env=(-e "GGML_CUDA_GDN_CHUNKED=${GGML_CUDA_GDN_CHUNKED:-1}" -e "GGML_CUDA_GDN_CHUNK_THRESHOLD=${GGML_CUDA_GDN_CHUNK_THRESHOLD:-16}")
  fi

  docker run --rm \
    --entrypoint bash \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add 992 \
    --group-add 44 \
    -v "${MODELS_HOST}:/app/models:ro" \
    -v "$OUT_DIR:/out" \
    -e HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE" \
    -e ROCM_PATH=/opt/rocm \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
    "${gdn_env[@]}" \
    "$IMAGE" -lc "
      set -euo pipefail
      OUT_BASE='/out/${workload}'
      BENCH=/app/build/bin/llama-bench
      ROCM=/opt/rocm/bin/rocprof
      bench_rc=0
      if command -v \"\$ROCM\" >/dev/null 2>&1; then
        echo 'Using rocprof --stats'
        \"\$ROCM\" --stats --timestamp on -o \"\${OUT_BASE}_trace.csv\" \
          \"\$BENCH\" -m '$MODEL_PATH' -r 1 --no-warmup -ngl -1 -fa on -b 2048 -t 8 -p $n_prompt -n $n_gen \
          > \"\${OUT_BASE}_bench.log\" 2>&1 || bench_rc=\$?
      else
        \"\$BENCH\" -m '$MODEL_PATH' -r 1 --no-warmup -ngl -1 -fa on -b 2048 -t 8 -p $n_prompt -n $n_gen \
          > \"\${OUT_BASE}_bench.log\" 2>&1 || bench_rc=\$?
      fi
      if test -f \"\${OUT_BASE}_trace.stats.csv\"; then
        python3 - <<'PY' \"\${OUT_BASE}_trace.stats.csv\" > \"\${OUT_BASE}_top_kernels.txt\"
import csv, sys
rows = []
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        try:
            rows.append({
                'name': row['Name'],
                'calls': int(row['Calls']),
                'total_ns': int(row['TotalDurationNs']),
                'avg_ns': int(row['AverageNs']),
                'pct': float(row['Percentage']),
            })
        except (KeyError, ValueError):
            pass
rows.sort(key=lambda r: r['total_ns'], reverse=True)
total_ns = sum(r['total_ns'] for r in rows)
calls = sum(r['calls'] for r in rows)
print(f'total_kernel_time_ns={total_ns}')
print(f'total_kernel_dispatches={calls}')
print('rank,calls,total_ms,avg_us,pct,name')
for i, r in enumerate(rows[:25], 1):
    print(f\"{i},{r['calls']},{r['total_ns']/1e6:.3f},{r['avg_ns']/1e3:.1f},{r['pct']:.2f},{r['name']}\")
PY
        cat \"\${OUT_BASE}_top_kernels.txt\"
      fi
      echo \"bench_exit_code=\$bench_rc\" | tee \"\${OUT_BASE}_meta.txt\"
    "

  docker start "$CONTAINER" >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    curl -sf "$HEALTH_URL" >/dev/null 2>&1 && break
    sleep 2
  done
}

{
  echo "Profile dir: $OUT_DIR"
  run_profile decode 0 128
  # Fork chunked GDN prefill crashes llama-bench under rocprof (missing gdn_chunk_cumsum symbol).
  if [[ "$BACKEND" == "fork" ]]; then
    run_profile prefill 512 0 0
    run_profile prefill_chunked 512 0 1 || true
  else
    run_profile prefill 512 0
  fi
  echo "Done: $OUT_DIR"
} 2>&1 | tee "$OUT_DIR/run.log"

echo "$OUT_DIR"
