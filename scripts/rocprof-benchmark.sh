#!/usr/bin/env bash
# rocprof capture for llama-server decode — serialized via profile-lock (one GPU run at a time).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${QWEN35_BENCH_URL:-http://127.0.0.1:50126}"
MODEL="${QWEN35_BENCH_MODEL:-Qwythos-9B-v2-MTP-Q8_0.gguf}"
TAG="${1:-baseline}"
OUT_DIR="$ROOT/var/profiles/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

exec "$ROOT/scripts/profile-lock.sh" -- bash -lc "
  set -euo pipefail
  echo 'rocprof run tag=$TAG url=$URL out=$OUT_DIR'
  if ! command -v rocprof >/dev/null 2>&1; then
    echo 'rocprof not installed; running HTTP benchmark only' >&2
    python3 '$ROOT/bench/profile_one_at_a_time.py' --url '$URL' --model '$MODEL' \
      --title 'rocprof-fallback-$TAG' --runs 2 --warmup 1 \
      --json-out '$OUT_DIR/bench.json'
    exit 0
  fi
  rocprof --stats --timestamp on -o '$OUT_DIR/trace.csv' \
    python3 '$ROOT/bench/profile_one_at_a_time.py' --url '$URL' --model '$MODEL' \
      --title 'rocprof-$TAG' --runs 2 --warmup 1 --json-out '$OUT_DIR/bench.json'
  echo 'Profile artifacts in $OUT_DIR'
"
