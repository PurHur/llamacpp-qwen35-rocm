#!/usr/bin/env bash
# PP/TPS benchmark with exclusive GPU lock (one run at a time).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/profile-lock.sh" -- \
  python3 "$ROOT/bench/profile_one_at_a_time.py" "$@"
