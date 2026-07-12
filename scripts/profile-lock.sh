#!/usr/bin/env bash
# Exclusive GPU profiling lock — only one benchmark/profile run at a time machine-wide.
set -euo pipefail

LOCK_FILE="${LLAMACPP_PROFILE_LOCK_FILE:-/tmp/llamacpp-gpu-profile.lock}"
LOCK_FD=200

exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  echo "ERROR: another llama.cpp GPU profile/benchmark is running (lock: $LOCK_FILE)." >&2
  echo "Wait for it to finish or remove stale lock if the holder crashed." >&2
  exit 1
fi

echo "$$" > "${LOCK_FILE}.pid"
trap 'rm -f "${LOCK_FILE}.pid"' EXIT

if [[ $# -eq 0 ]]; then
  echo "Acquired profile lock (pid $$). Run your command via: $0 -- <cmd...>"
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

exec "$@"
