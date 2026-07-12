#!/bin/bash
# Entrypoint for native llama.cpp server (C++ binary). OpenAI-compatible API.
set -e
export LD_LIBRARY_PATH="/app/build/bin:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"
HOST="${LLAMACPP_HOST:-0.0.0.0}"
PORT="${LLAMACPP_PORT:-8000}"
MODEL="${LLAMACPP_MODEL:-}"
N_CTX="${LLAMACPP_N_CTX:-16384}"
N_BATCH="${LLAMACPP_N_BATCH:-2048}"
N_THREADS="${LLAMACPP_N_THREADS:-8}"
N_GPU_LAYERS="${LLAMACPP_N_GPU_LAYERS:--1}"
N_CPU_MOE="${LLAMACPP_N_CPU_MOE:-0}"
FLASH_ATTN="${LLAMACPP_FLASH_ATTN:-on}"
CACHE_TYPE_K="${LLAMACPP_CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${LLAMACPP_CACHE_TYPE_V:-f16}"
THREADS_BATCH="${LLAMACPP_THREADS_BATCH:-16}"
POLL="${LLAMACPP_POLL:-0}"
NO_ESCAPE="${LLAMACPP_NO_ESCAPE:-1}"
EXTRA_ARGS_STR="${LLAMACPP_EXTRA_ARGS:-}"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "LLAMACPP_MODEL must point to a GGUF on disk"
  echo "Missing: ${MODEL:-<unset>}"
  sleep infinity
  exit 1
fi

ARGS=( --model "$MODEL" --host "$HOST" --port "$PORT" -ngl "$N_GPU_LAYERS" -c "$N_CTX" -b "$N_BATCH" -t "$N_THREADS" -ncmoe "$N_CPU_MOE" --flash-attn "$FLASH_ATTN" -ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V" --threads-batch "$THREADS_BATCH" --poll "$POLL" )
[ "$NO_ESCAPE" = "1" ] && ARGS+=(--no-escape)
if [ -n "$EXTRA_ARGS_STR" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $EXTRA_ARGS_STR )
  ARGS+=("${EXTRA_ARGS[@]}")
fi
exec /app/build/bin/llama-server "${ARGS[@]}"
