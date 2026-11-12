#!/usr/bin/env bash
# Identical request matrix for TP/EP and PLE CPU/GPU comparisons.
set -euo pipefail
DRY=0
if [ "${1:-}" = "--dry-run" ] && [ "$#" -eq 1 ]; then DRY=1
elif [ "$#" -ne 0 ]; then echo 'Usage: bench.sh [--dry-run]' >&2; exit 2
fi
PROFILE="${PROFILE:-tp4_ep4_ple_gpu}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${BENCH_MODEL:-${LLM_SERVED_MODEL_NAME:-orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4}}"
TOKENIZER="${BENCH_TOKENIZER:-${LLM_MODEL:-$MODEL}}"
RESULT_DIR="${RESULT_DIR:-results/$PROFILE}"
BENCH_MODE="${BENCH_MODE:-full}"
[[ "$PROFILE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid PROFILE' >&2; exit 2; }
case "$BENCH_MODE" in quick|full) ;; *) echo 'BENCH_MODE must be quick or full' >&2; exit 2;; esac
[ "$DRY" = 1 ] || [ -n "${API_KEY:-}" ] || { echo 'API_KEY must be set' >&2; exit 1; }
export OPENAI_API_KEY="${API_KEY:-dry-run-placeholder}"

run_case() {
  local input="$1" output="$2" concurrency="$3" prompts="$4"
  local filename="${PROFILE}-in${input}-out${output}-c${concurrency}.json"
  local args=(bench serve --backend openai-chat --base-url "$BASE_URL"
    --endpoint /v1/chat/completions --model "$MODEL" --tokenizer "$TOKENIZER"
    --trust-remote-code --dataset-name random
    --input-len "$input" --output-len "$output" --ignore-eos
    --max-concurrency "$concurrency" --num-prompts "$prompts"
    --num-warmups 4 --save-result --result-dir "$RESULT_DIR"
    --result-filename "$filename")
  if [ "$DRY" = 1 ]; then
    printf 'vllm'; printf ' %q' "${args[@]}"; printf '\n'
  else
    vllm "${args[@]}"
  fi
}

[ "$DRY" = 1 ] || mkdir -p "$RESULT_DIR"
if [ "$BENCH_MODE" = quick ]; then
  run_case 2048 256 1 8
  run_case 8192 512 16 32
  run_case 32768 512 8 16
  run_case 126976 256 1 2
else
  for spec in '2048 512' '8192 1024' '32768 1024'; do
    read -r input output <<<"$spec"
    for concurrency in 1 8 32; do
      prompts=$((concurrency * 4))
      [ "$prompts" -ge 32 ] || prompts=32
      run_case "$input" "$output" "$concurrency" "$prompts"
    done
  done
  # Keep total request length below the default 131072-token max model length.
  for concurrency in 1 4 8; do
    prompts=$((concurrency * 4))
    [ "$prompts" -ge 32 ] || prompts=32
    run_case 126976 2048 "$concurrency" "$prompts"
  done
fi
