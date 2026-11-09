#!/usr/bin/env bash
# Fixed-seed screens for prefill latency, decode speed, and mixed traffic.
set -euo pipefail

PROFILE="${PROFILE:-baseline}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MODEL="${BENCH_MODEL:-orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
TOKENIZER="${BENCH_TOKENIZER:-${LLM_MODEL:-$MODEL}}"
RESULT_DIR="${RESULT_DIR:-results/tuning/$PROFILE}"
[[ "$PROFILE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid PROFILE' >&2; exit 2; }
[ -n "${API_KEY:-}" ] || { echo 'API_KEY must be set' >&2; exit 1; }
export OPENAI_API_KEY="$API_KEY"
mkdir -p "$RESULT_DIR"

run_case() {
  local name="$1" input="$2" output="$3" concurrency="$4" prompts="$5"
  vllm bench serve --backend openai-chat --base-url "$BASE_URL" \
    --endpoint /v1/chat/completions --model "$MODEL" --tokenizer "$TOKENIZER" \
    --trust-remote-code --dataset-name random --seed 42 \
    --input-len "$input" --output-len "$output" --ignore-eos \
    --max-concurrency "$concurrency" --num-prompts "$prompts" \
    --num-warmups 4 --save-result --result-dir "$RESULT_DIR" \
    --result-filename "$name.json"
}

run_case prefill8k 8192 1 1 8
run_case prefill32k 32768 1 1 4
run_case decode2k 2048 1024 1 4
run_case mixed8k 8192 512 8 16
