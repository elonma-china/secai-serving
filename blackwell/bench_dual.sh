#!/usr/bin/env bash
# Run the same fixed-seed workload on both PCIe pairs at once.
set -euo pipefail

PROFILE="${PROFILE:-dual}"
RESULT_DIR="${RESULT_DIR:-results/dual/$PROFILE}"
MODEL="${BENCH_MODEL:-orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
TOKENIZER="${BENCH_TOKENIZER:-${LLM_MODEL:-$MODEL}}"
[[ "$PROFILE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid PROFILE' >&2; exit 2; }
[ -n "${API_KEY:-}" ] || { echo 'API_KEY must be set' >&2; exit 1; }
export OPENAI_API_KEY="$API_KEY"
mkdir -p "$RESULT_DIR"

run_one() {
  local case_name="$1" port="$2" input="$3" concurrency="$4" prompts="$5"
  vllm bench serve --backend openai-chat \
    --base-url "http://127.0.0.1:$port" --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer "$TOKENIZER" --trust-remote-code \
    --dataset-name random --seed 42 --input-len "$input" --output-len 512 \
    --ignore-eos --max-concurrency "$concurrency" --num-prompts "$prompts" \
    --num-warmups 4 --save-result --result-dir "$RESULT_DIR" \
    --result-filename "${case_name}-${port}.json" >"$RESULT_DIR/${case_name}-${port}.log" 2>&1
}

run_pair() {
  local case_name="$1" input="$2" concurrency="$3" prompts="$4" a b
  run_one "$case_name" 8001 "$input" "$concurrency" "$prompts" & a=$!
  run_one "$case_name" 8003 "$input" "$concurrency" "$prompts" & b=$!
  wait "$a"
  wait "$b"
  python3 - "$RESULT_DIR" "$case_name" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
name = sys.argv[2]
rows = [json.loads((directory / f"{name}-{port}.json").read_text()) for port in (8001, 8003)]
for port, row in zip((8001, 8003), rows):
    print(f"{name} port={port}: {row['output_throughput']:.1f} output tok/s; "
          f"duration={row['duration']:.2f}s; p99 TTFT={row['p99_ttft_ms']:.0f}ms")
combined = sum(row['total_output_tokens'] for row in rows) / max(row['duration'] for row in rows)
print(f"{name} approximate combined output tok/s: {combined:.1f}")
PY
}

run_pair mixed8k 8192 8 16
run_pair long32k 32768 4 8
