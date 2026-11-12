#!/usr/bin/env bash
# Same 8K/512 workload on both replicas; sweep simultaneous client requests.
set -euo pipefail

PROFILE="${PROFILE:-ccu}"
RESULT_DIR="${RESULT_DIR:-results/ccu/$PROFILE}"
MODEL="${BENCH_MODEL:-orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
TOKENIZER="${BENCH_TOKENIZER:-${LLM_MODEL:-$MODEL}}"
LEVELS="${CCU_LEVELS:-8 16 32 64}"
[[ "$PROFILE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid PROFILE' >&2; exit 2; }
[ -n "${API_KEY:-}" ] || { echo 'API_KEY must be set' >&2; exit 1; }
export OPENAI_API_KEY="$API_KEY"
mkdir -p "$RESULT_DIR"

run_one() {
  local port="$1" concurrency="$2" prompts="$3"
  vllm bench serve --backend openai-chat \
    --base-url "http://127.0.0.1:$port" --endpoint /v1/chat/completions \
    --model "$MODEL" --tokenizer "$TOKENIZER" --trust-remote-code \
    --dataset-name random --seed "$((42000 + concurrency))" \
    --input-len 8192 --output-len 512 \
    --ignore-eos --max-concurrency "$concurrency" --num-prompts "$prompts" \
    --num-warmups 4 --save-result --result-dir "$RESULT_DIR" \
    --result-filename "c${concurrency}-${port}.json" >"$RESULT_DIR/c${concurrency}-${port}.log" 2>&1
}

for concurrency in $LEVELS; do
  [[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CCU: $concurrency" >&2; exit 2; }
  prompts=$((2 * concurrency))
  run_one 8001 "$concurrency" "$prompts" & a=$!
  run_one 8003 "$concurrency" "$prompts" & b=$!
  wait "$a"
  wait "$b"
  python3 - "$RESULT_DIR" "$concurrency" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
concurrency = int(sys.argv[2])
rows = [json.loads((directory / f"c{concurrency}-{port}.json").read_text()) for port in (8001, 8003)]
combined = sum(row['total_output_tokens'] for row in rows) / max(row['duration'] for row in rows)
for port, row in zip((8001, 8003), rows):
    print(f"C{concurrency} port={port}: output={row['output_throughput']:.1f} tok/s, "
          f"TTFT p50/p99={row['median_ttft_ms']:.0f}/{row['p99_ttft_ms']:.0f}ms, "
          f"TPOT p50/p99={row['median_tpot_ms']:.1f}/{row['p99_tpot_ms']:.1f}ms")
print(f"C{2 * concurrency} total: approximate output={combined:.1f} tok/s")
PY
done
