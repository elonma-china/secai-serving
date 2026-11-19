#!/usr/bin/env bash
# Sequential single-GPU (GPU0) profile sweep at 131K context. Results: /workspace/results/sweep/<profile>/
# Usage: API_KEY=.. HF_TOKEN=.. bash sweep.sh [profile ...]     (default: all)
set -uo pipefail
M=secai1999/LLM-Autoscan-v1; PORT=8021; OUT=/workspace/results/sweep; ROOT=/workspace/secai
export HF_HOME=/workspace/hf-cache HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0
export VLLM_API_KEY="$API_KEY" OPENAI_API_KEY="$API_KEY"
declare -A EXTRA=(
  [base]=""
  [mtp1]="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":1}"
  [mtp2]="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":2}"
  [batch16k]="--max-num-batched-tokens 16384"
  [nvfp4kv]="--kv-cache-dtype nvfp4"
)
PROFILES=("$@"); [ ${#PROFILES[@]} -gt 0 ] || PROFILES=(base mtp1 mtp2 batch16k nvfp4kv)
mkdir -p "$OUT"
stop_server() { [ -f "$OUT/pid" ] && kill "$(cat "$OUT/pid")" 2>/dev/null; sleep 15; rm -f "$OUT/pid"; }
case_run() { # name in out conc prompts
  vllm bench serve --backend openai-chat --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
    --model "$M" --tokenizer "$M" --trust-remote-code --dataset-name random --seed 42 --input-len "$2" --output-len "$3" \
    --ignore-eos --max-concurrency "$4" --num-prompts "$5" --num-warmups 2 --save-result --result-dir "$D" --result-filename "$1.json" \
    >"$D/$1.log" 2>&1
}
for P in "${PROFILES[@]}"; do
  D="$OUT/$P"; rm -rf "$D"; mkdir -p "$D"; echo "=== $P $(date +%T)"
  # shellcheck disable=SC2086
  nohup setsid vllm serve "$M" --host 127.0.0.1 --port $PORT --trust-remote-code --gpu-memory-utilization 0.90 \
    --max-model-len 131072 --max-num-seqs 64 --enable-prefix-caching --language-model-only ${EXTRA[$P]} \
    >"$D/server.log" 2>&1 </dev/null & echo $! >"$OUT/pid"
  up=0; for i in $(seq 1 120); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$(cat "$OUT/pid")" 2>/dev/null || break; sleep 5; done
  if [ $up = 0 ]; then echo "$P: FAILED TO START: $(grep -h -E 'Error|error' "$D/server.log" | tail -2 | cut -c1-260)" | tee "$D/summary.txt"; stop_server; continue; fi
  grep -h "GPU KV cache size" "$D/server.log" | sed 's/.*GPU KV/GPU KV/' | tee "$D/summary.txt"
  case_run prefill32k 32768 1 1 4;  case_run prefill120k 120000 1 1 3
  case_run decode2k 2048 1024 1 4;  case_run mixed8k_c8 8192 512 8 16;  case_run mixed8k_c32 8192 512 32 48
  python3 "$ROOT/stack/needle.py" --url "http://127.0.0.1:$PORT" --model "$M" --words 21000 --depths 0.05,0.5,0.95 >"$D/needle.log" 2>&1
  python3 - "$D" >>"$D/summary.txt" <<'PY'
import json, sys, os
d = sys.argv[1]
for n in ("prefill32k", "prefill120k", "decode2k", "mixed8k_c8", "mixed8k_c32"):
    try: r = json.load(open(f"{d}/{n}.json"))
    except Exception: print(f"{n}: NO RESULT"); continue
    print(f"{n}: ttft_p50 {r.get('median_ttft_ms',0):.0f}ms | tpot_p50 {r.get('median_tpot_ms',0):.1f}ms | out {r.get('output_throughput',0):.0f} tok/s | done {r.get('completed')}/{r.get('completed',0)+r.get('failed',0) if 'failed' in r else '?'}")
try: print(open(f"{d}/needle.log").read().strip().splitlines()[-1])
except Exception: print("needle: NO RESULT")
PY
  cat "$D/summary.txt"; stop_server
done
echo SWEEP_DONE
