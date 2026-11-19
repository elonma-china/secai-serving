#!/usr/bin/env bash
# Sample SM clock / power / throttle reasons on GPU0 during a prefill window and a decode window.
# Output: /workspace/results/clocks/{prefill,decode}.csv   Usage: API_KEY=.. HF_TOKEN=.. bash clocks.sh
set -uo pipefail
M=secai1999/LLM-Autoscan-v1; PORT=8023; OUT=/workspace/results/clocks
export HF_HOME=/workspace/hf-cache HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY="$API_KEY" VLLM_API_KEY="$API_KEY"
mkdir -p "$OUT"; rm -f "$OUT"/*
vllm serve "$M" --host 127.0.0.1 --port $PORT --trust-remote-code --gpu-memory-utilization 0.90 --max-model-len 131072 --max-num-seqs 64 \
  --enable-prefix-caching --language-model-only --max-num-batched-tokens 16384 >"$OUT/server.log" 2>&1 </dev/null & SP=$!
for i in $(seq 1 160); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 5; done
b() { vllm bench serve --backend openai-chat --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --model "$M" --tokenizer "$M" --trust-remote-code \
  --dataset-name random --seed "$1" --input-len "$2" --output-len "$3" --ignore-eos --max-concurrency "$4" --num-prompts "$5" --num-warmups 0 >>"$OUT/bench.log" 2>&1; }
Q=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown
b 1 8192 1 1 2
for w in "prefill 32768 1 1 4" "decode 2048 512 32 32"; do set -- $w
  nvidia-smi -i 0 --query-gpu=$Q --format=csv -lms 500 >"$OUT/$1.csv" & NP=$!
  b 7 "$2" "$3" "$4" "$5"; kill $NP; sleep 1
done
kill $SP; sleep 10; echo CLOCKS_DONE
