#!/usr/bin/env bash
# A/B the SM120 FP8 prefill patch on GPU0 (stop llm-a first). Same server twice: VLLM_SM120_FP8_PREFILL=0 then =1.
# Output: /workspace/results/patchab/{off,on}.{needle,p32k,p120k}.log + summary lines.   Needs API_KEY, HF_TOKEN.
set -uo pipefail
M=secai1999/LLM-Autoscan-v1; PORT=8031; OUT=/workspace/results/patchab
export HF_HOME=/workspace/hf-cache HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY="$API_KEY" VLLM_API_KEY="$API_KEY"
mkdir -p "$OUT"; [ -n "${MODES:-}" ] || rm -f "$OUT"/*
python3 /workspace/secai/stack/patches/sm120_fp8_prefill.py apply
b() { vllm bench serve --backend openai-chat --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --model "$M" --tokenizer "$M" --trust-remote-code \
  --dataset-name random --seed 42 --input-len "$1" --output-len 1 --ignore-eos --max-concurrency 1 --num-prompts "$2" --num-warmups "$3" >"$4" 2>&1; }
for mode in ${MODES:-on off}; do
  flag=$([ $mode = on ] && echo 1 || echo 0)
  VLLM_SM120_FP8_PREFILL=$flag vllm serve "$M" --host 127.0.0.1 --port $PORT --trust-remote-code --gpu-memory-utilization 0.90 --max-model-len 131072 \
    --max-num-seqs 64 --enable-prefix-caching --language-model-only --max-num-batched-tokens 16384 >"$OUT/$mode.server.log" 2>&1 </dev/null & SP=$!
  up=0; for i in $(seq 1 160); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 5; done
  [ $up = 1 ] || { echo "$mode: server failed: $(grep -E 'Error|error' "$OUT/$mode.server.log" | tail -2 | cut -c1-240)" | tee -a "$OUT/summary.txt"; kill $SP 2>/dev/null; continue; }
  python3 /workspace/secai/stack/needle.py --url "http://127.0.0.1:$PORT" --model "$M" --words 21000 --depths 0.05,0.5,0.95 >"$OUT/$mode.needle.log" 2>&1
  b 32768 4 2 "$OUT/$mode.p32k.log"; b 120000 3 1 "$OUT/$mode.p120k.log"
  { echo "== $mode"; tail -1 "$OUT/$mode.needle.log"; grep -h "prompt_tokens" "$OUT/$mode.needle.log" | head -3 | cut -c1-120
    for t in p32k p120k; do echo -n "$t: "; grep -h "Median TTFT" "$OUT/$mode.$t.log"; done
    grep -h -c "fp8_paged\|sm120_fmha\|FMHA" "$OUT/$mode.server.log" | sed 's/^/fmha log lines: /'; } | tee -a "$OUT/summary.txt"
  kill $SP; sleep 20
done
echo PATCHAB_DONE
