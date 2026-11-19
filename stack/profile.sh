#!/usr/bin/env bash
# Nsight Systems profile of one LLM replica on GPU0: a prefill window and a decode window, each reduced to a
# per-kernel GPU time table (cuda_gpu_kern_sum). Output: /workspace/results/nsys/{prefill,decode}.{nsys-rep,csv}
# Usage: API_KEY=.. HF_TOKEN=.. bash profile.sh
set -uo pipefail
M=secai1999/LLM-Autoscan-v1; PORT=8021; OUT=/workspace/results/nsys; NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys
export HF_HOME=/workspace/hf-cache HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY="$API_KEY" VLLM_API_KEY="$API_KEY"
SIZES="1 2 4 8 16 24 32 40 48 56 64 128 192 256 320 384 448 512 1024 2048 4096 8192 12288 16384"
mkdir -p "$OUT"; rm -f "$OUT"/*
$NSYS launch --session-new=vl --trace=cuda,nvtx --cuda-graph-trace=node \
  vllm serve "$M" --host 127.0.0.1 --port $PORT --trust-remote-code --gpu-memory-utilization 0.90 --max-model-len 131072 \
  --max-num-seqs 64 --enable-prefix-caching --language-model-only --max-num-batched-tokens 16384 --cudagraph-capture-sizes $SIZES \
  >"$OUT/server.log" 2>&1 </dev/null &
for i in $(seq 1 160); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 5; done
bench() { vllm bench serve --backend openai-chat --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --model "$M" --tokenizer "$M" \
  --trust-remote-code --dataset-name random --seed "$1" --input-len "$2" --output-len "$3" --ignore-eos --max-concurrency "$4" --num-prompts "$5" --num-warmups "$6" >>"$OUT/bench.log" 2>&1; }
window() { # name seed in out conc prompts
  bench 1 "$3" "$4" "$5" 4 2
  $NSYS start --session=vl -o "$OUT/$1" --force-overwrite=true >/dev/null
  bench "$2" "$3" "$4" "$5" "$6" 0
  $NSYS stop --session=vl >/dev/null
  $NSYS stats --report cuda_gpu_kern_sum --format csv "$OUT/$1.nsys-rep" >"$OUT/$1.csv" 2>"$OUT/$1.stats.err"
}
window prefill 11 32768 1 1 3
window decode 22 2048 256 32 32
$NSYS shutdown --session=vl >/dev/null 2>&1; sleep 5; pkill -f "port $PORT" 2>/dev/null
echo PROFILE_DONE
