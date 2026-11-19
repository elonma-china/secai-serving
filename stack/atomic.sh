#!/usr/bin/env bash
# Atomic single-GPU comparison on GPU0 (stop llm-a first). Per profile: KV capacity, NLL on real text, needle at ~118K tokens, decode tok/s.
# Profiles: mtp3 | mtp2nvfp4 | mtp3nvfp4 (extra) ; off (FP8 attention patch disabled) | fp8attn | mtp1 | mtp2 | nvfp4kv  (all but off have FP8 attention on)
# Output: /workspace/results/atomic/summary.txt     Needs API_KEY, HF_TOKEN.  Usage: bash atomic.sh [profile ...]
set -uo pipefail
M=secai1999/LLM-Autoscan-v1; PORT=8032; OUT=/workspace/results/atomic; S=/workspace/secai/stack
export HF_HOME=/workspace/hf-cache HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 VLLM_API_KEY="$API_KEY"
declare -A EXTRA=( [off]="" [fp8attn]="" [mtp1]='--speculative-config {"method":"mtp","num_speculative_tokens":1}'
  [mtp2]='--speculative-config {"method":"mtp","num_speculative_tokens":2}' [nvfp4kv]="--kv-cache-dtype nvfp4"
  [mtp3]='--speculative-config {"method":"mtp","num_speculative_tokens":3}'
  [mtp2nvfp4]='--speculative-config {"method":"mtp","num_speculative_tokens":2} --kv-cache-dtype nvfp4'
  [mtp3nvfp4]='--speculative-config {"method":"mtp","num_speculative_tokens":3} --kv-cache-dtype nvfp4' )
PROFILES=("$@"); [ ${#PROFILES[@]} -gt 0 ] || PROFILES=(off fp8attn mtp1 mtp2 nvfp4kv)
mkdir -p "$OUT"
python3 $S/patches/sm120_fp8_prefill.py apply >/dev/null
for P in "${PROFILES[@]}"; do
  flag=$([ "$P" = off ] && echo 0 || echo 1)
  # shellcheck disable=SC2086
  VLLM_SM120_FP8_PREFILL=$flag vllm serve "$M" --host 127.0.0.1 --port $PORT --trust-remote-code --gpu-memory-utilization 0.90 --max-model-len 131072 \
    --max-num-seqs 64 --enable-prefix-caching --language-model-only --max-num-batched-tokens 16384 ${EXTRA[$P]} >"$OUT/$P.server.log" 2>&1 </dev/null & SP=$!
  up=0; for i in $(seq 1 160); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 5; done
  if [ $up = 0 ]; then echo "== $P: FAILED: $(grep -E 'Error' "$OUT/$P.server.log" | tail -1 | cut -c1-240)" | tee -a "$OUT/summary.txt"; kill $SP 2>/dev/null; sleep 15; continue; fi
  { echo "== $P"; grep -h "GPU KV cache size" "$OUT/$P.server.log" | head -1 | sed 's/.*GPU KV/GPU KV/'
    python3 $S/nll.py --url http://127.0.0.1:$PORT --model $M --tokens 24000 2>&1 | tail -1
    python3 $S/needle.py --url http://127.0.0.1:$PORT --model $M --words 21000 --depths 0.05,0.5,0.95 2>&1 | grep -E 'depth 0.50|NEEDLE'
    python3 $S/decode_test.py --url http://127.0.0.1:$PORT --model $M 2>&1 | tail -2
    grep -h -E "acceptance|Accepted" "$OUT/$P.server.log" | tail -1 | cut -c1-260; } | tee -a "$OUT/summary.txt"
  kill $SP; sleep 20
done
echo ATOMIC_DONE
