#!/usr/bin/env bash
# One OrcaRouter engine across either four GPUs or one of the host's PCIe pairs.
set -euo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ] && [ "$#" -eq 1 ]; then DRY=1
elif [ "$#" -ne 0 ]; then echo 'Usage: llm.sh [--dry-run]' >&2; exit 2
fi

MODEL="${LLM_MODEL:-orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
SERVED_MODEL_NAME="${LLM_SERVED_MODEL_NAME:-orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4}"
PARALLEL_MODE="${PARALLEL_MODE:-tp4_ep4}"
PLE_LOCATION="${PLE_LOCATION:-gpu}"
GPU_UTIL="${LLM_GPU_UTIL:-0.90}"
MAX_LEN="${LLM_MAX_MODEL_LEN:-131072}"
MAX_SEQS="${LLM_MAX_NUM_SEQS:-64}"
BATCH_TOKENS="${LLM_MAX_BATCHED_TOKENS:-16384}"
PORT="${LLM_PORT:-8001}"
SPECULATIVE_CONFIG="${LLM_SPECULATIVE_CONFIG:-}"
COMPILATION_CONFIG="${LLM_COMPILATION_CONFIG:-}"
KERNEL_CONFIG="${LLM_KERNEL_CONFIG:-}"
ASYNC_SCHEDULING="${LLM_ASYNC_SCHEDULING:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${LLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"

case "$PARALLEL_MODE" in
  tp4|tp4_ep4) TP=4; GPU_IDS="${LLM_GPU_IDS:-0,1,2,3}"; [ "$GPU_IDS" = 0,1,2,3 ] || { echo 'TP4 requires GPUs 0,1,2,3' >&2; exit 2; } ;;
  tp2|tp2_ep2) TP=2; GPU_IDS="${LLM_GPU_IDS:-0,1}"; case "$GPU_IDS" in 0,1|2,3) ;; *) echo 'TP2 requires PCIe pair 0,1 or 2,3 on this host' >&2; exit 2;; esac ;;
  *) echo 'PARALLEL_MODE must be tp4, tp4_ep4, tp2 or tp2_ep2' >&2; exit 2;;
esac
case "$PLE_LOCATION" in gpu|cpu) ;; *) echo 'PLE_LOCATION must be gpu or cpu' >&2; exit 2;; esac
case "$ASYNC_SCHEDULING" in 0|1) ;; *) echo 'LLM_ASYNC_SCHEDULING must be 0 or 1' >&2; exit 2;; esac
case "$DISABLE_CUSTOM_ALL_REDUCE" in 0|1) ;; *) echo 'LLM_DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1' >&2; exit 2;; esac
for value in "$MAX_LEN" "$MAX_SEQS" "$BATCH_TOKENS" "$PORT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'model length, sequence count, batch tokens and port must be positive integers' >&2; exit 2; }
done
[[ "$GPU_UTIL" =~ ^0\.[0-9]+$ ]] || { echo 'LLM_GPU_UTIL must be a decimal between 0 and 1' >&2; exit 2; }

ENGRAM='{"cpu_offload":false}'
[ "$PLE_LOCATION" = cpu ] && ENGRAM='{"cpu_offload":true}'
ARGS=(serve "$MODEL" --host 0.0.0.0 --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP" --trust-remote-code
  --engram-config "$ENGRAM"
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCH_TOKENS"
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder)
case "$PARALLEL_MODE" in *_ep*) ARGS+=(--enable-expert-parallel);; esac
[ -z "$SPECULATIVE_CONFIG" ] || ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
[ -z "$COMPILATION_CONFIG" ] || ARGS+=(--compilation-config "$COMPILATION_CONFIG")
[ -z "$KERNEL_CONFIG" ] || ARGS+=(--kernel-config "$KERNEL_CONFIG")
[ "$ASYNC_SCHEDULING" = 0 ] || ARGS+=(--async-scheduling)
[ "$DISABLE_CUSTOM_ALL_REDUCE" = 0 ] || ARGS+=(--disable-custom-all-reduce)
[ "${LLM_LANGUAGE_ONLY:-0}" = 1 ] && ARGS+=(--language-model-only)

if [ "$DRY" = 1 ]; then
  printf 'CUDA_VISIBLE_DEVICES=%s VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_API_KEY=<redacted> vllm' "$GPU_IDS"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

[ -n "${API_KEY:-}" ] || { echo 'API_KEY must be set' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi missing' >&2; exit 1; }
command -v vllm >/dev/null || { echo 'vllm missing' >&2; exit 1; }
command -v setsid >/dev/null || { echo 'setsid missing' >&2; exit 1; }
mapfile -t GPUS < <(nvidia-smi -i "$GPU_IDS" --query-gpu=name,memory.total --format=csv,noheader,nounits)
[ "${#GPUS[@]}" -eq "$TP" ] || { echo "LLM role needs $TP selected GPUs" >&2; exit 1; }
for row in "${GPUS[@]}"; do
  [[ "$row" == *"RTX PRO 6000"* ]] || { echo "unexpected GPU: $row" >&2; exit 1; }
  mib="${row##*, }"
  [[ "$mib" =~ ^[0-9]+$ ]] && [ "$mib" -ge 90000 ] || { echo "GPU has less than 90 GiB reported VRAM: $row" >&2; exit 1; }
done
python3 - <<'PY'
from importlib.metadata import version
import re
def number(name):
    return tuple(int(x) for x in re.match(r"\d+(?:\.\d+)*", version(name)).group().split("."))
if number("vllm") < (0, 30) or number("transformers") < (5, 16):
    raise SystemExit("Need vLLM with qwen4_exp support and transformers >=5.16")
PY
echo 'GPU topology:'
nvidia-smi topo -m
export CUDA_VISIBLE_DEVICES="$GPU_IDS" VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_API_KEY="$API_KEY"
LOG_DIR="${LOG_DIR:-/var/log/secai-blackwell}"
mkdir -p "$LOG_DIR"
nohup setsid vllm "${ARGS[@]}" >"$LOG_DIR/llm.log" 2>&1 </dev/null &
pid="$!"
echo "$pid" >"$LOG_DIR/llm.pid"
deadline=$((SECONDS + ${HEALTH_TIMEOUT:-3600}))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  kill -0 "$pid" 2>/dev/null || { echo "LLM exited; see $LOG_DIR/llm.log" >&2; exit 1; }
  [ "$SECONDS" -lt "$deadline" ] || { echo "LLM health timeout; see $LOG_DIR/llm.log" >&2; exit 1; }
  sleep 5
done
API_KEY="$API_KEY" LLM_MODEL="$SERVED_MODEL_NAME" LLM_PORT="$PORT" python3 - <<'PY'
import json
import os
import urllib.request

body = {"model": os.environ["LLM_MODEL"],
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 32, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}
request = urllib.request.Request(
    "http://127.0.0.1:" + os.environ["LLM_PORT"] + "/v1/chat/completions",
    json.dumps(body).encode(),
    {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["API_KEY"]})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=180) as response:
    result = json.load(response)
if not result.get("choices") or not (result["choices"][0].get("message", {}).get("content") or "").strip():
    raise SystemExit("LLM inference smoke returned no content")
print("LLM authenticated inference OK")
PY
