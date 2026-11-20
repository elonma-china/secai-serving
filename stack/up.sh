#!/usr/bin/env bash
# Final stack on ONE host with exactly 2x RTX PRO 6000 (96 GB). Everything is vLLM.
#   GPU0: LLM replica A                                  :8001
#   GPU1: LLM replica B :8003 + embedding :8000 + rerank :8002
#         + OvisOCR3 (loopback :18303 -> adapter :18083) + Hy-MT2 (loopback :18304 -> adapter :8004)
#   nginx :8005 -> A/B with session affinity (prefix-cache locality).
# STARTUP=serial (default) | parallel_aux | parallel: serial starts the GPU1 services one after another (each sizes its KV cache from
#   the memory it measures, which needs the others to be done). parallel_aux starts llm-a and the four auxiliary services at once,
#   then llm-b; parallel starts all six at once. Both use a fixed KV size per service (stack/kv_plan.py, --kv-cache-memory-bytes).
# KV offload per LLM replica: KV_OFFLOAD=native|tier|lmcache|none (default tier).
#   tier = vLLM TieringOffloadingSpec: pinned host RAM (KV_OFFLOAD_GIB) + NVMe fs tier (KV_FS_DIR, capped by KV_FS_MAX_GIB via kv_gc.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY=0
if [ "${1:-}" = "--dry-run" ] && [ "$#" -eq 1 ]; then DRY=1
elif [ "$#" -ne 0 ]; then echo 'Usage: up.sh [--dry-run]' >&2; exit 2
fi

LLM_MODEL="${LLM_MODEL:-secai1999/LLM-Autoscan-v1}"
EMBED_MODEL="${EMBED_MODEL:-secai1999/Embedding-Autoscan-v1}"
RERANK_MODEL="${RERANK_MODEL:-secai1999/Rerank-Autoscan-v1}"
OCR_MODEL="${OCR_MODEL:-secai1999/OCR-Autoscan-v1}"
MT_MODEL="${MT_MODEL:-secai1999/Translate-Autoscan-v1}"
STARTUP="${STARTUP:-serial}"; KV_PLAN="${KV_PLAN:-$ROOT/stack/kv_plan.json}"
case "$STARTUP" in serial|parallel_aux|parallel) ;; *) echo 'STARTUP must be serial, parallel_aux or parallel' >&2; exit 2;; esac
BIND_HOST="${BIND_HOST:-0.0.0.0}"   # interface of the public ports (8000/8001/8002/8003/8004/8005); 127.0.0.1 = reachable only via tunnel/reverse proxy
HEALTH_HOST="$BIND_HOST"; [ "$HEALTH_HOST" != 0.0.0.0 ] || HEALTH_HOST=127.0.0.1
LLM_MAX_LEN="${LLM_MAX_MODEL_LEN:-131072}"
LLM_MAX_SEQS="${LLM_MAX_NUM_SEQS:-64}"
LLM_UTIL_A="${LLM_UTIL_A:-0.90}"      # GPU0: LLM only
LLM_UTIL_B="${LLM_UTIL_B:-0.74}"      # GPU1: shares with the four aux models below
KV_OFFLOAD="${KV_OFFLOAD:-tier}"
KV_OFFLOAD_GIB="${KV_OFFLOAD_GIB:-55}"    # per replica; native backs it with /dev/shm (shared by both replicas)
KV_FS_DIR="${KV_FS_DIR:-/workspace/kvcache}"; KV_FS_MAX_GIB="${KV_FS_MAX_GIB:-80}"; KV_EVICTION="${KV_EVICTION:-lru}"   # RAM tier policy: lru | arc   # per replica
LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-}"
LLM_LANGUAGE_ONLY="${LLM_LANGUAGE_ONLY:-1}"  # 1 = drop vision tower (OCR is its own model)
LLM_BATCHED_TOKENS="${LLM_BATCHED_TOKENS:-16384}"  # larger prefill chunks; CUDA graphs captured up to the same size
LLM_MTP="${LLM_MTP:-2}"                       # N>0 enables MTP speculative decoding (measured: +86% decode at c=1, +64% at c=16); 0 = off
LLM_KV_DTYPE="${LLM_KV_DTYPE:-}"              # empty = checkpoint default (fp8); nvfp4 = +57-64% KV capacity but disables the FP8 prefill kernel (+52% prefill time at 118K)
export VLLM_SM120_FP8_PREFILL="${VLLM_SM120_FP8_PREFILL:-1}"   # FP8 attention prefill: -37% TTFT at 120K, NLL unchanged
LLM_EXTRA_ARGS="${LLM_EXTRA_ARGS:-}"          # e.g. speculative/reasoning flags, space separated
LOG_DIR="${LOG_DIR:-/workspace/logs}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
# MODEL_DIR set = serve the pre-fetched local copies (stack/fetch_models.py) under their repo names, fully offline.
MODEL_DIR="${MODEL_DIR:-}"
# Compile artifacts (torch.compile/Inductor, Triton, FlashInfer JIT) on the persistent disk: a restart on the same
# machine, or a machine restored with stack/cache_bundle.py, skips recompilation.
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_DIR/vllm}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}" \
  FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_DIR}"
src() { if [ -n "$MODEL_DIR" ]; then printf '%s' "$MODEL_DIR/${1##*/}"; else printf '%s' "$1"; fi; }
OCR_TOKEN_FILE="${OCR_TOKEN_FILE:-/workspace/secai-ovis-ocr.token}"
RERANK_OVERRIDES='{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
case "$KV_OFFLOAD" in native|tier|lmcache|none) ;; *) echo 'KV_OFFLOAD must be native, tier, lmcache or none' >&2; exit 2;; esac

CUDAGRAPH_SIZES="1 2 4 8 16 24 32 40 48 56 64 $(seq -s' ' 128 64 512) 1024 2048 4096 8192 12288 16384"
llm_args() { # port util lmcache_port
  local a=(serve "$(src "$LLM_MODEL")" --served-model-name "$LLM_MODEL" --host "$BIND_HOST" --port "$1" --trust-remote-code
    --gpu-memory-utilization "$2" --max-model-len "$LLM_MAX_LEN" --max-num-seqs "$LLM_MAX_SEQS"
    --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder)
  [ "$LLM_LANGUAGE_ONLY" = 0 ] || a+=(--language-model-only)
  [ "$LLM_BATCHED_TOKENS" = 0 ] || a+=(--max-num-batched-tokens "$LLM_BATCHED_TOKENS" --cudagraph-capture-sizes $CUDAGRAPH_SIZES)
  [ -z "$LLM_KV_DTYPE" ] || a+=(--kv-cache-dtype "$LLM_KV_DTYPE")
  [ "$LLM_MTP" = 0 ] || a+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$LLM_MTP}")
  case "$KV_OFFLOAD" in
    native) a+=(--kv-offloading-backend native --kv-offloading-size "$KV_OFFLOAD_GIB");;
    tier) a+=(--kv-transfer-config
      "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$((KV_OFFLOAD_GIB * 1073741824)),\"eviction_policy\":\"$KV_EVICTION\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$KV_FS_DIR/$1\"}]}}");;
    lmcache) a+=(--kv-transfer-config
      "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://localhost\",\"lmcache.mp.port\":$3}}");;
  esac
  # shellcheck disable=SC2206
  a+=($LLM_EXTRA_ARGS)
  printf '%s\0' "${a[@]}"
}
mapfile -d '' A_ARGS < <(llm_args 8001 "$LLM_UTIL_A" 5555)
mapfile -d '' B_ARGS < <(llm_args 8003 "$LLM_UTIL_B" 5556)
EMBED_ARGS=(serve "$(src "$EMBED_MODEL")" --served-model-name "$EMBED_MODEL" --host "$BIND_HOST" --port 8000
  --runner pooling --dtype float16 --max-model-len 1024 --gpu-memory-utilization 0.04)
RERANK_ARGS=(serve "$(src "$RERANK_MODEL")" --served-model-name "$RERANK_MODEL" --host "$BIND_HOST" --port 8002
  --runner pooling --hf-overrides "$RERANK_OVERRIDES"
  --chat-template "$ROOT/stack/templates/qwen3_reranker.jinja" --dtype float16 --max-model-len 2048
  --gpu-memory-utilization 0.04)
OCR_ARGS=(serve "$(src "$OCR_MODEL")" --served-model-name "$OCR_MODEL" --host 127.0.0.1 --port 18303
  --trust-remote-code --gpu-memory-utilization 0.10
  --max-model-len 32768 --max-num-seqs 8 --gdn-prefill-backend triton --limit-mm-per-prompt '{"image":2}'
  --mm-processor-kwargs '{"images_kwargs":{"min_pixels":200704,"max_pixels":8294400}}'
  --default-chat-template-kwargs '{"enable_thinking":false}')
MT_ARGS=(serve "$(src "$MT_MODEL")" --served-model-name "$MT_MODEL" --host 127.0.0.1 --port 18304
  --gpu-memory-utilization 0.05 --max-model-len 8192 --max-num-seqs 32)
fix_kv() { # array_name service: append the fixed KV size (parallel modes only; fails if the plan has no entry)
  local -n arr="$1"; local bytes
  bytes=$(python3 "$ROOT/stack/kv_plan.py" arg "$2" --plan "$KV_PLAN") || exit 1
  arr+=(--kv-cache-memory-bytes "$bytes")
}
if [ "$STARTUP" != serial ]; then
  fix_kv EMBED_ARGS embedding; fix_kv RERANK_ARGS reranker; fix_kv OCR_ARGS ovis-ocr; fix_kv MT_ARGS hy-mt
  [ "$STARTUP" != parallel ] || fix_kv B_ARGS llm-b
fi
OCR_PROMPT_ARGS=()
[ -z "$MODEL_DIR" ] || OCR_PROMPT_ARGS=(--prompt-file "$(src "$OCR_MODEL")/prompts/plain_ocr.txt")

show() { local gpu="$1"; shift; printf 'CUDA_VISIBLE_DEVICES=%s vllm' "$gpu"; printf ' %q' "$@"; printf ' --api-key <redacted>\n'; }
if [ "$DRY" = 1 ]; then
  if [ "$KV_OFFLOAD" = lmcache ]; then
    echo "lmcache server --host localhost --port 5555 --l1-size-gb $KV_OFFLOAD_GIB ${LMCACHE_CHUNK_SIZE:+--chunk-size $LMCACHE_CHUNK_SIZE}"
    echo "lmcache server --host localhost --port 5556 --l1-size-gb $KV_OFFLOAD_GIB ${LMCACHE_CHUNK_SIZE:+--chunk-size $LMCACHE_CHUNK_SIZE}"
  fi
  show 0 "${A_ARGS[@]}"; show 1 "${B_ARGS[@]}"
  show 1 "${EMBED_ARGS[@]}"; show 1 "${RERANK_ARGS[@]}"; show 1 "${OCR_ARGS[@]}"; show 1 "${MT_ARGS[@]}"
  echo "python3 $ROOT/stack/ocr_adapter.py --port 18083 ... ${OCR_PROMPT_ARGS[*]}"; echo "python3 $ROOT/stack/mt_adapter.py --port 8004"
  exit 0
fi

[ -n "${API_KEY:-}" ] || { echo 'API_KEY must be set' >&2; exit 1; }
for c in vllm nvidia-smi setsid curl; do command -v "$c" >/dev/null || { echo "$c missing" >&2; exit 1; }; done
mapfile -t GPUS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[ "${#GPUS[@]}" -eq 2 ] || { echo "Need exactly 2 GPUs, found ${#GPUS[@]}" >&2; exit 1; }
for row in "${GPUS[@]}"; do
  [[ "$row" == *"RTX PRO 6000"* ]] && [ "${row##*, }" -ge 90000 ] || { echo "unexpected GPU: $row" >&2; exit 1; }
done
# Dead replicas leave vllm_offload_* mmap files in /dev/shm (counted as RAM in use); drop them before checking space.
if ! pgrep -f 'vllm serve .*--port (8001|8003)' >/dev/null 2>&1 && [ "$DRY" = 0 ]; then rm -f /dev/shm/vllm_offload_* 2>/dev/null || true; fi
if [ "$KV_OFFLOAD" != none ]; then
  avail=$(free -g | awk '/^Mem:/{print $7}')
  [ "$avail" -ge $((KV_OFFLOAD_GIB * 2 + 32)) ] \
    || { echo "Need >= $((KV_OFFLOAD_GIB * 2 + 32)) GiB free RAM for 2x$KV_OFFLOAD_GIB GiB offload, have $avail" >&2; exit 1; }
fi
if [ "$KV_OFFLOAD" = native ] || [ "$KV_OFFLOAD" = tier ]; then
  # Both pin their host-RAM tier in /dev/shm. Crashed or killed replicas leave vllm_offload_* mmap files behind;
  # with no LLM replica running they are garbage, so drop them before checking the free space.
  shm=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
  [ $((KV_OFFLOAD_GIB * 2 + 4)) -le "$shm" ] \
    || { echo "$KV_OFFLOAD offload is pre-faulted in /dev/shm: 2x$KV_OFFLOAD_GIB GiB needs <= $((shm - 4)) GiB free, /dev/shm has ${shm}G free" >&2; exit 1; }
fi
mkdir -p "$LOG_DIR" "$HF_HOME" "$CACHE_DIR"
[ -z "${HF_TOKEN:-}" ] || export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
if [ -n "$MODEL_DIR" ]; then
  MODEL_DIR="$MODEL_DIR" python3 "$ROOT/stack/fetch_models.py" --offline \
    || { echo "models missing in $MODEL_DIR: run stack/fetch_models.py first" >&2; exit 1; }
  export HF_HUB_OFFLINE=1
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
T0=$SECONDS
mark() { printf '%s +%4ss %s\n' "$(date +%T)" "$((SECONDS - T0))" "$*" | tee -a "$LOG_DIR/startup-timeline.log"; }
mark "up.sh start"

start() { # name gpu args...
  local name="$1" gpu="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_API_KEY="$API_KEY" nohup setsid vllm "$@" >"$LOG_DIR/$name.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/$name.pid"
}
healthy() { # name port
  local deadline=$((SECONDS + ${HEALTH_TIMEOUT:-3600}))
  until curl -fsS "http://$HEALTH_HOST:$2/health" >/dev/null 2>&1; do
    kill -0 "$(cat "$LOG_DIR/$1.pid")" 2>/dev/null || { echo "$1 exited; see $LOG_DIR/$1.log" >&2; return 1; }
    [ "$SECONDS" -lt "$deadline" ] || { echo "$1 health timeout; see $LOG_DIR/$1.log" >&2; return 1; }
    sleep 3
  done
  mark "$1 healthy"
}

if [ "$KV_OFFLOAD" = lmcache ]; then
  command -v lmcache >/dev/null || { echo 'lmcache missing (pip install lmcache)' >&2; exit 1; }
  for p in 5555 5556; do
    nohup setsid lmcache server --host localhost --port "$p" --l1-size-gb "$KV_OFFLOAD_GIB" \
      ${LMCACHE_CHUNK_SIZE:+--chunk-size "$LMCACHE_CHUNK_SIZE"} >"$LOG_DIR/lmcache-$p.log" 2>&1 </dev/null &
    echo "$!" >"$LOG_DIR/lmcache-$p.pid"
  done
  sleep 5
fi

if [ "$VLLM_SM120_FP8_PREFILL" = 1 ] && [ "$DRY" = 0 ]; then
  python3 "$ROOT/stack/patches/sm120_fp8_prefill.py" apply
fi
if [ "$KV_OFFLOAD" = tier ]; then
  for p in 8001 8003; do
    mkdir -p "$KV_FS_DIR/$p"
    pkill -f "kv_gc.sh $KV_FS_DIR/$p" 2>/dev/null || true
    nohup setsid bash "$(dirname "$0")/kv_gc.sh" "$KV_FS_DIR/$p" "$KV_FS_MAX_GIB" >"$LOG_DIR/kv_gc-$p.log" 2>&1 </dev/null &
  done
fi

ensure() { # name gpu port args... ; skip services that are already serving (idempotent re-run)
  local name="$1" gpu="$2" port="$3"; shift 3
  if curl -fsS "http://$HEALTH_HOST:$port/health" >/dev/null 2>&1; then echo "$name already healthy"; return; fi
  start "$name" "$gpu" "$@"
}
# GPU0 holds only replica A, so it loads and compiles while the GPU1 services start.
ensure llm-a 0 8001 "${A_ARGS[@]}"
case "$STARTUP" in
  serial)
    # GPU1: small aux models first, one at a time, so their memory is claimed before replica B profiles its budget.
    ensure embedding 1 8000 "${EMBED_ARGS[@]}"; healthy embedding 8000
    ensure reranker 1 8002 "${RERANK_ARGS[@]}"; healthy reranker 8002
    ensure ovis-ocr 1 18303 "${OCR_ARGS[@]}"; healthy ovis-ocr 18303
    ensure hy-mt 1 18304 "${MT_ARGS[@]}"; healthy hy-mt 18304
    ensure llm-b 1 8003 "${B_ARGS[@]}";;
  parallel_aux)
    # Fixed KV sizes: the four aux services no longer depend on each other. llm-b still waits, so it measures after them.
    ensure embedding 1 8000 "${EMBED_ARGS[@]}"; ensure reranker 1 8002 "${RERANK_ARGS[@]}"
    ensure ovis-ocr 1 18303 "${OCR_ARGS[@]}"; ensure hy-mt 1 18304 "${MT_ARGS[@]}"
    healthy embedding 8000; healthy reranker 8002; healthy ovis-ocr 18303; healthy hy-mt 18304
    ensure llm-b 1 8003 "${B_ARGS[@]}";;
  parallel)
    ensure embedding 1 8000 "${EMBED_ARGS[@]}"; ensure reranker 1 8002 "${RERANK_ARGS[@]}"
    ensure ovis-ocr 1 18303 "${OCR_ARGS[@]}"; ensure hy-mt 1 18304 "${MT_ARGS[@]}"
    ensure llm-b 1 8003 "${B_ARGS[@]}"
    healthy embedding 8000; healthy reranker 8002; healthy ovis-ocr 18303; healthy hy-mt 18304;;
esac
healthy llm-a 8001; healthy llm-b 8003

listening() { curl -s -o /dev/null --max-time 2 "http://$HEALTH_HOST:$1/" 2>/dev/null; }   # any HTTP answer
(umask 077; [ -f "$OCR_TOKEN_FILE" ] || python3 -c 'import secrets;print(secrets.token_urlsafe(32))' >"$OCR_TOKEN_FILE")
if ! listening 18083; then
  API_KEY="$API_KEY" nohup setsid python3 "$ROOT/stack/ocr_adapter.py" --port 18083 --backend-url http://127.0.0.1:18303 \
    --token-file "$OCR_TOKEN_FILE" "${OCR_PROMPT_ARGS[@]}" >"$LOG_DIR/ocr-adapter.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/ocr-adapter.pid"
fi
if ! listening 8004; then
  API_KEY="$API_KEY" nohup setsid python3 "$ROOT/stack/mt_adapter.py" --host "$BIND_HOST" --port 8004 >"$LOG_DIR/mt-adapter.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/mt-adapter.pid"
fi
sed "s/^    listen 8005;/    listen $BIND_HOST:8005;/" "$ROOT/stack/nginx.conf" >/etc/nginx/conf.d/secai-llm.conf && nginx -t && { nginx -s reload 2>/dev/null || nginx; }
healthy llm-a 8005 2>/dev/null || curl -fsS "http://$HEALTH_HOST:8005/health" >/dev/null
mark "stack up"
echo "Stack up. Public: 8005 LLM, 8000 embed, 8002 rerank, 8004 translate. OCR 18083 stays loopback (SSH tunnel)."
