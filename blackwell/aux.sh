#!/usr/bin/env bash
# RTX 5090 role: resident embedding/rerank/OvisOCR3 with private OCR adapter.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY=0
if [ "${1:-}" = "--dry-run" ] && [ "$#" -eq 1 ]; then DRY=1
elif [ "$#" -ne 0 ]; then echo 'Usage: aux.sh [--dry-run]' >&2; exit 2
fi

EMBED_MODEL="${EMBED_MODEL:-DuyTa/sec-embedding}"
RERANK_MODEL="${RERANK_MODEL:-DuyTa/sec-rerank}"
OCR_MODEL="ATH-MaaS/OvisOCR3"
EMBED_UTIL="${EMBED_GPU_UTIL:-0.14}"
RERANK_UTIL="${RERANK_GPU_UTIL:-0.14}"
OCR_UTIL="${OCR_GPU_UTIL:-0.30}"
OCR="${ENABLE_OCR:-1}"
LOG_DIR="${LOG_DIR:-/var/log/secai-blackwell}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
OCR_TOKEN_FILE="${OCR_TOKEN_FILE:-/workspace/secai-ovis-ocr.token}"
RERANK_TEMPLATE="$ROOT/templates/qwen3_reranker.jinja"
RERANK_OVERRIDES='{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
case "$OCR" in 0|1) ;; *) echo 'ENABLE_OCR must be 0 or 1' >&2; exit 2;; esac
for util in "$EMBED_UTIL" "$RERANK_UTIL" "$OCR_UTIL"; do
  [[ "$util" =~ ^0\.[0-9]+$ ]] || { echo 'GPU utilization must be a decimal between 0 and 1' >&2; exit 2; }
done
EMBED_ARGS=(serve "$EMBED_MODEL" --host 0.0.0.0 --port 8000 --runner pooling
  --dtype float16 --max-model-len 1024 --gpu-memory-utilization "$EMBED_UTIL")
RERANK_ARGS=(serve "$RERANK_MODEL" --host 0.0.0.0 --port 8002 --runner pooling
  --hf-overrides "$RERANK_OVERRIDES" --chat-template "$RERANK_TEMPLATE"
  --dtype float16 --max-model-len 2048 --gpu-memory-utilization "$RERANK_UTIL")
OCR_ARGS=(serve "$OCR_MODEL" --host 127.0.0.1 --port 18303 --trust-remote-code
  --gpu-memory-utilization "$OCR_UTIL" --max-model-len 32768 --max-num-seqs 8
  --gdn-prefill-backend triton --limit-mm-per-prompt '{"image":2}'
  --mm-processor-kwargs '{"images_kwargs":{"min_pixels":200704,"max_pixels":8294400}}'
  --default-chat-template-kwargs '{"enable_thinking":false}')
ADAPTER_ARGS=("$ROOT/blackwell/ocr_adapter.py" --port 18083 --backend-url http://127.0.0.1:18303
  --token-file "$OCR_TOKEN_FILE")

if [ "$DRY" = 1 ]; then
  printf 'CUDA_VISIBLE_DEVICES=0 vllm'; printf ' %q' "${EMBED_ARGS[@]}" --api-key '<redacted>'; printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=0 vllm'; printf ' %q' "${RERANK_ARGS[@]}" --api-key '<redacted>'; printf '\n'
  if [ "$OCR" = 1 ]; then
    printf 'CUDA_VISIBLE_DEVICES=0 vllm'; printf ' %q' "${OCR_ARGS[@]}" --api-key '<redacted>'; printf '\n'
    printf 'python3'; printf ' %q' "${ADAPTER_ARGS[@]}"; printf '\n'
  fi
  exit 0
fi

[ -n "${API_KEY:-}" ] || { echo 'API_KEY must be set' >&2; exit 1; }
command -v setsid >/dev/null || { echo 'setsid missing' >&2; exit 1; }
[ -f "$RERANK_TEMPLATE" ] || { echo "Missing reranker template: $RERANK_TEMPLATE" >&2; exit 1; }
mapfile -t GPUS < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[ "${#GPUS[@]}" -eq 1 ] && [[ "${GPUS[0]}" == *"RTX 5090"* ]] \
  || { echo 'Aux role needs exactly one RTX 5090' >&2; exit 1; }
mkdir -p "$LOG_DIR" "$HF_HOME"
export CUDA_VISIBLE_DEVICES=0 HF_HOME
[ -z "${HF_TOKEN:-}" ] || export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

start() {
  local name="$1"; shift
  nohup setsid vllm "$@" --api-key "$API_KEY" >"$LOG_DIR/$name.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/$name.pid"
}
healthy() {
  local name="$1" port="$2" deadline=$((SECONDS + ${HEALTH_TIMEOUT:-1800}))
  until curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$(cat "$LOG_DIR/$name.pid")" 2>/dev/null || { echo "$name exited; see $LOG_DIR/$name.log" >&2; return 1; }
    [ "$SECONDS" -lt "$deadline" ] || { echo "$name health timeout; see $LOG_DIR/$name.log" >&2; return 1; }
    sleep 3
  done
}
start embedding "${EMBED_ARGS[@]}"; healthy embedding 8000
start reranker "${RERANK_ARGS[@]}"; healthy reranker 8002
API_KEY="$API_KEY" PYTHONPATH="$ROOT/vast" python3 - "$EMBED_MODEL" "$RERANK_MODEL" <<'PY'
import os
import sys
from resident_smoke import probe

token = os.environ["API_KEY"]
probe("http://127.0.0.1:8000", "/v1/embeddings", sys.argv[1],
      {"input": ["SECAI readiness"]}, "embed", token)
probe("http://127.0.0.1:8002", "/rerank", sys.argv[2],
      {"query": "SECAI", "documents": ["SECAI readiness", "Unrelated document"],
       "top_n": 1}, "rerank", token)
PY
if [ "$OCR" = 1 ]; then
  start ovis-ocr "${OCR_ARGS[@]}"; healthy ovis-ocr 18303
  (umask 077; python3 - "$OCR_TOKEN_FILE" <<'PY'
import os
import secrets
import sys
path = sys.argv[1]
if not os.path.exists(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as sink:
        sink.write(secrets.token_urlsafe(32) + "\n")
if os.stat(path).st_mode & 0o077:
    raise SystemExit("OCR token file must be private")
PY
  )
  nohup setsid python3 "${ADAPTER_ARGS[@]}" >"$LOG_DIR/ocr-adapter.log" 2>&1 </dev/null &
  echo "$!" >"$LOG_DIR/ocr-adapter.pid"
  healthy ocr-adapter 18083
  OCR_TOKEN_FILE="$OCR_TOKEN_FILE" python3 - <<'PY'
import base64
import io
import json
import os
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw

image = Image.new("RGB", (512, 512), "white")
ImageDraw.Draw(image).text((30, 200), "SECAI 2026", fill="black", font_size=48)
buffer = io.BytesIO()
image.save(buffer, format="PNG")
body = {"model": "ATH-MaaS/OvisOCR3",
        "image_base64": base64.b64encode(buffer.getvalue()).decode()}
token = Path(os.environ["OCR_TOKEN_FILE"]).read_text().strip()
request = urllib.request.Request("http://127.0.0.1:18083/v1/ocr",
    json.dumps(body).encode(), {"Content-Type": "application/json",
                              "Authorization": "Bearer " + token})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=180) as response:
    result = json.load(response)
if not result.get("blocks") or not result["blocks"][0].get("text", "").strip():
    raise SystemExit("OvisOCR3 inference smoke returned empty text")
print("OvisOCR3 authenticated inference OK")
PY
fi
echo 'Aux services ready: embedding :8000, reranker :8002; OvisOCR3 private :18083 when enabled.'
