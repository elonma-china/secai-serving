#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  serving-up-4090.sh — thuê 1× RTX 4090 trên vast.ai, host CHỈ embedding + reranker
#  cho secai-agent-gateway (LLM đã dùng DeepSeek API, KHÔNG self-host).
#
#  Kết quả = 2 URL (embed :8000, rerank :8002) + API key → set vào .env của gateway:
#     EMBEDDING_API_URL / RERANKER_API_URL / EMBEDDING_API_KEY / RERANKER_API_KEY
#
#  Khác serving-up.sh (2×A100 full stack): 1 GPU, 1 replica/service, bind vLLM
#  thẳng ra port công cộng (KHÔNG nginx), chờ đúng 2 endpoint.
#
#  Dùng:
#     ./serving-up-4090.sh --dry-run      # in offer + lệnh, KHÔNG thuê (miễn phí)
#     ./serving-up-4090.sh --yes          # lấy offer 4090 rẻ nhất, thuê luôn
#     ./serving-up-4090.sh --offer <ID>   # thuê đúng offer
#  Cờ: --max-price (0.7) · --disk (40) · --api-key (sk-secai2026) · --hf-token
#      --gpu-util (0.35 mỗi service) · --timeout (1800)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
. "$ROOT/lib.sh"

LABEL="secai-embed-rerank"
BASE_IMAGE="vllm/vllm-openai:latest"
EMBED_MODEL="BAAI/bge-m3"                       # BẮT BUỘC — index KB dựng bằng nó
RERANK_MODEL="Qwen/Qwen3-Reranker-0.6B"

OFFER=""; YES=0; MAX_PRICE="0.7"; DISK=40; API_KEY="sk-secai2026"; HF_TOKEN=""
GPU_UTIL="0.35"; TIMEOUT=1800; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --offer)     OFFER="$2"; shift 2;;
  --yes)       YES=1; shift;;
  --max-price) MAX_PRICE="$2"; shift 2;;
  --disk)      DISK="$2"; shift 2;;
  --api-key)   API_KEY="$2"; shift 2;;
  --hf-token)  HF_TOKEN="$2"; shift 2;;
  --gpu-util)  GPU_UTIL="$2"; shift 2;;
  --timeout)   TIMEOUT="$2"; shift 2;;
  --dry-run)   DRY=1; shift;;
  -h|--help)   sed -n '2,26p' "$0"; exit 0;;
  *) die "tham số lạ: $1" "xem ./serving-up-4090.sh --help";;
esac; done

say "══ Preflight ══"
ensure_tools; ensure_vastai; ensure_apikey
ok "API key serving: ${API_KEY:0:6}*** · giá trần \$$MAX_PRICE/h · disk ${DISK}GB · 1×RTX_4090"

# ── on_start (chạy TRÊN instance): embed :8000 + rerank :8002, bind thẳng public ─
RERANK_HF_OVERRIDES='{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'
JINJA_B64="$(base64 -w0 "$ROOT/templates/qwen3_reranker.jinja")"
read -r -d '' ONSTART <<ONSTART_EOF || true
set -e
mkdir -p /var/log/vllm /workspace/hf-cache
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c 'import hf_transfer' 2>/dev/null || pip install -q hf_transfer 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
[ -n "\$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"
echo "$JINJA_B64" | base64 -d > /tmp/qwen3_reranker.jinja
echo "═══ secai embed+rerank on_start @ \$(date -u +%FT%TZ) ═══"
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 --api-key "\$API_KEY" --trust-remote-code \
  --model $EMBED_MODEL --runner pooling --dtype float16 \
  --max-model-len 512 --gpu-memory-utilization $GPU_UTIL \
  > /var/log/vllm/embedding.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8002 --api-key "\$API_KEY" --trust-remote-code \
  --model $RERANK_MODEL --runner pooling \
  --hf-overrides '$RERANK_HF_OVERRIDES' --chat-template /tmp/qwen3_reranker.jinja \
  --dtype float16 --max-model-len 4096 --gpu-memory-utilization $GPU_UTIL \
  > /var/log/vllm/reranker.log 2>&1 < /dev/null &
echo "═══ on_start launched embed :8000 + rerank :8002 ═══"
ONSTART_EOF

ENVSTR="-p 8000:8000 -p 8002:8002 -e API_KEY=$API_KEY -e HF_HUB_ENABLE_HF_TRANSFER=1"
[ -n "$HF_TOKEN" ] && ENVSTR="$ENVSTR -e HF_TOKEN=$HF_TOKEN"

# ── search offer 1×4090 ───────────────────────────────────────────────────────
QUERY="num_gpus=1 gpu_name=RTX_4090 gpu_ram>=24 disk_space>=$DISK dph_total<=$MAX_PRICE rentable=true verified=true"
search_table(){
  "$VASTAI" search offers "$1" -o dph_total --raw 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
d=d if isinstance(d,list) else d.get("offers",[])
for i,o in enumerate(d[:10],1):
    print("%2d) id=%-9s %-10s x%s  disk=%4.0fG  ↓%5.0fMbps  $%.3f/h  rel=%.3f  %s"%(
      i,o["id"],o.get("gpu_name",""),o.get("num_gpus"),o.get("disk_space") or 0,
      o.get("inet_down") or 0,o.get("dph_total") or 0,o.get("reliability2") or 0,
      str(o.get("geolocation"))[:20]))'
}
if [ -z "$OFFER" ]; then
  say "══ Search offer 1×RTX_4090 ≤\$$MAX_PRICE/h ══"
  TABLE="$(search_table "$QUERY")"
  [ -z "$TABLE" ] && { warn "0 offer verified — thử KHÔNG verified..."; QUERY="${QUERY/ verified=true/}"; TABLE="$(search_table "$QUERY")"; }
  [ -n "$TABLE" ] || die "không có offer 4090 nào ≤\$$MAX_PRICE/h" "nới --max-price hoặc thử lại"
  echo "$TABLE"
  OFFER="$(echo "$TABLE" | head -1 | sed 's/.*id=\([0-9]*\).*/\1/')"
fi
say "offer chọn: $OFFER"

if [ "$DRY" = "1" ]; then
  say "══ DRY-RUN — lệnh sẽ chạy (KHÔNG thuê) ══"
  echo "$VASTAI create instance $OFFER --image $BASE_IMAGE --disk $DISK --ssh --direct --cancel-unavail \\"
  echo "  --label $LABEL --env '$ENVSTR' --onstart-cmd '<embed+rerank on_start, $(printf '%s' "$ONSTART" | wc -l) dòng>'"
  exit 0
fi

# ── thuê ──────────────────────────────────────────────────────────────────────
say "══ Thuê 1×RTX_4090 (offer $OFFER) ══"
ERRF="$(mktemp)"; trap 'rm -f "$ERRF"' EXIT
CREATED="$("$VASTAI" create instance "$OFFER" --image "$BASE_IMAGE" --disk "$DISK" --ssh --direct \
  --cancel-unavail --label "$LABEL" --env "$ENVSTR" --onstart-cmd "$ONSTART" --raw 2>"$ERRF")"
IID="$(printf '%s' "$CREATED" | python3 -c 'import re,sys;m=re.search(r"\"new_contract\"\s*:\s*(\d+)",sys.stdin.read());print(m.group(1) if m else "")')"
[ -n "$IID" ] || { warn "stdout: $(printf '%s' "$CREATED" | head -c 300)"; warn "stderr: $(head -c 300 "$ERRF")"; \
  die "không đọc được instance id" "KIỂM TRA: $VASTAI show instances — có thể đã tạo và ĐANG TÍNH TIỀN"; }
ok "instance id=$IID (label=$LABEL) — bắt đầu TÍNH TIỀN"
say "log boot: $VASTAI logs $IID"

# ── chờ port 8000 (embed) map rồi chờ embed+rerank 200 ────────────────────────
say "══ Chờ map port + 2 vLLM healthy (≤$((TIMEOUT/60))') ══"
deadline=$(( $(date +%s) + 900 )); IP=""; P_EMB=""; P_RR=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  read -r ip pl pe pr st <<<"$(inst_read "$IID")"
  case "$st" in GONE|exited|failed|created_canceled|offline) die "instance $IID chết khi provisioning (state=$st)" "$VASTAI show instances; chạy lại chọn offer khác";; esac
  [ "$pe" != "-" ] && [ -n "$pe" ] && { IP="$ip"; P_EMB="$pe"; P_RR="$pr"; ok "port map: ip=$IP embed=$P_EMB rerank=$P_RR"; break; }
  say "  chờ map port... (state=$st)"; sleep 20
done
[ -n "$P_EMB" ] || die "hết hạn chờ map port ($VASTAI logs $IID)" "instance $IID vẫn TÍNH TIỀN; bỏ: ./serving-down.sh $IID"

deadline=$(( $(date +%s) + TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  read -r ip2 pl2 pe2 pr2 _ <<<"$(inst_read "$IID")"
  [ "$pe2" != "-" ] && [ -n "$pe2" ] && { IP="$ip2"; P_EMB="$pe2"; P_RR="$pr2"; }
  c2=$(probe "http://$IP:$P_EMB" "$API_KEY"); c3=$(probe "http://$IP:$P_RR" "$API_KEY")
  say "  serving: embed=$c2 rerank=$c3"
  case "$c2$c3" in *401*) die "serving 401 — API key lệch" "so --api-key với env template";; esac
  [ "$c2" = "200" ] && [ "$c3" = "200" ] && { READY=1; break; }
  sleep 20
done
[ "${READY:-0}" = "1" ] || die "serving chưa sẵn sàng sau ${TIMEOUT}s ($VASTAI logs $IID)" \
  "instance $IID vẫn TÍNH TIỀN; xem log (tải weights chậm), probe lại: ./serving-status.sh; bỏ: ./serving-down.sh $IID"

DPH="$("$VASTAI" show instance "$IID" --raw 2>/dev/null | python3 -c 'import json,sys
try: print("%.3f"%(json.load(sys.stdin).get("dph_total") or 0))
except Exception: print("?")')"
echo
ok "EMBED+RERANK SẴN SÀNG — instance $IID (~\$$DPH/h)"
echo "──────────────────────────────────────────────────────────────"
echo "  EMBED URL   : http://$IP:$P_EMB"
echo "  RERANK URL  : http://$IP:$P_RR"
echo "  API KEY     : $API_KEY"
echo "──────────────────────────────────────────────────────────────"
echo "▶ Set vào .env của gateway (secai-agent-gateway):"
echo "  EMBEDDING_API_URL=http://$IP:$P_EMB"
echo "  RERANKER_API_URL=http://$IP:$P_RR"
echo "  EMBEDDING_API_KEY=$API_KEY"
echo "  RERANKER_API_KEY=$API_KEY"
echo
echo "  Trạng thái: $VASTAI show instance $IID · Dừng tiền: ./serving-down.sh $IID"
