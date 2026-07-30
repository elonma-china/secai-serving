#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  serving-up-cyber-f1.sh — thuê 2×A100 80GB và dựng FULL STACK secai với LLM
#  = DuyTa/Cyber-F1-AWQ (Qwen3.5-MoE AWQ int4). Kết quả = 3 URL (LLM/embed/rerank).
#
#  Khác serving-up.sh: KHÔNG clone code từ GitHub tag — nhúng THẲNG (base64) bản
#  vast/on_start_cyber_f1.sh + templates/qwen3_reranker.jinja của repo LOCAL vào
#  --onstart-cmd. Nhờ vậy serve-args LLM mới (language-model-only, reasoning-parser
#  qwen3, tool-call-parser qwen3_coder, enable_thinking=false, len 65536) chạy ngay,
#  khỏi phải push tag mới. Kiến trúc giữ nguyên: 1 replica/GPU (TP=1) + nginx LB.
#
#  Model DuyTa/Cyber-F1-AWQ là PUBLIC → instance kéo được KHÔNG cần HF_TOKEN
#  (ta cố tình KHÔNG đưa token lên máy thuê).
#
#  Dùng:
#     ./serving-up-cyber-f1.sh --dry-run        # in offer + onstart, KHÔNG thuê
#     ./serving-up-cyber-f1.sh --yes            # lấy offer 2×A100 rẻ nhất, thuê luôn
#     ./serving-up-cyber-f1.sh --offer <ID>     # thuê đúng offer
#     ./serving-up-cyber-f1.sh --yes --replace-old   # sau khi healthy → destroy 4090 cũ
#  Cờ: --max-price (2.5) · --disk (100) · --api-key (sk-secai2026) · --max-model-len
#      (65536) · --gpu-util (0.80) · --timeout (2400) · --replace-old (destroy label
#      secai-embed-rerank sau khi stack mới 3/3 healthy)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
. "$ROOT/lib.sh"

LABEL="secai-serving-cyberf1"
OLD_EMBED_LABEL="secai-embed-rerank"
BASE_IMAGE="vllm/vllm-openai:v0.25.1"
LLM_MODEL="DuyTa/Cyber-F1-AWQ"
ONSTART_SRC="$ROOT/vast/on_start_cyber_f1.sh"
JINJA_SRC="$ROOT/templates/qwen3_reranker.jinja"

OFFER=""; YES=0; MAX_PRICE="2.5"; DISK=100; API_KEY="sk-secai2026"
MAX_LEN=65536; GPU_UTIL="0.80"; TIMEOUT=2400; DRY=0; REPLACE_OLD=0
while [ $# -gt 0 ]; do case "$1" in
  --offer)        OFFER="$2"; shift 2;;
  --yes)          YES=1; shift;;
  --max-price)    MAX_PRICE="$2"; shift 2;;
  --disk)         DISK="$2"; shift 2;;
  --api-key)      API_KEY="$2"; shift 2;;
  --max-model-len)MAX_LEN="$2"; shift 2;;
  --gpu-util)     GPU_UTIL="$2"; shift 2;;
  --timeout)      TIMEOUT="$2"; shift 2;;
  --replace-old)  REPLACE_OLD=1; shift;;
  --dry-run)      DRY=1; shift;;
  -h|--help)      sed -n '2,32p' "$0"; exit 0;;
  *) die "tham số lạ: $1" "xem ./serving-up-cyber-f1.sh --help";;
esac; done

say "══ Preflight ══"
ensure_tools; ensure_vastai; ensure_apikey
[ -f "$ONSTART_SRC" ] || die "thiếu $ONSTART_SRC" "kiểm tra repo"
[ -f "$JINJA_SRC" ]   || die "thiếu $JINJA_SRC"   "kiểm tra repo"
ok "LLM=$LLM_MODEL · len=$MAX_LEN · util=$GPU_UTIL · key=${API_KEY:0:6}*** · trần \$$MAX_PRICE/h · disk ${DISK}GB"

OLD_SAME="$(list_instances "$LABEL" | cut -f1 | tr '\n' ' ')"
[ -n "${OLD_SAME// /}" ] && warn "đã có instance label=$LABEL: $OLD_SAME (đang TÍNH TIỀN)"

# ── chọn offer 2×A100 ──────────────────────────────────────────────────────────
QUERY="num_gpus=2 gpu_name in [A100_SXM4,A100_PCIE] gpu_ram>=80 disk_space>=$DISK dph_total<=$MAX_PRICE inet_down>=3000 rentable=true verified=true"
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
  say "══ Search offer 2×A100 80GB ≤\$$MAX_PRICE/h ══"
  TABLE="$(search_table "$QUERY")"
  [ -z "$TABLE" ] && { warn "0 offer verified — thử KHÔNG verified..."; QUERY="${QUERY/ verified=true/}"; TABLE="$(search_table "$QUERY")"; }
  [ -n "$TABLE" ] || die "không có offer 2×A100 80GB ≤\$$MAX_PRICE/h" "nới --max-price hoặc thử lại"
  echo "$TABLE"
  OFFER="$(echo "$TABLE" | head -1 | sed 's/.*id=\([0-9]*\).*/\1/')"
  if [ "$YES" != "1" ] && [ "$DRY" != "1" ]; then
    { [ -r /dev/tty ] && [ -w /dev/tty ]; } || die "không có TTY để xác nhận offer" "chạy lại với --yes hoặc --offer <ID>"
    printf '\033[0;36m[serving]\033[0m Chọn offer id (Enter = %s — rẻ nhất): ' "$OFFER" >/dev/tty
    read -r ans </dev/tty || ans=""; [ -n "$ans" ] && OFFER="$ans"
  fi
fi
say "offer chọn: $OFFER"

# ── onstart-cmd: nhúng on_start_cyber_f1.sh + jinja (base64), decode & chạy ─────
# gzip TRƯỚC base64: vast giới hạn --onstart-cmd ≤ 16384B; base64 thô của on_start
# (~18KB) vượt limit → gzip (~7KB) rồi mới base64. Instance giải nén: base64 -d | gunzip.
ONSTART_B64="$(gzip -9 -c "$ONSTART_SRC" | base64 -w0)"
JINJA_B64="$(gzip -9 -c "$JINJA_SRC" | base64 -w0)"
read -r -d '' ONSTART <<ONSTART_EOF || true
set -e
mkdir -p /opt/secai/vast /opt/secai/templates
echo $ONSTART_B64 | base64 -d | gunzip > /opt/secai/vast/on_start_cyber_f1.sh
echo $JINJA_B64 | base64 -d | gunzip > /opt/secai/templates/qwen3_reranker.jinja
chmod +x /opt/secai/vast/on_start_cyber_f1.sh
bash /opt/secai/vast/on_start_cyber_f1.sh
ONSTART_EOF

ENVSTR="-p 8001:8001 -p 8000:8000 -p 8002:8002 -e API_KEY=$API_KEY -e HF_HUB_ENABLE_HF_TRANSFER=1"
ENVSTR="$ENVSTR -e LLM_MODEL=$LLM_MODEL -e LLM_MAX_MODEL_LEN=$MAX_LEN -e LLM_GPU_UTIL=$GPU_UTIL"

if [ "$DRY" = "1" ]; then
  say "══ DRY-RUN — lệnh sẽ chạy (KHÔNG thuê) ══"
  echo "$VASTAI create instance $OFFER --image $BASE_IMAGE --disk $DISK --ssh --direct --cancel-unavail \\"
  echo "  --label $LABEL --env '$ENVSTR' \\"
  echo "  --onstart-cmd '<inline: decode on_start_cyber_f1.sh ($(printf '%s' "$ONSTART_B64" | wc -c)B b64) + jinja, then run>'"
  echo
  say "onstart (đã giải mã, để review):"
  printf '%s\n' "$ONSTART" | sed 's/^/    /'
  exit 0
fi

# ── thuê ───────────────────────────────────────────────────────────────────────
say "══ Thuê 2×A100 (offer $OFFER) ══"
ERRF="$(mktemp)"; trap 'rm -f "$ERRF"' EXIT
CREATED="$("$VASTAI" create instance "$OFFER" --image "$BASE_IMAGE" --disk "$DISK" --ssh --direct \
  --cancel-unavail --label "$LABEL" --env "$ENVSTR" --onstart-cmd "$ONSTART" --raw 2>"$ERRF")"
IID="$(printf '%s' "$CREATED" | python3 -c 'import re,sys;m=re.search(r"\"new_contract\"\s*:\s*(\d+)",sys.stdin.read());print(m.group(1) if m else "")')"
[ -n "$IID" ] || { warn "stdout: $(printf '%s' "$CREATED" | head -c 300)"; warn "stderr: $(head -c 300 "$ERRF")"; \
  die "không đọc được instance id" "KIỂM TRA: $VASTAI show instances — có thể đã tạo và ĐANG TÍNH TIỀN"; }
ok "instance id=$IID (label=$LABEL) — bắt đầu TÍNH TIỀN"
say "log boot: $VASTAI logs $IID   (marker xong: 'on_start done')"

# ── chờ 3 endpoint healthy ─────────────────────────────────────────────────────
say "══ Chờ map port (≤25') rồi 6 vLLM healthy (≤$((TIMEOUT/60))') ══"
wait_ports "$IID" 1500; WP=$?
case $WP in
  1) die "hết hạn chờ map port ($VASTAI logs $IID)" "instance $IID vẫn TÍNH TIỀN; bỏ: ./serving-down.sh $IID";;
  2) die "instance $IID chết khi provisioning" "$VASTAI show instances; chạy lại chọn offer khác";;
esac
wait_serving "$IID" "$API_KEY" "$TIMEOUT" \
  || die "serving chưa sẵn sàng sau ${TIMEOUT}s ($VASTAI logs $IID)" \
         "instance $IID vẫn TÍNH TIỀN; xem log (tải ~24GB chậm / arch mới), probe lại: ./serving-status.sh; bỏ: ./serving-down.sh $IID"

# ── (tuỳ chọn) destroy 4090 embed cũ SAU KHI stack mới healthy ─────────────────
if [ "$REPLACE_OLD" = "1" ]; then
  OLD_IDS="$(list_instances "$OLD_EMBED_LABEL" | cut -f1 | tr '\n' ' ')"
  for oid in $OLD_IDS; do
    [ -n "$oid" ] || continue
    say "destroy 4090 embed cũ $oid (--replace-old)"
    "$VASTAI" destroy instance "$oid" -y >/dev/null 2>&1 && ok "đã destroy $oid" || warn "destroy $oid fail — ./serving-down.sh $oid"
  done
fi

DPH="$("$VASTAI" show instance "$IID" --raw 2>/dev/null | python3 -c 'import json,sys
try: print("%.2f"%(json.load(sys.stdin).get("dph_total") or 0))
except Exception: print("?")')"
echo
ok "FULL STACK SẴN SÀNG — instance $IID (~\$$DPH/h) · LLM=$LLM_MODEL"
echo "──────────────────────────────────────────────────────────────"
echo "  LLM URL     : http://$IP:$P_LLM/v1   (model id: $LLM_MODEL)"
echo "  EMBED URL   : http://$IP:$P_EMB"
echo "  RERANK URL  : http://$IP:$P_RR"
echo "  API KEY     : $API_KEY"
echo "──────────────────────────────────────────────────────────────"
echo "▶ smoke test LLM (thinking off mặc định):"
echo "  curl -s http://$IP:$P_LLM/v1/chat/completions -H 'Authorization: Bearer $API_KEY' \\"
echo "    -H 'Content-Type: application/json' -d '{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":64}'"
echo
echo "  Trạng thái: ./serving-status.sh · Dừng tiền: ./serving-down.sh $IID"
[ "$REPLACE_OLD" = "1" ] || echo "  4090 embed cũ vẫn chạy — dừng khi đã trỏ app sang stack mới: ./serving-down.sh <4090_ID>"
