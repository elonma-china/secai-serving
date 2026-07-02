#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  serving-up.sh — thuê 2×A100 trên vast.ai và dựng serving secai
#  (LLM + embedding + reranker: 6 replica vLLM + nginx LB). Kết quả = 3 URL.
#
#  Dự án này CHỈ lo serving. App secai nhận 3 URL đó qua:
#     cd /opt/nghiem_thu/secai-customer-2026-06-23-A100 && ./deploy.sh --llm <URL>/v1 --embed <URL> --rerank <URL>
#
#  Code chạy trên instance pin tại tag **v1.0** (= commit ccb6ca9) của fork
#  justtuananh/infra_serving_quick_start — bất biến, push nhánh sau này không ảnh hưởng.
#
#  Dùng:
#     ./serving-up.sh                    # search → chọn máy → thuê → chờ → in 3 URL
#     ./serving-up.sh --offer <ID>       # thuê đúng offer id
#     ./serving-up.sh --yes              # tự lấy offer rẻ nhất, không hỏi
#     ./serving-up.sh --dry-run          # chỉ in lệnh, KHÔNG thuê
#
#  Cờ:
#     --offer <ID>        thuê đúng offer này (bỏ search/confirm)
#     --yes               không hỏi — chọn offer rẻ nhất đạt filter
#     --max-price <$/h>   trần giá search (mặc định 2.5)
#     --disk <GB>         đĩa instance (mặc định 100; cần ≥80: image ~10G + weights ~21G)
#     --api-key <KEY>     bearer key serving (mặc định sk-secai2026)
#     --hf-token <TOKEN>  HuggingFace token (tuỳ chọn — 3 model đều public)
#     --thin              dùng thin image justtuananh/secai-serving:vast (build trước
#                         bằng vast/build-push.sh); mặc định clone-based
#     --replace           serving MỚI healthy xong → destroy instance secai-serving CŨ
#     --timeout <giây>    hạn chờ serving sẵn sàng (mặc định 1800)
#     --dry-run           in kế hoạch rồi thoát (không thuê, không tốn tiền)
#
#  Chuẩn bị 1 lần: ~/.local/bin/vastai set api-key <KEY>   (https://cloud.vast.ai/account/)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
. "$ROOT/lib.sh"

LABEL="secai-serving"
TAG="v1.1"
REPO_FORK="https://github.com/justtuananh/infra_serving_quick_start.git"
THIN_IMAGE="justtuananh/secai-serving:vast"
BASE_IMAGE="vllm/vllm-openai:latest"
BUNDLE_DIR="/opt/nghiem_thu/secai-customer-2026-06-23-A100"

# ── args ──────────────────────────────────────────────────────────────────────
OFFER=""; YES=0; MAX_PRICE="2.5"; DISK=100; API_KEY="sk-secai2026"; HF_TOKEN=""
THIN=0; REPLACE=0; TIMEOUT=1800; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --offer)     OFFER="$2"; shift 2;;
  --yes)       YES=1; shift;;
  --max-price) MAX_PRICE="$2"; shift 2;;
  --disk)      DISK="$2"; shift 2;;
  --api-key)   API_KEY="$2"; shift 2;;
  --hf-token)  HF_TOKEN="$2"; shift 2;;
  --thin)      THIN=1; shift;;
  --replace)   REPLACE=1; shift;;
  --timeout)   TIMEOUT="$2"; shift 2;;
  --dry-run)   DRY=1; shift;;
  -h|--help)   sed -n '2,36p' "$0"; exit 0;;
  *) die "tham số lạ: $1" "xem ./serving-up.sh --help";;
esac; done

# ── preflight ─────────────────────────────────────────────────────────────────
say "══ Preflight ══"
ensure_tools; ensure_vastai; ensure_apikey
ok "API key serving: ${API_KEY:0:6}*** · giá trần \$$MAX_PRICE/h · disk ${DISK}GB · version $TAG"

OLD_IDS="$(list_instances "$LABEL" | cut -f1 | tr '\n' ' ')"
[ -n "${OLD_IDS// /}" ] && warn "đã có instance label=$LABEL: $OLD_IDS (đang TÍNH TIỀN; --replace để tự thay sau khi cái mới healthy)"

# ── chọn offer ────────────────────────────────────────────────────────────────
QUERY="num_gpus=2 gpu_name in [A100_SXM4,A100_PCIE] gpu_ram>=80 disk_space>=$DISK dph_total<=$MAX_PRICE rentable=true verified=true"
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
  if [ -z "$TABLE" ]; then
    warn "0 offer verified — thử KHÔNG verified..."
    QUERY="${QUERY/ verified=true/}"
    TABLE="$(search_table "$QUERY")"
  fi
  [ -n "$TABLE" ] || die "không có offer 2×A100 80GB nào ≤\$$MAX_PRICE/h" "nới --max-price hoặc thử lại sau"
  echo "$TABLE"
  OFFER="$(echo "$TABLE" | head -1 | sed 's/.*id=\([0-9]*\).*/\1/')"
  if [ "$YES" != "1" ] && [ "$DRY" != "1" ]; then
    # không có TTY (cron/ssh không -t) thì KHÔNG được tự thuê — tiền thật
    { [ -r /dev/tty ] && [ -w /dev/tty ]; } || die "không có TTY để xác nhận chọn máy" \
        "chạy lại với --yes (tự lấy rẻ nhất) hoặc --offer <ID>"
    printf '\033[0;36m[serving]\033[0m Chọn offer id (Enter = %s — rẻ nhất): ' "$OFFER" >/dev/tty
    read -r ans </dev/tty || ans=""
    [ -n "$ans" ] && OFFER="$ans"
  fi
fi
say "offer chọn: $OFFER"

# ── build lệnh create ─────────────────────────────────────────────────────────
ENVSTR="-p 8001:8001 -p 8000:8000 -p 8002:8002 -e API_KEY=$API_KEY -e HF_HUB_ENABLE_HF_TRANSFER=1 -e LLM_MAX_MODEL_LEN=40960 -e LLM_GPU_UTIL=0.85"
[ -n "$HF_TOKEN" ] && ENVSTR="$ENVSTR -e HF_TOKEN=$HF_TOKEN"
if [ "$THIN" = "1" ]; then
  IMAGE="$THIN_IMAGE"; ONSTART="bash /opt/serving/vast/on_start.sh"
else
  IMAGE="$BASE_IMAGE"
  ONSTART="set -e; git clone --depth 1 -b $TAG $REPO_FORK /opt/iss || true; bash /opt/iss/vast/on_start.sh"
fi

if [ "$DRY" = "1" ]; then
  say "══ DRY-RUN — lệnh sẽ chạy (KHÔNG thuê) ══"
  echo "$VASTAI create instance $OFFER --image $IMAGE --disk $DISK --ssh --direct --cancel-unavail \\"
  echo "  --label $LABEL --env '$ENVSTR' --onstart-cmd '$ONSTART'"
  exit 0
fi

# ── thuê (parse stdout/stderr TÁCH RIÊNG — stderr warning không phá JSON) ──────
say "══ Thuê instance (offer $OFFER, image $IMAGE) ══"
ERRF="$(mktemp)"; trap 'rm -f "$ERRF"' EXIT
CREATED="$("$VASTAI" create instance "$OFFER" --image "$IMAGE" --disk "$DISK" --ssh --direct \
  --cancel-unavail --label "$LABEL" --env "$ENVSTR" --onstart-cmd "$ONSTART" --raw 2>"$ERRF")"
IID="$(printf '%s' "$CREATED" | python3 -c '
import re,sys
m=re.search(r"\"new_contract\"\s*:\s*(\d+)", sys.stdin.read())
print(m.group(1) if m else "")')"
if [ -z "$IID" ]; then
  warn "stdout: $(printf '%s' "$CREATED" | head -c 300)"
  warn "stderr: $(head -c 300 "$ERRF")"
  die "không đọc được instance id từ kết quả create" \
      "KIỂM TRA NGAY: $VASTAI show instances — instance CÓ THỂ đã được tạo và ĐANG TÍNH TIỀN. Nếu chưa có thì offer đã bị thuê mất → chạy lại chọn offer khác."
fi
ok "instance id=$IID (label=$LABEL) — bắt đầu TÍNH TIỀN từ bây giờ"
say "theo dõi log boot: $VASTAI logs $IID   (marker xong: 'on_start done')"

# ── chờ sẵn sàng ──────────────────────────────────────────────────────────────
say "══ Chờ map port (≤15') rồi chờ 6 vLLM healthy (≤$((TIMEOUT/60))') ══"
wait_ports "$IID" 900; WP=$?
case $WP in
  1) die "hết hạn chờ map port (xem: $VASTAI logs $IID)" \
        "instance $IID vẫn TÍNH TIỀN; bỏ thì: ./serving-down.sh $IID";;
  2) die "instance $IID không còn chạy (bị huỷ/hỏng khi provisioning — thường KHÔNG tính tiền)" \
        "xác nhận bằng: $VASTAI show instances; rồi chạy lại chọn offer khác";;
esac
wait_serving "$IID" "$API_KEY" "$TIMEOUT" \
  || die "serving chưa sẵn sàng sau ${TIMEOUT}s (xem: $VASTAI logs $IID)" \
         "instance $IID vẫn TÍNH TIỀN; kiểm tra log — thường do tải weights chậm, thử chờ thêm rồi probe lại bằng ./serving-status.sh; bỏ thì ./serving-down.sh $IID"

# ── thay instance cũ (tuỳ chọn) ───────────────────────────────────────────────
if [ "$REPLACE" = "1" ] && [ -n "${OLD_IDS// /}" ]; then
  for oid in $OLD_IDS; do
    [ "$oid" = "$IID" ] && continue
    say "destroy instance cũ $oid (--replace)"
    "$VASTAI" destroy instance "$oid" >/dev/null 2>&1 \
      && ok "đã destroy $oid" || warn "destroy $oid thất bại — tự xử lý: ./serving-down.sh $oid"
  done
fi

# ── KẾT QUẢ: 3 URL cho bundle ─────────────────────────────────────────────────
DPH="$("$VASTAI" show instance "$IID" --raw 2>/dev/null | python3 -c 'import json,sys
try: print("%.2f"%(json.load(sys.stdin).get("dph_total") or 0))
except Exception: print("?")')"
echo
ok  "SERVING SẴN SÀNG — instance $IID (~\$$DPH/h)"
echo "──────────────────────────────────────────────────────────────"
echo "  LLM URL     : http://$IP:$P_LLM/v1"
echo "  EMBED URL   : http://$IP:$P_EMB"
echo "  RERANK URL  : http://$IP:$P_RR"
echo "  API KEY     : $API_KEY"
echo "──────────────────────────────────────────────────────────────"
echo "▶ Trỏ app secai vào (trên máy chạy app):"
echo "  cd $BUNDLE_DIR && ./deploy.sh \\"
echo "      --llm http://$IP:$P_LLM/v1 --embed http://$IP:$P_EMB --rerank http://$IP:$P_RR \\"
echo "      --api-key $API_KEY"
echo
echo "  Trạng thái: ./serving-status.sh · Dừng tiền: ./serving-down.sh $IID"
