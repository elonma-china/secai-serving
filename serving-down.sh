#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  serving-down.sh — dừng tính tiền instance vast.
#     ./serving-down.sh <ID>            # destroy 1 instance (hỏi xác nhận)
#     ./serving-down.sh --all           # destroy MỌI instance label secai-*
#     ./serving-down.sh <ID> --stop     # chỉ stop (giữ đĩa/weights, phí lưu trữ nhỏ)
#     thêm --yes để bỏ hỏi xác nhận
#  ⚠ destroy = mất weights đã tải trên instance (app secai KHÔNG mất gì — Qdrant/
#    Neo4j nằm ở máy chạy app). Dựng lại: ./serving-up.sh (~10-15').
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
. "$ROOT/lib.sh"

IDS=(); ALL=0; ACTION="destroy"; YES=0
while [ $# -gt 0 ]; do case "$1" in
  --all)  ALL=1; shift;;
  --stop) ACTION="stop"; shift;;
  --yes)  YES=1; shift;;
  -h|--help) sed -n '2,10p' "$0"; exit 0;;
  *) IDS+=("$1"); shift;;
esac; done

ensure_tools; ensure_vastai; ensure_apikey

if [ "$ALL" = "1" ]; then
  mapfile -t IDS < <(list_instances | awk -F'\t' '$2 ~ /^secai-/ {print $1}')
  [ "${#IDS[@]}" -gt 0 ] || { say "không có instance label secai-* nào."; exit 0; }
fi
[ "${#IDS[@]}" -gt 0 ] || die "thiếu instance id" "xem id bằng ./serving-status.sh; dùng: ./serving-down.sh <ID> hoặc --all"

say "sẽ $ACTION các instance: ${IDS[*]}"
list_instances | grep -E "^($(IFS='|'; echo "${IDS[*]}"))	" || true
if [ "$YES" != "1" ]; then
  { [ -r /dev/tty ] && [ -w /dev/tty ]; } || die "không có TTY để xác nhận" "thêm --yes nếu chắc chắn"
  printf '\033[0;33m[serving] Xác nhận %s? (gõ yes): \033[0m' "$ACTION" >/dev/tty
  read -r ans </dev/tty || ans=""
  [ "$ans" = "yes" ] || { say "huỷ bỏ."; exit 0; }
fi

RC=0
for id in "${IDS[@]}"; do
  if "$VASTAI" "$ACTION" instance "$id" $([ "$ACTION" = destroy ] && echo -y) >/dev/null 2>&1; then ok "$ACTION $id xong"
  else warn "$ACTION $id THẤT BẠI — thử tay: $VASTAI $ACTION instance $id"; RC=1; fi
done
exit $RC
