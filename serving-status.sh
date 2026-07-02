#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  serving-status.sh — xem instance vast + probe 3 endpoint của mọi instance
#  label secai-serving. Read-only. Dùng: ./serving-status.sh [--api-key KEY]
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
. "$ROOT/lib.sh"

API_KEY="sk-secai2026"
while [ $# -gt 0 ]; do case "$1" in
  --api-key) API_KEY="$2"; shift 2;;
  -h|--help) sed -n '2,6p' "$0"; exit 0;;
  *) die "tham số lạ: $1" "dùng: ./serving-status.sh [--api-key KEY]";;
esac; done

ensure_tools; ensure_vastai; ensure_apikey

say "══ Instance vast của tài khoản ══"
OUT="$(list_instances)"
if [ -n "$OUT" ]; then printf '%s\n' "id	label	gpu	giá	state	ip" "$OUT" | { command -v column >/dev/null && column -t -s "$(printf '\t')" || cat; }
else say "  (không có instance nào — không tính tiền)"; fi

for iid in $(list_instances "secai-serving" | cut -f1); do
  say "══ Probe serving instance $iid ══"
  read -r ip pl pe pr st <<<"$(inst_read "$iid")"
  if [ "$pl" = "-" ] || [ -z "$pl" ]; then
    warn "  chưa map port (state=$st) — xem: $VASTAI logs $iid"
    continue
  fi
  printf '  LLM    http://%s:%s/v1  → http=%s\n'  "$ip" "$pl" "$(probe "http://$ip:$pl" "$API_KEY")"
  printf '  EMBED  http://%s:%s      → http=%s\n' "$ip" "$pe" "$(probe "http://$ip:$pe" "$API_KEY")"
  printf '  RERANK http://%s:%s      → http=%s\n' "$ip" "$pr" "$(probe "http://$ip:$pr" "$API_KEY")"
done
