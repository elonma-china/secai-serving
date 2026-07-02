#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  lib.sh — hàm dùng chung cho serving-*.sh (KHÔNG chạy trực tiếp)
#  Dự án: /opt/secai-serving-vast — serving secai trên vast.ai (version secai-2026-07)
# ═══════════════════════════════════════════════════════════════════════════════
say(){  printf '\033[0;36m[serving]\033[0m %s\n' "$*"; }
ok(){   printf '\033[0;32m[serving] [OK] %s\033[0m\n' "$*"; }
warn(){ printf '\033[0;33m[serving] [WARN] %s\033[0m\n' "$*"; }
die(){  printf '\033[0;31m[serving] [FAIL] %s\033[0m\n' "$1" >&2
        printf '\033[0;31m>>> CẦN NGƯỜI DÙNG XỬ LÝ: %s\033[0m\n' "${2:-$1}" >&2; exit 1; }

# ── công cụ bắt buộc ───────────────────────────────────────────────────────────
ensure_tools(){
  command -v curl >/dev/null 2>&1 || die "thiếu curl" "apt-get install -y curl"
  command -v python3 >/dev/null 2>&1 || die "thiếu python3" "apt-get install -y python3"
}

# ── vastai CLI: tìm / tự cài (pip --user; fallback --break-system-packages) ────
VASTAI=""
find_vastai(){
  for c in vastai "$HOME/.local/bin/vastai"; do
    command -v "$c" >/dev/null 2>&1 && { VASTAI="$c"; return 0; }
    [ -x "$c" ] && { VASTAI="$c"; return 0; }
  done
  return 1
}
ensure_vastai(){
  find_vastai && return 0
  say "vastai CLI chưa có — cài qua pip --user ..."
  python3 -m pip install --user -U vastai >/dev/null 2>&1 \
    || python3 -m pip install --user --break-system-packages -U vastai >/dev/null 2>&1 \
    || die "không cài được vastai CLI" "chạy tay: python3 -m pip install --user --break-system-packages vastai"
  find_vastai || die "cài xong nhưng không tìm thấy vastai" "kiểm tra ~/.local/bin trong PATH"
  ok "đã cài vastai CLI ($VASTAI)"
}
ensure_apikey(){
  local f="$HOME/.config/vastai/vast_api_key"
  [ -s "$f" ] || die "chưa có vast API key ($f trống/không tồn tại)" \
    "lấy key ở https://cloud.vast.ai/account/ rồi chạy: $VASTAI set api-key <KEY>"
  "$VASTAI" show user --raw 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
if d.get("error"): raise SystemExit(1)
print("  tài khoản: %s · số dư: $%.2f"%(d.get("username") or d.get("email"),(d.get("credit") or d.get("balance") or 0)))' \
    || die "API key không hợp lệ (show user lỗi)" "kiểm tra key tại https://cloud.vast.ai/account/"
}

# ── đọc instance: in "ip p8001 p8000 p8002 state" ('-' nếu chưa có) ────────────
inst_read(){ # <instance_id>
  "$VASTAI" show instance "$1" --raw 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print("- - - - GONE"); raise SystemExit
if not isinstance(d,dict) or d.get("error"): print("- - - - GONE"); raise SystemExit
p=d.get("ports") or {}
def hp(x):
    for k,v in p.items():
        if k.split("/")[0]==x and v: return v[0].get("HostPort","-")
    return "-"
ip=d.get("public_ipaddr") or "-"
print(ip, hp("8001"), hp("8000"), hp("8002"), d.get("actual_status") or d.get("cur_state") or "unknown")'
}

# ── chờ port map. return: 0=ok · 1=timeout · 2=instance biến mất/chết ──────────
# LƯU Ý: public_ipaddr lúc provisioning có thể SAI (bug gặp thực tế 2026-07-02) —
# wait_serving sẽ re-read định kỳ để tự sửa.
wait_ports(){ # <instance_id> <timeout_s>  → set IP,P_LLM,P_EMB,P_RR
  local iid="$1" deadline=$(( $(date +%s) + ${2:-900} )) gone=0
  IP=""; P_LLM=""; P_EMB=""; P_RR=""
  while [ "$(date +%s)" -lt "$deadline" ]; do
    read -r ip pl pe pr st <<<"$(inst_read "$iid")"
    case "$st" in
      GONE|exited|failed|created_canceled|offline)
        gone=$((gone+1))
        [ "$gone" -ge 3 ] && { warn "instance $iid không còn chạy (state=$st)"; return 2; };;
      *) gone=0;;
    esac
    if [ "$pl" != "-" ] && [ -n "$pl" ]; then
      IP="$ip"; P_LLM="$pl"; P_EMB="$pe"; P_RR="$pr"
      ok "port đã map: ip=$IP llm=$P_LLM embed=$P_EMB rerank=$P_RR"
      return 0
    fi
    say "  chờ map port... (state=$st)"; sleep 20
  done
  return 1
}

# ── probe 1 endpoint /v1/models → http code (không bao giờ in đôi) ─────────────
probe(){ # <base_url> <api_key>
  local c
  c="$(curl -s -m 8 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $2" "$1/v1/models" 2>/dev/null)"
  echo "${c:-000}"
}

# ── chờ cả 3 dịch vụ 200; die nếu BẤT KỲ endpoint 401; tự refresh IP/port ─────
wait_serving(){ # <instance_id> <api_key> <timeout_s>   (dùng IP/P_* toàn cục từ wait_ports)
  local iid="$1" key="$2" deadline=$(( $(date +%s) + ${3:-1800} )) n=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    n=$((n+1))
    # mỗi 3 vòng: re-read instance — IP/port có thể đổi sau provisioning
    if [ $((n % 3)) -eq 1 ] && [ $n -gt 1 ]; then
      read -r ip2 pl2 pe2 pr2 _ <<<"$(inst_read "$iid")"
      if [ "$pl2" != "-" ] && [ -n "$pl2" ] && { [ "$ip2" != "$IP" ] || [ "$pl2" != "$P_LLM" ]; }; then
        warn "endpoint đổi: $IP:$P_LLM → $ip2:$pl2 (vast cập nhật lúc provisioning)"
        IP="$ip2"; P_LLM="$pl2"; P_EMB="$pe2"; P_RR="$pr2"
      fi
    fi
    local c1 c2 c3
    c1=$(probe "http://$IP:$P_LLM" "$key"); c2=$(probe "http://$IP:$P_EMB" "$key"); c3=$(probe "http://$IP:$P_RR" "$key")
    say "  serving: llm=$c1 embed=$c2 rerank=$c3"
    case "$c1$c2$c3" in *401*)
      die "serving trả 401 — API key không khớp (llm=$c1 embed=$c2 rerank=$c3)" \
          "API_KEY của instance khác key script dùng để probe; kiểm tra --api-key và env template";;
    esac
    [ "$c1" = "200" ] && [ "$c2" = "200" ] && [ "$c3" = "200" ] && return 0
    sleep 20
  done
  return 1
}

# ── liệt kê instance (TSV: id label gpu giá state ip) — lọc theo label nếu có ──
list_instances(){ # [label]
  "$VASTAI" show instances --raw 2>/dev/null | python3 -c '
import json,sys
want=sys.argv[1] if len(sys.argv)>1 else ""
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
d=d if isinstance(d,list) else d.get("instances",[])
for i in d:
    if want and (i.get("label") or "")!=want: continue
    print("%s\t%s\t%sx %s\t$%.3f/h\t%s\t%s"%(i.get("id"),i.get("label") or "-",
      i.get("num_gpus"),i.get("gpu_name"),i.get("dph_total") or 0,
      i.get("actual_status") or i.get("cur_state"),i.get("public_ipaddr") or "-"))' "${1:-}"
}
