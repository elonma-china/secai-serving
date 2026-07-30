#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  show-logs.sh — stream log REQUEST của serving vast (Cyber-F1) theo THỜI GIAN THỰC
#
#  Mục đích: bằng chứng "dự án gọi server THẬT" — SSH vào instance vast rồi tail -F
#  đồng thời log của cả 4 tầng, gắn nhãn + tô màu từng dòng:
#     [nginx ]  LB access.log     — mỗi request từ gateway đập vào (IP client, path, status)
#     [llm    ] Cyber-F1 vLLM      — POST /v1/chat/completions + throughput (tokens/s, running reqs)
#     [embed  ] bge-m3 vLLM        — POST /v1/embeddings
#     [rerank ] Qwen3-Reranker     — POST /rerank
#
#  IP/port SSH KHÔNG hardcode — đọc động qua `vastai ssh-url` (vast đổi mỗi lần restart).
#
#  Dùng:
#     ./show-logs.sh                 # tail -F cả 4 tầng (mặc định)
#     ./show-logs.sh --requests      # chỉ dòng HTTP request (gọn cho demo nghiệm thu)
#     ./show-logs.sh --llm           # chỉ LLM ; ghép được --embed --rerank --nginx --onstart
#     ./show-logs.sh -n 100          # in 100 dòng lịch sử trước khi follow (mặc định 20)
#     ./show-logs.sh --no-follow     # in log hiện tại rồi thoát (không follow)
#     ./show-logs.sh --id 44942210   # chỉ định instance ; mặc định auto theo label secai-serving
#     ./show-logs.sh --no-color      # tắt màu (khi pipe ra file)
#  Ctrl-C để dừng — tiến trình tail phía remote tự được dọn (SIGHUP).
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
. "$ROOT/lib.sh"

# ── tham số ────────────────────────────────────────────────────────────────────
LABEL="secai-serving"          # lọc instance theo label (như serving-status.sh)
IID=""                         # instance id (rỗng = auto theo label)
N=20                           # số dòng lịch sử in trước khi follow
FOLLOW=1                       # 1 = tail -F ; 0 = in 1 lần rồi thoát
REQ_ONLY=0                     # 1 = chỉ dòng HTTP request
COLOR=1; [ -t 1 ] || COLOR=0   # auto tắt màu khi không phải terminal
SEL_NGINX=0 SEL_LLM=0 SEL_EMBED=0 SEL_RERANK=0 SEL_ONSTART=0 ANY_SEL=0

while [ $# -gt 0 ]; do case "$1" in
  --id)         IID="$2"; shift 2;;
  -n|--lines)   N="$2"; shift 2;;
  --no-follow|--once) FOLLOW=0; shift;;
  -r|--requests) REQ_ONLY=1; shift;;
  --nginx)      SEL_NGINX=1;   ANY_SEL=1; shift;;
  --llm)        SEL_LLM=1;     ANY_SEL=1; shift;;
  --embed)      SEL_EMBED=1;   ANY_SEL=1; shift;;
  --rerank)     SEL_RERANK=1;  ANY_SEL=1; shift;;
  --onstart)    SEL_ONSTART=1; ANY_SEL=1; shift;;
  --no-color)   COLOR=0; shift;;
  -h|--help)    sed -n '2,30p' "$0"; exit 0;;
  *)            die "tham số lạ: $1" "xem: ./show-logs.sh --help";;
esac; done

# mặc định (không chỉ định gì): nginx + llm + embed + rerank
if [ "$ANY_SEL" = 0 ]; then SEL_NGINX=1 SEL_LLM=1 SEL_EMBED=1 SEL_RERANK=1; fi

ensure_tools; ensure_vastai; ensure_apikey
command -v ssh >/dev/null 2>&1 || die "thiếu ssh client" "apt-get install -y openssh-client"

# ── 1) resolve instance id ─────────────────────────────────────────────────────
# Khớp theo TIỀN TỐ label (secai-serving*) + ưu tiên instance đang running — không
# dùng list_instances (khớp tuyệt đối) vì label thật là secai-serving-cyberf1.
if [ -z "$IID" ]; then
  IID="$("$VASTAI" show instances --raw 2>/dev/null | python3 -c '
import json,sys
want=sys.argv[1]
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
d=d if isinstance(d,list) else d.get("instances",[])
c=[i for i in d if (i.get("label") or "").startswith(want)]
run=[i for i in c if (i.get("actual_status") or i.get("cur_state"))=="running"]
pick=(run or c)
if pick: print(pick[0].get("id"))' "$LABEL")"
  [ -n "$IID" ] || die "không thấy instance label bắt đầu bằng '$LABEL'" \
      "bật serving: ./serving-up-cyber-f1.sh  — hoặc chỉ định: ./show-logs.sh --id <ID>"
fi
LBL="$("$VASTAI" show instance "$IID" --raw 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("label") or "-")
except Exception: print("-")')"
say "instance vast: $IID (label=$LBL)"

# ── 2) lấy ssh host/port ĐỘNG qua vastai ssh-url ──────────────────────────────
SSH_URL="$("$VASTAI" ssh-url "$IID" 2>/dev/null)"   # dạng: ssh://root@HOST:PORT
[ -n "$SSH_URL" ] || die "không lấy được ssh-url cho instance $IID" \
    "kiểm tra: $VASTAI show instance $IID"
SSH_USER="$(printf '%s' "$SSH_URL" | sed -E 's#^ssh://([^@]+)@.*#\1#')"
SSH_HOST="$(printf '%s' "$SSH_URL" | sed -E 's#^ssh://[^@]+@([^:]+):.*#\1#')"
SSH_PORT="$(printf '%s' "$SSH_URL" | sed -E 's#^ssh://[^@]+@[^:]+:([0-9]+).*#\1#')"
[ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] || die "parse ssh-url lỗi: $SSH_URL" "$SSH_URL"
say "ssh endpoint: $SSH_USER@$SSH_HOST:$SSH_PORT"

SSH_BASE=(-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
          -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=5
          -p "$SSH_PORT" "$SSH_USER@$SSH_HOST")

# ── 3) preflight: xác nhận SSH vào được (báo lỗi rõ nếu chưa gắn key) ──────────
say "kiểm tra kết nối SSH ..."
PRE="$(ssh -o BatchMode=yes "${SSH_BASE[@]}" 'echo __ok__' 2>&1)"
case "$PRE" in
  *__ok__*) ok "SSH OK";;
  *) die "không SSH vào instance được: ${PRE##*$'\n'}" \
        "gắn key vào instance: $VASTAI attach ssh $IID \"\$(cat ~/.ssh/id_ed25519.pub)\"  (chờ ~30s rồi chạy lại)";;
esac

# ── 4) dựng lệnh remote: tail -Fq mỗi tầng, gắn nhãn, dọn tiến trình khi thoát ─
LOG_DIR="/var/log/vllm"
TAILFLAGS="-q"; [ "$FOLLOW" = 1 ] && TAILFLAGS="-Fq"
REQPIPE=""; [ "$REQ_ONLY" = 1 ] && REQPIPE='| grep --line-buffered -aiE "(GET|POST|PUT|DELETE|PATCH) /"'

RUN_LINES=""
addsvc(){ RUN_LINES="$RUN_LINES
run '$1' $2"; }
[ "$SEL_NGINX"   = 1 ] && addsvc 'nginx ' "/var/log/nginx/access.log"
[ "$SEL_LLM"     = 1 ] && addsvc 'llm    ' "$LOG_DIR/llm-*.log"
[ "$SEL_EMBED"   = 1 ] && addsvc 'embed  ' "$LOG_DIR/embedding-*.log"
[ "$SEL_RERANK"  = 1 ] && addsvc 'rerank ' "$LOG_DIR/reranker-*.log"
[ "$SEL_ONSTART" = 1 ] && addsvc 'onstrt' "/var/log/vast_onstart.log"

REMOTE="$(cat <<REMOTE_EOF
PIDS=
trap 'kill \$PIDS 2>/dev/null' EXIT INT TERM HUP
run(){
  tag="\$1"; shift
  have=0; for f in "\$@"; do [ -e "\$f" ] && have=1; done
  if [ "\$have" = 1 ]; then
    tail $TAILFLAGS -n $N "\$@" 2>/dev/null $REQPIPE | sed -u "s|^|[\$tag] |" &
    PIDS="\$PIDS \$!"
  else
    echo "[\$tag] (chưa có file log khớp: \$*)"
  fi
}
$RUN_LINES
wait
REMOTE_EOF
)"

# ── 5) tô màu theo nhãn tầng (nginx=cyan llm=green embed=vàng rerank=tím, lỗi=đỏ) ─
# Dùng vòng `while read` của bash, KHÔNG dùng awk: mawk block-buffer INPUT (gom cả block
# mới xử lý) → log nhỏ giọt bị kẹt, phá real-time. `read` đọc từng dòng ngay khi tới.
colorize(){
  local line c
  while IFS= read -r line; do
    if [ "$COLOR" = 0 ]; then printf '%s\n' "$line"; continue; fi
    case "$line" in
      "[nginx"*)  c=36;;   # cyan
      "[llm"*)    c=32;;   # green
      "[embed"*)  c=33;;   # vàng
      "[rerank"*) c=35;;   # tím
      "[onstrt"*) c=34;;   # xanh dương
      *)          c=0;;
    esac
    case "$line" in
      *ERROR*|*Traceback*|*" 500 "*|*" 502 "*|*" 503 "*|*" 504 "*|*CANCELLED*) c=31;;  # đỏ
    esac
    printf '\033[0;%sm%s\033[0m\n' "$c" "$line"
  done
}

# ── 6) banner + stream ────────────────────────────────────────────────────────
SVC=""
[ "$SEL_NGINX" = 1 ]   && SVC="$SVC nginx"
[ "$SEL_LLM" = 1 ]     && SVC="$SVC llm"
[ "$SEL_EMBED" = 1 ]   && SVC="$SVC embed"
[ "$SEL_RERANK" = 1 ]  && SVC="$SVC rerank"
[ "$SEL_ONSTART" = 1 ] && SVC="$SVC onstart"
ok "đang follow log [$SVC ] trên $SSH_HOST:$SSH_PORT — $([ "$FOLLOW" = 1 ] && echo 'Ctrl-C để dừng' || echo 'chế độ --no-follow')"
[ "$REQ_ONLY" = 1 ] && say "lọc: chỉ dòng HTTP request"
say "gợi ý: mở terminal khác bắn thử → curl -s http://127.0.0.1:8131/health ; hoặc gọi 1 request qua gateway để thấy dòng chạy real-time"
echo "────────────────────────────────────────────────────────────────────────────"

# Plain ssh + `-n` (stdin ← /dev/null): KHÔNG -tt (pty nuốt output tail); KHÔNG `tr`
# (block-buffer 4KB, phá real-time); `-n` để nếu chạy nền/nohup ssh khỏi bị SIGTTIN.
# Khi ssh đóng, sshd HUP lệnh remote → trap dọn các tiến trình tail (xem $REMOTE).
ssh -n "${SSH_BASE[@]}" "$REMOTE" | colorize
