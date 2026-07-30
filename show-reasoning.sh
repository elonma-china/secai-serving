#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  show-reasoning.sh — STREAM chuỗi suy luận (reasoning trace) của Cyber-F1 real-time
#
#  Gọi thẳng vLLM trên vast (server THẬT, đúng endpoint dự án dùng), bật thinking rồi
#  stream token-by-token: phần "nghĩ" (delta.reasoning) tô XÁM, câu trả lời
#  (delta.content) tô XANH — cho demo "model đang suy luận".
#
#  Serving để enable_thinking=false mặc định → script tự gửi
#  chat_template_kwargs={"enable_thinking":true} mỗi request để bật reasoning.
#  Config (URL/KEY/MODEL) đọc từ gateway .env — KHÔNG hardcode secret.
#
#  Dùng:
#     ./show-reasoning.sh "Câu hỏi của bạn ..."     # stream reasoning + answer
#     ./show-reasoning.sh                            # dùng prompt demo bảo mật mặc định
#     ./show-reasoning.sh --no-think "1+1?"          # tắt thinking (đối chứng: không có trace)
#     ./show-reasoning.sh --no-stream "..."          # lấy trọn gói rồi in (không stream)
#     ./show-reasoning.sh -m 4096 -t 0.6 "..."       # chỉnh max_tokens / temperature
#     ./show-reasoning.sh --system "Bạn là ..." "..." # thêm system prompt
#     ./show-reasoning.sh --raw "..."                # tắt màu
#  Ctrl-C để dừng.
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"

# ── màu + log ──────────────────────────────────────────────────────────────────
COLOR=1; [ -t 1 ] || COLOR=0
say(){  printf '\033[0;36m[reason]\033[0m %s\n' "$*" >&2; }
die(){  printf '\033[0;31m[reason] [FAIL] %s\033[0m\n' "$1" >&2
        printf '\033[0;31m>>> CẦN NGƯỜI DÙNG XỬ LÝ: %s\033[0m\n' "${2:-$1}" >&2; exit 1; }

# ── tham số ────────────────────────────────────────────────────────────────────
ENVFILE="$ROOT/../.env"          # gateway .env (secai-agent-gateway/.env)
THINK=true                       # bật reasoning
STREAM=true                      # stream real-time
MAXTOK=2048
TEMP=0.6                         # Qwen3 khuyến nghị 0.6 khi thinking
SYSTEM=""
PROMPT=""
DEFAULT_PROMPT="Một máy chủ web để lộ header 'Server: Apache/2.4.49'. Phân tích nguy cơ bảo mật và cách khai thác khả dĩ, nêu rõ các bước."

while [ $# -gt 0 ]; do case "$1" in
  --no-think)   THINK=false; shift;;
  --think)      THINK=true; shift;;
  --no-stream)  STREAM=false; shift;;
  -m|--max-tokens) MAXTOK="$2"; shift 2;;
  -t|--temp)    TEMP="$2"; shift 2;;
  --system)     SYSTEM="$2"; shift 2;;
  --env)        ENVFILE="$2"; shift 2;;
  --raw|--no-color) COLOR=0; shift;;
  -h|--help)    sed -n '2,34p' "$0"; exit 0;;
  --)           shift; PROMPT="$*"; break;;
  -*)           die "tham số lạ: $1" "xem: ./show-reasoning.sh --help";;
  *)            PROMPT="${PROMPT:+$PROMPT }$1"; shift;;
esac; done
[ -n "$PROMPT" ] || PROMPT="$DEFAULT_PROMPT"

command -v curl >/dev/null 2>&1 || die "thiếu curl" "apt-get install -y curl"
command -v jq   >/dev/null 2>&1 || die "thiếu jq"   "apt-get install -y jq"
[ -f "$ENVFILE" ] || die "không thấy .env: $ENVFILE" "chỉ định: ./show-reasoning.sh --env <path> ..."

# ── đọc config từ .env (grep, KHÔNG source cả file để tránh side-effect) ───────
getenv(){ grep -E "^$1=" "$ENVFILE" | tail -n1 | cut -d= -f2-; }
BASE_URL="$(getenv QWEN_BASE_URL)"
API_KEY="$(getenv QWEN_API_KEY)"
MODEL="$(getenv QWEN_MODEL)"
[ -n "$BASE_URL" ] && [ -n "$MODEL" ] || die "thiếu QWEN_BASE_URL/QWEN_MODEL trong $ENVFILE" "kiểm tra .env"

# ── preflight: server sống? (tái dùng gợi ý serving-status.sh) ────────────────
CODE="$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE_URL/models" -H "Authorization: Bearer $API_KEY" 2>/dev/null)"
[ "$CODE" = 200 ] || die "LLM endpoint không phản hồi 200 (got '$CODE') tại $BASE_URL" \
    "kiểm tra serving: ./serving-status.sh  — hoặc bật lại: ./serving-up-cyber-f1.sh"

# ── dựng body request bằng jq (an toàn với prompt có dấu nháy/xuống dòng) ──────
BODY="$(jq -n --arg model "$MODEL" --arg p "$PROMPT" --arg sys "$SYSTEM" \
  --argjson think "$THINK" --argjson stream "$STREAM" \
  --argjson mx "$MAXTOK" --argjson temp "$TEMP" '
  {
    model: $model,
    messages: ((if $sys != "" then [{role:"system",content:$sys}] else [] end)
               + [{role:"user",content:$p}]),
    chat_template_kwargs: { enable_thinking: $think },
    stream: $stream,
    max_tokens: $mx,
    temperature: $temp
  }')"

# ── màu ────────────────────────────────────────────────────────────────────────
if [ "$COLOR" = 1 ]; then
  GRAY=$'\033[90m'; GRN=$'\033[0;32m'; BOLD=$'\033[1m'; CYN=$'\033[0;36m'; RST=$'\033[0m'
else GRAY=""; GRN=""; BOLD=""; CYN=""; RST=""; fi

# ── banner ─────────────────────────────────────────────────────────────────────
say "endpoint : $BASE_URL   model: $MODEL"
say "thinking : $THINK   stream: $STREAM   max_tokens: $MAXTOK   temp: $TEMP"
say "prompt   : $PROMPT"
printf '%s────────────────────────────────────────────────────────────────────────────%s\n' "$CYN" "$RST" >&2

# ── STREAM ─────────────────────────────────────────────────────────────────────
# jq phát mỗi delta thành:  <0x01>R<reasoning>  và/hoặc  <0x01>C<content>  (giữ xuống
# dòng). bash tách theo 0x01 → đổi tầng thì in header, còn lại in thẳng token real-time.
if [ "$STREAM" = true ]; then
  JQPROG='(.choices[0].delta // {}) as $d
    | (if (($d.reasoning // "") != "") then "R" + $d.reasoning else "" end)
    + (if (($d.content   // "") != "") then "C" + $d.content   else "" end)'
  curl -s -N -m 600 "$BASE_URL/chat/completions" \
       -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d "$BODY" \
  | grep --line-buffered '^data: ' | sed -u 's/^data: //' | grep -v --line-buffered '^\[DONE\]' \
  | jq -j --unbuffered "$JQPROG" \
  | {
      mode=""
      while IFS= read -r -d $'\x01' seg || [ -n "$seg" ]; do
        [ -z "$seg" ] && continue
        m=${seg:0:1}; text=${seg:1}
        if [ "$m" != "$mode" ]; then
          if [ "$m" = R ]; then
            printf '%s💭 REASONING TRACE — chuỗi suy luận của Cyber-F1%s\n%s' "$BOLD$GRAY" "$RST" "$GRAY"
          elif [ "$m" = C ]; then
            printf '%s\n\n%s✅ ANSWER — câu trả lời%s\n%s' "$RST" "$BOLD$GRN" "$RST" "$GRN"
          fi
          mode=$m
        fi
        printf '%s' "$text"
      done
      printf '%s\n' "$RST"
      [ -z "$mode" ] && say "(không nhận được token nào — thử --no-stream để xem lỗi, hoặc kiểm tra serving)"
    }
  exit 0
fi

# ── NON-STREAM ─────────────────────────────────────────────────────────────────
RESP="$(curl -s -m 600 "$BASE_URL/chat/completions" \
        -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d "$BODY")"
ERR="$(printf '%s' "$RESP" | jq -r '.error.message // empty' 2>/dev/null)"
[ -n "$ERR" ] && die "LLM error: $ERR" "kiểm tra body/model/endpoint"
REASONING="$(printf '%s' "$RESP" | jq -r '.choices[0].message.reasoning // ""' 2>/dev/null)"
ANSWER="$(printf '%s' "$RESP" | jq -r '.choices[0].message.content // ""' 2>/dev/null)"

if [ -n "$REASONING" ]; then
  printf '%s💭 REASONING TRACE — chuỗi suy luận của Cyber-F1%s\n%s%s%s\n' "$BOLD$GRAY" "$RST" "$GRAY" "$REASONING" "$RST"
else
  say "(không có reasoning trace — thinking=$THINK)"
fi
printf '\n%s✅ ANSWER — câu trả lời%s\n%s%s%s\n' "$BOLD$GRN" "$RST" "$GRN" "${ANSWER:-(rỗng — max_tokens có thể đã hết cho phần reasoning; tăng -m)}" "$RST"
