#!/usr/bin/env bash
# Copy the standalone pinned bundle to an EXISTING Vast SSH endpoint, then
# install/restart only the private MT/OCR supervisor. Never rent/destroy/Docker.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST=""; PORT=22; IDENTITY=""; ACTION=install; DRY=0
REMOTE_ROOT=/workspace/secai-specialist-services
ARGS=()
while [ $# -gt 0 ]; do case "$1" in
  --host) HOST="$2"; shift 2;;
  --port) PORT="$2"; shift 2;;
  --identity) IDENTITY="$2"; shift 2;;
  --action) ACTION="$2"; shift 2;;
  --root) REMOTE_ROOT="$2"; shift 2;;
  --dry-run) DRY=1; shift;;
  --gpu|--python|--mt-token-file|--ocr-token-file|--ocr-model-path|--mt-model-path|--mt-server-path|--cuda-compat-path|--shared-lock|--expected-incumbents|--mt-port|--ocr-port)
    ARGS+=("$1" "$2"); shift 2;;
  --disable-mt|--disable-ocr|--reuse-artifacts|--smoke) ARGS+=("$1"); shift;;
  -h|--help) sed -n '2,4p' "$0"; echo "Usage: $0 --host root@IP --port PORT [--action install|start|stop|restart|boot|ready] [--root DIR] [--identity KEY] [bootstrap options] [--dry-run]"; exit 0;;
  *) echo "Unknown option: $1" >&2; exit 1;;
esac; done
[[ "$HOST" =~ ^[A-Za-z0-9_.@:-]+$ ]] || { echo 'Set a valid --host root@IP' >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo 'Invalid SSH port' >&2; exit 1; }
[[ "$REMOTE_ROOT" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo 'Unsafe remote root' >&2; exit 1; }
case "$ACTION" in install|start|stop|restart|boot|ready) ;; *) echo 'Invalid action' >&2; exit 1;; esac
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
[ -z "$IDENTITY" ] || SSH_OPTS+=(-i "$IDENTITY")
printf -v COMMAND '%q ' python3 "$REMOTE_ROOT/bootstrap.py" "$ACTION" --root "$REMOTE_ROOT" "${ARGS[@]}"
if [ "$DRY" = 1 ]; then
  echo "DRY-RUN: upload pinned runtime bundle/helpers to $HOST:$PORT:$REMOTE_ROOT (install only)"
  echo "ssh $HOST -p $PORT $COMMAND"
  exit 0
fi
if [ "$ACTION" = install ]; then
  # Refuse to overwrite helpers of a running supervisor during a reinstall.
  ssh "${SSH_OPTS[@]}" -p "$PORT" "$HOST" "mkdir -p '$REMOTE_ROOT'; if [ -f '$REMOTE_ROOT/bootstrap.py' ]; then python3 '$REMOTE_ROOT/bootstrap.py' stop --root '$REMOTE_ROOT'; fi"
  scp "${SSH_OPTS[@]}" -P "$PORT" "$ROOT"/vast/gpu_specialists/{bootstrap.py,worker.py,supervisor.py,runtime-manifest.json,runtime.tar.gz} "$HOST:$REMOTE_ROOT/"
fi
ssh "${SSH_OPTS[@]}" -p "$PORT" "$HOST" "$COMMAND"
