#!/usr/bin/env bash
# One command on a target host: check the machine -> pull the pinned image -> start it -> wait until every model answers.
# It stops BEFORE pulling (36 GB) if the machine does not meet the requirements.
#   docker/deploy.sh [--workspace DIR] [--image REF] [--generate-key] [--check-only] [--no-smoke] [--timeout SEC]
#   docker/deploy.sh --down                 stop the stack (logs, caches and key in the workspace are kept)
# Defaults: image = docker/release.env (repo@digest), workspace = $SECAI_WORKSPACE or /srv/secai-serving.
# --generate-key creates $WORKSPACE/api.key (mode 600) if missing; the key is never printed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/release.env"
IMAGE="${SECAI_SERVING_IMAGE:-$SECAI_SERVING_REPO@$SECAI_SERVING_DIGEST}"
WS="${SECAI_WORKSPACE:-/srv/secai-serving}"; TIMEOUT=3600; CHECK_ONLY=0; SMOKE=1; GENKEY=0; DOWN=0; NAME=secai-serving
while [ $# -gt 0 ]; do case "$1" in
  --workspace) WS="$2"; shift 2;; --image) IMAGE="$2"; shift 2;; --timeout) TIMEOUT="$2"; shift 2;;
  --check-only) CHECK_ONLY=1; shift;; --no-smoke) SMOKE=0; shift;; --generate-key) GENKEY=1; shift;; --down) DOWN=1; shift;;
  -h|--help) sed -n 2,7p "$0"; exit 0;; *) echo "unknown argument: $1 (see --help)" >&2; exit 2;; esac; done
export SECAI_SERVING_IMAGE="$IMAGE" SECAI_WORKSPACE="$WS"
compose() { docker compose -f "$HERE/compose.yaml" "$@"; }

if [ "$DOWN" = 1 ]; then compose down; exit 0; fi

if [ "$GENKEY" = 1 ] && [ ! -s "$WS/api.key" ]; then
  mkdir -p "$WS"; (umask 077; openssl rand -hex 32 >"$WS/api.key"); echo "created $WS/api.key (mode 600)"
fi

echo "== 1/4 checking this machine"
if ! "$HERE/preflight.sh" --image "$IMAGE" --workspace "$WS"; then
  echo; echo "Not deploying: fix the FAIL items above and run again (nothing was pulled or started)." >&2; exit 1
fi
[ "$CHECK_ONLY" = 0 ] || { echo "check-only: machine is ready for $IMAGE"; exit 0; }

echo "== 2/4 pulling $IMAGE"
docker pull "$IMAGE"
echo "image revision: $(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"

echo "== 3/4 starting"
compose up -d
deadline=$((SECONDS + TIMEOUT)); next=0
until docker exec "$NAME" curl -fsS --max-time 5 http://127.0.0.1:8005/health >/dev/null 2>&1; do
  state=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
  if [ "$state" != running ]; then echo "container is $state; last log lines:" >&2; docker logs --tail 40 "$NAME" >&2 || true; exit 1; fi
  if [ "$SECONDS" -ge "$deadline" ]; then echo "not healthy after ${TIMEOUT}s; see: docker logs $NAME / $WS/logs/boot.log" >&2; exit 1; fi
  if [ "$SECONDS" -ge "$next" ]; then echo "  waiting... $(tail -n 1 "$WS/logs/startup-timeline.log" 2>/dev/null || echo booting)"; next=$((SECONDS + 30)); fi
  sleep 5
done
echo "stack is up"; cat "$WS/logs/startup-timeline.log" 2>/dev/null || true

if [ "$SMOKE" = 1 ]; then
  echo "== 4/4 smoke: one real request per model"
  docker exec "$NAME" python3 /opt/secai/stack/smoke.py || { echo "NOT READY: a model did not answer (see above)" >&2; exit 1; }
fi
echo "DEPLOYED $IMAGE  (LLM :8005, embedding :8000, rerank :8002, translate :8004, OCR :18083; all need the API key)"
