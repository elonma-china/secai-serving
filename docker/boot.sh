#!/usr/bin/env bash
# Boot the whole stack from the prebuilt image (docker run / compose entrypoint, or a Vast.ai onstart command):
#   1. fetch the pinned models (parallel, skipped when already on disk)  || restore the compile-cache bundle
#   2. stack/up.sh
#   3. watchdog: a service that stays unhealthy triggers down.sh + up.sh (a partial restart cannot pass up.sh's
#      RAM / /dev/shm checks while the other replica holds its pinned offload memory). After MAX_RESTARTS it exits
#      non-zero so a container restart policy takes over.
# CACHE_SAVE=1 (HF_TOKEN with write access): when no cache bundle matched, publish one once the stack is up. Doing
# it here, not by hand over ssh, keeps the env (and so the bundle key) identical to the one the stack ran with.
# CACHE_SEED: a compile-cache seed baked into the image (Dockerfile.warm) is copied into $CACHE_DIR when it is empty.
# Secrets: API_KEY, or API_KEY_FILE (default /workspace/api.key). HF_TOKEN is optional (needed for the cache bundle
# if its repo is private).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LOG_DIR="${LOG_DIR:-/workspace/logs}" MODEL_DIR="${MODEL_DIR:-/workspace/models}" CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$CACHE_DIR"
# Service logs also go to stdout (prefixed), so `docker logs` / journald / a log shipper see them. Files stay for post-mortems.
if [ "${LOG_STDOUT:-1}" = 1 ]; then
  for n in llm-a llm-b embedding reranker ovis-ocr hy-mt mt-adapter ocr-adapter; do
    (tail -q -n 0 -F "$LOG_DIR/$n.log" 2>/dev/null | sed -u "s/^/[$n] /") &
  done
fi
exec > >(tee -a "$LOG_DIR/boot.log") 2>&1
log() { echo "$(date +%T) boot: $*"; }

[ -n "${HF_TOKEN:-}" ] || unset HF_TOKEN
t0=$SECONDS
if [ "${SKIP_FETCH:-0}" != 1 ]; then python3 "$ROOT/stack/fetch_models.py" & fetch=$!; fi
if [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ] && [ -d "${CACHE_SEED:-}" ] && [ -n "$(ls -A "$CACHE_SEED" 2>/dev/null)" ]; then
  log "seeding compile cache from the image"; cp -a "$CACHE_SEED/." "$CACHE_DIR/"
elif [ "${CACHE_RESTORE:-1}" = 1 ] && [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
  rc=0; python3 "$ROOT/stack/cache_bundle.py" restore || rc=$?
  [ "$rc" = 3 ] && cache_miss=1
  [ "$rc" = 0 ] || [ "$rc" = 3 ] || log "cache restore failed; continuing without it"
fi
# cached host code built for another CPU crashes with "Illegal instruction": drop it so it is rebuilt here
if [ -f "$ROOT/docker/host_code.sh" ]; then . "$ROOT/docker/host_code.sh"; msg=$(drop_foreign_host_code "$CACHE_DIR"); [ -z "$msg" ] || log "$msg"; fi

# The key may arrive after boot (e.g. scp'd to a fresh Vast instance, so it never shows up in the instance env/UI):
# downloads already run meanwhile, so waiting for it costs nothing.
if [ -z "${API_KEY:-}" ]; then
  f="${API_KEY_FILE:-/workspace/api.key}"; deadline=$((SECONDS + ${API_KEY_WAIT:-1800}))
  until [ -s "$f" ]; do
    [ "$SECONDS" -lt "$deadline" ] || { log "no API_KEY and no key file $f"; exit 1; }
    [ "$((SECONDS % 60))" -ge 5 ] || log "waiting for $f"; sleep 5
  done
  API_KEY="$(tr -d '\r\n' <"$f")"
fi
export API_KEY

[ -z "${fetch:-}" ] || wait "$fetch" || { log "model fetch failed"; exit 1; }
log "models + cache ready in $((SECONDS - t0))s"

bash "$ROOT/stack/up.sh"
log "stack up in $((SECONDS - t0))s since boot"
if [ "${cache_miss:-0}" = 1 ] && [ "${CACHE_SAVE:-0}" = 1 ]; then
  python3 "$ROOT/stack/cache_bundle.py" save >>"$LOG_DIR/cache-save.log" 2>&1 &
  log "publishing the compile-cache bundle in the background (cache-save.log)"
fi
[ "${WATCHDOG:-1}" = 1 ] || exit 0

unhealthy() {
  local p
  local h="${BIND_HOST:-0.0.0.0}"; [ "$h" != 0.0.0.0 ] || h=127.0.0.1
  for p in 8000 8001 8002 8003; do
    curl -fsS --max-time 10 "http://$h:$p/health" >/dev/null 2>&1 || { echo "$p"; return 0; }
  done
  for p in 18303 18304; do
    curl -fsS --max-time 10 "http://127.0.0.1:$p/health" >/dev/null 2>&1 || { echo "$p"; return 0; }
  done
  for p in 8004 18083 8005; do   # adapters / nginx: any HTTP answer counts
    curl -s -o /dev/null --max-time 10 "http://$h:$p/" 2>/dev/null || { echo "$p"; return 0; }
  done
  return 1
}
fails=0; restarts=0
while sleep "${WATCH_INTERVAL:-15}"; do
  if port=$(unhealthy); then
    fails=$((fails + 1)); log "port $port unhealthy ($fails/${WATCH_FAILS:-4})"
  else
    fails=0; continue
  fi
  [ "$fails" -ge "${WATCH_FAILS:-4}" ] || continue
  restarts=$((restarts + 1))
  [ "$restarts" -le "${MAX_RESTARTS:-3}" ] || { log "giving up after $((restarts - 1)) restarts"; exit 1; }
  log "restarting the stack (restart $restarts)"
  bash "$ROOT/stack/down.sh"
  bash "$ROOT/stack/up.sh" || { log "up.sh failed"; exit 1; }
  fails=0
done
