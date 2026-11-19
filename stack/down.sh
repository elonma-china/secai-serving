#!/usr/bin/env bash
# Stop every service up.sh started (each runs in its own session, so kill the whole process group), then drop the
# pinned KV offload files. nginx is left running; it only proxies.
set -uo pipefail
LOG_DIR="${LOG_DIR:-/workspace/logs}"
pids=()
for f in "$LOG_DIR"/*.pid; do
  [ -f "$f" ] || continue
  p=$(cat "$f"); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"; pids+=("$p"); }
  rm -f "$f"
done
pkill -f 'kv_gc.sh ' 2>/dev/null || true
for _ in $(seq 30); do
  alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && break; sleep 2
done
for p in "${pids[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
rm -f /dev/shm/vllm_offload_* 2>/dev/null || true
echo "stopped ${#pids[@]} services"
