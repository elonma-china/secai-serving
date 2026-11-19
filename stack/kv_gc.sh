#!/usr/bin/env bash
# Keeps the vLLM fs KV tier under a size cap (the tier itself has no capacity limit): deletes the oldest block files.
# Usage: kv_gc.sh <dir> <max_gib> [interval_s]     A load of a just-deleted block fails safely (vLLM treats it as a miss).
# No `set -e`: files vanish under us while vLLM writes, and one failed find/rm must never kill the loop.
DIR="$1"; MAX=$(( $2 * 1073741824 )); EVERY="${3:-10}"
while true; do
  used=$(du -sb "$DIR" 2>/dev/null | cut -f1)
  if [ "${used:-0}" -gt $(( MAX * 95 / 100 )) ]; then          # start early: vLLM writes faster than a slow loop trims
    over=$(( used - MAX * 80 / 100 ))                          # trim down to 80% of the cap
    find "$DIR" -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -n | while read -r _ size path; do
      [ "$over" -gt 0 ] || break
      rm -f "$path" 2>/dev/null; over=$(( over - size ))
    done
  fi
  sleep "$EVERY"
done
