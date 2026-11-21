#!/usr/bin/env bash
# On a RUNNING stack (2x RTX PRO 6000): exercise every model, including long prompts that compile the lazily-built
# kernels, then pack $CACHE_DIR into a seed for `docker/build.sh warm`. Fails (no seed) if any model does not answer.
#   make_seed.sh [out=/workspace/cache-seed.tar.zst]
set -euo pipefail
OUT="${1:-/workspace/cache-seed.tar.zst}"; CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
python3 "$(dirname "$0")/smoke.py" --warm
python3 "$(dirname "$0")/smoke.py" --warm >/dev/null   # second pass: anything still compiling lazily is now cached
. "$(dirname "$0")/../docker/host_code.sh"; record_host_code_cpu "$CACHE_DIR"   # boot drops CPU-specific objects on other CPUs
tar -I 'zstd -T0 -10' -cf "$OUT.tmp" -C "$CACHE_DIR" . && mv "$OUT.tmp" "$OUT"
echo "seed: $OUT ($(du -h "$OUT" | cut -f1)), $(find "$CACHE_DIR" -type f | wc -l) files"
