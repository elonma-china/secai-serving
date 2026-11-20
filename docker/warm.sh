#!/usr/bin/env bash
# Make a compile-cache seed with the release image on a GPU host that has docker (the target machine itself, or a build
# box with 2x RTX PRO 6000). Cold-starts the stack with an EMPTY cache, so the seed holds everything needed.
#   docker/warm.sh <image> [seed_out=./cache-seed.tar.zst]
# Then: docker/build.sh warm ./cache-seed.tar.zst [--push]. On Vast (no docker inside the instance) run the same two steps
# by hand: boot the image, `docker exec`-less: ssh in, run /opt/secai/stack/make_seed.sh, scp the file back.
set -euo pipefail
IMAGE="${1:?usage: warm.sh <image> [seed_out]}"; OUT="${2:-./cache-seed.tar.zst}"; name="secai-warm-$$"
ws=$(mktemp -d /var/tmp/secai-warm.XXXXXX); umask 077; openssl rand -hex 32 >"$ws/api.key"
trap 'docker rm -f "$name" >/dev/null 2>&1; rm -rf "$ws"' EXIT
docker run -d --name "$name" --gpus all --network host --shm-size 120g --ulimit memlock=-1 -v "$ws:/workspace" "$IMAGE" >/dev/null
t0=$SECONDS
until docker exec "$name" curl -fsS http://127.0.0.1:8005/health >/dev/null 2>&1; do
  docker inspect -f '{{.State.Running}}' "$name" | grep -q true || { docker logs --tail 50 "$name"; exit 1; }
  sleep 10
done
echo "cold start to healthy: $((SECONDS - t0))s"
docker exec "$name" /opt/secai/stack/make_seed.sh /workspace/cache-seed.tar.zst
cp "$ws/cache-seed.tar.zst" "$OUT"; echo "seed written to $OUT"
