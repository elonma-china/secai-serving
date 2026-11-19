#!/usr/bin/env bash
# Build the deployable image, tagged by commit: <repo>:<sha>-<yyyymmdd>.
#   docker/build.sh release [--push]            vLLM + code + 5 models (shard-per-layer). ~44 GB. Tags the build and :latest.
#   docker/build.sh warm <seed.tar.zst> [--push] release + compile-cache seed made on a real 2x PRO 6000 (docker/warm.sh).
#                                               Tags <tag>-warm and moves :latest to it.
# REPO defaults to ghcr.io/elonma-china/secai-serving. Pushing needs `docker login ghcr.io` (token with write:packages);
# a dirty tree gets a -dirty suffix and is never pushed. The package stays private unless you change it on GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${REPO:-ghcr.io/elonma-china/secai-serving}"
mode="${1:-}"; shift || true
sha=$(git rev-parse --short HEAD); dirty=""
git diff --quiet HEAD -- stack docker .dockerignore || dirty="-dirty"
tag="$REPO:$sha$dirty-$(date +%Y%m%d)"
case "$mode" in
  release)
    docker build -f docker/Dockerfile --build-arg GIT_SHA="$sha$dirty" -t "$tag" -t "$REPO:latest" .
    final="$tag";;
  warm)
    seed="${1:-}"; shift || true; [ -s "$seed" ] || { echo "usage: $0 warm <seed.tar.zst> [--push]" >&2; exit 2; }
    docker image inspect "$tag" >/dev/null 2>&1 || { echo "build the release image first: $0 release" >&2; exit 1; }
    ctx=$(mktemp -d); trap 'rm -rf "$ctx"' EXIT
    mkdir "$ctx/cache-seed"; tar -I 'zstd -d -T0' -xf "$seed" -C "$ctx/cache-seed"
    cp docker/Dockerfile.warm "$ctx/Dockerfile"
    docker build --build-arg BASE="$tag" -t "$tag-warm" -t "$REPO:latest" "$ctx"
    final="$tag-warm";;
  *) sed -n 2,8p "$0" >&2; exit 2;;
esac
echo "built $final"
if [ "${1:-}" = --push ]; then
  [ -z "$dirty" ] || { echo "refusing to push a dirty build; commit first" >&2; exit 1; }
  docker push "$final" && docker push "$REPO:latest"
  docker image inspect "$final" --format '{{index .RepoDigests 0}}'
fi
