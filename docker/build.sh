#!/usr/bin/env bash
# Build the serving image, tagged by commit: <repo>:<sha>-<yyyymmdd> (+ :latest). --push uploads both tags.
#   docker/build.sh [--push]        REPO=ghcr.io/elonma-china/secai-serving by default
# Push needs `docker login ghcr.io` (a token with write:packages). A dirty tree gets a -dirty suffix and is never pushed.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${REPO:-ghcr.io/elonma-china/secai-serving}"
sha=$(git rev-parse --short HEAD); dirty=""
git diff --quiet HEAD -- stack docker .dockerignore || dirty="-dirty"
tag="$REPO:$sha$dirty-$(date +%Y%m%d)"
docker build -f docker/Dockerfile --build-arg GIT_SHA="$sha$dirty" -t "$tag" -t "$REPO:latest" .
echo "built $tag"
if [ "${1:-}" = --push ]; then
  [ -z "$dirty" ] || { echo "refusing to push a dirty build; commit first" >&2; exit 1; }
  docker push "$tag" && docker push "$REPO:latest"
  docker image inspect "$tag" --format '{{index .RepoDigests 0}}'
fi
