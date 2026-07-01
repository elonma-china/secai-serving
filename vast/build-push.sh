#!/usr/bin/env bash
# ── Build & push thin vast image lên Docker Hub ───────────────────────────────
#   IMAGE=justtuananh/secai-serving:vast  ./vast/build-push.sh
#
# CHẠY TRÊN MÁY CÓ DOCKER DAEMON: một vast.ai VM instance, hoặc host bất kỳ có
# docker. LƯU Ý: vast.ai *Docker-instance thường* KHÔNG có daemon (giống lý do
# không chạy docker-compose) → dùng bản **VM**, hoặc buildah/kaniko (daemonless).
#
# Cách nhanh trên 1 instance vast tạm:
#   git clone --depth 1 -b justtuananh https://github.com/justtuananh/infra_serving_quick_start.git iss
#   cd iss && IMAGE=justtuananh/secai-serving:vast bash vast/build-push.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
IMAGE="${IMAGE:-justtuananh/secai-serving:vast}"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo manual)"
SHA_TAG="${IMAGE%:*}:justtuananh-${SHA}"

echo "Repo:  $ROOT"
echo "Image: $IMAGE   (+ pin: $SHA_TAG)"

command -v docker >/dev/null 2>&1 || { echo "LỖI: không có 'docker' — chạy trên host có Docker daemon (vast VM), hoặc dùng buildah/kaniko."; exit 1; }
docker info >/dev/null 2>&1 || { echo "LỖI: Docker daemon không chạy/không truy cập được (vast Docker-instance thường không cấp daemon → dùng bản VM)."; exit 1; }

echo "── docker build (context = repo root, gồm vast/ templates/ nginx/) ──"
docker build -f vast/Dockerfile -t "$IMAGE" -t "$SHA_TAG" .

echo "── docker login (bỏ qua nếu đã login) ──"
docker login

echo "── push ──"
docker push "$IMAGE"
docker push "$SHA_TAG"
echo "✓ pushed $IMAGE  và  $SHA_TAG"
echo "  Template vast: image=$IMAGE · SSH · On-start: bash /opt/serving/vast/on_start.sh"
