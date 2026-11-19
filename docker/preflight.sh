#!/usr/bin/env bash
# Run on the TARGET host before deploying. Checks everything the image cannot check for itself, and prints a fix for
# each failure. Exit 0 = safe to `docker compose up`. Usage: preflight.sh [--image REF] [--workspace DIR]
set -uo pipefail
IMAGE="${SECAI_SERVING_IMAGE:-ghcr.io/elonma-china/secai-serving:latest}"; WS="${SECAI_WORKSPACE:-/srv/secai-serving}"
while [ $# -gt 0 ]; do case "$1" in --image) IMAGE="$2"; shift 2;; --workspace) WS="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
bad=0
ok()   { printf 'ok    %s\n' "$*"; }
fail() { printf 'FAIL  %s\n      fix: %s\n' "$1" "$2"; bad=$((bad + 1)); }

command -v docker >/dev/null && docker info >/dev/null 2>&1 && ok "docker daemon reachable" || fail "docker not usable by this user" "install Docker Engine >= 24 and add the user to the docker group"
docker compose version >/dev/null 2>&1 && ok "docker compose plugin" || fail "docker compose plugin missing" "install docker-compose-plugin"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  mapfile -t gpus < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
  good=0; for g in "${gpus[@]}"; do [[ "$g" == *"RTX PRO 6000"* ]] && [ "${g##*, }" -ge 90000 ] && good=$((good + 1)); done
  [ "${#gpus[@]}" -eq 2 ] && [ "$good" -eq 2 ] && ok "2x RTX PRO 6000 (>= 90 GB each)" \
    || fail "need exactly 2x RTX PRO 6000 of >= 90 GB, found: ${gpus[*]:-none}" "use a host with exactly two RTX PRO 6000 (96 GB)"
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
  [ -n "$cuda" ] && [ "$(printf '%s\n13.2\n' "$cuda" | sort -V | head -1)" = 13.2 ] && ok "driver supports CUDA $cuda (image needs >= 13.2)" \
    || fail "driver supports CUDA ${cuda:-unknown}, image needs >= 13.2" "upgrade the NVIDIA driver"
else
  fail "nvidia-smi missing or failing" "install the NVIDIA driver"
fi
docker info 2>/dev/null | grep -qi nvidia && ok "nvidia container runtime registered" \
  || fail "docker has no nvidia runtime" "install nvidia-container-toolkit, then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
avail=$(free -g | awk '/^Mem:/{print $7}')
[ "${avail:-0}" -ge 142 ] && ok "RAM available ${avail} GiB (>= 142 for 2x55 GiB pinned KV offload + headroom)" \
  || fail "only ${avail:-0} GiB RAM available, need >= 142" "free memory or use a host with >= 192 GiB RAM"
mkdir -p "$WS" 2>/dev/null; free_gb=$(df -BG --output=avail "$WS" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 250 ] && ok "workspace $WS has ${free_gb} GB free (>= 250: KV tier 2x80 GiB, caches, logs)" \
  || fail "workspace $WS has ${free_gb:-0} GB free, need >= 250" "point SECAI_WORKSPACE at a bigger NVMe volume"
docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null); dfree=$(df -BG --output=avail "${docker_root:-/var/lib/docker}" 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${dfree:-0}" -ge 60 ] && ok "docker root has ${dfree} GB free (image ~44 GB)" || fail "docker root has ${dfree:-0} GB free, need >= 60 for the image" "free space or move Docker's data-root"
busy=0
for port in 8000 8001 8002 8003 8004 8005 18083 18303 18304; do
  ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN && { busy=1; fail "port $port is already in use" "stop the process using it (ss -ltnp 'sport = :$port')"; }
done; [ "$busy" -eq 0 ] && ok "ports 8000-8005, 18083, 18303, 18304 free"
[ -s "$WS/api.key" ] && ok "api key present ($WS/api.key)" || fail "no API key at $WS/api.key" "umask 077; openssl rand -hex 32 > $WS/api.key (or put your key there, chmod 600)"
[ -s "$WS/api.key" ] && [ "$(stat -c %a "$WS/api.key")" != 600 ] && fail "$WS/api.key is not mode 600" "chmod 600 $WS/api.key"
if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then ok "image $IMAGE is pullable"; else
  docker image inspect "$IMAGE" >/dev/null 2>&1 && ok "image $IMAGE present locally" || fail "cannot pull $IMAGE" "docker login ghcr.io -u <user> (token with read:packages), or set SECAI_SERVING_IMAGE"; fi
[ "$bad" -eq 0 ] && { echo "PREFLIGHT OK"; exit 0; } || { echo "PREFLIGHT FAILED: $bad problem(s)"; exit 1; }
