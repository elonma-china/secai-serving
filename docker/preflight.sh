#!/usr/bin/env bash
# Run on the TARGET host before deploying. Checks everything the image cannot check for itself, and prints a fix for
# each failure. Exit 0 = safe to `docker compose up`. Usage: preflight.sh [--image REF] [--workspace DIR]
# --in-container: run INSIDE an already started image (a rented Vast.ai instance, where there is no Docker host to check):
# skips the Docker checks and the API key (it is installed only after this check passes), and checks /dev/shm, the
# workspace volume and the models baked into the image instead. `docker/vast.sh check <instance>` runs it over ssh.
set -uo pipefail
IMAGE="${SECAI_SERVING_IMAGE:-ghcr.io/elonma-china/secai-serving:latest}"; WS="${SECAI_WORKSPACE:-}"; INC=0
while [ $# -gt 0 ]; do case "$1" in --image) IMAGE="$2"; shift 2;; --workspace) WS="$2"; shift 2;; --in-container) INC=1; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
if [ "$INC" = 1 ]; then WS="${WS:-/workspace}"; else WS="${WS:-/srv/secai-serving}"; fi
bad=0
ok()   { printf 'ok    %s\n' "$*"; }
fail() { printf 'FAIL  %s\n      fix: %s\n' "$1" "$2"; bad=$((bad + 1)); }

if [ "$INC" = 0 ]; then
  command -v docker >/dev/null && docker info >/dev/null 2>&1 && ok "docker daemon reachable" || fail "docker not usable by this user" "install Docker Engine >= 24 and add the user to the docker group"
  docker compose version >/dev/null 2>&1 && ok "docker compose plugin" || fail "docker compose plugin missing" "install docker-compose-plugin"
fi
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
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2048 {n++} END {print n+0}')
  [ "$busy" -eq 0 ] && ok "GPUs are idle (< 2 GiB in use)" \
    || fail "$busy GPU(s) already have > 2 GiB in use (another process holds them)" "stop it (nvidia-smi shows who), or the stack cannot size its memory"
fi
[ "$INC" = 1 ] || { docker info 2>/dev/null | grep -qi nvidia && ok "nvidia container runtime registered" \
  || fail "docker has no nvidia runtime" "install nvidia-container-toolkit, then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"; }
avail=$(free -g | awk '/^Mem:/{print $7}')
[ "${avail:-0}" -ge 142 ] && ok "RAM available ${avail} GiB (>= 142 for 2x55 GiB pinned KV offload + headroom)" \
  || fail "only ${avail:-0} GiB RAM available, need >= 142" "free memory or use a host with >= 192 GiB RAM"
if [ "$INC" = 1 ]; then
  mkdir -p "$WS" 2>/dev/null
  # The image is already unpacked here; the workspace holds the 2x80 GiB NVMe KV tier + compile cache + logs.
  NEED_WS=180; ws_free=$(df -BG --output=avail "$WS" 2>/dev/null | tail -1 | tr -dc 0-9)
  [ "${ws_free:-0}" -ge "$NEED_WS" ] && ok "workspace $WS has ${ws_free} GB free (>= $NEED_WS)" \
    || fail "workspace $WS has ${ws_free:-0} GB free, need >= $NEED_WS" "rent with a bigger disk (DISK_GB=250 in docker/vast.sh)"
  shm=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9)
  [ "${shm:-0}" -ge 114 ] && ok "/dev/shm has ${shm} GB free (>= 114 for 2x55 GiB KV offload)" \
    || fail "/dev/shm has ${shm:-0} GB free, need >= 114" "use a host that gives the container >= 128 GB of /dev/shm"
  stack_dir="${SECAI_STACK_DIR:-/opt/secai/stack}"
  if [ -f "$stack_dir/up.sh" ] && MODEL_DIR="${MODEL_DIR:-/opt/secai/models}" python3 "$stack_dir/fetch_models.py" --offline >/dev/null 2>&1; then
    ok "image code and all pinned models present (${SECAI_IMAGE_REVISION:-revision unknown})"
  else
    fail "image code or baked models missing under ${stack_dir%/stack}" "the instance does not run the released image: recreate it with docker/vast.sh create"
  fi
else
  mkdir -p "$WS" 2>/dev/null
  docker_root=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null); docker_root=${docker_root:-/var/lib/docker}
  avail() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc 0-9; }
  # Image: ~36 GB compressed blobs + ~70 GB unpacked at peak. Workspace: 2x80 GiB NVMe KV tier + compile cache + logs.
  NEED_IMG=110; NEED_WS=250; ws_free=$(avail "$WS"); img_free=$(avail "$docker_root")
  if [ "$(stat -c %d "$WS" 2>/dev/null)" = "$(stat -c %d "$docker_root" 2>/dev/null)" ]; then
    [ "${ws_free:-0}" -ge $((NEED_IMG + NEED_WS)) ] && ok "disk: ${ws_free} GB free for image + workspace (>= $((NEED_IMG + NEED_WS)), same filesystem)" \
      || fail "disk: ${ws_free:-0} GB free, need >= $((NEED_IMG + NEED_WS)) (image $NEED_IMG + workspace $NEED_WS on one filesystem)" "free space, move Docker's data-root, or point SECAI_WORKSPACE at another NVMe volume"
  else
    [ "${ws_free:-0}" -ge "$NEED_WS" ] && ok "workspace $WS has ${ws_free} GB free (>= $NEED_WS)" || fail "workspace $WS has ${ws_free:-0} GB free, need >= $NEED_WS" "point SECAI_WORKSPACE at a bigger NVMe volume"
    [ "${img_free:-0}" -ge "$NEED_IMG" ] && ok "docker root has ${img_free} GB free (>= $NEED_IMG)" || fail "docker root has ${img_free:-0} GB free, need >= $NEED_IMG for the image" "free space or move Docker's data-root"
  fi
fi
busy=0
for port in 8000 8001 8002 8003 8004 8005 18083 18303 18304; do
  ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN && { busy=1; fail "port $port is already in use" "stop the process using it (ss -ltnp 'sport = :$port')"; }
done; [ "$busy" -eq 0 ] && ok "ports 8000-8005, 18083, 18303, 18304 free"
if [ "$INC" = 0 ]; then
  [ -s "$WS/api.key" ] && ok "api key present ($WS/api.key)" || fail "no API key at $WS/api.key" "umask 077; openssl rand -hex 32 > $WS/api.key (or put your key there, chmod 600)"
  [ -s "$WS/api.key" ] && [ "$(stat -c %a "$WS/api.key")" != 600 ] && fail "$WS/api.key is not mode 600" "chmod 600 $WS/api.key"
  if docker manifest inspect "$IMAGE" >/dev/null 2>&1; then ok "image $IMAGE is pullable"; else
    docker image inspect "$IMAGE" >/dev/null 2>&1 && ok "image $IMAGE present locally" || fail "cannot pull $IMAGE" "docker login ghcr.io -u <user> (token with read:packages), or set SECAI_SERVING_IMAGE"; fi
fi
[ "$bad" -eq 0 ] && { echo "PREFLIGHT OK"; exit 0; } || { echo "PREFLIGHT FAILED: $bad problem(s)"; exit 1; }
