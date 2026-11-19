#!/usr/bin/env bash
# Rent and boot the prebuilt image on Vast.ai (2x RTX PRO 6000). Nothing is rented without --yes.
#   docker/vast.sh offers                         # list matching offers, cheapest first
#   docker/vast.sh create <offer_id> [--yes]      # print (or run) the create command
#   docker/vast.sh key <instance_id> <key_file>   # copy the API key to /workspace/api.key (boot waits for it)
#   docker/vast.sh tunnel <instance_id>           # local 38005/38000/38002/38004/38083 -> instance ports
# The API key never goes into the instance env (visible in the Vast UI); boot.sh downloads models while it waits.
set -euo pipefail
IMAGE="${SECAI_SERVING_IMAGE:-ghcr.io/elonma-china/secai-serving:latest}"
GPU_NAME="${GPU_NAME:-RTX_PRO_6000_WS}"   # also e.g. RTX_PRO_6000_S (server edition)
DISK_GB="${DISK_GB:-250}"   # 32 GB models + 2x80 GiB NVMe KV tier + compile cache + logs
ONSTART='mkdir -p /workspace/logs; env | grep -E "^(HF_TOKEN|SECAI_|KV_|LLM_|CACHE_|MODEL_DIR)" >/workspace/.boot-env 2>/dev/null; chmod 600 /workspace/.boot-env; (set -a; . /workspace/.boot-env; set +a; nohup /opt/secai/docker/boot.sh >/dev/null 2>&1 &)'

ssh_target() { # instance id -> "port host"
  vastai show instance "$1" --raw | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ssh_port"], d["ssh_host"])'
}

case "${1:-}" in
  offers)
    vastai search offers 'num_gpus=2 gpu_name='"$GPU_NAME"' gpu_ram>=90 cpu_ram>=200 disk_space>='"$DISK_GB"' inet_down>=500 reliability>0.98 rentable=true' -o 'dph+' | head -15;;
  create)
    [ -n "${2:-}" ] || { echo "usage: $0 create <offer_id> [--yes]" >&2; exit 2; }
    cmd=(vastai create instance "$2" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --onstart-cmd "$ONSTART")
    [ -z "${GHCR_LOGIN:-}" ] || cmd+=(--login "$GHCR_LOGIN")   # e.g. '-u USER -p TOKEN ghcr.io' for a private image
    if [ "${3:-}" = --yes ]; then "${cmd[@]}"; else printf '%q ' "${cmd[@]}" | sed 's/-p [^ ]* ghcr/-p <redacted> ghcr/'; echo; echo "(dry run; add --yes to rent)"; fi;;
  key)
    [ -n "${2:-}" ] && [ -s "${3:-}" ] || { echo "usage: $0 key <instance_id> <key_file>" >&2; exit 2; }
    read -r port host < <(ssh_target "$2")
    ssh -p "$port" -o StrictHostKeyChecking=accept-new "root@$host" 'umask 077; cat >/workspace/api.key.tmp && mv /workspace/api.key.tmp /workspace/api.key' <"$3"
    echo "key installed; follow progress: ssh -p $port root@$host tail -f /workspace/logs/boot.log";;
  tunnel)
    [ -n "${2:-}" ] || { echo "usage: $0 tunnel <instance_id>" >&2; exit 2; }
    read -r port host < <(ssh_target "$2")
    exec ssh -N -p "$port" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -L 127.0.0.1:38005:127.0.0.1:8005 -L 127.0.0.1:38000:127.0.0.1:8000 -L 127.0.0.1:38002:127.0.0.1:8002 \
      -L 127.0.0.1:38004:127.0.0.1:8004 -L 127.0.0.1:38083:127.0.0.1:18083 "root@$host";;
  *) sed -n '2,8p' "$0" >&2; exit 2;;
esac
