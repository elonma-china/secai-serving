#!/usr/bin/env bash
# Rent and boot the prebuilt image on Vast.ai (2x RTX PRO 6000). Nothing is rented without --yes.
#   docker/vast.sh offers                         # list matching offers, cheapest first
#   docker/vast.sh check-offer <offer_id>         # check one offer against every criterion before renting (free)
#   docker/vast.sh create <offer_id> [--yes]      # check the offer, then print (or run) the create command
#   docker/vast.sh check <instance_id>            # preflight INSIDE the rented machine (GPUs, RAM, /dev/shm, disk, models)
#   docker/vast.sh key <instance_id> <key_file>   # runs `check`, then copies the key to /workspace/api.key: boot waits for
#                                                 # it, so no model is loaded before the machine passed the check
#   docker/vast.sh ports <instance_id>            # public URL of every service (needs the API key; no TLS)
#   docker/vast.sh tunnel <instance_id>           # optional private access: local 38005/38000/38002/38004/38083 -> instance ports
#   docker/vast.sh destroy <instance_id>          # stop billing (asks first; --yes skips the question)
# The API key never goes into the instance env (visible in the Vast UI); boot.sh downloads models while it waits.
set -euo pipefail
# shellcheck disable=SC1091
. "$(dirname "$0")/release.env"
IMAGE="${SECAI_SERVING_IMAGE:-$SECAI_SERVING_REPO@$SECAI_SERVING_DIGEST}"   # public package: no registry login needed
GPU_NAME="${GPU_NAME:-RTX_PRO_6000_WS}"   # also e.g. RTX_PRO_6000_S (server edition)
DISK_GB="${DISK_GB:-250}"   # 32 GB models + 2x80 GiB NVMe KV tier + compile cache + logs
ONSTART='mkdir -p /workspace/logs; env | grep -E "^(HF_TOKEN|SECAI_|KV_|LLM_|CACHE_|MODEL_DIR)" >/workspace/.boot-env 2>/dev/null; chmod 600 /workspace/.boot-env; (set -a; . /workspace/.boot-env; set +a; nohup /opt/secai/docker/boot.sh >/dev/null 2>&1 &)'

HERE="$(cd "$(dirname "$0")" && pwd)"
ssh_target() { # instance id -> "port host"
  vastai show instance "$1" --raw | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ssh_port"], d["ssh_host"])'
}

case "${1:-}" in
  offers)
    vastai search offers "$(python3 "$HERE/vast_check.py" query --disk "$DISK_GB" --gpu-name "$GPU_NAME")" -o 'dph+' | head -15;;
  check-offer)
    [ -n "${2:-}" ] || { echo "usage: $0 check-offer <offer_id>" >&2; exit 2; }
    vastai search offers "gpu_name=$GPU_NAME rentable=true" --limit 1000 --raw | python3 "$HERE/vast_check.py" offer "$2" --disk "$DISK_GB";;
  check)
    [ -n "${2:-}" ] || { echo "usage: $0 check <instance_id>" >&2; exit 2; }
    read -r port host < <(ssh_target "$2")
    # the local preflight.sh is piped in, so the check is the one of this checkout even on an older image
    ssh -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "root@$host" 'bash -s -- --in-container' <"$HERE/preflight.sh";;
  destroy)
    [ -n "${2:-}" ] || { echo "usage: $0 destroy <instance_id> [--yes]" >&2; exit 2; }
    if [ "${3:-}" != --yes ]; then read -r -p "destroy instance $2 (all its data is lost)? [y/N] " a; [ "$a" = y ] || exit 1; fi
    vastai destroy instance "$2" -y
    n=$(vastai show instances --raw | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))'); echo "instances left on the account: $n";;
  create)
    [ -n "${2:-}" ] || { echo "usage: $0 create <offer_id> [--yes]" >&2; exit 2; }
    "$0" check-offer "$2" || { echo "not renting offer $2" >&2; exit 1; }
    # the image EXPOSEs 8000/8002/8004/8005 (Vast maps those); -p adds the OCR port to the mapping, so images built before it was EXPOSEd work too
    cmd=(vastai create instance "$2" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --env '-p 18083:18083' --onstart-cmd "$ONSTART")
    [ -z "${GHCR_LOGIN:-}" ] || cmd+=(--login "$GHCR_LOGIN")   # e.g. '-u USER -p TOKEN ghcr.io' for a private image
    if [ "${3:-}" = --yes ]; then "${cmd[@]}"; else printf '%q ' "${cmd[@]}" | sed 's/-p [^ ]* ghcr/-p <redacted> ghcr/'; echo; echo "(dry run; add --yes to rent)"; fi;;
  key)
    [ -n "${2:-}" ] && [ -s "${3:-}" ] || { echo "usage: $0 key <instance_id> <key_file>" >&2; exit 2; }
    "$0" check "$2" || { echo "machine check failed: key NOT installed, nothing started. Destroy it: $0 destroy $2" >&2; exit 1; }
    read -r port host < <(ssh_target "$2")
    ssh -p "$port" -o StrictHostKeyChecking=accept-new "root@$host" 'umask 077; cat >/workspace/api.key.tmp && mv /workspace/api.key.tmp /workspace/api.key' <"$3"
    echo "key installed; follow progress: ssh -p $port root@$host tail -f /workspace/logs/boot.log";;
  ports)
    [ -n "${2:-}" ] || { echo "usage: $0 ports <instance_id>" >&2; exit 2; }
    vastai show instance "$2" --raw | python3 "$HERE/vast_check.py" ports
    echo "all need 'Authorization: Bearer <api key>'; the ports have no TLS: do not send real data over them";;
  tunnel)
    [ -n "${2:-}" ] || { echo "usage: $0 tunnel <instance_id>" >&2; exit 2; }
    read -r port host < <(ssh_target "$2")
    exec ssh -N -p "$port" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -L 127.0.0.1:38005:127.0.0.1:8005 -L 127.0.0.1:38000:127.0.0.1:8000 -L 127.0.0.1:38002:127.0.0.1:8002 \
      -L 127.0.0.1:38004:127.0.0.1:8004 -L 127.0.0.1:38083:127.0.0.1:18083 "root@$host";;
  *) sed -n '2,13p' "$0" >&2; exit 2;;
esac
