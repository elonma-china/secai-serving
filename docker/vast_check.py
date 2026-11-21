"""Is a Vast.ai offer good enough for this stack? Checked BEFORE renting (costs nothing).

  vastai search offers 'gpu_name=RTX_PRO_6000_WS rentable=true' --raw | vast_check.py offer <offer_id> [--disk 250]
  vast_check.py query [--disk 250]      # the search query docker/vast.sh offers uses (same criteria)

Each criterion prints ok/FAIL with what the offer has and what is needed. Exit 0 = every criterion passes, 1 = at least
one fails, 2 = the offer is not in the input (already rented or gone). The second check, on the rented machine itself,
is `docker/preflight.sh --in-container` (docker/vast.sh check).
"""
import argparse, json, sys

GB = 1000   # Vast reports gpu_ram / cpu_ram in MB


def criteria(disk_gb):
    """(label, test(offer) -> bool, shown value(offer), requirement)."""
    return [
        ("GPU count", lambda o: o.get("num_gpus") == 2, lambda o: o.get("num_gpus"), "exactly 2"),
        ("GPU model", lambda o: "RTX PRO 6000" in str(o.get("gpu_name", "")).replace("_", " "), lambda o: o.get("gpu_name"), "RTX PRO 6000"),
        ("VRAM per GPU", lambda o: (o.get("gpu_ram") or 0) >= 90 * GB, lambda o: f"{(o.get('gpu_ram') or 0) / GB:.1f} GB", ">= 90 GB"),
        ("CUDA (driver)", lambda o: (o.get("cuda_max_good") or 0) >= 13.2, lambda o: o.get("cuda_max_good"), ">= 13.2"),
        ("host RAM", lambda o: (o.get("cpu_ram") or 0) >= 200 * GB, lambda o: f"{(o.get('cpu_ram') or 0) / GB:.0f} GB", ">= 200 GB (2x55 GiB pinned KV offload + headroom)"),
        ("disk", lambda o: (o.get("disk_space") or 0) >= disk_gb, lambda o: f"{o.get('disk_space') or 0:.0f} GB", f">= {disk_gb} GB"),
        ("download speed", lambda o: (o.get("inet_down") or 0) >= 500, lambda o: f"{o.get('inet_down') or 0:.0f} Mbps", ">= 500 Mbps (36 GB image)"),
        ("reliability", lambda o: (o.get("reliability2") or o.get("reliability") or 0) > 0.98,
         lambda o: f"{(o.get('reliability2') or o.get('reliability') or 0) * 100:.1f}%", "> 98%"),
        ("rentable", lambda o: bool(o.get("rentable", True)), lambda o: o.get("rentable", True), "yes"),
    ]


def check(offer, disk_gb=250):
    """[(label, ok, value, requirement)] for one offer dict."""
    return [(label, bool(test(offer)), shown(offer), need) for label, test, shown, need in criteria(disk_gb)]


def query(disk_gb=250, gpu_name="RTX_PRO_6000_WS"):
    return (f"num_gpus=2 gpu_name={gpu_name} gpu_ram>=90 cuda_vers>=13.2 cpu_ram>=200 disk_space>={disk_gb} "
            "inet_down>=500 reliability>0.98 rentable=true")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offer"); o.add_argument("offer_id", type=int); o.add_argument("--disk", type=int, default=250)
    q = sub.add_parser("query"); q.add_argument("--disk", type=int, default=250); q.add_argument("--gpu-name", default="RTX_PRO_6000_WS")
    a = p.parse_args(argv)
    if a.cmd == "query":
        print(query(a.disk, a.gpu_name)); return 0
    offers = json.load(sys.stdin)
    offer = next((x for x in offers if x.get("id") == a.offer_id), None)
    if offer is None:
        print(f"offer {a.offer_id} not found (rented by someone else or gone): run `docker/vast.sh offers` again", file=sys.stderr)
        return 2
    rows = check(offer, a.disk)
    print(f"offer {a.offer_id}: {offer.get('num_gpus')}x {offer.get('gpu_name')}, ${offer.get('dph_total', 0):.3f}/h, {offer.get('geolocation')}")
    for label, ok, value, need in rows:
        print(f"{'ok  ' if ok else 'FAIL'}  {label:15} {str(value):22} need {need}")
    bad = sum(not ok for _, ok, _, _ in rows)
    print("OFFER OK" if not bad else f"OFFER REJECTED: {bad} criterion(s) not met, do not rent it")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
