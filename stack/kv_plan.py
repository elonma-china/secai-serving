"""Fixed KV-cache sizes per service, so the GPU1 services can start in parallel.

Why: by default vLLM sizes each KV cache from the GPU memory it *measures* after warm-up ("extensible KV cache"), i.e.
budget minus everything that became resident since the process started. Processes starting at the same time on one GPU
count each other's allocations as their own and end up with a smaller, unpredictable cache. That is why up.sh starts the
GPU1 services one after another. `--kv-cache-memory-bytes` fixes the size and skips the measurement, so the start order
stops mattering. The sizes come from a serial run on this exact hardware/vLLM build (the image pins both):

  kv_plan.py learn --logs /workspace/logs --out stack/kv_plan.json   # read "Extended KV cache to N blocks (X GiB)" per service
  kv_plan.py arg hy-mt                                               # bytes for --kv-cache-memory-bytes (exit 1 if unknown)
"""
import argparse, json, math, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "kv_plan.json"
SERVICES = ("llm-a", "llm-b", "embedding", "reranker", "ovis-ocr", "hy-mt")
EXTENDED = re.compile(r"Extended KV cache to (\d+) blocks \(([0-9.]+) GiB\)")
GIB = 1 << 30
MARGIN = 0.99   # the measured size minus 1%: vLLM rounds the log to 0.01 GiB and a too-large fixed size is an OOM, a smaller one only fewer tokens


def parse_log(text):
    """(blocks, gib) of the LAST 'Extended KV cache' line (a restarted engine logs it again), or None."""
    found = EXTENDED.findall(text)
    return (int(found[-1][0]), float(found[-1][1])) if found else None


def learn(logs_dir, services=SERVICES):
    plan, missing = {}, []
    for name in services:
        f = pathlib.Path(logs_dir) / f"{name}.log"
        got = parse_log(re.sub(r"\x1b\[[0-9;]*m", "", f.read_text(errors="replace"))) if f.is_file() else None
        if got is None:
            missing.append(name)
        else:
            plan[name] = {"blocks": got[0], "gib": got[1]}
    return plan, missing


def kv_bytes(plan, service):
    entry = plan.get("kv", {}).get(service)
    if not entry:
        raise KeyError(service)
    return int(math.floor(entry["gib"] * MARGIN * GIB))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arg"); a.add_argument("service"); a.add_argument("--plan", default=str(DEFAULT_PLAN))
    l = sub.add_parser("learn"); l.add_argument("--logs", required=True); l.add_argument("--out", default=str(DEFAULT_PLAN))
    l.add_argument("--note", default="")
    args = p.parse_args(argv)
    if args.cmd == "arg":
        try:
            plan = json.loads(pathlib.Path(args.plan).read_text())
            print(kv_bytes(plan, args.service))
        except (OSError, ValueError, KeyError) as e:
            print(f"kv_plan: no fixed KV size for '{args.service}' in {args.plan} ({type(e).__name__}); "
                  f"run a serial start and `kv_plan.py learn`, or use STARTUP=serial", file=sys.stderr)
            return 1
        return 0
    plan, missing = learn(args.logs)
    if missing:
        print(f"kv_plan: no 'Extended KV cache' line in the logs of: {', '.join(missing)}", file=sys.stderr)
        return 1
    out = {"note": args.note or "measured on a serial start; see kv_plan.py", "margin": MARGIN, "kv": plan}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["gib"] for k, v in plan.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
