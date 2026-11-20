"""Fixed KV-cache sizes per service, so the GPU1 services can start in parallel.

Why: by default vLLM sizes each KV cache from the GPU memory it *measures* after warm-up ("extensible KV cache"), i.e.
budget minus everything that became resident since the process started. Processes starting at the same time on one GPU
count each other's allocations as their own and end up with a smaller, unpredictable cache. That is why up.sh starts the
GPU1 services one after another. `--kv-cache-memory-bytes` fixes the size and skips the measurement, so the start order
stops mattering. The sizes come from a serial run on this exact hardware/vLLM build (the image pins both):

  kv_plan.py learn --logs /workspace/logs --out stack/kv_plan.json   # read "Extended KV cache to N blocks (X GiB)" per service
  kv_plan.py arg hy-mt                                               # bytes for --kv-cache-memory-bytes (empty: no KV cache; exit 1 if unknown)
  kv_plan.py check                                                   # exit 0 if the plan was learned on this exact config (see below)

The plan stores a fingerprint of the configuration it was learned on: the GPU model, the vLLM version and the full serial launch plan
(`STARTUP=serial up.sh --dry-run`: models, memory fractions, context length, ports, flags). up.sh's default STARTUP=auto starts in
parallel only when the fingerprint matches, and otherwise falls back to serial: a changed setting can never meet a stale fixed size.
"""
import argparse, hashlib, json, math, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "kv_plan.json"
SERVICES = ("llm-a", "llm-b", "embedding", "reranker", "ovis-ocr", "hy-mt")
EXTENDED = re.compile(r"Extended KV cache to (\d+) blocks \(([0-9.]+) GiB\)")
GIB = 1 << 30
MARGIN = 0.99   # the measured size minus 1%: vLLM rounds the log to 0.01 GiB and a too-large fixed size is an OOM, a smaller one only fewer tokens


def config_key():
    """Fingerprint of GPU model + vLLM version + the serial launch plan of this up.sh with the current environment."""
    env = {**os.environ, "STARTUP": "serial"}
    def out(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    try:
        from importlib.metadata import version
        vllm = version("vllm")
    except Exception:
        vllm = ""
    material = "\n".join([out(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]), vllm, out(["bash", str(ROOT / "up.sh"), "--dry-run"])])
    return hashlib.sha256(material.encode()).hexdigest()[:20]


def parse_log(text):
    """(blocks, gib) of the LAST 'Extended KV cache' line (a restarted engine logs it again), or None."""
    found = EXTENDED.findall(text)
    return (int(found[-1][0]), float(found[-1][1])) if found else None


def learn(logs_dir, services=SERVICES):
    """({service: {blocks, gib} or {none: True}}, [services whose log is missing or never finished starting]).
    A service that started ("Application startup complete") without any KV line has no KV cache (encoder models)."""
    plan, missing = {}, []
    for name in services:
        f = pathlib.Path(logs_dir) / f"{name}.log"
        text = re.sub(r"\x1b\[[0-9;]*m", "", f.read_text(errors="replace")) if f.is_file() else ""
        got = parse_log(text)
        if got is not None:
            plan[name] = {"blocks": got[0], "gib": got[1]}
        elif "Application startup complete" in text:
            plan[name] = {"none": True}
        else:
            missing.append(name)
    return plan, missing


def kv_bytes(plan, service):
    """Fixed size in bytes, or None for a service without a KV cache; KeyError if the plan does not know the service."""
    entry = plan.get("kv", {}).get(service)
    if not entry:
        raise KeyError(service)
    if entry.get("none"):
        return None
    return int(math.floor(entry["gib"] * MARGIN * GIB))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arg"); a.add_argument("service"); a.add_argument("--plan", default=str(DEFAULT_PLAN))
    l = sub.add_parser("learn"); l.add_argument("--logs", required=True); l.add_argument("--out", default=str(DEFAULT_PLAN))
    l.add_argument("--note", default="")
    c = sub.add_parser("check"); c.add_argument("--plan", default=str(DEFAULT_PLAN))
    args = p.parse_args(argv)
    if args.cmd == "check":
        try:
            want = json.loads(pathlib.Path(args.plan).read_text()).get("key")
        except (OSError, ValueError):
            print(f"no KV plan at {args.plan}", file=sys.stderr); return 1
        if not want:
            print("the KV plan has no configuration fingerprint (re-run `kv_plan.py learn`)", file=sys.stderr); return 1
        if want != config_key():
            print("the KV plan was learned on a different configuration (GPU, vLLM version or launch settings changed)", file=sys.stderr); return 1
        return 0
    if args.cmd == "arg":
        try:
            plan = json.loads(pathlib.Path(args.plan).read_text())
            size = kv_bytes(plan, args.service)
            if size is not None:        # empty output = "no KV cache, add no flag"
                print(size)
        except (OSError, ValueError, KeyError) as e:
            print(f"kv_plan: no fixed KV size for '{args.service}' in {args.plan} ({type(e).__name__}); "
                  f"run a serial start and `kv_plan.py learn`, or use STARTUP=serial", file=sys.stderr)
            return 1
        return 0
    plan, missing = learn(args.logs)
    if missing:
        print(f"kv_plan: no 'Extended KV cache' line in the logs of: {', '.join(missing)}", file=sys.stderr)
        return 1
    out = {"note": args.note or "measured on a serial start; see kv_plan.py", "key": config_key(), "margin": MARGIN, "kv": plan}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v.get("gib", "no KV cache") for k, v in plan.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
