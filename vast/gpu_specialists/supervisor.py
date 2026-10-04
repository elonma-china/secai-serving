"""Small root-owned supervisor: restart only its two CPU ASGI workers."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main(root):
    root = Path(root)
    lock = (root / "supervisor.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("specialist supervisor already running")
    config = json.loads((root / "config.json").read_text())
    pid_file = root / "supervisor.pid"
    pid_file.write_text(str(os.getpid()))
    stopping = False
    children = {}
    retry_at = {}

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            for name in ("mt", "ocr"):
                if not config["enabled"][name]:
                    continue
                child = children.get(name)
                if child is not None and child.poll() is not None:
                    print(f"{name} exited {child.returncode}; restarting after 5s", flush=True)
                    del children[name]
                    retry_at[name] = time.monotonic() + 5
                if name not in children and time.monotonic() >= retry_at.get(name, 0):
                    with (root / "logs" / f"{name}.log").open("ab") as log:
                        children[name] = subprocess.Popen(
                            [config["python"], str(root / "worker.py"), str(root), name],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            time.sleep(.5)
    finally:
        # Each worker first cancels/reaps its own disposable GPU child. Do not
        # kill process groups or GPU host PIDs; a held GPU flock must survive.
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            child.wait()
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1])
