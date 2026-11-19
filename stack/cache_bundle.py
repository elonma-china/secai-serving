"""Save/restore the compile caches ($CACHE_DIR: torch.compile/Inductor, Triton, FlashInfer JIT) via a private HF repo.

A new machine otherwise recompiles every service on first start. The bundle key hashes everything that invalidates
those caches: vLLM version, GPU model, and the exact launch plan (`up.sh --dry-run`, i.e. models and all flags).
A different key is simply a miss; vLLM then compiles as usual and `save` can publish the new bundle.

  cache_bundle.py key                 # print the key for this machine/config
  cache_bundle.py restore             # download + unpack if a bundle exists; exit 3 = no bundle for this key
  cache_bundle.py save                # pack $CACHE_DIR and upload it (needs HF_TOKEN with write access)
Env: CACHE_DIR (/workspace/cache), CACHE_REPO (secai1999/serving-compile-cache), HF_TOKEN.
"""
import hashlib, os, pathlib, shutil, subprocess, sys, tarfile, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent
CACHE_DIR = pathlib.Path(os.environ.get("CACHE_DIR", "/workspace/cache"))
REPO = os.environ.get("CACHE_REPO", "secai1999/serving-compile-cache")


def _out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def key_material():
    gpus = _out(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    vllm = _out([sys.executable, "-c", "import vllm; print(vllm.__version__)"])
    plan = _out(["bash", str(ROOT / "up.sh"), "--dry-run"])
    if not (gpus and vllm and plan):
        raise SystemExit("cannot compute cache key (need nvidia-smi, vllm and a working up.sh --dry-run)")
    return f"gpus={gpus}\nvllm={vllm}\nplan={plan}\n"


def key():
    return hashlib.sha256(key_material().encode()).hexdigest()[:20]


def _zstd():
    return shutil.which("zstd")


def restore():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    k, t0 = key(), time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE_DIR.parent) as tmp:
        for name in (f"{k}.tar.zst", f"{k}.tar.gz"):
            try:
                path = hf_hub_download(REPO, name, repo_type="model", local_dir=tmp)
                break
            except (EntryNotFoundError, RepositoryNotFoundError):
                continue
            except Exception as e:  # network/auth problems must not block the boot: vLLM just compiles
                print(f"cache restore skipped: {type(e).__name__}: {e}"); return 0
        else:
            print(f"no cache bundle for key {k}; vLLM will compile (run `cache_bundle.py save` once the stack is up)")
            return 3
        if name.endswith(".zst"):
            subprocess.run(["tar", "-I", "zstd -d -T0", "-xf", path, "-C", str(CACHE_DIR)], check=True)
        else:
            with tarfile.open(path) as tf:
                tf.extractall(CACHE_DIR, filter="data")
    print(f"restored cache bundle {name} in {time.time() - t0:.0f}s")
    return 0


def save():
    from huggingface_hub import HfApi
    if not CACHE_DIR.is_dir() or not any(CACHE_DIR.iterdir()):
        raise SystemExit(f"{CACHE_DIR} is empty: start the stack first")
    k, api = key(), HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE_DIR.parent) as tmp:
        if _zstd():
            out = pathlib.Path(tmp) / f"{k}.tar.zst"
            subprocess.run(["tar", "-I", "zstd -T0 -10", "-cf", str(out), "-C", str(CACHE_DIR), "."], check=True)
        else:
            out = pathlib.Path(tmp) / f"{k}.tar.gz"
            subprocess.run(["tar", "-czf", str(out), "-C", str(CACHE_DIR), "."], check=True)
        (pathlib.Path(tmp) / f"{k}.key.txt").write_text(key_material())
        for f in (out, pathlib.Path(tmp) / f"{k}.key.txt"):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=REPO, repo_type="model")
        print(f"saved {out.name} ({out.stat().st_size / 1e6:.0f} MB) to {REPO}")
    return 0


def main(argv):
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    cmd = argv[0] if argv else ""
    if cmd == "key":
        print(key()); return 0
    if cmd == "restore":
        return restore()
    if cmd == "save":
        return save()
    print(__doc__, file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
