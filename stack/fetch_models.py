"""Download all five models in parallel into $MODEL_DIR/<repo name>, pinned to stack/models.lock.

vLLM otherwise pulls each model lazily, one service at a time, while the previous one compiles; fetching all ~32 GB
up front in parallel (Xet, high-performance mode) takes the network off the critical path. A finished model gets a
`.secai-revision` marker, so a re-run on a warm disk is a no-op. Usage: fetch_models.py [--dry-run] [--offline] [VAR ...]   (VAR e.g. LLM_MODEL: only those)
"""
import argparse, os, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent
MARKER = ".secai-revision"


def load_lock(path=ROOT / "models.lock", env=os.environ):
    """[(env_var, repo, revision)]; an env override of the repo uses <VAR>_REVISION (default main)."""
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].split()
        if not line:
            continue
        var, repo, rev = line
        if env.get(var, repo) != repo:
            repo, rev = env[var], env.get(f"{var}_REVISION", "main")
        out.append((var, repo, rev))
    return out


def target(model_dir, repo):
    return pathlib.Path(model_dir) / repo.rsplit("/", 1)[-1]


def is_done(model_dir, repo, rev):
    m = target(model_dir, repo) / MARKER
    return m.is_file() and m.read_text().strip() == f"{repo}@{rev}"


def fetch(model_dir, repo, rev, log=print):
    from huggingface_hub import HfApi, snapshot_download
    dest = target(model_dir, repo)
    if is_done(model_dir, repo, rev):
        log(f"{repo}: present ({rev[:12]})"); return dest
    t0 = time.time()
    snapshot_download(repo, revision=rev, local_dir=str(dest), max_workers=16)
    # Size check against the Hub metadata catches truncated files; Xet already verifies chunk hashes.
    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    bad = [s.rfilename for s in info.siblings if s.size is not None and
           (not (dest / s.rfilename).is_file() or (dest / s.rfilename).stat().st_size != s.size)]
    if bad:
        raise RuntimeError(f"{repo}: size mismatch for {bad[:5]}")
    (dest / MARKER).write_text(f"{repo}@{rev}\n")
    gb = sum(s.size or 0 for s in info.siblings) / 1e9
    log(f"{repo}: {gb:.1f} GB in {time.time() - t0:.0f}s"); return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="print what would be fetched")
    p.add_argument("--offline", action="store_true", help="only check that every pinned model is already present")
    p.add_argument("only", nargs="*", help="limit to these env var names (default: all)")
    a = p.parse_args(argv)
    model_dir = os.environ.get("MODEL_DIR", "/workspace/models")
    models = [m for m in load_lock() if not a.only or m[0] in a.only]
    if not models:
        p.error(f"no model matches {a.only}")
    if a.dry_run:
        for var, repo, rev in models:
            print(f"{var}={repo}@{rev} -> {target(model_dir, repo)}")
        return 0
    if a.offline:
        missing = [repo for _, repo, rev in models if not is_done(model_dir, repo, rev)]
        print("missing: " + ", ".join(missing) if missing else "all models present")
        return 1 if missing else 0
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    t0 = time.time()
    log = lambda msg: print(time.strftime("%H:%M:%S"), msg, flush=True)
    with ThreadPoolExecutor(len(models)) as ex:
        futs = [ex.submit(fetch, model_dir, repo, rev, log) for _, repo, rev in models]
        errors = [f.exception() for f in futs if f.exception()]
    for e in errors:
        log(f"ERROR {e}")
    log(f"fetch done in {time.time() - t0:.0f}s, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
