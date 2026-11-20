"""Release a code / compile-cache-seed change as ONE extra layer on top of an image that is already in the registry.

Rebuilding the image locally needs the 36 GB base on disk (~90 GB unpacked) and a re-push. Because the code and the cache seed sit above the
model layers, a change to them is just a new top layer: this tool builds that layer, uploads it plus a new config and manifest straight to
the registry (like `crane append`), and tags the result. Nothing else is touched; the base image and its tag stay valid for rollback.

  append_layer.py --base repo@sha256:... --tag NEW_TAG --revision GITSHA \\
      --add stack=/opt/secai/stack --add cache-seed-dir=/opt/secai/cache-seed [--dry-run]

--add LOCAL=DEST adds a file or a whole directory (mode bits and exec flags kept, owner root, fixed mtime). Files override the same paths of lower
layers. The package must already exist and the GitHub token (`gh auth token`, scope write:packages) never gets printed.
"""
import argparse, copy, datetime, gzip, hashlib, io, json, os, pathlib, subprocess, sys, tarfile, urllib.error, urllib.parse, urllib.request

OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
OCI_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def build_layer(adds, mtime=0):
    """(gzipped layer bytes, diff_id) for [(local_path, dest_path)]; deterministic for equal inputs."""
    raw = io.BytesIO()
    seen_dirs = set()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        def add_dir(name):
            parts = pathlib.PurePosixPath(name).parts
            for i in range(1, len(parts) + 1):
                d = "/".join(parts[:i])
                if d not in seen_dirs:
                    seen_dirs.add(d)
                    ti = tarfile.TarInfo(d + "/"); ti.type = tarfile.DIRTYPE; ti.mode = 0o755; ti.mtime = mtime
                    tar.addfile(ti)
        def add_file(path, name):
            add_dir(str(pathlib.PurePosixPath(name).parent))
            ti = tarfile.TarInfo(name); data = path.read_bytes()
            ti.size = len(data); ti.mode = 0o755 if os.access(path, os.X_OK) else 0o644; ti.mtime = mtime
            tar.addfile(ti, io.BytesIO(data))
        for local, dest in adds:
            local = pathlib.Path(local); dest = dest.strip("/")
            if local.is_dir():
                for f in sorted(p for p in local.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"):
                    add_file(f, f"{dest}/{f.relative_to(local).as_posix()}")
            elif local.is_file():
                add_file(local, dest)
            else:
                raise SystemExit(f"--add: {local} does not exist")
    raw_bytes = raw.getvalue()
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, compresslevel=9) as g:
        g.write(raw_bytes)
    return gz.getvalue(), sha(raw_bytes)


class Registry:
    def __init__(self, repo, token, host="ghcr.io"):
        self.repo, self.base = repo, f"https://{host}/v2/{repo}"
        self.h = {"Authorization": f"Bearer {token}"}

    def _req(self, url, method="GET", data=None, headers=None):
        req = urllib.request.Request(url, data=data, method=method, headers={**self.h, **(headers or {})})
        return urllib.request.urlopen(req, timeout=600)

    def get_json(self, path, accept=OCI_MANIFEST):
        with self._req(f"{self.base}/{path}", headers={"Accept": accept}) as r:
            return json.loads(r.read()), r.headers.get("Docker-Content-Digest")

    def has_blob(self, digest):
        try:
            self._req(f"{self.base}/blobs/{digest}", "HEAD").close(); return True
        except urllib.error.HTTPError as e:
            if e.code == 404: return False
            raise

    def put_blob(self, data):
        digest = sha(data)
        if self.has_blob(digest):
            return digest
        with self._req(f"{self.base}/blobs/uploads/", "POST", b"") as r:
            loc = r.headers["Location"]
        loc = loc if loc.startswith("http") else "https://ghcr.io" + loc
        url = loc + ("&" if "?" in loc else "?") + "digest=" + urllib.parse.quote(digest)
        self._req(url, "PUT", data, {"Content-Type": "application/octet-stream", "Content-Length": str(len(data))}).close()
        return digest

    def put_manifest(self, tag, manifest_bytes):
        with self._req(f"{self.base}/manifests/{tag}", "PUT", manifest_bytes, {"Content-Type": OCI_MANIFEST}) as r:
            return r.headers.get("Docker-Content-Digest")


def new_config(config, diff_id, revision, note):
    c = copy.deepcopy(config)
    c["rootfs"]["diff_ids"].append(diff_id)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    c["created"] = now
    c.setdefault("history", []).append({"created": now, "created_by": f"append_layer.py {note}"})
    if revision:
        env = [e for e in c["config"].get("Env", []) if not e.startswith("SECAI_IMAGE_REVISION=")]
        c["config"]["Env"] = env + [f"SECAI_IMAGE_REVISION={revision}"]
        c["config"].setdefault("Labels", {})["org.opencontainers.image.revision"] = revision
    return c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="repo@sha256:digest of the image to extend, e.g. ghcr.io/elonma-china/secai-serving@sha256:...")
    p.add_argument("--tag", required=True); p.add_argument("--revision", default="")
    p.add_argument("--add", action="append", required=True, metavar="LOCAL=DEST"); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    host_repo, _, digest = a.base.partition("@")
    host, _, repo = host_repo.partition("/")
    adds = [tuple(x.split("=", 1)) for x in a.add]
    layer, diff_id = build_layer(adds)
    print(f"layer: {len(layer) / 1e6:.1f} MB gzip, {sum(1 for _ in adds)} --add, diff_id {diff_id[:19]}")
    if a.dry_run:
        return 0
    gh = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    import base64
    basic = base64.b64encode(f"x:{gh}".encode()).decode()
    q = urllib.parse.urlencode({"service": host, "scope": f"repository:{repo}:push,pull"})
    with urllib.request.urlopen(urllib.request.Request(f"https://{host}/token?{q}", headers={"Authorization": f"Basic {basic}"})) as r:
        token = json.load(r)["token"]
    reg = Registry(repo, token, host)
    manifest, got = reg.get_json(f"manifests/{digest}")
    if manifest.get("mediaType") != OCI_MANIFEST or got != digest:
        raise SystemExit(f"base must be a single OCI manifest with digest {digest} (got {manifest.get('mediaType')} {got})")
    config, _ = reg.get_json(f"blobs/{manifest['config']['digest']}", accept="application/json")
    layer_digest = reg.put_blob(layer)
    cfg_bytes = json.dumps(new_config(config, diff_id, a.revision, f"{a.tag}"), separators=(",", ":")).encode()
    cfg_digest = reg.put_blob(cfg_bytes)
    m2 = copy.deepcopy(manifest)
    m2["layers"].append({"mediaType": OCI_LAYER, "digest": layer_digest, "size": len(layer)})
    m2["config"] = {"mediaType": OCI_CONFIG, "digest": cfg_digest, "size": len(cfg_bytes)}
    out = reg.put_manifest(a.tag, json.dumps(m2, separators=(",", ":")).encode())
    missing = [l["digest"][:19] for l in m2["layers"] + [m2["config"]] if not reg.has_blob(l["digest"])]
    print(f"pushed {host_repo}:{a.tag}  digest {out}  layers {len(m2['layers'])}  missing blobs: {missing or 'none'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
