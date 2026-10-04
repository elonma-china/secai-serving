"""Rebuild the deterministic runtime bundle from a reviewed gateway commit."""
import argparse
import ast
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-repo", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    def git(*command):
        return subprocess.check_output(["git", "-C", args.gateway_repo, *command])
    revision = git("rev-parse", args.revision + "^{commit}").decode().strip()
    prefix = git("rev-parse", "--show-prefix").decode().strip()
    current = json.loads((HERE / "runtime-manifest.json").read_text())
    manifest = {"gateway_revision": revision, "files": {}}
    buffer = io.BytesIO()
    pins_source = None
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in current["files"]:
            data = git("show", revision + ":" + prefix + name)
            if name == "evals/multilang/gpu_specialist_server.py":
                pins_source = data
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    bundle = gzip.compress(buffer.getvalue(), compresslevel=9, mtime=0)
    manifest["bundle_sha256"] = hashlib.sha256(bundle).hexdigest()
    values = {node.targets[0].id: ast.literal_eval(node.value)
              for node in ast.parse(pins_source).body if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in {"SERVER_SHA256", "RUNTIME_SO_HASHES"}}
    manifest["mt_pins"] = {"server_sha256": values["SERVER_SHA256"], "libraries": values["RUNTIME_SO_HASHES"]}
    (HERE / "runtime.tar.gz").write_bytes(bundle)
    (HERE / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Pinned gateway {revision}; bundle {len(bundle)} bytes; SHA256 {manifest['bundle_sha256']}")


if __name__ == "__main__":
    main()
