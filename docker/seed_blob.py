"""Copy one public Docker Hub blob into a GHCR repo (digest-verified). Usage: seed_blob.py <sha256:digest> <src_repo> <dst_repo>
Fixes `docker push` failing with "NotFound: content digest ..." for a base-image layer missing from the local store.
The GitHub token comes from `gh auth token` (needs write:packages) and is never printed."""
import base64, hashlib, json, subprocess, sys, urllib.parse, urllib.request
digest, src_repo, dst_repo = sys.argv[1], sys.argv[2], sys.argv[3]
def get(url, headers=None, data=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    return urllib.request.urlopen(req, timeout=300)
tok = json.load(get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{src_repo}:pull"))["token"]
blob = get(f"https://registry-1.docker.io/v2/{src_repo}/blobs/{digest}", {"Authorization": f"Bearer {tok}"}).read()
assert "sha256:" + hashlib.sha256(blob).hexdigest() == digest, "downloaded blob digest mismatch"
gh = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
basic = base64.b64encode(f"elonma-china:{gh}".encode()).decode()
gt = json.load(get(f"https://ghcr.io/token?service=ghcr.io&scope=repository:{dst_repo}:push,pull", {"Authorization": f"Basic {basic}"}))["token"]
H = {"Authorization": f"Bearer {gt}"}
r = get(f"https://ghcr.io/v2/{dst_repo}/blobs/uploads/", H, b"", "POST")
loc = r.headers["Location"]; loc = loc if loc.startswith("http") else "https://ghcr.io" + loc
put = loc + ("&" if "?" in loc else "?") + "digest=" + urllib.parse.quote(digest)
r = get(put, {**H, "Content-Type": "application/octet-stream", "Content-Length": str(len(blob))}, blob, "PUT")
print("uploaded", digest[:19], len(blob), "bytes, HTTP", r.status)
chk = urllib.request.Request(f"https://ghcr.io/v2/{dst_repo}/blobs/{digest}", headers=H, method="HEAD")
print("exists on ghcr: HTTP", urllib.request.urlopen(chk).status)
