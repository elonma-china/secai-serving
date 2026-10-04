"""Install and control the accepted on-demand GPU workers (no Vast API calls)."""
import argparse
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OCR_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
OCR_SHA = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
MT_REV = "ab8472660ac61fac25f1af43fac2599d52a8a775"
MT_SHA = "9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b"
LLAMA_REV = "11fe02151f79c41d0d4af7da708755d73b9c0da6"
COMPAT_SHA = "f27223c58d4c0d2ead3c2d747eb30a530c449f89c42e16b23e9bef04f3e6dc2e"
DEPENDENCIES = ["transformers==5.13.1", "fastapi==0.136.3", "pydantic==2.13.4",
                "Pillow==12.3.0", "uvicorn==0.51.0", "huggingface-hub==1.23.0",
                "safetensors==0.8.0", "accelerate==1.14.0", "httpx==0.28.1"]


def run(*args, **kwargs):
    return subprocess.check_output([str(arg) for arg in args], text=True, **kwargs).strip()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path, expected):
    if sha(path) != expected:
        raise RuntimeError(f"artifact checksum mismatch: {path}")


def private_json(path, data):
    temporary = path.with_suffix(".new")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as sink:
        json.dump(data, sink, indent=2)
        sink.write("\n")
    temporary.replace(path)


def extract_source(root):
    manifest = json.loads((HERE / "runtime-manifest.json").read_text())
    check_hash(HERE / "runtime.tar.gz", manifest["bundle_sha256"])
    destination = root / "source"
    with tarfile.open(HERE / "runtime.tar.gz", "r:gz") as archive:
        for member in archive:
            if member.name not in manifest["files"] or not member.isfile():
                raise RuntimeError("unexpected runtime bundle member")
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    for name, digest in manifest["files"].items():
        check_hash(destination / name, digest)
    if HERE / "runtime-manifest.json" != root / "runtime-manifest.json":
        shutil.copyfile(HERE / "runtime-manifest.json", root / "runtime-manifest.json")


def token_file(root, supplied, name):
    path = Path(supplied).resolve() if supplied else root / f"{name}.token"
    if not path.exists():
        if supplied:
            raise RuntimeError(f"supplied token file missing: {path}")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as sink:
            sink.write(secrets.token_urlsafe(32) + "\n")
    import re
    if path.stat().st_mode & 0o077 or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", path.read_text().strip()):
        raise RuntimeError(f"invalid/private token file required: {path}")
    return str(path)


def capture_gpu(config, gpu_id, expected_count):
    lock_path = Path(config["environment"]["OCR_SHARED_GPU_LOCK_PATH"])
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Re-use the accepted durable cleanup veto before adopting a baseline.
        sys.path.insert(0, str(Path(config["root"]) / "source"))
        from deploy.gpu_stage.owner import GpuStageOwner
        from deploy.gpu_stage.coordinator import GpuBudget
        GpuStageOwner(lock_path, GpuBudget("baseline", frozenset()), None)._check_history()
        smi = shutil.which("nvidia-smi")
        line = run(smi, "--id", gpu_id, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits")
        if len(line.splitlines()) != 1:
            raise RuntimeError("select exactly one specialist GPU")
        uuid, free = [part.strip() for part in line.split(",")]
        rows = run(smi, "--id", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits")
        pids = sorted(int(pid) for pid in rows.splitlines())
        if len(pids) != expected_count or int(free) < 8192:
            raise RuntimeError("GPU baseline requires expected incumbent count and >=8192 MiB free")
        config["gpu"] = {"uuid": uuid, "incumbents": pids}
        config["environment"]["CUDA_VISIBLE_DEVICES"] = uuid
        config["environment"]["SECAI_SPECIALIST_GPU_UUID"] = uuid
        root = Path(config["root"])
        (root / "bin").mkdir(exist_ok=True)
        wrapper = root / "bin/nvidia-smi"
        import shlex
        wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(smi) +
                           ' --id "$SECAI_SPECIALIST_GPU_UUID" "$@"\n')
        wrapper.chmod(0o700)


def prepare_python(args, root):
    # Resolving a venv's Python symlink would bypass that venv's site-packages.
    python = Path(os.path.abspath(args.python)) if args.python else root / "venv/bin/python"
    if args.python and (not python.is_file() or not os.access(python, os.X_OK)):
        raise RuntimeError("explicit --python must name an existing executable")
    if not python.exists():
        if args.reuse_artifacts:
            raise RuntimeError("--reuse-artifacts requires an existing --python or installed venv")
        run(sys.executable, "-m", "venv", "--system-site-packages", root / "venv")
    pins = DEPENDENCIES if not args.disable_ocr else [pin for pin in DEPENDENCIES
           if pin.split("==")[0] in {"fastapi", "pydantic", "uvicorn", "httpx", "huggingface-hub"}]
    if not args.reuse_artifacts:
        # Keep vLLM's environment intact; all auxiliary additions live in a venv.
        run(python, "-m", "pip", "install", *pins)
        if not args.disable_ocr and run(python, "-c", "import importlib.util; print(__import__('torch').__version__ if importlib.util.find_spec('torch') else '')") != "2.11.0+cu130":
            run(python, "-m", "pip", "install", "torch==2.11.0", "--index-url",
                "https://download.pytorch.org/whl/cu130")
    code = ("from importlib.metadata import version\n"
            "pins=" + repr(pins) + "\n"
            "for pin in pins:\n name,v=pin.split('=='); assert version(name)==v,(name,version(name),v)\n")
    if not args.disable_ocr:
        code += "import torch; assert torch.__version__=='2.11.0+cu130',torch.__version__\n"
    run(python, "-c", code)
    return str(python)


def prepare_models(args, python):
    ocr = Path(args.ocr_model_path)
    mt = Path(args.mt_model_path)
    if not args.reuse_artifacts:
        code = "from huggingface_hub import snapshot_download,hf_hub_download; "
        if not args.disable_ocr:
            code += f"snapshot_download('Qwen/Qwen3-VL-2B-Instruct',revision={OCR_REV!r},local_dir={str(ocr)!r}," + \
                    "allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE*'],max_workers=2); "
        if not args.disable_mt:
            code += f"hf_hub_download('tencent/Hy-MT2-7B-GGUF','Hy-MT2-7B-Q4_K_M.gguf',revision={MT_REV!r},local_dir={str(mt.parent)!r})"
        run(python, "-c", code)
    if not args.disable_ocr:
        check_hash(ocr / "model.safetensors", OCR_SHA)
    if not args.disable_mt:
        check_hash(mt, MT_SHA)


def prepare_mt(args):
    server = Path(args.mt_server_path)
    llama = server.parents[2]
    built = False
    if not server.exists() and not args.reuse_artifacts:
        compiler = ("/usr/local/cuda-13.0/bin/nvcc" if Path("/usr/local/cuda-13.0/bin/nvcc").exists()
                    else shutil.which("nvcc"))
        if compiler is None or "release 13.0" not in run(compiler, "--version"):
            # Only in a fresh install; --reuse-artifacts never installs packages.
            keyring = Path(args.root) / "cuda-keyring.deb"
            urllib.request.urlretrieve("https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
                                       "cuda-keyring_1.1-1_all.deb", keyring)
            run("dpkg", "-i", keyring)
            run("apt-get", "update", "-qq")
            run("apt-get", "install", "-y", "-qq", "cuda-toolkit-13-0=13.0.2-1")
            compiler = "/usr/local/cuda-13.0/bin/nvcc"
        if not (llama / ".git").exists():
            run("git", "clone", "https://github.com/ggml-org/llama.cpp.git", llama)
        run("git", "-C", llama, "checkout", "--detach", LLAMA_REV)
        run("cmake", "-S", llama, "-B", server.parent.parent, "-DGGML_CUDA=ON",
            "-DCMAKE_CUDA_COMPILER=" + compiler,
            "-DCMAKE_CUDA_ARCHITECTURES=80", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release")
        run("cmake", "--build", server.parent.parent, "--target", "llama-server", "-j", "4")
        built = True
    if run("git", "-C", llama, "rev-parse", "HEAD") != LLAMA_REV:
        raise RuntimeError("llama.cpp source revision mismatch")
    compat = Path(args.cuda_compat_path)
    if not compat.exists() and not args.reuse_artifacts:
        # The accepted CUDA forward-compat library, extracted without apt changes.
        package = llama / "cuda-compat-13-0.deb"
        urllib.request.urlretrieve("https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
                                   "cuda-compat-13-0_580.95.05-0ubuntu1_amd64.deb", package)
        extraction = Path(args.root) / "cuda-compat"
        run("dpkg-deb", "--extract", package, extraction)
        compat = extraction / "usr/local/cuda-13.0/compat/libcuda.so.580.95.05"
    check_hash(compat, COMPAT_SHA)
    libraries = {path.name: sha(path) for path in server.parent.glob("*.so*") if not path.is_symlink()}
    if not libraries or not os.access(server, os.X_OK):
        raise RuntimeError("CUDA llama-server/shared libraries missing")
    generated_manifest = server.parent.parent / "secai-runtime-manifest.json"
    if not built:
        if generated_manifest.exists():
            pins = json.loads(generated_manifest.read_text())
            if pins["llama_revision"] != LLAMA_REV:
                raise RuntimeError("generated runtime source revision mismatch")
        else:
            pins = json.loads((HERE / "runtime-manifest.json").read_text())["mt_pins"]
        check_hash(server, pins["server_sha256"])
        for name, expected in pins["libraries"].items():
            check_hash(server.parent / name, expected)
    artifacts = {"server": str(server), "server_sha256": sha(server),
            "model": args.mt_model_path, "model_sha256": MT_SHA,
            "compat": str(compat), "compat_sha256": COMPAT_SHA,
            "libraries": libraries, "llama_revision": LLAMA_REV}
    if built:
        artifacts["compiler"] = run(compiler, "--version")
        private_json(generated_manifest, artifacts)
    return artifacts


def attest(config):
    root = Path(config["root"])
    manifest = json.loads((root / "runtime-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        check_hash(root / "source" / name, digest)
    if config["enabled"]["ocr"]:
        check_hash(Path(config["environment"]["OCR_MODEL_PATH"]) / "model.safetensors", OCR_SHA)
        for path, digest in config["ocr_artifacts"].items():
            check_hash(path, digest)
    if config["enabled"]["mt"]:
        artifacts = config["mt_artifacts"]
        for key in ("server", "model", "compat"):
            check_hash(artifacts[key], artifacts[key + "_sha256"])
        for name, digest in artifacts["libraries"].items():
            check_hash(Path(artifacts["server"]).parent / name, digest)


def is_running(root):
    path = root / "supervisor.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def start(root):
    if is_running(root):
        return
    config = json.loads((root / "config.json").read_text())
    attest(config)
    with (root / "logs/supervisor.log").open("ab") as log:
        subprocess.Popen([config["python"], str(root / "supervisor.py"), str(root)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    until = time.monotonic() + 10
    while not is_running(root):
        if time.monotonic() > until:
            raise RuntimeError("specialist supervisor did not start")
        time.sleep(.1)


def stop(root):
    if not is_running(root):
        return
    pid = int((root / "supervisor.pid").read_text())
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if str(root / "supervisor.py").encode() not in command:
        raise RuntimeError("supervisor PID identity mismatch")
    os.kill(pid, signal.SIGTERM)
    until = time.monotonic() + 150
    while is_running(root):
        if time.monotonic() > until:
            raise RuntimeError("cleanup still pending; supervisor left alive")
        time.sleep(.5)


def post(config, name, body, token=True, timeout=4):
    path = "/v1/translate" if name == "mt" else "/v1/ocr"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + Path(config[name + "_token_file"]).read_text().strip()
    req = urllib.request.Request(f"http://127.0.0.1:{config[name + '_port']}{path}",
                                 json.dumps(body).encode(), headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def ready(root, smoke=False):
    config = json.loads((root / "config.json").read_text())
    attest(config)
    for name in ("mt", "ocr"):
        if not config["enabled"][name]:
            continue
        body = {"text": "test", "source_lang": "en", "target_lang": "en"} if name == "mt" else {
            "model": "invalid-readiness-model", "image_base64": ""}
        # Check actual private routes and model identity validation, not /health.
        if post(config, name, body, token=False)[0] != 401:
            raise RuntimeError(f"{name} auth readiness failed")
        expected = 200 if name == "mt" else 422
        if post(config, name, body)[0] != expected:
            raise RuntimeError(f"{name} route readiness failed")
        if smoke:
            if name == "mt":
                body["text"] = "Hello."
                body["target_lang"] = "vi"
            else:
                encoded = run(config["python"], "-c", "import io,base64; from PIL import Image,ImageDraw; "
                              "im=Image.new('RGB',(512,192),'white'); "
                              "ImageDraw.Draw(im).text((20,55),'SECAI 2026',fill='black',font_size=40); "
                              "buf=io.BytesIO(); im.save(buf,format='PNG'); "
                              "print(base64.b64encode(buf.getvalue()).decode())")
                body = {"model": "Qwen/Qwen3-VL-2B-Instruct", "calibration_prompt": "conditional",
                        "image_base64": encoded}
            status, result = post(config, name, body, timeout=120)
            content = result.get("translated_text") if name == "mt" else " ".join(
                block.get("text", "") for block in result.get("blocks", []))
            if status != 200 or not content.strip():
                raise RuntimeError(f"{name} inference smoke failed: HTTP {status}")
    print("specialist artifact/auth/route readiness OK" + ("; sequential inference OK" if smoke else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "start", "stop", "restart", "boot", "ready"])
    parser.add_argument("--root", default="/workspace/secai-specialist-services")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--expected-incumbents", type=int, default=3)
    parser.add_argument("--python")
    parser.add_argument("--mt-token-file")
    parser.add_argument("--ocr-token-file")
    parser.add_argument("--ocr-model-path", default="/workspace/secai-translation-ocr/qwen3vl-model")
    parser.add_argument("--mt-model-path", default="/workspace/secai-gpu-specialists/model/Hy-MT2-7B-Q4_K_M.gguf")
    parser.add_argument("--mt-server-path", default="/workspace/secai-gpu-specialists/runtime/llama.cpp/build-cuda/bin/llama-server")
    parser.add_argument("--cuda-compat-path", default="/usr/local/cuda-13.0/compat/libcuda.so.580.95.05")
    parser.add_argument("--shared-lock", default="/workspace/secai-gpu-specialists/admission.lock")
    parser.add_argument("--mt-port", type=int, default=18868)
    parser.add_argument("--ocr-port", type=int, default=18083)
    parser.add_argument("--disable-mt", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--reuse-artifacts", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.dry_run:
        print(f"DRY-RUN {args.action}: root={root}; GPU={args.gpu}; MT=127.0.0.1:{args.mt_port}; "
              f"OCR=127.0.0.1:{args.ocr_port}; shared-lock={args.shared_lock}; reuse={args.reuse_artifacts}")
        return
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    (root / "logs").mkdir(mode=0o700, exist_ok=True)
    if args.action == "install":
        if is_running(root):
            raise RuntimeError("stop existing specialist supervisor before reinstalling")
        extract_source(root)
        if not args.reuse_artifacts:
            if shutil.disk_usage(root).free < 30 * 1024**3:
                raise RuntimeError("fresh specialist install requires >=30 GiB free disk")
            missing = [name for name in ("cmake", "git", "ss", "gcc") if not shutil.which(name)]
            if missing:
                run("apt-get", "update", "-qq")
                run("apt-get", "install", "-y", "-qq", "cmake", "git", "iproute2", "build-essential", "python3-venv")
        python = prepare_python(args, root)
        for filename in ("worker.py", "supervisor.py", "bootstrap.py"):
            if HERE / filename != root / filename:
                shutil.copyfile(HERE / filename, root / filename)
        config = {"root": str(root), "python": python, "gpu_selector": args.gpu,
                  "expected_incumbents": args.expected_incumbents,
                  "mt_port": args.mt_port, "ocr_port": args.ocr_port,
                  "enabled": {"mt": not args.disable_mt, "ocr": not args.disable_ocr},
                  "mt_token_file": token_file(root, args.mt_token_file, "mt"),
                  "ocr_token_file": token_file(root, args.ocr_token_file, "ocr"),
                  "environment": {"GPU_MT_ON_DEMAND_ENABLED": "1", "OCR_MODEL_ID": "Qwen/Qwen3-VL-2B-Instruct",
                                  "OCR_MODEL_PATH": args.ocr_model_path, "OCR_MAX_PEAK_MIB": "6144",
                                  "OCR_SHARED_GPU_LOCK_PATH": args.shared_lock, "HF_HUB_OFFLINE": "1",
                                  "TRANSFORMERS_OFFLINE": "1", "GPU_STAGE_LOG_DIR": str(root / "logs"),
                                  "LD_LIBRARY_PATH": str(Path(args.cuda_compat_path).parent) + ":/usr/local/cuda-13.0/lib64"}}
        config["environment"]["GPU_STAGE_MT_TOKEN_FILE"] = config["mt_token_file"]
        prepare_models(args, python)
        if not args.disable_ocr:
            config["ocr_artifacts"] = {str(path): sha(path) for path in Path(args.ocr_model_path).iterdir()
                                       if path.is_file() and path.suffix in {".json", ".jinja", ".txt", ".model"}}
        if not args.disable_mt:
            config["mt_artifacts"] = prepare_mt(args)
            config["environment"]["LD_LIBRARY_PATH"] = str(Path(config["mt_artifacts"]["compat"]).parent) + ":/usr/local/cuda-13.0/lib64"
        capture_gpu(config, args.gpu, args.expected_incumbents)
        private_json(root / "config.json", config)
        start(root)
    elif args.action in ("stop", "restart"):
        stop(root)
        if args.action == "restart":
            start(root)
    elif args.action == "boot":
        config = json.loads((root / "config.json").read_text())
        if not is_running(root):
            attest(config)
            capture_gpu(config, config["gpu_selector"], config["expected_incumbents"])
            private_json(root / "config.json", config)
            start(root)
    elif args.action == "start":
        start(root)
    if args.action in ("ready", "install", "start", "restart", "boot"):
        until = time.monotonic() + 60
        while True:
            try:
                ready(root, args.smoke)
                break
            except (OSError, urllib.error.URLError, RuntimeError):
                if time.monotonic() >= until:
                    raise
                time.sleep(1)


if __name__ == "__main__":
    main()
