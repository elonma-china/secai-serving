import json
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "stack" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mt = load("mt_adapter")


def test_default_prompt():
    p = mt.build_prompt("你好", "Vietnamese")
    assert p.startswith("Translate the following text into Vietnamese.") and p.endswith("你好")


def test_glossary_prompt():
    p = mt.build_prompt("hello", "French", glossary={"hello": "bonjour"})
    assert "hello translates to bonjour" in p and "ONLY output" in p


def test_single_mode_only():
    for kwargs in ({"style": "formal", "context": "x"}, {"glossary": {"a": "b"}, "format_type": "JSON"}):
        try:
            mt.build_prompt("t", "German", **kwargs)
        except ValueError:
            continue
        raise AssertionError("mixed modes accepted")


def test_rejects_bad_input():
    for text, lang in (("", "German"), ("x", ""), ("x" * 70_000, "German")):
        try:
            mt.build_prompt(text, lang)
        except ValueError:
            continue
        raise AssertionError("bad input accepted")


def test_sampling_matches_model_card():
    r = mt.build_request("hi")
    assert (r["temperature"], r["top_p"], r["top_k"], r["repetition_penalty"]) == (0.7, 0.6, 20, 1.05)


def dry(**env):
    import os
    out = subprocess.run(["bash", str(ROOT / "stack/up.sh"), "--dry-run"], capture_output=True, text=True,
                         env={**os.environ, **env}, check=True)
    return out.stdout


def test_dry_run_native():
    out = dry(KV_OFFLOAD="native")
    assert "--kv-offloading-backend native" in out and "CUDA_VISIBLE_DEVICES=0" in out
    assert out.count("--tensor-parallel-size") == 0 and "<redacted>" in out


def test_dry_run_lmcache_and_none():
    assert "LMCacheMPConnector" in dry(KV_OFFLOAD="lmcache") and "lmcache server" in dry(KV_OFFLOAD="lmcache")
    assert "kv-offloading" not in dry(KV_OFFLOAD="none")


def test_dry_run_final_defaults():
    out = dry()  # final stack defaults: tiered offload + MTP 2 + larger batches
    assert "TieringOffloadingSpec" in out and "/workspace/kvcache/8001" in out and "/workspace/kvcache/8003" in out
    assert "--max-num-batched-tokens 16384" in out and "--cudagraph-capture-sizes" in out
    assert '"num_speculative_tokens":2' in out.replace("\\", "") and "kv-cache-dtype" not in out
    assert "speculative-config" not in dry(LLM_MTP="0") and "--kv-cache-dtype nvfp4" in dry(LLM_KV_DTYPE="nvfp4")


def test_dry_run_uses_secai1999_model_names():
    out = dry()
    for name in ("secai1999/LLM-Autoscan-v1", "secai1999/Embedding-Autoscan-v1", "secai1999/Rerank-Autoscan-v1",
                 "secai1999/OCR-Autoscan-v1", "secai1999/Translate-Autoscan-v1"):
        assert name in out, name
    for old in ("orcarouter/", "DuyTa/", "ATH-MaaS/", "tencent/"):
        assert old not in out, old


def test_adapter_default_models():
    assert mt.MODEL == "secai1999/Translate-Autoscan-v1"
    ocr = load("ocr_adapter")
    assert ocr.MODEL == "secai1999/OCR-Autoscan-v1" and ocr.PREVIOUS_MODEL == "ATH-MaaS/OvisOCR3"



def test_gateway_protocol_language_codes():
    assert mt.LANG_NAMES["vi"] == "Vietnamese" and mt.LANG_NAMES["zh"] == "Chinese"
    assert all(mt.LANG_NAMES[c] for c in ("vi", "en", "zh", "ja", "ko", "fr", "de", "es", "ru", "ar"))


def test_gateway_protocol_end_to_end():
    import threading, urllib.request
    from http.server import ThreadingHTTPServer
    seen = {}
    class Fake(mt.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["prompt"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["messages"][0]["content"]
            body = json.dumps({"choices": [{"message": {"content": " xin chao "}}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass
    fake = ThreadingHTTPServer(("127.0.0.1", 0), Fake); threading.Thread(target=fake.serve_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mt.make_handler("tok", f"http://127.0.0.1:{fake.server_port}"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    def post(body):
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/v1/translate", json.dumps(body).encode(),
                                     {"Authorization": "Bearer tok", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r: return r.status, json.load(r)
        except urllib.error.HTTPError as e: return e.code, None
    try:
        status, out = post({"text": "hello", "source_lang": "en", "target_lang": "vi"})
        assert status == 200 and out == {"translated_text": "xin chao", "model": mt.MODEL}
        assert "into Vietnamese" in seen["prompt"]
        status, out = post({"text": "hello", "target_lang": "Vietnamese"})
        assert status == 200 and out == {"model": mt.MODEL, "translation": "xin chao"}
        assert post({"text": "hello", "source_lang": "en", "target_lang": "xx"})[0] == 422
    finally:
        srv.shutdown(); fake.shutdown()


def test_dry_run_local_model_dir_keeps_served_names():
    out = dry(MODEL_DIR="/models")
    for repo in ("LLM", "Embedding", "Rerank", "OCR", "Translate"):
        name = f"secai1999/{repo}-Autoscan-v1"
        assert f"serve /models/{repo}-Autoscan-v1 --served-model-name {name}" in out, name
    assert "--prompt-file /models/OCR-Autoscan-v1/prompts/plain_ocr.txt" in out
    hub = dry()  # without MODEL_DIR the Hub names are served directly
    assert "serve secai1999/LLM-Autoscan-v1 --served-model-name secai1999/LLM-Autoscan-v1" in hub and "--prompt-file" not in hub


def test_replica_a_starts_before_the_gpu1_chain():
    src = (ROOT / "stack/up.sh").read_text()
    a, emb, b = (src.index(f"ensure {n} ") for n in ("llm-a", "embedding", "llm-b"))
    assert a < emb < b


fm = load("fetch_models")


def test_models_lock_pins_every_model():
    lock = fm.load_lock(env={})
    assert [v for v, _, _ in lock] == ["LLM_MODEL", "EMBED_MODEL", "RERANK_MODEL", "OCR_MODEL", "MT_MODEL"]
    assert all(r.startswith("secai1999/") and len(rev) == 40 for _, r, rev in lock)
    up = (ROOT / "stack/up.sh").read_text()
    assert all(f'{v}:-{r}}}' in up for v, r, _ in lock)  # up.sh defaults == lock file
    over = dict((v, (r, rev)) for v, r, rev in fm.load_lock(env={"MT_MODEL": "x/y", "MT_MODEL_REVISION": "abc"}))
    assert over["MT_MODEL"] == ("x/y", "abc") and over["LLM_MODEL"][1] == lock[0][2]


def test_fetch_marker_and_offline_check(tmp_path=None):
    import os, tempfile
    d = Path(tmp_path or tempfile.mkdtemp())
    repo, rev = "secai1999/LLM-Autoscan-v1", "a" * 40
    assert not fm.is_done(d, repo, rev)
    (d / "LLM-Autoscan-v1").mkdir(parents=True)
    (d / "LLM-Autoscan-v1" / fm.MARKER).write_text(f"{repo}@{rev}\n")
    assert fm.is_done(d, repo, rev) and not fm.is_done(d, repo, "b" * 40)
    out = subprocess.run([sys.executable, str(ROOT / "stack/fetch_models.py"), "--offline"], capture_output=True,
                         text=True, env={**os.environ, "MODEL_DIR": str(d)})
    assert out.returncode == 1 and "missing:" in out.stdout


def test_bind_host_limits_public_ports():
    out = dry(BIND_HOST="127.0.0.1")
    assert "0.0.0.0" not in out and out.count("--host 127.0.0.1") == 6
    assert "--host 0.0.0.0" in dry() and "--host 127.0.0.1" in dry()


def test_dockerfile_bakes_every_model_with_shard_layers():
    df = (ROOT / "docker/Dockerfile").read_text()
    assert df.index("fetch_models.py LLM_MODEL --include") < df.index("COPY stack /opt/secai/stack")  # models below code
    for i in range(1, 6):
        assert f"model-0000{i}-of-00005.safetensors" in df
    assert "EMBED_MODEL RERANK_MODEL OCR_MODEL MT_MODEL" in df and "fetch_models.py --offline" in df
    assert "HF_HUB_OFFLINE" not in df.split("ARG GIT_SHA")[1]  # cache_bundle restore needs the Hub


def test_fetch_include_patterns_are_repeatable():
    import argparse
    out = subprocess.run([sys.executable, str(ROOT / "stack/fetch_models.py"), "--dry-run", "LLM_MODEL"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.count("\n") == 1


def _fake_host(tmp, gpu="NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887", smoke_rc=0):
    """Fake docker/nvidia-smi/free/df/ss on PATH so deploy.sh can be driven end to end without a GPU or a registry."""
    import os, stat
    bindir, log = Path(tmp) / "bin", Path(tmp) / "calls.log"
    bindir.mkdir()
    def tool(name, body):
        f = bindir / name; f.write_text("#!/usr/bin/env bash\n" + body); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    tool("nvidia-smi", f'''case "$*" in
  *name,memory.total*) printf '%s\\n' "{gpu}" "{gpu}";;
  *memory.used*) printf '0\\n0\\n';;
  *) echo "| NVIDIA-SMI 595.00   Driver Version: 595.00   CUDA Version: 13.2 |";; esac\n''')
    tool("free", 'echo "Mem: 500 10 400 0 90 400"')
    tool("df", 'echo Avail; echo 2000G')
    tool("ss", "exit 0")
    tool("docker", f'''echo "$*" >> {log}
case "$1 $2" in
  "info -f") echo /var/lib/docker;; "info "*|"info") echo "Runtimes: nvidia runc";;
  "compose version"|"manifest inspect"|"compose -f"|"pull "*) ;;
esac
[ "$1" = info ] && [ -z "$2" ] && echo "Runtimes: nvidia runc"
[ "$1" = image ] && echo cbe8629
[ "$1" = inspect ] && echo running
[ "$1" = exec ] && [[ "$*" == *smoke.py* ]] && {{ echo READY; exit {smoke_rc}; }}
exit 0\n''')
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}, log


def _deploy(tmp, *args, **kw):
    env, log = _fake_host(tmp, **kw)
    out = subprocess.run(["bash", str(ROOT / "docker/deploy.sh"), "--workspace", str(Path(tmp) / "ws"), "--generate-key", *args],
                         capture_output=True, text=True, env=env, timeout=120)
    calls = log.read_text().splitlines() if log.exists() else []
    return out, calls


def test_deploy_pulls_then_starts_then_smokes_on_a_good_host():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out, calls = _deploy(tmp)
        assert out.returncode == 0, out.stdout + out.stderr
        pull = next(i for i, c in enumerate(calls) if c.startswith("pull "))
        up = next(i for i, c in enumerate(calls) if " up -d" in c)
        smoke = next(i for i, c in enumerate(calls) if "smoke.py" in c)
        assert pull < up < smoke and "DEPLOYED" in out.stdout and "@sha256:" in calls[pull]
        key = Path(tmp) / "ws/api.key"
        assert key.read_text().strip() not in out.stdout and oct(key.stat().st_mode & 0o777) == "0o600"


def test_deploy_refuses_before_pulling_when_the_machine_is_not_suitable():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out, calls = _deploy(tmp, gpu="NVIDIA GeForce RTX 4090, 24564")
        assert out.returncode == 1 and "Not deploying" in out.stderr and "FAIL  need exactly 2x RTX PRO 6000" in out.stdout
        assert not any(c.startswith("pull ") or " up -d" in c for c in calls)


def test_deploy_check_only_never_pulls_and_failed_smoke_is_not_ready():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out, calls = _deploy(tmp, "--check-only")
        assert out.returncode == 0 and "machine is ready" in out.stdout and not any(c.startswith("pull ") for c in calls)
    with tempfile.TemporaryDirectory() as tmp:
        out, calls = _deploy(tmp, smoke_rc=1)
        assert out.returncode == 1 and "NOT READY" in out.stderr and "DEPLOYED" not in out.stdout


def test_release_env_pins_a_digest_and_compose_requires_an_image():
    env = dict(l.split("=", 1) for l in (ROOT / "docker/release.env").read_text().splitlines() if l and not l.startswith("#"))
    assert env["SECAI_SERVING_REPO"] == "ghcr.io/elonma-china/secai-serving" and len(env["SECAI_SERVING_DIGEST"]) == 71
    assert "SECAI_SERVING_IMAGE:?" in (ROOT / "docker/compose.yaml").read_text()
    assert "latest" not in (ROOT / "docker/deploy.sh").read_text().split("set -euo")[1]


kvp = load("kv_plan")
SAMPLE_LOG = ("(EngineCore pid=1) INFO 11-20 02:39:56 [extensible_kv_cache.py:693] Memory after warmup: 7.0 GiB free, 0.06 GiB committed\n"
              "\x1b[32m(EngineCore pid=1) INFO 11-20 02:39:57 [extensible_kv_cache.py:537] Extended KV cache to 1000 blocks (10.00 GiB).\x1b[0m\n"
              "(EngineCore pid=1) INFO 11-20 03:00:00 [extensible_kv_cache.py:537] Extended KV cache to 2000 blocks (44.90 GiB).\n")


def _plan(tmp, **gib):
    plan = {"kv": {n: {"blocks": 1, "gib": g} for n, g in gib.items()}}
    f = Path(tmp) / "plan.json"; f.write_text(json.dumps(plan)); return f


def test_kv_plan_parses_the_last_extended_line_and_keeps_a_margin():
    assert kvp.parse_log(SAMPLE_LOG.replace("\x1b[32m", "").replace("\x1b[0m", "")) == (2000, 44.9)
    assert kvp.parse_log("nothing here") is None
    assert kvp.kv_bytes({"kv": {"x": {"gib": 10.0}}}, "x") == int(10.0 * 0.99 * (1 << 30))


def test_kv_plan_learn_reads_service_logs_and_reports_missing():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "hy-mt.log").write_text(SAMPLE_LOG)  # includes ANSI colour codes
        plan, missing = kvp.learn(tmp, ("hy-mt", "llm-b"))
        assert plan == {"hy-mt": {"blocks": 2000, "gib": 44.9}} and missing == ["llm-b"]
        out = subprocess.run([sys.executable, str(ROOT / "stack/kv_plan.py"), "learn", "--logs", tmp, "--out", str(Path(tmp) / "o.json")],
                             capture_output=True, text=True)
        assert out.returncode == 1 and "llm-b" in out.stderr


def test_startup_modes_fix_kv_only_where_the_order_no_longer_protects_it():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        plan = _plan(tmp, **{"embedding": 1.0, "reranker": 1.0, "ovis-ocr": 6.0, "hy-mt": 2.0, "llm-b": 44.0})
        assert "kv-cache-memory-bytes" not in dry()                                        # serial default: measured, as before
        aux = dry(STARTUP="parallel_aux", KV_PLAN=str(plan))
        assert aux.count("--kv-cache-memory-bytes") == 4 and "46772193853" not in aux      # llm-b still measures
        par = dry(STARTUP="parallel", KV_PLAN=str(plan))
        assert par.count("--kv-cache-memory-bytes") == 5 and "46772193853" in par


def test_parallel_start_refuses_without_a_kv_plan_instead_of_guessing():
    import os
    out = subprocess.run(["bash", str(ROOT / "stack/up.sh"), "--dry-run"], capture_output=True, text=True,
                         env={**os.environ, "STARTUP": "parallel", "KV_PLAN": "/nonexistent/plan.json"})
    assert out.returncode == 1 and "no fixed KV size" in out.stderr
    bad = subprocess.run(["bash", str(ROOT / "stack/up.sh"), "--dry-run"], capture_output=True, text=True, env={**os.environ, "STARTUP": "fast"})
    assert bad.returncode == 2


def test_up_sh_starts_every_service_before_waiting_in_parallel_modes():
    src = (ROOT / "stack/up.sh").read_text()
    par = src.split("  parallel)")[1].split(";;\nesac")[0]
    assert par.index("ensure llm-b") < par.index("healthy embedding")            # all six launched before the first wait
    aux = src.split("  parallel_aux)")[1].split("  parallel)")[0]
    assert aux.index("healthy hy-mt") < aux.index("ensure llm-b")                  # llm-b waits for the aux services


def test_shell_scripts_parse():
    for s in ("stack/up.sh", "stack/down.sh", "stack/kv_gc.sh", "docker/boot.sh", "docker/vast.sh", "docker/preflight.sh", "docker/build.sh", "docker/deploy.sh"):
        subprocess.run(["bash", "-n", str(ROOT / s)], check=True)


sim = load("agent_sim")


def test_repo_chain_is_resolvable():
    r = sim.Repo(5, hops=4, file_words=120)
    path, hops = r.root, 0
    while True:
        text = r.run("read_file", {"path": path})
        line = next(l for l in text.splitlines() if l.startswith(r.key + ":"))
        value = line.split(":", 1)[1].strip()
        if not value.startswith("ref:"):
            break
        path, hops = value[4:].split("#")[0].strip(), hops + 1
    assert hops == 4 and value == r.answer
    assert r.run("read_file", {"path": "nope"}).startswith("ERROR")
    assert r.run("search", {"query": r.answer}) == r.paths[-1]


def test_repo_is_deterministic_and_bulky():
    a, b = sim.Repo(9, 3, 600), sim.Repo(9, 3, 600)
    assert a.files == b.files and a.answer == b.answer and len(a.files[a.root].split()) > 500


def test_stream_parser_tool_call_and_usage():
    sse = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
           b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}',
           b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\\":\\"x\\"}"}}]}}]}',
           b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3}}', b"data: [DONE]"]
    first, msg, usage = sim.parse_stream((i * 0.1, l) for i, l in enumerate(sse))
    assert first == 0.1 and usage["prompt_tokens"] == 10
    call = msg["tool_calls"][0]
    assert call["id"] == "c1" and call["function"]["name"] == "read_file" and json.loads(call["function"]["arguments"]) == {"path": "x"}
