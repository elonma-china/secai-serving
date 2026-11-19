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
