"""Correctness gate for KV offload to host RAM (run against ONE replica, e.g. :8001).

1. Ask a long prompt at temperature 0 (reference).
2. Flood the replica with many other long prompts so the reference prefix is evicted from GPU KV.
3. Ask the reference prompt again; the answer must match the reference and the offload tier must report hits.
Repeat for --rounds with fresh prompts. Exit 1 on any mismatch: a cache that is fast but wrong is worse than none.
Usage: API_KEY=... python3 kv_check.py --url http://127.0.0.1:8001 --model <served model>
"""

import argparse
import json
import os
import random
import re
import sys
import urllib.request

WORDS = [w for w in ("amber cobalt delta ember falcon garnet harbor indigo jasper kestrel lagoon meadow nebula "
                     "onyx prairie quartz raven sable tundra umber velvet willow xenon yonder zephyr").split()]


def post(url, key, body):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=900) as r:
        return json.load(r)


def metrics(url, key):
    req = urllib.request.Request(url + "/metrics", headers={"Authorization": "Bearer " + key})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=30) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not re.search(r"(external_prefix_cache|kv_offload|prefix_cache_hits)", line):
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


def prompt(rng, words):
    body = " ".join(rng.choice(WORDS) + str(rng.randrange(10_000)) for _ in range(words))
    return f"Memorize this ledger, then answer the question at the end.\n{body}\nQuestion: reply with the first three tokens of the ledger."


def ask(url, key, model, text):
    r = post(url + "/v1/chat/completions", key, {
        "model": model, "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 48,
        "chat_template_kwargs": {"enable_thinking": False}})
    return r["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8001")
    p.add_argument("--model", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--words", type=int, default=6000, help="approx prompt size (~2 tokens/word)")
    p.add_argument("--flood", type=int, default=80, help="distinct long prompts used to evict GPU KV")
    p.add_argument("--seed", type=int, default=20261118)
    a = p.parse_args()
    key = os.environ["API_KEY"]
    rng = random.Random(a.seed)
    bad = 0
    restores = []
    for rnd in range(a.rounds):
        ref_prompt = prompt(rng, a.words)
        ref = ask(a.url, key, a.model, ref_prompt)
        before = metrics(a.url, key)
        for _ in range(a.flood):
            ask(a.url, key, a.model, prompt(rng, a.words))
        again = ask(a.url, key, a.model, ref_prompt)
        after = metrics(a.url, key)
        gained = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0) > 0}
        ok = again == ref
        bad += not ok
        # Only a CPU->GPU transfer or an external prefix hit proves KV was restored from the offload tier.
        restored = {k: round(v) for k, v in gained.items()
                    if ("CPU_to_GPU" in k and "bytes_total" in k) or "external_prefix_cache_hits" in k}
        restores.append(bool(restored))
        print(f"round {rnd}: {'MATCH' if ok else 'MISMATCH'} | restored from offload: {restored or 'NONE'} "
              f"| stored to offload bytes: "
              f"{round(sum(v for k, v in gained.items() if 'GPU_to_CPU' in k and 'bytes_total' in k))}", flush=True)
        if not ok:
            print(f"  ref  : {ref!r}\n  again: {again!r}")
    if bad:
        sys.exit(f"FAIL: {bad}/{a.rounds} rounds returned different output after offload restore")
    if not any(restores):
        sys.exit("INCONCLUSIVE: outputs match but nothing was restored from the offload tier (no CPU->GPU bytes, "
                 "no external prefix hits). The flood probably evicted the reference from the offload tier too, "
                 "or the GPU cache never evicted it. Resize --flood or lower GPU KV capacity and retry.")
    print("PASS: outputs identical after restore from the offload tier")


if __name__ == "__main__":
    main()
