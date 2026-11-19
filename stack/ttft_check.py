"""Time-to-first-token for the same long prompt served from: cold recompute, GPU prefix cache,
RAM offload tier, filesystem offload tier. Needs a replica with a small GPU KV cache and a RAM tier larger
than it. Flood sizes are in prompts of ~--words words; pick them from the logged tier sizes.
Usage: API_KEY=.. python3 ttft_check.py --url http://127.0.0.1:8011 --model M --gpu-flood 25 --ram-flood 55
"""
import argparse, json, os, random, statistics, sys, time, urllib.request

W = "amber cobalt delta ember falcon garnet harbor indigo jasper kestrel lagoon meadow nebula onyx prairie quartz raven sable tundra umber velvet willow xenon yonder zephyr".split()


def prompt(rng, n):
    body = " ".join(rng.choice(W) + str(rng.randrange(10_000)) for _ in range(n))
    return f"Memorize this ledger.\n{body}\nQuestion: reply with the first three tokens of the ledger."


def ttft(url, key, model, text):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps({
        "model": model, "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 8,
        "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=900) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content":"' in line and b'"content":""' not in line:
                dt = time.time() - t0
                for _ in r:  # drain
                    pass
                return dt
    raise RuntimeError("no content token received")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8011")
    p.add_argument("--model", required=True)
    p.add_argument("--words", type=int, default=4000)
    p.add_argument("--gpu-flood", type=int, default=25, help="prompts that evict the GPU cache but fit in RAM tier")
    p.add_argument("--ram-flood", type=int, default=55, help="prompts that also overflow the RAM tier")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--seed", type=int, default=77)
    a = p.parse_args()
    key = os.environ["API_KEY"]
    rng = random.Random(a.seed)
    res = {"cold": [], "gpu_hit": [], "ram_hit": [], "fs_hit": []}
    for rep in range(a.reps):
        P = prompt(rng, a.words)
        res["cold"].append(ttft(a.url, key, a.model, P))
        res["gpu_hit"].append(ttft(a.url, key, a.model, P))
        for _ in range(a.gpu_flood):
            ttft(a.url, key, a.model, prompt(rng, a.words))
        res["ram_hit"].append(ttft(a.url, key, a.model, P))
        for _ in range(a.ram_flood):
            ttft(a.url, key, a.model, prompt(rng, a.words))
        res["fs_hit"].append(ttft(a.url, key, a.model, P))
        print(f"rep {rep}: " + " | ".join(f"{k} {v[-1]:.2f}s" for k, v in res.items()), flush=True)
    print("MEDIAN TTFT: " + " | ".join(f"{k} {statistics.median(v):.2f}s" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
