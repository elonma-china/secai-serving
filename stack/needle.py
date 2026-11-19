"""Needle-in-a-haystack gate at long context: a random passkey is hidden at several depths of a
~--words-word prompt; the model must return it. Quantized KV that loses long-range detail fails here.
Usage: API_KEY=.. python3 needle.py --url http://127.0.0.1:8021 --model M --words 21000"""
import argparse, json, os, random, secrets, sys, time, urllib.request

W = "amber cobalt delta ember falcon garnet harbor indigo jasper kestrel lagoon meadow nebula onyx prairie quartz raven sable tundra umber velvet willow xenon yonder zephyr".split()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8021")
    p.add_argument("--model", required=True)
    p.add_argument("--words", type=int, default=21000)
    p.add_argument("--depths", default="0.05,0.25,0.5,0.75,0.95")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    key = os.environ["API_KEY"]
    rng = random.Random(a.seed)
    ok = total = 0
    for depth in map(float, a.depths.split(",")):
        passkey = secrets.token_hex(3)
        words = [rng.choice(W) + str(rng.randrange(10_000)) for _ in range(a.words)]
        words.insert(int(depth * len(words)), f"IMPORTANT: the secret passkey is {passkey}. Remember it.")
        text = " ".join(words) + "\n\nWhat is the secret passkey? Reply with only the passkey."
        body = {"model": a.model, "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": 24,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        t0 = time.time()
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=1200) as r:
            res = json.load(r)
        ans = res["choices"][0]["message"]["content"]
        hit = passkey in ans
        ok += hit; total += 1
        print(f"needle depth {depth:.2f}: {'OK ' if hit else 'MISS'} prompt_tokens={res['usage']['prompt_tokens']} "
              f"{time.time() - t0:.1f}s answer={ans.strip()[:40]!r}", flush=True)
    print(f"NEEDLE {ok}/{total}", flush=True)
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
