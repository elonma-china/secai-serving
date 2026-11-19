"""Quality gate: mean per-token NLL of the model on real text (concatenated vLLM python sources, deterministic order).
Compare the same text across configs; a lossy change (FP8 Q, quantized KV) shows up as a higher NLL.
Usage: API_KEY=.. python3 nll.py --url http://127.0.0.1:8031 --model M --tokens 24000"""
import argparse, glob, json, math, os, urllib.request

def main():
    p = argparse.ArgumentParser(); p.add_argument("--url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--tokens", type=int, default=24000); a = p.parse_args()
    root = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
    text = ""
    for f in sorted(glob.glob(root + "/**/*.py", recursive=True)):
        text += open(f, errors="ignore").read() + "\n"
        if len(text) > a.tokens * 3.2: break
    text = text[: int(a.tokens * 3.2)]
    body = {"model": a.model, "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    req = urllib.request.Request(a.url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["API_KEY"]})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=1800) as r:
        res = json.load(r)
    lps = res["choices"][0]["prompt_logprobs"][1:]
    vals = [next(iter(d.values()))["logprob"] for d in lps if d]  # prompt_logprobs=0 -> only the actual prompt token
    print(f"NLL tokens={len(vals)} mean_nll={-sum(vals) / len(vals):.5f} ppl={math.exp(-sum(vals) / len(vals)):.4f}")

if __name__ == "__main__":
    main()
