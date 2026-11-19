"""Decode throughput on natural prompts (speculative decoding needs real text; random tokens give ~0 acceptance).
Usage: API_KEY=.. python3 decode_test.py --url http://127.0.0.1:8031 --model M"""
import argparse, json, os, threading, time, urllib.request

PROMPTS = [
    "Explain how a B-tree works and when to prefer it over a hash index.", "Write a Python function that merges overlapping intervals, with tests.",
    "Giải thích sự khác nhau giữa TCP và UDP bằng tiếng Việt, kèm ví dụ ứng dụng.", "Summarize the CAP theorem and give one example system for each trade-off.",
    "Write a bash script that rotates log files and keeps seven days of history.", "Describe how a hybrid linear-attention transformer differs from a standard one.",
    "Viết một đoạn mã Go đọc file CSV lớn và thống kê theo cột.", "Explain gradient checkpointing and its memory/compute trade-off.",
    "Write a SQL query that finds the second highest salary per department.", "Explain what a KV cache is in LLM inference and why it grows with context.",
    "Write a short email asking a vendor to extend a deadline by one week.", "Compare REST and gRPC for internal microservice communication.",
]

def one(url, key, model, prompt, out):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t = time.time()
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=900) as r:
        res = json.load(r)
    out.append((res["usage"]["completion_tokens"], time.time() - t))

def run(url, key, model, conc, n):
    out, sem, ths = [], threading.Semaphore(conc), []
    def w(p):
        with sem: one(url, key, model, p, out)
    t0 = time.time()
    for i in range(n):
        th = threading.Thread(target=w, args=(PROMPTS[i % len(PROMPTS)],)); th.start(); ths.append(th)
    for th in ths: th.join()
    wall = time.time() - t0; toks = sum(o[0] for o in out)
    return toks / wall, wall / max(1, toks) * conc * 1000

def main():
    p = argparse.ArgumentParser(); p.add_argument("--url", required=True); p.add_argument("--model", required=True); a = p.parse_args()
    key = os.environ["API_KEY"]; run(a.url, key, a.model, 1, 2)
    for conc, n in ((1, 6), (16, 32)):
        tps, tpot = run(a.url, key, a.model, conc, n)
        print(f"decode c={conc}: {tps:.0f} tok/s total, ~{tpot:.1f} ms/token/request", flush=True)

if __name__ == "__main__":
    main()
