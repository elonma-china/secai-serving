"""Agentic load test: N concurrent agents solve a multi-hop "follow the config references" task with tool calls.

Each agent has a long shared system prompt (prefix sharing), a private synthetic repo, and three tools:
  search(query) -> file paths containing the text      read_file(path) -> file content (large, grows the context)
  submit_answer(value) -> ends the task; success iff value equals the final literal at the end of the chain.
The context grows by one big file per hop, so a run walks each session toward --hops * --file-words * ~5.6 tokens.
Reported: success rate, tool-call errors, per-turn TTFT p50/p95/max, prompt-token totals, cache-hit deltas from /metrics.
Usage: API_KEY=.. python3 agent_sim.py --url http://127.0.0.1:8005 --model M --agents 16 --hops 7 --file-words 2000
Send X-Session-Id per agent (done here) so the nginx front keeps each session on one replica.
"""
import argparse, hashlib, json, os, random, statistics, threading, time, urllib.request

WORDS = "amber cobalt delta ember falcon garnet harbor indigo jasper kestrel lagoon meadow nebula onyx prairie quartz raven sable tundra umber velvet willow xenon yonder zephyr".split()

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "List repository files whose content contains the text.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "submit_answer", "description": "Submit the final value. Ends the task.",
     "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}},
]


def system_prompt(words):
    rng = random.Random(1)
    rules = " ".join(rng.choice(WORDS) for _ in range(words))
    return ("You are a careful configuration auditor working in a repository through tools. Never guess: read files.\n"
            "Values may be literals or references written as `ref: <path>#<key>`; follow references until you reach a literal.\n"
            "Call exactly one tool per turn. When you know the final literal value, call submit_answer.\n"
            f"Repository style notes (ignore for the task): {rules}")


class Repo:
    """Deterministic synthetic repository for one agent: a reference chain plus bulky distractor text."""

    def __init__(self, seed, hops, file_words):
        rng = random.Random(seed)
        self.files = {}
        self.paths = [f"services/{rng.choice(WORDS)}{rng.randrange(1000)}/{rng.choice(WORDS)}{i}.yaml" for i in range(hops)]
        self.answer = str(rng.randrange(10_000, 99_999))
        self.service = f"{rng.choice(WORDS)}{rng.randrange(100)}"
        self.root = "config/root.yaml"
        self.key = "timeout_ms"
        chain = [self.root] + self.paths
        for i, path in enumerate(chain):
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            value = f"ref: {nxt}#{self.key}" if nxt else self.answer
            noise = "\n".join("# " + " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(12))
                              for _ in range(max(1, file_words // 12)))
            decoy = f"{self.key}_legacy: {rng.randrange(10_000, 99_999)}\nretries: {rng.randrange(9)}\n"
            self.files[path] = f"service: {self.service}\n{decoy}{self.key}: {value}\n{noise}\n"
        self.task = (f"What is the final literal value of `{self.key}` for service `{self.service}`? "
                     f"Start at `{self.root}` and follow the `ref:` references. Submit it with submit_answer.")

    def run(self, name, args):
        try:
            if name == "read_file":
                return self.files.get(args["path"], f"ERROR: no such file {args['path']!r}")
            if name == "search":
                hits = [p for p, t in self.files.items() if str(args["query"]) in t]
                return "\n".join(hits[:20]) or "(no matches)"
        except (KeyError, TypeError):
            pass
        return f"ERROR: bad arguments for {name}"


def parse_stream(lines):
    """Parse OpenAI SSE lines -> (first_token_time_or_None, message, usage). `lines` yields (t, bytes)."""
    content, calls, usage, first = [], {}, None, None
    for t, line in lines:
        if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices", []):
            d = ch.get("delta", {})
            if first is None and (d.get("content") or d.get("tool_calls") or d.get("reasoning_content") or d.get("reasoning")):
                first = t
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
    msg = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        msg["tool_calls"] = [{"id": c["id"] or f"call_{i}", "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in sorted(calls.items())]
    return first, msg, usage


def chat(url, key, model, messages, session, thinking):
    body = {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0.2, "max_tokens": 1024, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key, "X-Session-Id": session})
    t0 = time.time()
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=1800) as r:
        first, msg, usage = parse_stream((time.time() - t0, line) for line in r)
    return (first if first is not None else time.time() - t0), msg, usage, time.time() - t0


def agent(i, a, key, sysmsg, out):
    repo = Repo(a.seed * 1000 + i, a.hops, a.file_words)
    rng = random.Random(i)
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": repo.task}]
    rec = {"agent": i, "turns": [], "ok": False, "bad_calls": 0, "done": False, "error": None}
    # Hashed id: sequential ids ("agent-1", "agent-2", ...) cluster on nginx's hash ring and land on one replica.
    session = hashlib.sha1(f"{a.seed}:{i}".encode()).hexdigest()
    t_start = time.time()
    try:
        for _ in range(a.max_turns):
            ttft, msg, usage, total = chat(a.url, key, a.model, msgs, session, a.thinking)
            rec["turns"].append({"ttft": ttft, "total": total, "prompt": (usage or {}).get("prompt_tokens", 0),
                                 "out": (usage or {}).get("completion_tokens", 0)})
            msgs.append(msg)
            if not msg.get("tool_calls"):
                msgs.append({"role": "user", "content": "Call a tool (read_file, search or submit_answer)."})
                rec["bad_calls"] += 1
                continue
            for tc in msg["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError
                except ValueError:
                    rec["bad_calls"] += 1
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "ERROR: arguments are not valid JSON"})
                    continue
                if tc["function"]["name"] == "submit_answer":
                    rec["ok"] = str(args.get("value", "")).strip() == repo.answer
                    rec["done"] = True
                    break
                time.sleep(rng.uniform(a.tool_min, a.tool_max))  # tool execution latency
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": repo.run(tc["function"]["name"], args)})
            if rec["done"]:
                break
    except Exception as e:  # keep one failing agent from sinking the run
        rec["error"] = repr(e)[:200]
    rec["wall"] = time.time() - t_start
    out.append(rec)


def metric_sums(urls, key):
    sums = {}
    for u in urls:
        req = urllib.request.Request(u + "/metrics", headers={"Authorization": "Bearer " + key})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=30) as r:
            for line in r.read().decode().splitlines():
                if line.startswith(("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                                    "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total",
                                    "vllm:kv_offload")):
                    name, _, v = line.rpartition(" ")
                    sums[name.split("{")[0]] = sums.get(name.split("{")[0], 0) + float(v)
    return sums


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8005")
    p.add_argument("--model", required=True)
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--hops", type=int, default=7)
    p.add_argument("--file-words", type=int, default=2000)
    p.add_argument("--system-words", type=int, default=1200)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--tool-min", type=float, default=0.5)
    p.add_argument("--tool-max", type=float, default=3.0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--metrics-urls", default="", help="comma-separated replica URLs for cache-hit deltas")
    p.add_argument("--json-out", default="")
    a = p.parse_args()
    key = os.environ["API_KEY"]
    murls = [u for u in a.metrics_urls.split(",") if u]
    before = metric_sums(murls, key) if murls else {}
    out, sysmsg, threads = [], system_prompt(a.system_words), []
    t0 = time.time()
    for i in range(a.agents):
        t = threading.Thread(target=agent, args=(i, a, key, sysmsg, out))
        t.start(); threads.append(t)
        time.sleep(0.2)
    for t in threads:
        t.join()
    wall = time.time() - t0
    after = metric_sums(murls, key) if murls else {}
    turns = [x for r in out for x in r["turns"]]
    ttft = [x["ttft"] for x in turns]
    summary = {
        "agents": a.agents, "completed": sum(r["done"] for r in out), "correct": sum(r["ok"] for r in out),
        "errors": sum(1 for r in out if r["error"]), "bad_tool_calls": sum(r["bad_calls"] for r in out),
        "turns_total": len(turns), "turns_per_agent_median": statistics.median(len(r["turns"]) for r in out),
        "max_prompt_tokens": max((x["prompt"] for x in turns), default=0),
        "prompt_tokens_total": sum(x["prompt"] for x in turns), "output_tokens_total": sum(x["out"] for x in turns),
        "ttft_s": {"p50": pct(ttft, .5), "p95": pct(ttft, .95), "max": max(ttft, default=float("nan"))},
        "agent_wall_s": {"p50": pct([r["wall"] for r in out], .5), "max": max(r["wall"] for r in out)},
        "run_wall_s": wall, "output_tok_per_s": sum(x["out"] for x in turns) / wall,
        "cache_delta": {k: after.get(k, 0) - before.get(k, 0) for k in after},
        "error_samples": [r["error"] for r in out if r["error"]][:3],
    }
    print(json.dumps(summary, indent=2))
    if a.json_out:
        json.dump({"summary": summary, "agents": out}, open(a.json_out, "w"))


if __name__ == "__main__":
    main()
