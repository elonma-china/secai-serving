"""Authenticated inference probes for all three resident service contracts."""
import argparse
import json
import math
import os
import urllib.request


def probe(url, path, model, body, kind, token):
    request = urllib.request.Request(url.rstrip("/") + path,
        json.dumps({"model": model, **body}).encode(),
        {"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        result = json.load(response)
    if result.get("model", model) != model:
        raise RuntimeError(f"{kind} returned unexpected model")
    if kind == "llm":
        if not result.get("choices", [{}])[0].get("message", {}).get("content", "").strip():
            raise RuntimeError("LLM inference returned no content")
    elif kind == "embed":
        vector = result.get("data", [{}])[0].get("embedding")
        if not isinstance(vector, list) or not vector or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in vector):
            raise RuntimeError("embedding inference returned invalid vector")
    elif kind == "rerank":
        rows = result.get("results", result.get("data"))
        if not isinstance(rows, list) or not rows or any(
                type(row.get("index")) is not int or row["index"] not in {0, 1}
                or type(row.get("relevance_score")) not in (int, float)
                or not math.isfinite(row["relevance_score"]) for row in rows):
            raise RuntimeError("reranker inference returned invalid scores")
    print(f"{kind} authenticated inference OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("llm", "embed", "rerank"):
        parser.add_argument("--" + name + "-url", required=True)
    parser.add_argument("--llm-model", default="DuyTa/Cyber-F1-AWQ")
    parser.add_argument("--embed-model", default="DuyTa/sec-embedding")
    parser.add_argument("--rerank-model", default="DuyTa/sec-rerank")
    args = parser.parse_args()
    token = os.environ["API_KEY"]
    probe(args.llm_url, "/v1/chat/completions", args.llm_model,
          {"messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 16,
           "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, "llm", token)
    probe(args.embed_url, "/v1/embeddings", args.embed_model, {"input": ["SECAI readiness"]}, "embed", token)
    probe(args.rerank_url, "/rerank", args.rerank_model,
          {"query": "SECAI", "documents": ["SECAI readiness", "Unrelated document"], "top_n": 1}, "rerank", token)


if __name__ == "__main__":
    main()
