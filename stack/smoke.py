"""Readiness smoke: one real inference call per model, so "READY" means the models answer, not that a port listens.
Exit 0 only if every check passes. Usage: smoke.py [--host 127.0.0.1] [--key-file /workspace/api.key] [--ocr-token-file FILE]
  chat (via nginx + each replica), embedding, rerank, translation adapter, OCR adapter (skipped if its token file is absent).
Synthetic inputs only. Respects the same model names and ports as up.sh."""
import argparse, base64, io, json, os, pathlib, sys, time, urllib.error, urllib.request

LLM, EMBED, RERANK = ("secai1999/LLM-Autoscan-v1", "secai1999/Embedding-Autoscan-v1", "secai1999/Rerank-Autoscan-v1")
OCR = "secai1999/OCR-Autoscan-v1"


def post(url, body, key, timeout=180):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def png_with_text(text):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(img).text((30, 60), text, fill="black", font=ImageFont.load_default(size=72))
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()


def checks(host, key, ocr_token, warm=False):
    u = lambda port, path: f"http://{host}:{port}{path}"
    chat = {"model": LLM, "messages": [{"role": "user", "content": "Reply with the single word: pong"}], "max_tokens": 16, "temperature": 0}
    def llm(port):
        out = post(u(port, "/v1/chat/completions"), chat, key)
        text = out["choices"][0]["message"]["content"]
        assert isinstance(text, str) and text.strip(), "empty completion"
        return text.strip()[:40]
    yield "llm via nginx :8005", lambda: llm(8005)
    yield "llm replica A :8001", lambda: llm(8001)
    yield "llm replica B :8003", lambda: llm(8003)
    def embed():
        v = post(u(8000, "/v1/embeddings"), {"model": EMBED, "input": ["secai smoke"]}, key)["data"][0]["embedding"]
        assert len(v) > 100 and any(abs(x) > 0 for x in v), "bad embedding"
        return f"dim={len(v)}"
    if warm:  # long prompts on each replica compile the lazily-built prefill kernels (FP8 FMHA kicks in at >= 256 query tokens)
        def long_prompt(port):
            filler = " ".join(f"record {i} alpha beta gamma delta" for i in range(3000))
            body = {"model": LLM, "messages": [{"role": "user", "content": filler + "\nSummarize in one word."}], "max_tokens": 8, "temperature": 0}
            out = post(u(port, "/v1/chat/completions"), body, key, timeout=900)
            return f"prompt_tokens={out['usage']['prompt_tokens']}"
        yield "warm: long prompt A :8001", lambda: long_prompt(8001)
        yield "warm: long prompt B :8003", lambda: long_prompt(8003)
    yield "embedding :8000", embed
    def rerank():
        r = post(u(8002, "/v1/rerank"), {"model": RERANK, "query": "what is the capital of France",
                 "documents": ["Paris is the capital of France.", "Bananas are yellow."]}, key)["results"]
        best = max(r, key=lambda x: x["relevance_score"])
        assert best["index"] == 0, f"wrong ranking {r}"
        return f"top={best['index']} score={best['relevance_score']:.2f}"
    yield "rerank :8002", rerank
    def translate():
        out = post(u(8004, "/v1/translate"), {"text": "Good morning", "source_lang": "en", "target_lang": "vi"}, key)
        assert out["translated_text"].strip() and out["model"], out
        return out["translated_text"].strip()[:40]
    yield "translate :8004", translate
    if ocr_token:
        def ocr():
            b64 = base64.b64encode(png_with_text("SECAI 2026")).decode()
            out = post(u(18083, "/v1/ocr"), {"model": OCR, "image_base64": b64}, ocr_token)
            text = " ".join(b["text"] for b in out["blocks"])
            assert text.strip(), "OCR returned no text"
            return text.strip()[:40]
        yield "ocr :18083", ocr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--key-file", default=os.environ.get("API_KEY_FILE", "/workspace/api.key"))
    p.add_argument("--ocr-token-file", default=os.environ.get("OCR_TOKEN_FILE", "/workspace/secai-ovis-ocr.token"))
    p.add_argument("--warm", action="store_true", help="also send a ~20K-token prompt to each replica (for cache seeding)")
    a = p.parse_args()
    key = os.environ.get("API_KEY") or pathlib.Path(a.key_file).read_text().strip()
    tok = pathlib.Path(a.ocr_token_file)
    ocr_token = tok.read_text().strip() if tok.is_file() else None
    bad = 0
    for name, fn in checks(a.host, key, ocr_token, a.warm):
        t0 = time.time()
        try:
            detail = fn(); print(f"PASS {name:24s} {time.time() - t0:5.1f}s  {detail}")
        except Exception as e:  # report every failure, not just the first
            bad += 1; print(f"FAIL {name:24s} {time.time() - t0:5.1f}s  {type(e).__name__}: {e}")
    if not ocr_token:
        print("SKIP ocr :18083 (no token file)")
    print("READY" if not bad else f"NOT READY: {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
