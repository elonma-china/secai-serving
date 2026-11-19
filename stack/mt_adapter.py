"""Hy-MT2 translation adapter: builds the model card's task prompts, calls vLLM.

POST /v1/translate
  {"text": str, "target_lang": "Vietnamese",          # full English language name
   "glossary": {"src term": "target term"},            # optional -> Terminology prompt
   "style": "formal",                                  # optional -> Style prompt
   "context": "background text",                       # optional -> Background prompt
   "format_type": "JSON"}                              # optional -> Structured Data prompt
Hy-MT2 has no default system prompt, so the prompt is the whole instruction.

Gateway protocol (secai_gateway translation/multilang specialist client): when the body carries "source_lang",
language codes (vi, en, zh, ja, ko, fr, de, es, ru, ar) are mapped to language names and the reply is exactly
{"translated_text": str, "model": str}. Without "source_lang" the reply is {"model", "translation"} as before.
"""

import argparse
import hmac
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("MT_MODEL", "secai1999/Translate-Autoscan-v1")
MAX_TEXT_CHARS = 60_000
SAMPLING = {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repetition_penalty": 1.05}
LANG_NAMES = {"vi": "Vietnamese", "en": "English", "zh": "Chinese", "zh-Hans": "Chinese", "ja": "Japanese",
              "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian", "ar": "Arabic"}
ONLY = "Note that you must **ONLY output the translated result without any additional explanation**"


def build_prompt(text: str, target_lang: str, glossary: dict | None = None, style: str | None = None,
                 context: str | None = None, format_type: str | None = None) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError("text must be a non-empty string")
    if not isinstance(target_lang, str) or not target_lang.strip() or len(target_lang) > 64:
        raise ValueError("target_lang must be a language name")
    modes = [bool(glossary), bool(style), bool(context), bool(format_type)]
    if sum(modes) > 1:
        raise ValueError("use only one of glossary, style, context, format_type per request")
    if format_type:
        return (f"### Task\nTranslate the user-facing text within the following {format_type} data into {target_lang}.\n\n"
                "### Strict Rules\n1. **Structure Preservation:** You MUST preserve the original "
                f"{format_type} data structure, nesting, hierarchy, and indentation exactly as they are.\n"
                "2. **Selective Translation:** Translate ONLY the visible, user-facing text content/values.\n"
                "3. **Strict Non-Translation:** NEVER translate or alter code tags, keys, properties, object names, "
                "or variable placeholders. Leave them exactly in their original English/code form.\n\n"
                f"### Source Data\n{text}")
    if context:
        return (f"[Background Information]\n{context}\n\nPlease translate the following text into {target_lang}, "
                f"taking the provided background information into consideration.\n\n[Source Text]\n{text}")
    if style:
        return (f"Please translate the following text into {target_lang}. Note that the translation style must "
                f"strictly conform to [{style}]:\n\n{text}")
    if glossary:
        if not isinstance(glossary, dict) or len(glossary) > 200:
            raise ValueError("glossary must be an object with at most 200 entries")
        pairs = "\n".join(f"{k} translates to {v}" for k, v in glossary.items())
        return (f"Reference the following translations:\n{pairs}\n\nTranslate the following text into "
                f"{target_lang}. {ONLY}:\n\n{text}")
    return (f"Translate the following text into {target_lang}. Note that you should **only output the "
            f"translated result without any additional explanation**:\n\n{text}")


def build_request(prompt: str, max_tokens: int = 4096) -> dict:
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, **SAMPLING}


def make_handler(token: str, backend_url: str):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, body: dict):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_json(200, {"ready": True, "model": MODEL}) if self.path == "/health" \
                else self.send_json(404, {"detail": "Not found"})

        def do_POST(self):
            if self.path != "/v1/translate":
                return self.send_json(404, {"detail": "Not found"})
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                return self.send_json(401, {"detail": "Unauthorized"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 1_000_000:
                    raise ValueError("request too large or empty")
                req = json.loads(self.rfile.read(size))
                gateway = "source_lang" in req
                if gateway and req["source_lang"] not in LANG_NAMES:
                    raise ValueError("unsupported source_lang")
                target = LANG_NAMES.get(req["target_lang"], req["target_lang"]) if isinstance(req["target_lang"], str) else req["target_lang"]
                if gateway and req["target_lang"] not in LANG_NAMES:
                    raise ValueError("unsupported target_lang")
                prompt = build_prompt(req["text"], target, req.get("glossary"), req.get("style"),
                                      req.get("context"), req.get("format_type"))
                payload = build_request(prompt, min(int(req.get("max_tokens", 4096)), 8192))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                return self.send_json(422, {"detail": "Invalid translate request"})
            backend = urllib.request.Request(backend_url.rstrip("/") + "/v1/chat/completions",
                                             json.dumps(payload).encode(),
                                             {"Content-Type": "application/json", "Authorization": "Bearer " + token})
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(backend, timeout=180) as response:
                    completion = json.load(response)
                text = completion["choices"][0]["message"]["content"].strip()
                self.send_json(200, {"translated_text": text, "model": MODEL} if gateway else {"model": MODEL, "translation": text})
            except (urllib.error.URLError, ValueError, KeyError, IndexError, TypeError):
                self.send_json(502, {"detail": "Translation backend failed"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--backend-url", default="http://127.0.0.1:18304")
    args = parser.parse_args()
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(os.environ["API_KEY"], args.backend_url)).serve_forever()


if __name__ == "__main__":
    main()
