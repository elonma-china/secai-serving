"""Private /v1/ocr compatibility route backed by OvisOCR3's vLLM chat API."""

import argparse
import base64
import binascii
import hmac
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODEL = "ATH-MaaS/OvisOCR3"
LEGACY_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def build_request(image_base64: str, prompt: str, model: str = MODEL) -> dict:
    if len(image_base64) > 15_000_000:
        raise ValueError("image too large")
    try:
        image = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("invalid base64 image") from error
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ValueError("image too large or empty")
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif image.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("image must be PNG, JPEG or WebP")
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_base64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": int(os.environ.get("OCR_MAX_TOKENS", "4096")),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def to_ocr_response(completion: dict, requested_model: str = MODEL) -> dict:
    content = completion["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("OCR backend returned non-text content")
    text = content.strip()
    return {"model": requested_model, "blocks": [] if not text else [
        {"text": text, "bbox": None, "confidence": None}]}


def make_handler(token: str, backend_key: str, backend_url: str, prompt: str):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"ready": True, "model": MODEL})
            else:
                self.send_json(404, {"detail": "Not found"})

        def do_POST(self):
            if self.path != "/v1/ocr":
                self.send_json(404, {"detail": "Not found"})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_json(401, {"detail": "Unauthorized"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 15_100_000:
                    raise ValueError("request too large or empty")
                request = json.loads(self.rfile.read(size))
                requested_model = request.get("model")
                if requested_model not in (MODEL, LEGACY_MODEL):
                    raise ValueError("unsupported OCR model")
                calibration = request.get("calibration_prompt", "standard")
                if calibration not in ("standard", "conditional", "logical", "arabic"):
                    raise ValueError("unsupported calibration prompt")
                task_prompt = prompt
                if calibration == "logical":
                    task_prompt += "\nKeep identifiers and IP addresses exactly as printed."
                elif calibration == "arabic":
                    task_prompt += "\nKeep Arabic in logical reading order and preserve Latin identifiers."
                payload = build_request(request["image_base64"], task_prompt)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self.send_json(422, {"detail": "Invalid OCR request"})
                return
            backend_request = urllib.request.Request(
                backend_url.rstrip("/") + "/v1/chat/completions",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + backend_key},
            )
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(backend_request, timeout=180) as response:
                    completion = json.load(response)
                self.send_json(200, to_ocr_response(completion, requested_model))
            except (urllib.error.URLError, ValueError, KeyError, IndexError, TypeError):
                self.send_json(502, {"detail": "OCR backend failed"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--backend-url", default="http://127.0.0.1:18303")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if not token or args.token_file.stat().st_mode & 0o077:
        raise SystemExit("private OCR token file required")
    if args.prompt_file:
        prompt_path = args.prompt_file
    else:
        from huggingface_hub import hf_hub_download
        prompt_path = Path(hf_hub_download(MODEL, "prompts/plain_ocr.txt"))
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    ThreadingHTTPServer(("127.0.0.1", args.port),
                        make_handler(token, os.environ["API_KEY"], args.backend_url, prompt)).serve_forever()


if __name__ == "__main__":
    main()
