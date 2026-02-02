#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


def _http_json(url: str, timeout_s: int = 5) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_ready(base_url: str, timeout_s: int) -> Optional[str]:
    deadline = time.time() + timeout_s
    models_url = f"{base_url}/models"
    while time.time() < deadline:
        try:
            payload = _http_json(models_url, timeout_s=2)
            data = payload.get("data") or []
            if data and isinstance(data, list) and isinstance(data[0], dict) and data[0].get("id"):
                return str(data[0]["id"])
        except Exception:
            time.sleep(1)
    return None


def _encode_multipart(fields: Dict[str, str], file_field: Tuple[str, str, bytes]) -> Tuple[bytes, str]:
    boundary = f"----vllm-qwen3asr-{int(time.time() * 1000)}"
    crlf = "\r\n"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.append(f"--{boundary}{crlf}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'.encode())
        parts.append(value.encode())
        parts.append(crlf.encode())

    filename, content_type, content = file_field
    parts.append(f"--{boundary}{crlf}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"{crlf}'.encode()
    )
    parts.append(f"Content-Type: {content_type}{crlf}{crlf}".encode())
    parts.append(content)
    parts.append(crlf.encode())
    parts.append(f"--{boundary}--{crlf}".encode())
    body = b"".join(parts)
    return body, boundary


def _post_transcribe(base_url: str, model_id: str, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    filename = os.path.basename(audio_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    body, boundary = _encode_multipart({"model": model_id}, (filename, content_type, audio_bytes))
    req = urllib.request.Request(
        f"{base_url}/audio/transcriptions",
        method="POST",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            text = payload.get("text")
            if isinstance(text, str):
                return text
            return json.dumps(payload, ensure_ascii=False)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test vLLM OpenAI-compatible transcriptions API with Qwen3-ASR.")
    parser.add_argument("--model", default="./model/Qwen3-ASR-1.7B", help="Local path or HF model id to serve.")
    parser.add_argument("--audio", required=True, help="Path to an audio file (wav/mp3/flac...).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to wait for vLLM server readiness.")
    parser.add_argument("--no-start-server", action="store_true", help="Assume vLLM server is already running.")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERROR: --audio not found: {args.audio}", file=sys.stderr)
        return 1
    if (args.model.startswith("./") or args.model.startswith("/")) and not os.path.exists(args.model):
        print(f"ERROR: --model not found: {args.model}", file=sys.stderr)
        return 1

    check_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    base_url = f"http://{check_host}:{args.port}/v1"
    proc: Optional[subprocess.Popen[str]] = None

    def _shutdown() -> None:
        nonlocal proc
        if not proc or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=30)
        except Exception:
            proc.kill()

    try:
        if not args.no_start_server:
            cmd = ["vllm", "serve", args.model, "--host", args.host, "--port", str(args.port)]
            print("Launching:", " ".join(cmd), flush=True)
            proc = subprocess.Popen(cmd)

        model_id = _wait_ready(base_url, args.timeout)
        if not model_id:
            print(f"ERROR: vLLM not ready within {args.timeout}s: {base_url}", file=sys.stderr)
            _shutdown()
            return 2

        text = _post_transcribe(base_url, model_id, args.audio)
        print("MODEL:", model_id)
        print("TEXT:", text)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if proc:
            _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

