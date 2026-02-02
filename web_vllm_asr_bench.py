#!/usr/bin/env python3
import argparse
import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple


INDEX_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Qwen3-ASR vLLM Bench</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }
      .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
      button { padding: 10px 14px; font-size: 14px; cursor: pointer; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      code, pre { background: #f4f4f5; padding: 2px 6px; border-radius: 6px; }
      pre { padding: 12px; overflow: auto; }
      .card { border: 1px solid #e4e4e7; border-radius: 12px; padding: 16px; max-width: 920px; }
      .muted { color: #71717a; }
      .k { font-weight: 600; width: 160px; display: inline-block; }
      .err { color: #b91c1c; white-space: pre-wrap; }
      .ok { color: #166534; }
    </style>
  </head>
  <body>
    <h2>Qwen3-ASR vLLM Bench</h2>
    <div class="card">
      <h3 style="margin-top: 0;">Microphone Recording</h3>
      <div class="row">
        <button id="btnStart">Start recording</button>
        <button id="btnStop" disabled>Stop & transcribe</button>
        <span class="muted">Records mic audio, encodes WAV 16k mono.</span>
      </div>
      <div style="height: 8px"></div>
      <div id="status" class="muted">Idle.</div>
      <div style="height: 4px"></div>
      <div id="recordingTime" class="muted" style="font-size: 18px; font-weight: 600;">00:00</div>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h3 style="margin-top: 0;">Upload Audio File</h3>
      <div class="row">
        <input type="file" id="fileInput" accept="audio/*" style="padding: 8px;" />
        <button id="btnUpload">Upload & transcribe</button>
        <span class="muted">Supports WAV, MP3, FLAC, etc.</span>
      </div>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h3 style="margin-top: 0;">Results</h3>
      <div>
        <div><span class="k">Audio seconds</span><span id="audioSec">-</span></div>
        <div><span class="k">vLLM seconds</span><span id="vllmSec">-</span></div>
        <div><span class="k">Client seconds</span><span id="clientSec">-</span></div>
        <div><span class="k">RTF (vLLM/audio)</span><span id="rtf">-</span></div>
      </div>
      <div style="height: 12px"></div>
      <div class="k">Transcript</div>
      <pre id="out" class="muted">-</pre>
      <div id="err" class="err"></div>
    </div>

    <script>
      const btnStart = document.getElementById("btnStart");
      const btnStop = document.getElementById("btnStop");
      const btnUpload = document.getElementById("btnUpload");
      const fileInput = document.getElementById("fileInput");
      const statusEl = document.getElementById("status");
      const recordingTimeEl = document.getElementById("recordingTime");
      const outEl = document.getElementById("out");
      const errEl = document.getElementById("err");
      const audioSecEl = document.getElementById("audioSec");
      const vllmSecEl = document.getElementById("vllmSec");
      const clientSecEl = document.getElementById("clientSec");
      const rtfEl = document.getElementById("rtf");

      let recording = false;
      let mediaRecorder = null;
      let audioChunks = [];
      let stream = null;
      let recordingStartTime = 0;
      let recordingTimer = null;

      function setStatus(msg, ok=false) {
        statusEl.textContent = msg;
        statusEl.className = ok ? "ok" : "muted";
      }

      function setError(msg) {
        errEl.textContent = msg || "";
      }

      function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
      }

      function updateRecordingTime() {
        if (recording) {
          const elapsed = (Date.now() - recordingStartTime) / 1000;
          recordingTimeEl.textContent = formatTime(elapsed);
        }
      }

      function clearResults() {
        outEl.textContent = "-";
        audioSecEl.textContent = "-";
        vllmSecEl.textContent = "-";
        clientSecEl.textContent = "-";
        rtfEl.textContent = "-";
        setError("");
      }

      function downsampleBuffer(buffer, inRate, outRate) {
        if (outRate >= inRate) return buffer;
        const ratio = inRate / outRate;
        const outLen = Math.round(buffer.length / ratio);
        const out = new Float32Array(outLen);
        let offsetResult = 0;
        let offsetBuffer = 0;
        while (offsetResult < out.length) {
          const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
          let accum = 0, count = 0;
          for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
          }
          out[offsetResult] = accum / Math.max(1, count);
          offsetResult++;
          offsetBuffer = nextOffsetBuffer;
        }
        return out;
      }

      function floatTo16BitPCM(floatBuf) {
        const out = new Int16Array(floatBuf.length);
        for (let i = 0; i < floatBuf.length; i++) {
          let s = Math.max(-1, Math.min(1, floatBuf[i]));
          out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return out;
      }

      function encodeWav(pcm16, sampleRate) {
        const bytesPerSample = 2;
        const blockAlign = bytesPerSample * 1;
        const buffer = new ArrayBuffer(44 + pcm16.length * bytesPerSample);
        const view = new DataView(buffer);
        function writeStr(off, s) { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); }

        writeStr(0, "RIFF");
        view.setUint32(4, 36 + pcm16.length * bytesPerSample, true);
        writeStr(8, "WAVE");
        writeStr(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);   // PCM
        view.setUint16(22, 1, true);   // mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, 16, true);  // bits
        writeStr(36, "data");
        view.setUint32(40, pcm16.length * bytesPerSample, true);

        let offset = 44;
        for (let i = 0; i < pcm16.length; i++, offset += 2) view.setInt16(offset, pcm16[i], true);
        return buffer;
      }

      function arrayBufferToBase64(buf) {
        const bytes = new Uint8Array(buf);
        const chunk = 0x8000;
        let binary = "";
        for (let i = 0; i < bytes.length; i += chunk) {
          binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
        }
        return btoa(binary);
      }

      async function startRec() {
        clearResults();
        setStatus("Requesting microphone...");
        recordingTimeEl.textContent = "00:00";

        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];

        // Use MediaRecorder instead of deprecated ScriptProcessorNode
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm';

        mediaRecorder = new MediaRecorder(stream, { mimeType });

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            audioChunks.push(e.data);
          }
        };

        mediaRecorder.start(100); // Collect data every 100ms
        recording = true;
        recordingStartTime = Date.now();
        recordingTimer = setInterval(updateRecordingTime, 100);
        btnStart.disabled = true;
        btnStop.disabled = false;
        btnUpload.disabled = true;
        fileInput.disabled = true;
        setStatus("Recording...");
      }

      async function stopRecAndTranscribe() {
        btnStop.disabled = true;
        setStatus("Stopping recording...");

        recording = false;
        if (recordingTimer) {
          clearInterval(recordingTimer);
          recordingTimer = null;
        }
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
          mediaRecorder.stop();
        }
        if (stream) {
          stream.getTracks().forEach(t => t.stop());
        }

        await new Promise(resolve => {
          if (mediaRecorder) {
            mediaRecorder.onstop = resolve;
          } else {
            resolve();
          }
        });

        setStatus("Processing audio...");

        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const inputSampleRate = audioBuffer.sampleRate;
        const channelData = audioBuffer.getChannelData(0);
        const targetRate = 16000;
        const downsampled = downsampleBuffer(channelData, inputSampleRate, targetRate);
        const pcm16 = floatTo16BitPCM(downsampled);
        const wav = encodeWav(pcm16, targetRate);
        const audioSec = (pcm16.length / targetRate);
        audioSecEl.textContent = audioSec.toFixed(3);

        const payload = {
          wav_base64: arrayBufferToBase64(wav),
          filename: "recording.wav",
        };

        setStatus("Transcribing via vLLM...");
        const t0 = performance.now();
        const resp = await fetch("/api/transcribe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const t1 = performance.now();
        clientSecEl.textContent = ((t1 - t0) / 1000).toFixed(3);

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          setStatus("Failed.");
          setError(data.error || `HTTP ${resp.status}`);
          btnStart.disabled = false;
          btnUpload.disabled = false;
          fileInput.disabled = false;
          return;
        }

        outEl.textContent = data.text || "";
        vllmSecEl.textContent = (data.vllm_seconds ?? 0).toFixed(3);
        rtfEl.textContent = (data.rtf ?? 0).toFixed(3);
        setStatus(`OK (model: ${data.model || "?"})`, true);
        btnStart.disabled = false;
        btnUpload.disabled = false;
        fileInput.disabled = false;
      }

      btnStart.addEventListener("click", async () => {
        try { await startRec(); } catch (e) {
          setStatus("Failed.");
          setError(String(e));
          btnStart.disabled = false;
          btnStop.disabled = true;
        }
      });

      btnStop.addEventListener("click", async () => {
        try { await stopRecAndTranscribe(); } catch (e) {
          setStatus("Failed.");
          setError(String(e));
          btnStart.disabled = false;
          btnStop.disabled = true;
          btnUpload.disabled = false;
          fileInput.disabled = false;
        }
      });

      async function uploadAndTranscribe() {
        const file = fileInput.files[0];
        if (!file) {
          setError("Please select a file first.");
          return;
        }

        clearResults();
        btnUpload.disabled = true;
        btnStart.disabled = true;
        fileInput.disabled = true;
        setStatus("Processing uploaded file...");

        try {
          const arrayBuffer = await file.arrayBuffer();
          const audioContext = new (window.AudioContext || window.webkitAudioContext)();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

          const inputSampleRate = audioBuffer.sampleRate;
          const channelData = audioBuffer.getChannelData(0);
          const targetRate = 16000;
          const downsampled = downsampleBuffer(channelData, inputSampleRate, targetRate);
          const pcm16 = floatTo16BitPCM(downsampled);
          const wav = encodeWav(pcm16, targetRate);
          const audioSec = (pcm16.length / targetRate);
          audioSecEl.textContent = audioSec.toFixed(3);

          const payload = {
            wav_base64: arrayBufferToBase64(wav),
            filename: file.name,
          };

          setStatus("Transcribing via vLLM...");
          const t0 = performance.now();
          const resp = await fetch("/api/transcribe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const t1 = performance.now();
          clientSecEl.textContent = ((t1 - t0) / 1000).toFixed(3);

          const data = await resp.json().catch(() => ({}));
          if (!resp.ok) {
            setStatus("Failed.");
            setError(data.error || `HTTP ${resp.status}`);
            btnStart.disabled = false;
            btnUpload.disabled = false;
            fileInput.disabled = false;
            return;
          }

          outEl.textContent = data.text || "";
          vllmSecEl.textContent = (data.vllm_seconds ?? 0).toFixed(3);
          rtfEl.textContent = (data.rtf ?? 0).toFixed(3);
          setStatus(`OK (model: ${data.model || "?"})`, true);
          btnStart.disabled = false;
          btnUpload.disabled = false;
          fileInput.disabled = false;
        } catch (e) {
          setStatus("Failed.");
          setError(String(e));
          btnStart.disabled = false;
          btnUpload.disabled = false;
          fileInput.disabled = false;
        }
      }

      btnUpload.addEventListener("click", uploadAndTranscribe);
    </script>
  </body>
</html>
"""


def _http_json(url: str, timeout_s: int = 5) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
    parts.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"{crlf}'.encode())
    parts.append(f"Content-Type: {content_type}{crlf}{crlf}".encode())
    parts.append(content)
    parts.append(crlf.encode())
    parts.append(f"--{boundary}--{crlf}".encode())
    body = b"".join(parts)
    return body, boundary


class App:
    def __init__(self, vllm_base_url: str, model_id: Optional[str]) -> None:
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self._model_id = model_id
        self._lock = threading.Lock()

    def get_model_id(self) -> str:
        if self._model_id:
            return self._model_id
        with self._lock:
            if self._model_id:
                return self._model_id
            payload = _http_json(f"{self.vllm_base_url}/models", timeout_s=5)
            data = payload.get("data") or []
            if not data or not isinstance(data, list) or not isinstance(data[0], dict) or not data[0].get("id"):
                raise RuntimeError(f"Unexpected /models payload: {payload!r}")
            self._model_id = str(data[0]["id"])
            return self._model_id

    def transcribe_wav(self, wav_bytes: bytes) -> Dict[str, Any]:
        audio_seconds: Optional[float] = None
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    audio_seconds = frames / float(rate)
        except Exception:
            audio_seconds = None

        model_id = self.get_model_id()
        body, boundary = _encode_multipart({"model": model_id}, ("recording.wav", "audio/wav", wav_bytes))

        req = urllib.request.Request(
            f"{self.vllm_base_url}/audio/transcriptions",
            method="POST",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        t1 = time.perf_counter()

        if isinstance(payload, dict) and payload.get("error"):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or json.dumps(err, ensure_ascii=False)
            else:
                msg = str(err)
            raise RuntimeError(msg)

        text = payload.get("text")
        if not isinstance(text, str):
            text = json.dumps(payload, ensure_ascii=False)

        vllm_seconds = t1 - t0
        rtf: Optional[float] = None
        if audio_seconds and audio_seconds > 0:
            rtf = vllm_seconds / audio_seconds

        return {
            "model": model_id,
            "text": text,
            "audio_seconds": audio_seconds,
            "vllm_seconds": vllm_seconds,
            "rtf": rtf,
        }


class Handler(BaseHTTPRequestHandler):
    server: "ThreadingHTTPServer"  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            return
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/transcribe":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty body"})
            return
        if length > 50 * 1024 * 1024:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "payload too large"})
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
            return

        b64 = payload.get("wav_base64")
        if not isinstance(b64, str) or not b64:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing wav_base64"})
            return

        try:
            wav_bytes = base64.b64decode(b64, validate=True)
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid base64"})
            return

        try:
            app: App = getattr(self.server, "app")  # type: ignore[assignment]
            result = app.transcribe_wav(wav_bytes)
            self._send_json(HTTPStatus.OK, result)
        except Exception as e:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(e)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny web UI to record audio and benchmark vLLM Qwen3-ASR.")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="qwen3-asr", help="Model id to use (default: qwen3-asr, matches test_vllm_transcriptions.py default).")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.listen, args.port), Handler)
    httpd.app = App(args.vllm_base_url, args.model_id)  # type: ignore[attr-defined]

    host = "127.0.0.1" if args.listen in ("0.0.0.0", "::") else args.listen
    print(f"Open: http://{host}:{args.port}/")
    print(f"Proxying to vLLM: {args.vllm_base_url}")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
