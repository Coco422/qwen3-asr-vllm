#!/usr/bin/env python3
"""
Qwen3-ASR 流式推理 Web UI

适合远程服务器使用，支持：
- 上传音频文件进行流式转录
- 实时显示流式输出
- 性能指标统计
"""

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Qwen3-ASR 流式推理</title>
    <style>
      body {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        margin: 24px;
        background: #f9fafb;
      }
      .container { max-width: 960px; margin: 0 auto; }
      .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      }
      h2 { margin-top: 0; color: #111827; }
      h3 { margin-top: 0; color: #374151; font-size: 18px; }
      .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
      button {
        padding: 10px 16px;
        font-size: 14px;
        cursor: pointer;
        border: none;
        border-radius: 6px;
        background: #3b82f6;
        color: white;
        font-weight: 500;
      }
      button:hover { background: #2563eb; }
      button:disabled { opacity: 0.5; cursor: not-allowed; background: #9ca3af; }
      input[type="file"] { padding: 8px; border: 1px solid #d1d5db; border-radius: 6px; }
      .status { padding: 8px 12px; border-radius: 6px; font-size: 14px; }
      .status.idle { background: #f3f4f6; color: #6b7280; }
      .status.processing { background: #dbeafe; color: #1e40af; }
      .status.success { background: #d1fae5; color: #065f46; }
      .status.error { background: #fee2e2; color: #991b1b; }
      .streaming-output {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 12px;
        min-height: 100px;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 13px;
        white-space: pre-wrap;
        word-break: break-word;
      }
      .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
      .metric { padding: 12px; background: #f9fafb; border-radius: 6px; }
      .metric-label { font-size: 12px; color: #6b7280; margin-bottom: 4px; }
      .metric-value { font-size: 20px; font-weight: 600; color: #111827; }
    </style>
  </head>
  <body>
    <div class="container">
      <h2>Qwen3-ASR 流式推理</h2>

      <div class="card">
        <h3>上传音频文件</h3>
        <div class="row">
          <input type="file" id="fileInput" accept="audio/*" />
          <button id="btnUpload">开始流式转录</button>
        </div>
        <div id="status" class="status idle">等待上传...</div>
      </div>

      <div class="card">
        <h3>流式输出</h3>
        <div id="streamOutput" class="streaming-output">等待开始...</div>
      </div>

      <div class="card">
        <h3>性能指标</h3>
        <div class="metrics">
          <div class="metric">
            <div class="metric-label">音频时长</div>
            <div class="metric-value" id="audioDuration">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">处理时间</div>
            <div class="metric-value" id="processTime">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">RTF</div>
            <div class="metric-value" id="rtf">-</div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const fileInput = document.getElementById('fileInput');
      const btnUpload = document.getElementById('btnUpload');
      const statusEl = document.getElementById('status');
      const streamOutput = document.getElementById('streamOutput');
      const audioDuration = document.getElementById('audioDuration');
      const processTime = document.getElementById('processTime');
      const rtf = document.getElementById('rtf');

      function setStatus(msg, type = 'idle') {
        statusEl.textContent = msg;
        statusEl.className = `status ${type}`;
      }

      function appendOutput(text) {
        streamOutput.textContent += text;
        streamOutput.scrollTop = streamOutput.scrollHeight;
      }

      function clearOutput() {
        streamOutput.textContent = '';
        audioDuration.textContent = '-';
        processTime.textContent = '-';
        rtf.textContent = '-';
      }

      async function uploadAndStream() {
        const file = fileInput.files[0];
        if (!file) {
          setStatus('请先选择文件', 'error');
          return;
        }

        clearOutput();
        btnUpload.disabled = true;
        fileInput.disabled = true;
        setStatus('正在处理...', 'processing');

        try {
          const formData = new FormData();
          formData.append('file', file);

          const t0 = performance.now();
          const resp = await fetch('/api/stream', {
            method: 'POST',
            body: formData,
          });

          if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
          }

          const reader = resp.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.text) {
                  appendOutput(data.text);
                }
                if (data.metrics) {
                  const m = data.metrics;
                  if (m.audio_duration) audioDuration.textContent = m.audio_duration.toFixed(2) + 's';
                  if (m.process_time) processTime.textContent = m.process_time.toFixed(2) + 's';
                  if (m.rtf) rtf.textContent = m.rtf.toFixed(3);
                }
              }
            }
          }

          const t1 = performance.now();
          setStatus('完成', 'success');
        } catch (e) {
          setStatus('错误: ' + e.message, 'error');
        } finally {
          btnUpload.disabled = false;
          fileInput.disabled = false;
        }
      }

      btnUpload.addEventListener('click', uploadAndStream);
    </script>
  </body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server: "ThreadingHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/stream":
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        # TODO: 实现流式处理
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        msg = "data: " + json.dumps({"text": "流式推理功能开发中...\n"}) + "\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-ASR 流式推理 Web UI")
    parser.add_argument("--listen", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7861, help="监听端口")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1", help="vLLM 服务器地址")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.listen, args.port), Handler)

    host = "127.0.0.1" if args.listen in ("0.0.0.0", "::") else args.listen
    print(f"流式推理 Web UI: http://{host}:{args.port}/")
    print(f"vLLM 服务器: {args.vllm_base_url}")
    print()
    print("注意: 当前版本为演示界面，流式推理功能开发中")
    print("vLLM 的 OpenAI API 暂不直接支持流式音频推理")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
