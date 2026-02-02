#!/usr/bin/env python3
"""
Qwen3-ASR 流式推理测试 (连接 vLLM 服务器)

使用方法：
1. 先启动 vLLM 服务器：
   python test_vllm_transcriptions.py --no-start-server

2. 运行本脚本测试流式推理：
   python test_vllm_streaming.py --audio /path/to/audio.wav
"""

import argparse
import json
import sys
import time
import urllib.request
import wave
from typing import Optional


def _http_json_post(url: str, data: dict, timeout_s: int = 30) -> dict:
    """发送 JSON POST 请求"""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_wav_file(path: str) -> tuple[bytes, int, int]:
    """读取 WAV 文件，返回 (音频数据, 采样率, 时长秒数)"""
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        duration = n_frames / sample_rate if sample_rate > 0 else 0
        return audio_data, sample_rate, duration


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 vLLM 流式推理")
    parser.add_argument("--audio", required=True, help="音频文件路径 (WAV)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="vLLM 服务器地址")
    parser.add_argument("--model", default="qwen3-asr", help="模型名称")
    parser.add_argument("--chunk-ms", type=int, default=1000, help="流式块大小（毫秒）")
    args = parser.parse_args()

    print(f"读取音频文件: {args.audio}")
    try:
        audio_data, sample_rate, duration = _read_wav_file(args.audio)
    except Exception as e:
        print(f"错误: 无法读取音频文件: {e}", file=sys.stderr)
        return 1

    print(f"音频时长: {duration:.2f}秒, 采样率: {sample_rate}Hz")
    print(f"流式块大小: {args.chunk_ms}ms")
    print(f"连接到: {args.base_url}")
    print()

    # TODO: 实现流式推理逻辑
    print("注意: vLLM 的 OpenAI API 目前不直接支持流式音频推理")
    print("建议使用 qwen-asr 包的流式 API 或等待 vLLM 支持")

    return 0


if __name__ == "__main__":
    sys.exit(main())

