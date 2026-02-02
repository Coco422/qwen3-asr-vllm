#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-ASR 流式推理示例 (vLLM backend)

需要安装: pip install qwen-asr[vllm]
"""

import io
import urllib.request
from typing import Tuple

import numpy as np
import soundfile as sf

from qwen_asr import Qwen3ASRModel


ASR_MODEL_PATH = "./model/Qwen3-ASR-0.6B"
URL_EN = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"


def build_vllm_asr(
    model_path: str,
    *,
    gpu_memory_utilization: float = 0.8,
    max_new_tokens: int = 32,
    max_model_len: int = 2048,
) -> Qwen3ASRModel:
    """
    Workaround helper.

    `qwen-asr` currently loads the processor with `fix_mistral_regex=True`, but under some Transformers
    versions this crashes with `KeyError: 'fix_mistral_regex'`. This function loads the processor without
    that flag and constructs the Qwen3ASRModel manually.
    """
    from vllm import LLM as vLLM
    from vllm import SamplingParams

    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    llm = vLLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    processor = Qwen3ASRProcessor.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    return Qwen3ASRModel(
        backend="vllm",
        model=llm,
        processor=processor,
        sampling_params=sampling_params,
        forced_aligner=None,
        max_inference_batch_size=-1,
        max_new_tokens=max_new_tokens,
    )


def _download_audio_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
        return np.asarray(wav, dtype=np.float32), int(sr)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """简单重采样到 16kHz (使用线性插值)"""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def run_streaming_case(asr: Qwen3ASRModel, wav16k: np.ndarray, step_ms: int) -> None:
    """运行流式推理测试"""
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))

    print(f"\n===== 流式步长 = {step_ms} ms =====")
    state = asr.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=2.0,
    )

    pos = 0
    call_id = 0
    while pos < wav16k.shape[0]:
        seg = wav16k[pos : pos + step]
        pos += seg.shape[0]
        call_id += 1
        asr.streaming_transcribe(seg, state)
        print(f"[调用 {call_id:03d}] 语言={state.language!r} 文本={state.text!r}")

    asr.finish_streaming_transcribe(state)
    print(f"[最终] 语言={state.language!r} 文本={state.text!r}")


def main() -> None:
    """主函数：演示不同步长的流式推理"""
    # 流式推理仅支持 vLLM，不支持强制对齐
    # 使用 0.6B 模型和优化参数以节省显存（适合 RTX 3060）
    asr = build_vllm_asr(ASR_MODEL_PATH, gpu_memory_utilization=0.8, max_model_len=2048, max_new_tokens=32)

    print("下载测试音频...")
    audio_bytes = _download_audio_bytes(URL_EN)
    wav, sr = _read_wav_from_bytes(audio_bytes)
    wav16k = _resample_to_16k(wav, sr)

    # 测试不同的流式步长
    for step_ms in [500, 1000, 2000, 4000]:
        run_streaming_case(asr, wav16k, step_ms)


if __name__ == "__main__":
    main()
