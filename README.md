# Qwen3-ASR 测试脚本

本地模型路径：
- `./model/Qwen3-ASR-0.6B`
- `./model/Qwen3-ASR-1.7B`

## 安装依赖

推荐 Python 3.12

### 方式 1：官方 qwen-asr 包

```bash
uv pip install -U qwen-asr
```

### 方式 2：vLLM（推荐）

```bash
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

uv pip install "vllm[audio]"
```

### 常见问题

**SSL 证书错误**：
```bash
uv pip install -U --reinstall certifi
sudo apt-get update && sudo apt-get install -y ca-certificates
```

**Triton 找不到 C 编译器**：
```bash
sudo apt update && sudo apt install build-essential
```

## 使用方法

### 1. vLLM 转录测试

启动 vLLM 服务器并测试转录：

```bash
python test_vllm_transcriptions.py
```

常用参数：
- `--model ./model/Qwen3-ASR-1.7B` - 指定模型
- `--audio /path/to/audio.wav` - 测试音频文件
- `--gpu-memory-utilization 0.7` - 降低显存占用（默认 0.8）
- `--max-model-len 2048` - 限制上下文长度节省显存
- `--served-model-name qwen3-asr` - 自定义模型名称

输出指标：
- `AUDIO_SECONDS`: 音频时长
- `ELAPSED_SECONDS`: 处理耗时
- `RTF`: 实时率（处理时间/音频时长）

### 2. Web 录音测试界面

启动 vLLM 后，运行 Web UI：

```bash
python web_vllm_asr_bench.py
```

打开 `http://127.0.0.1:7860/`，功能：
- 麦克风实时录音转录（显示录音时长）
- 上传音频文件测试
- 显示性能指标（RTF、处理时间等）

### 3. 流式推理示例

**使用 qwen-asr 包（需要加载模型到显存）**：
```bash
python example_streaming.py
```

演示不同步长（500ms/1s/2s/4s）的流式语音识别。

**注意**：此脚本会直接加载模型，已针对 RTX 3060 优化：
- `gpu_memory_utilization=0.8`
- `max_model_len=2048`

**流式推理 Web UI（开发中）**：
```bash
python web_vllm_streaming.py
```

#### 官方demo

https://huggingface.co/Qwen/Qwen3-ASR-1.7B?#streaming-demo

```bash
qwen-asr-demo-streaming \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```



### 4. 官方 Demo 测试

```bash
python test_official_demo.py
```

参数：
- `--backend transformers|vllm` - 后端选择
- `--ip 0.0.0.0 --port 8000` - 绑定地址
- `--stay` - 保持运行

## 模型说明

### Qwen3-ASR
语音转文本模型，支持中英文识别。

### Qwen3-ForcedAligner
强制对齐模型，用于生成词级/字符级时间戳。适用场景：
- 精确字幕生成
- 卡拉 OK 歌词同步
- 语音数据集标注

精度优于传统方法（WhisperX、Nemo-ForcedAligner 等）。

## 参考文档

- [Qwen3-ASR-1.7B 模型卡片](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ASR-0.6B 模型卡片](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [官方仓库](https://github.com/QwenLM/Qwen3-ASR)
- [vLLM 使用指南](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html)
- [vLLM OpenAI API 文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)





