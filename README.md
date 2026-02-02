# qwen3asr - quick smoke tests

You said the model weights are already on the server:

- `./model/Qwen3-ASR-0.6B`
- `./model/Qwen3-ASR-1.7B`

This repo only contains two tiny Python smoke-test scripts.

## Dependencies

The scripts only use Python stdlib, but they expect upstream CLIs to exist in `PATH`.

Recommended Python: 3.12 (matches upstream docs).

### Official demo (`qwen-asr-demo`)

```bash
uv venv
source .venv/bin/activate
uv pip install -U qwen-asr
```

### vLLM (day-0 support)

Follow the official vLLM recipe / model card (nightly wheels). Example for `cu129`:

```bash
uv venv
source .venv/bin/activate

uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

uv pip install "vllm[audio]"
```

If `/v1/audio/transcriptions` returns `{"error":{"message":"Please install vllm[audio] for audio support"...}}`,
it means the server environment is missing the audio extra; install it in the same `.venv` and restart `vllm serve`.

### Troubleshooting: SSL CA errors (Gradio/HTTPX)

If `qwen-asr-demo` crashes with a traceback mentioning `ssl.create_default_context(...)` /
`load_verify_locations(...)`, try reinstalling `certifi`:

```bash
uv pip install -U --reinstall certifi
# or: pip install -U --force-reinstall certifi
```

On Ubuntu/Debian you may also need OS CA certificates:

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates
```

## 1) Test the official demo launcher (`qwen-asr-demo`)

This checks whether the official web UI demo can start and becomes reachable over HTTP.

```bash
python test_official_demo.py
```

Useful flags:

- `--backend transformers|vllm` (default: `transformers`)
- `--ip 0.0.0.0 --port 8000` (bind to all interfaces)
- `--grace 0` (don’t wait after reachable; default waits a bit to catch early crashes)
- `--stay` (keep it running instead of auto-shutdown)

## 2) Test vLLM transcriptions API (day-0 support)

This script starts `vllm serve` (unless `--no-start-server` is set), waits for `/v1/models`, then calls:
`POST /v1/audio/transcriptions` with a local audio file.

```bash
python test_vllm_transcriptions.py

# Optional: test with a real audio file
# python test_vllm_transcriptions.py --audio /path/to/audio.wav
```

It prints timing metrics to help with perf checks:

- `AUDIO_SECONDS`: input audio duration (WAV only; auto-generated WAV always supported)
- `ELAPSED_SECONDS`: wall time for the transcription request
- `RTF`: `ELAPSED_SECONDS / AUDIO_SECONDS`

## 3) Web recorder UI (proxy to vLLM)

If you want an "official-demo-like" web page to record audio and see latency/RTF:

1) Start vLLM (example):

```bash
vllm serve ./model/Qwen3-ASR-0.6B --host 0.0.0.0 --port 8000
```

2) Start the tiny web UI:

```bash
python web_vllm_asr_bench.py
```

Open `http://127.0.0.1:7860/` and click “Start recording”.

## References (official docs)

- Qwen3-ASR model card (includes `qwen-asr` usage + demo + vLLM): https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3-ASR-0.6B model card: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Official repo (install-from-source): https://github.com/QwenLM/Qwen3-ASR
- vLLM recipe for Qwen3-ASR: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html
- vLLM OpenAI-compatible server docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- vLLM multimodal inputs docs: https://docs.vllm.ai/usage/multimodal_inputs/
