# qwen3asr - quick smoke tests

You said the model weights are already on the server:

- `./model/Qwen3-ASR-0.6B`
- `./model/Qwen3-ASR-1.7B`

This repo only contains two tiny Python smoke-test scripts.

## 1) Test the official demo launcher (`qwen-asr-demo`)

This checks whether the official web UI demo can start and becomes reachable over HTTP.

```bash
python test_official_demo.py --asr-checkpoint ./model/Qwen3-ASR-1.7B
```

Useful flags:

- `--backend transformers|vllm` (default: `transformers`)
- `--ip 0.0.0.0 --port 8000` (bind to all interfaces)
- `--stay` (keep it running instead of auto-shutdown)

## 2) Test vLLM transcriptions API (day-0 support)

This script starts `vllm serve` (unless `--no-start-server` is set), waits for `/v1/models`, then calls:
`POST /v1/audio/transcriptions` with a local audio file.

```bash
python test_vllm_transcriptions.py --model ./model/Qwen3-ASR-1.7B --audio /path/to/audio.wav
```

## References (official docs)

- Qwen3-ASR model card (includes `qwen-asr` usage + demo + vLLM): https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3-ASR-0.6B model card: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Official repo (install-from-source): https://github.com/QwenLM/Qwen3-ASR
- vLLM recipe for Qwen3-ASR: https://docs.vllm.ai/en/latest/getting_started/examples/audio_language/qwen3_asr.html
- vLLM OpenAI-compatible server docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- vLLM multimodal inputs (incl. local media paths): https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html

