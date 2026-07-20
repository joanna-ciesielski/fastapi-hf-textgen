# FastAPI + Hugging Face Text Generation API

A small, clean proof of concept: an open-source Hugging Face LLM served behind a
FastAPI endpoint. Send a text prompt, get a generated response back as clean JSON.

Built CPU-friendly by default (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB download) so it
runs on a laptop or any small VM — no GPU required. The model is swappable with a
single environment variable.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First startup downloads the model weights (~1 GB); subsequent starts are fast.
Interactive API docs are served at http://localhost:8000/docs.

## Usage

Generate text:

```bash
curl -s http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about the ocean.", "max_new_tokens": 64, "temperature": 0.7}'
```

Response:

```json
{
  "generated_text": "Waves whisper softly...",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "prompt_chars": 31,
  "generated_chars": 84,
  "elapsed_ms": 2140
}
```

Health check:

```bash
curl -s http://localhost:8000/health
# {"status":"ok","model":"Qwen/Qwen2.5-0.5B-Instruct","model_loaded":true}
```

## Configuration

| Variable          | Default                       | Purpose                                   |
| ----------------- | ----------------------------- | ----------------------------------------- |
| `MODEL_ID`        | `Qwen/Qwen2.5-0.5B-Instruct`  | Any HF causal-LM / chat model             |
| `SKIP_MODEL_LOAD` | unset                         | `1` = start API without loading the model |

Request parameters (validated, with clear 422 errors):

| Field            | Default | Range      |
| ---------------- | ------- | ---------- |
| `prompt`         | —       | 1–8000 chars |
| `max_new_tokens` | 128     | 1–512      |
| `temperature`    | 0.7     | 0.0–2.0 (0 = deterministic) |

## Error handling

All errors come back as JSON, never stack traces:

- `422` — invalid input (missing/empty prompt, out-of-range parameters)
- `503` — model still loading (`{"error": "model_not_loaded", ...}`)
- `500` — inference failure (`{"error": "generation_failed", ...}`)

Inference runs in a worker thread, so the event loop (health checks, docs)
stays responsive during generation, and pipeline access is serialized for
thread safety.

## Tests

The test suite runs in seconds and does not download any model — the generator
is stubbed while the full request/validation/error path is exercised.

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

## Docker

```bash
docker build -t hf-textgen .
docker run -p 8000:8000 hf-textgen
# or with a different model:
docker run -p 8000:8000 -e MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct hf-textgen
```

## Beyond the POC

Deliberately out of scope here, but natural next steps: streaming responses
(SSE), request queuing/batching, auth, rate limiting, and swapping in a
dedicated inference server (vLLM / TGI) behind the same API contract.
