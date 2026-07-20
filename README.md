# FastAPI + Hugging Face Text Generation API

A small, clean proof of concept: an open-source Hugging Face LLM served behind a
FastAPI endpoint. Send a text prompt, get a generated response back as clean JSON.

Built CPU-friendly by default (`Qwen/Qwen2.5-0.5B-Instruct`, ~1 GB download) so it
runs on a laptop or any small VM — no GPU required. The model is swappable with a
single environment variable.

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU wheel, much smaller
pip install -r requirements.txt
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

Health and readiness:

```bash
curl -s http://localhost:8000/health   # liveness: 200 as soon as the process is up
curl -s http://localhost:8000/ready    # readiness: 200 only once the model is loaded (503 before)
```

## Configuration

| Variable                     | Default                      | Purpose                                              |
| ---------------------------- | ---------------------------- | ---------------------------------------------------- |
| `MODEL_ID`                   | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal-LM / chat model                        |
| `MAX_CONCURRENT_GENERATIONS` | `1`                          | Generations running at once; extras queue cheaply    |
| `GENERATION_MAX_TIME_S`      | `120`                        | Wall-clock cap per generation (`0` = off)            |
| `GENERATION_QUEUE_TIMEOUT_S` | `30`                         | Max queue wait before `503 server_busy` (`0` = wait) |
| `SKIP_MODEL_LOAD`            | unset                        | `1` = start API without loading the model            |

Invalid values fall back to defaults with a logged warning — bad config never
crashes a request.

Request parameters (validated, with clear 422 errors):

| Field            | Default | Range      |
| ---------------- | ------- | ---------- |
| `prompt`         | —       | 1–8000 chars |
| `max_new_tokens` | 128     | 1–512      |
| `temperature`    | 0.7     | 0.0–2.0 (0 = deterministic) |

## Error handling

All errors come back as JSON, never stack traces:

- `422` — invalid input (missing/empty prompt, out-of-range parameters, unknown fields)
- `503` — model still loading (`model_not_loaded`) or at capacity (`server_busy`, with `Retry-After`)
- `500` — inference failure (`generation_failed`) or anything unforeseen (`internal_error`, no leaked internals)

Unknown request fields are rejected (422) so typos like `max_tokens` fail
loudly instead of being silently ignored.

## Concurrency model

Inference is CPU-bound, so it runs in a worker thread — the event loop
(health checks, docs, queued requests) stays responsive during generation.
An async semaphore (`MAX_CONCURRENT_GENERATIONS`, default 1) bounds how many
generations occupy threads at once; excess requests wait on the event loop
without consuming threads, then run in arrival order — bounded by a queue
timeout that returns `503 server_busy` (the acquire is written to be
cancellation-safe, so a timed-out request can never leak a capacity permit).
Each generation is also wall-clock capped (`GENERATION_MAX_TIME_S`) so a slow
CPU can't run unbounded.

State (semaphore, loaded model) is per-process by design: run one process per
container and scale horizontally behind a load balancer using `/ready`.

## Tests

The test suite runs in seconds and does not download any model — the generator
is stubbed while the full request/validation/error/backpressure path is
exercised. The same suite runs in CI (GitHub Actions) on Python 3.11 and 3.12.

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

## Docker

Runs as a non-root user, includes a container HEALTHCHECK, and keeps model
weights in a mountable cache so they survive restarts:

```bash
docker build -t hf-textgen .
docker run -p 8000:8000 -v hf-cache:/srv/.cache hf-textgen
# or with a different model:
docker run -p 8000:8000 -v hf-cache:/srv/.cache -e MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct hf-textgen
```

## Beyond the POC

Deliberately out of scope here, but natural next steps: streaming responses
(SSE), request queuing/batching, auth, rate limiting, and swapping in a
dedicated inference server (vLLM / TGI) behind the same API contract.
