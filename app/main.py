"""FastAPI service exposing a Hugging Face LLM for text generation.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_ID                    Hugging Face model to serve
                                (default: Qwen/Qwen2.5-0.5B-Instruct)
    MAX_CONCURRENT_GENERATIONS  Generations allowed to run in worker threads at
                                once; further requests queue on the event loop
                                without holding threads (default: 1).
    GENERATION_MAX_TIME_S       Wall-clock cap per generation, 0 = off
                                (default: 120).
    GENERATION_QUEUE_TIMEOUT_S  Max seconds a request may wait for capacity
                                before receiving 503 server_busy; 0 = wait
                                indefinitely (default: 30).
    SKIP_MODEL_LOAD             Set to "1" to start the API without loading the
                                model (used by tests; /generate returns 503).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import env_float, env_int
from app.model import GenerationError, TextGenerator
from app.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    generator = TextGenerator()
    app.state.generator = generator
    # Bounds how many generations may occupy worker threads simultaneously.
    # Without this, N concurrent requests would each hold a thread-pool slot
    # while waiting on the model lock, starving the shared pool. With it,
    # excess requests await the semaphore on the event loop instead.
    max_concurrent = env_int("MAX_CONCURRENT_GENERATIONS", 1)
    app.state.generation_semaphore = asyncio.Semaphore(max(1, max_concurrent))
    if os.environ.get("SKIP_MODEL_LOAD") != "1":
        # Load in a worker thread so startup can't block the event loop.
        await run_in_threadpool(generator.load)
    yield


app = FastAPI(
    title="HF Text Generation API",
    description="Small proof-of-concept: a Hugging Face LLM behind a FastAPI endpoint.",
    version="1.2.1",
    lifespan=lifespan,
)


async def _acquire_with_timeout(semaphore: asyncio.Semaphore, timeout_s: float) -> bool:
    """Acquire ``semaphore`` within ``timeout_s`` seconds; return False on timeout.

    Deliberately not ``asyncio.wait_for(semaphore.acquire(), ...)``: on
    cancellation there is a narrow race (pre-3.12 especially) where the permit
    is acquired just as the timeout cancels the task, leaking the permit and
    permanently shrinking capacity. Here the acquired-anyway case is detected
    and the permit released.
    """
    if timeout_s <= 0:  # 0 (or negative) = wait indefinitely
        await semaphore.acquire()
        return True
    acquire_task = asyncio.create_task(semaphore.acquire())
    done, _pending = await asyncio.wait({acquire_task}, timeout=timeout_s)
    if acquire_task in done:
        return True
    acquire_task.cancel()
    try:
        await acquire_task
    except asyncio.CancelledError:
        return False
    # The task completed between the timeout and the cancel: give the permit back.
    semaphore.release()
    return False


def _not_loaded_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="model_not_loaded",
            detail="The model has not finished loading. Try again shortly.",
        ).model_dump(),
    )


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="generation_failed", detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Consistent JSON error shape for anything unforeseen — no leaked internals."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred.",
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness: the process is up (model may still be loading)."""
    generator: TextGenerator = request.app.state.generator
    return HealthResponse(
        status="ok",
        model=generator.model_id,
        model_loaded=generator.is_loaded,
    )


@app.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "Model not loaded yet."}},
)
async def ready(request: Request) -> HealthResponse | JSONResponse:
    """Readiness: 200 only once the model can serve requests."""
    generator: TextGenerator = request.app.state.generator
    if not generator.is_loaded:
        return _not_loaded_response()
    return HealthResponse(status="ready", model=generator.model_id, model_loaded=True)


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        422: {"description": "Validation error (bad prompt/parameters)."},
        500: {"model": ErrorResponse, "description": "Model inference failed."},
        503: {
            "model": ErrorResponse,
            "description": "Model not loaded yet, or server at capacity (server_busy).",
        },
    },
)
async def generate(
    payload: GenerateRequest, request: Request
) -> GenerateResponse | JSONResponse:
    generator: TextGenerator = request.app.state.generator
    if not generator.is_loaded:
        return _not_loaded_response()

    start = time.perf_counter()
    # Semaphore first (cheap wait on the event loop, no thread held), threadpool
    # second: CPU-bound inference runs off the loop, and only up to
    # MAX_CONCURRENT_GENERATIONS requests occupy threads at a time. Waiting is
    # bounded: past the queue timeout the client gets an honest 503 with
    # Retry-After instead of piling up indefinitely.
    semaphore: asyncio.Semaphore = request.app.state.generation_semaphore
    queue_timeout_s = env_float("GENERATION_QUEUE_TIMEOUT_S", 30.0)
    if not await _acquire_with_timeout(semaphore, queue_timeout_s):
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "5"},
            content=ErrorResponse(
                error="server_busy",
                detail="Generation capacity is saturated. Please retry shortly.",
            ).model_dump(),
        )
    try:
        text = await run_in_threadpool(
            generator.generate,
            payload.prompt,
            payload.max_new_tokens,
            payload.temperature,
        )
    finally:
        semaphore.release()
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "generated %d chars from %d-char prompt in %d ms",
        len(text),
        len(payload.prompt),
        elapsed_ms,
    )

    return GenerateResponse(
        generated_text=text,
        model=generator.model_id,
        prompt_chars=len(payload.prompt),
        generated_chars=len(text),
        elapsed_ms=elapsed_ms,
    )
