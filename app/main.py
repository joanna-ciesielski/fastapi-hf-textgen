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
    max_concurrent = int(os.environ.get("MAX_CONCURRENT_GENERATIONS", "1"))
    app.state.generation_semaphore = asyncio.Semaphore(max(1, max_concurrent))
    if os.environ.get("SKIP_MODEL_LOAD") != "1":
        # Load in a worker thread so startup can't block the event loop.
        await run_in_threadpool(generator.load)
    yield


app = FastAPI(
    title="HF Text Generation API",
    description="Small proof-of-concept: a Hugging Face LLM behind a FastAPI endpoint.",
    version="1.1.0",
    lifespan=lifespan,
)


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
        503: {"model": ErrorResponse, "description": "Model not loaded yet."},
    },
)
async def generate(
    payload: GenerateRequest, request: Request
) -> GenerateResponse | JSONResponse:
    generator: TextGenerator = request.app.state.generator
    if not generator.is_loaded:
        return _not_loaded_response()

    start = time.perf_counter()
    # Semaphore first (cheap, non-blocking wait on the event loop), threadpool
    # second: CPU-bound inference runs off the loop, and only up to
    # MAX_CONCURRENT_GENERATIONS requests occupy threads at a time.
    async with request.app.state.generation_semaphore:
        text = await run_in_threadpool(
            generator.generate,
            payload.prompt,
            payload.max_new_tokens,
            payload.temperature,
        )
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
