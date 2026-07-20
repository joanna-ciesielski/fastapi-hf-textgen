"""FastAPI service exposing a Hugging Face LLM for text generation.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_ID         Hugging Face model to serve (default: Qwen/Qwen2.5-0.5B-Instruct)
    SKIP_MODEL_LOAD  Set to "1" to start the API without loading the model
                     (used by the test suite; /generate returns 503).
"""

from __future__ import annotations

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
    if os.environ.get("SKIP_MODEL_LOAD") != "1":
        # Load in a worker thread so startup can't block the event loop.
        await run_in_threadpool(generator.load)
    yield


app = FastAPI(
    title="HF Text Generation API",
    description="Small proof-of-concept: a Hugging Face LLM behind a FastAPI endpoint.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="generation_failed", detail=str(exc)).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    generator: TextGenerator = app.state.generator
    return HealthResponse(
        status="ok",
        model=generator.model_id,
        model_loaded=generator.is_loaded,
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        422: {"description": "Validation error (bad prompt/parameters)."},
        500: {"model": ErrorResponse, "description": "Model inference failed."},
        503: {"model": ErrorResponse, "description": "Model not loaded yet."},
    },
)
async def generate(request: GenerateRequest) -> GenerateResponse | JSONResponse:
    generator: TextGenerator = app.state.generator
    if not generator.is_loaded:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="model_not_loaded",
                detail="The model has not finished loading. Try again shortly.",
            ).model_dump(),
        )

    start = time.perf_counter()
    # Inference is CPU-bound and synchronous: run it in the threadpool so the
    # event loop stays responsive (health checks, docs, other requests).
    text = await run_in_threadpool(
        generator.generate,
        request.prompt,
        request.max_new_tokens,
        request.temperature,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return GenerateResponse(
        generated_text=text,
        model=generator.model_id,
        prompt_chars=len(request.prompt),
        generated_chars=len(text),
        elapsed_ms=elapsed_ms,
    )
