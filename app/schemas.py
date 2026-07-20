"""Request/response schemas for the text-generation API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """A single text-generation request."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The input text prompt.",
        examples=["Write a haiku about the ocean."],
    )
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = deterministic (greedy).",
    )


class GenerateResponse(BaseModel):
    """A successful generation result."""

    generated_text: str
    model: str
    prompt_chars: int
    generated_chars: int
    elapsed_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
