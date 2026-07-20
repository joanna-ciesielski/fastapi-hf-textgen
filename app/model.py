"""Thin wrapper around a Hugging Face text-generation pipeline.

Transformers/torch are imported lazily inside ``load()`` so that the API
module can be imported (e.g. by the test suite, which stubs the generator)
without the heavy ML dependencies being loaded.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


class GenerationError(RuntimeError):
    """Raised when the underlying model fails to generate text."""


class TextGenerator:
    """Loads a Hugging Face causal LM and generates text from prompts."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._pipeline = None
        # transformers pipelines are not guaranteed thread-safe; FastAPI may
        # serve concurrent requests, so serialize access to the pipeline.
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        """Download (first run) and initialize the model. Idempotent."""
        if self._pipeline is not None:
            return
        from transformers import pipeline  # lazy import, see module docstring

        logger.info("Loading model %s (first run may download weights)...", self.model_id)
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            device="cpu",
        )
        logger.info("Model %s loaded.", self.model_id)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate a completion for ``prompt`` and return only the new text."""
        if self._pipeline is None:
            raise GenerationError("Model is not loaded.")

        from transformers import GenerationConfig  # lazy import, see module docstring

        greedy = temperature == 0.0
        # Chat-tuned models produce much better results via their chat template;
        # base models without one get the raw prompt instead.
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        use_chat = bool(getattr(tokenizer, "chat_template", None))
        model_input = [{"role": "user", "content": prompt}] if use_chat else prompt
        # An explicit GenerationConfig avoids the deprecated loose-kwargs path
        # (removed in future transformers releases).
        max_time_s = float(os.environ.get("GENERATION_MAX_TIME_S", "120"))
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            # Graceful wall-clock cap: generation stops (mid-output) rather
            # than running unbounded on slow CPUs. 0 disables the cap.
            max_time=max_time_s if max_time_s > 0 else None,
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None),
        )
        try:
            with self._lock:
                outputs = self._pipeline(
                    model_input,
                    generation_config=generation_config,
                    return_full_text=False,
                )
            text = outputs[0]["generated_text"]
            # Chat pipelines may return a message dict instead of a string.
            if isinstance(text, list):
                text = text[-1].get("content", "") if text else ""
            elif isinstance(text, dict):
                text = text.get("content", "")
            return str(text).strip()
        except GenerationError:
            raise
        except Exception as exc:  # torch/transformers raise many types
            logger.exception("Generation failed")
            raise GenerationError(f"Model inference failed: {exc}") from exc
