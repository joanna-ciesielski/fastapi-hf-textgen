"""Safe environment-variable parsing.

A malformed value (e.g. GENERATION_MAX_TIME_S=abc) must never crash a
request — it degrades to the documented default with a logged warning.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s", name, raw, default)
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s", name, raw, default)
        return default
