# Small proof-of-concept image (CPU inference).
FROM python:3.11-slim

WORKDIR /srv

# Install torch from the CPU-only index FIRST (its own step): with a plain
# --extra-index-url, pip can still resolve the multi-GB CUDA wheel from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as non-root; keep model weights in a stable, mountable cache path so
# they survive container restarts (mount a volume at /srv/.cache).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /srv/.cache/huggingface \
    && chown -R appuser:appuser /srv
USER appuser
ENV HF_HOME=/srv/.cache/huggingface \
    MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
