# Small proof-of-concept image (CPU inference).
FROM python:3.11-slim

WORKDIR /srv

# Install torch from the CPU-only index FIRST (its own step): with a plain
# --extra-index-url, pip can still resolve the multi-GB CUDA wheel from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
