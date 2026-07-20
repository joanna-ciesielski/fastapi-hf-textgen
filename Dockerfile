# Small proof-of-concept image (CPU inference).
FROM python:3.11-slim

WORKDIR /srv

# CPU-only torch keeps the image several GB smaller than the default wheel.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app ./app

ENV MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
