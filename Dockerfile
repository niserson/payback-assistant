# Lightweight, reproducible image (~60 MB compressed): slim base, no build stage needed
# because the service is pure Python.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/srv/hf \
    ONNX_THREADS=2

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Bake everything heavy at build time (deterministic; no cold-start downloads):
# catalog -> EmbeddingGemma q4 download + product embeddings -> classifier heads.
RUN python -m app.catalog && python -m app.semantic && python -m app.intent_model

# Run as non-root (security baseline).
RUN useradd --no-create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
