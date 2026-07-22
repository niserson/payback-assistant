# Torch-free image with the EmbeddingGemma q4 encoder baked in.
# Layer order is deliberate: deps and the 190 MB model download sit ABOVE the
# app code, so code-only changes rebuild in seconds instead of re-downloading.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/srv/hf \
    ONNX_THREADS=2

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model weights layer: independent of app code, cached across code changes.
RUN python -c "from huggingface_hub import hf_hub_download as d; \
    r='onnx-community/embeddinggemma-300m-ONNX'; \
    d(r, 'onnx/model_q4.onnx'); d(r, 'onnx/model_q4.onnx_data'); \
    d(r, 'tokenizer.json'); d(r, 'tokenizer_config.json'); d(r, 'special_tokens_map.json'); d(r, 'config.json')"

COPY app ./app

# Deterministic build-time artifacts: catalog -> product embeddings -> classifier heads.
RUN python -m app.catalog && python -m app.semantic && python -m app.intent_model

# Run as non-root (security baseline).
RUN useradd --no-create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
