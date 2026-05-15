FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SAFE_RESPONSE_STATE_DB=/app/artifacts/api_state.sqlite3 \
    SAFE_RESPONSE_CONVERSATION_STORE=/app/artifacts/conversation_memory/conversations.json \
    SAFE_RESPONSE_CPU_DTYPE=float16 \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
    && pip install -r requirements-docker.txt

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
