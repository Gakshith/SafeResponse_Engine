# Deployment Runbook

## Local API

```bash
venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Docker

The Docker image installs `requirements-docker.txt` plus the CPU PyTorch wheel,
runs as a non-root user, exposes `/health`, and stores API state in SQLite at
`/app/artifacts/api_state.sqlite3`.

Build:

```bash
docker build -t saferesponse-engine .
```

Run:

```bash
docker run --rm -p 8000:8000 saferesponse-engine
```

Production-like local run:

```bash
docker run --rm -p 8000:8000 \
  -e SAFE_RESPONSE_API_KEY="replace-me" \
  -e SAFE_RESPONSE_ALLOWED_ORIGINS="https://your-domain.com" \
  -e SAFE_RESPONSE_RATE_LIMIT_PER_MINUTE=60 \
  -e SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 \
  -v saferesponse-artifacts:/app/artifacts \
  -v saferesponse-hf-cache:/app/.cache/huggingface \
  saferesponse-engine
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Environment Variables

`SAFE_RESPONSE_ALLOWED_ORIGINS`

Comma-separated CORS origin list. Defaults to `*`.

Example:

```bash
SAFE_RESPONSE_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:8000
```

`SAFE_RESPONSE_API_KEY`

Optional shared API key. When set, all non-public endpoints require
`X-API-Key: <value>`. `/` and `/health` remain public.

`SAFE_RESPONSE_RATE_LIMIT_PER_MINUTE`

Optional per-IP/API-key in-process rate limit. Use an external gateway or load
balancer for distributed production rate limiting.

`SAFE_RESPONSE_STATE_DB`

SQLite path for API job and metrics state. Defaults to
`artifacts/api_state.sqlite3` locally and `/app/artifacts/api_state.sqlite3` in
Docker.

`SAFE_RESPONSE_CONVERSATION_STORE`

JSON conversation-history path used by the chat UI. Defaults to
`artifacts/conversation_memory/conversations.json` locally and
`/app/artifacts/conversation_memory/conversations.json` in Docker. The store is
protected with a file lock and atomic writes so multiple worker threads do not
corrupt history.

`SAFE_RESPONSE_LOG_FILE`

Optional file log path. Logs go to stdout by default, which is preferred for
Docker and most deployment platforms.

`SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS`

Set to `1` for first-run Hugging Face downloads. In production, prefer a
pre-warmed Hugging Face cache volume or baked model artifacts.

`SAFE_RESPONSE_CPU_DTYPE`

Optional CPU model dtype override. Docker defaults this to `float16` to keep the
Qwen 0.5B pipeline inside smaller memory limits. Use `float32` for the default
CPU precision when the host has enough RAM.

## Artifact Behavior

Runtime artifacts are generated under `artifacts/` and are ignored by git and
Docker. Use a Docker volume or persistent disk for `/app/artifacts` so queued
jobs, metrics, and generated pipeline outputs survive container replacement.

## Model Image

The runtime image includes PyTorch, Transformers, PEFT, sentence-transformers,
FAISS, and model-loading dependencies. It intentionally installs the CPU PyTorch
wheel to avoid pulling CUDA libraries into the default local image. Allocate at
least 4 GB of Docker memory for health/API checks and extractive pipeline
routes. Full Qwen CPU generation was observed to be OOM-killed by Docker
Desktop at a 4 GB VM limit; use 8 GB or more for full model-backed local runs.

EC2/AWS deployment is currently out of scope for this cleaned project. If the
project is deployed later, use the same Docker image behind a platform that
provides enough RAM for model loading, persistent artifact/cache storage, an API
key, locked-down CORS, and HTTPS at the edge.
