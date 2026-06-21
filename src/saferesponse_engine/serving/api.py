import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from saferesponse_engine import logger
from saferesponse_engine.components.chat_engine import (
    PipelineBusyError,
    SafeResponseChatEngine,
)
from saferesponse_engine.serving.state_store import APIStateStore


ENGINE_LOCK = threading.Lock()
ENGINE: SafeResponseChatEngine | None = None
STATE_STORE = APIStateStore()
MAX_JOBS = int(os.getenv("SAFE_RESPONSE_MAX_JOBS", "100"))
JOB_TTL_SECONDS = int(os.getenv("SAFE_RESPONSE_JOB_TTL_SECONDS", "3600"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("SAFE_RESPONSE_RATE_LIMIT_PER_MINUTE", "0"))
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT_LOCK = threading.RLock()
PUBLIC_PATHS = {"/", "/health"}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    run_pipeline: bool = Field(
        default=False,
        description=(
            "When true, run Stages 2-7 for this query. When false, return the "
            "latest cached Stage 7 artifact."
        ),
    )


class JobRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(default=None, max_length=128)
    run_pipeline: bool = Field(
        default=True,
        description="Run the SafeResponse pipeline for this chat turn.",
    )


def _allowed_origins() -> list[str]:
    raw = os.getenv("SAFE_RESPONSE_ALLOWED_ORIGINS", "*")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


app = FastAPI(
    title="SafeResponse Engine API",
    version="0.1.0",
    description="API wrapper for the SafeResponse hallucination-reduction pipeline.",
)

ALLOWED_ORIGINS = _allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path("static")
TEMPLATE_DIR = Path("templates")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _api_key() -> str | None:
    key = os.getenv("SAFE_RESPONSE_API_KEY", "").strip()
    return key or None


def _is_public_path(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/static/")


def _rate_limit_key(request: Request) -> str:
    provided_key = request.headers.get("x-api-key")
    if provided_key:
        return f"api-key:{provided_key}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def _is_rate_limited(key: str, now: float | None = None) -> bool:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return False
    now = now or time.time()
    cutoff = now - 60
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS[key]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            return True
        bucket.append(now)
        return False


@app.middleware("http")
async def request_controls(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started_at = time.time()

    required_key = _api_key()
    if required_key and not _is_public_path(request.url.path):
        provided_key = request.headers.get("x-api-key")
        if provided_key != required_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API key.", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )

    if _is_rate_limited(_rate_limit_key(request)):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded.", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        {
            "event": "api_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_seconds": round(time.time() - started_at, 6),
        }
    )
    return response


def _engine() -> SafeResponseChatEngine:
    global ENGINE
    with ENGINE_LOCK:
        if ENGINE is None:
            ENGINE = SafeResponseChatEngine()
        return ENGINE


@app.on_event("startup")
def _warmup_on_startup() -> None:
    # Warm the model + retrieval index in a background thread so the first user
    # query is fast without blocking server startup. Best-effort (never fatal).
    def _warm() -> None:
        try:
            _engine().warmup()
        except Exception:
            logger.warning("[Serving] Background warmup failed.", exc_info=True)

    threading.Thread(target=_warm, name="saferesponse-warmup", daemon=True).start()


def _prune_jobs(now: float | None = None) -> None:
    STATE_STORE.prune_jobs(
        max_jobs=MAX_JOBS,
        ttl_seconds=JOB_TTL_SECONDS,
        now=now,
    )


def _record_metrics(started_at: float, response: dict[str, Any] | None = None) -> None:
    latency = round(time.time() - started_at, 6)
    decision = "ERROR"
    risk_score = None
    if response:
        decision = str(response.get("decision") or "UNKNOWN")
        risk_score = response.get("risk_score")
    STATE_STORE.record_metric(
        latency_seconds=latency,
        decision=decision,
        risk_score=float(risk_score) if isinstance(risk_score, (int, float)) else None,
        success=response is not None,
    )


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, PipelineBusyError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_job(job_id: str, query: str) -> None:
    STATE_STORE.update_job(job_id, status="running", started_at=time.time())
    started_at = time.time()
    try:
        engine = _engine()
        output = engine.run_pipeline_for_query(query)
        response = engine.api_response(output)
        STATE_STORE.update_job(
            job_id,
            status="completed",
            completed_at=time.time(),
            result=response,
        )
        _record_metrics(started_at, response)
    except Exception as exc:
        STATE_STORE.update_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            error=str(exc),
        )
        _record_metrics(started_at, None)


def _run_chat_job(job_id: str, request: ChatRequest) -> None:
    STATE_STORE.update_job(job_id, status="running", started_at=time.time())
    started_at = time.time()
    try:
        response = _engine().chat(
            message=request.message,
            conversation_id=request.conversation_id,
            run_pipeline=request.run_pipeline,
        )
        STATE_STORE.update_job(
            job_id,
            status="completed",
            completed_at=time.time(),
            result=response,
        )
        _record_metrics(started_at, response)
    except Exception as exc:
        STATE_STORE.update_job(
            job_id,
            status="failed",
            completed_at=time.time(),
            error=str(exc),
        )
        _record_metrics(started_at, None)


@app.get("/health")
def health() -> dict[str, Any]:
    return _engine().status()


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return STATE_STORE.metrics_summary()


@app.get("/")
def index() -> FileResponse:
    index_path = TEMPLATE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI template not found.")
    return FileResponse(index_path)


@app.get("/v1/final-response")
def get_final_response() -> dict[str, Any]:
    started_at = time.time()
    try:
        engine = _engine()
        response = engine.api_response(engine.load_final_output())
        _record_metrics(started_at, response)
        return response
    except Exception as exc:
        _record_metrics(started_at, None)
        _raise_http_error(exc)


@app.post("/v1/query")
def query(request: QueryRequest) -> dict[str, Any]:
    started_at = time.time()
    try:
        response = _engine().query(
            query=request.query,
            run_pipeline=request.run_pipeline,
        )
        _record_metrics(started_at, response)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        _record_metrics(started_at, None)
        _raise_http_error(exc)


@app.post("/v1/query/jobs")
def create_query_job(
    request: JobRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _prune_jobs()
    job_id = str(uuid.uuid4())
    STATE_STORE.create_job(job_id=job_id, kind="query", query=request.query)
    background_tasks.add_task(_run_job, job_id, request.query)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/query/jobs/{job_id}")
def get_query_job(job_id: str) -> dict[str, Any]:
    _prune_jobs()
    job = STATE_STORE.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/v1/chat/jobs")
def create_chat_job(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    _prune_jobs()
    job_id = str(uuid.uuid4())
    STATE_STORE.create_job(job_id=job_id, kind="chat", query=request.message)
    background_tasks.add_task(_run_chat_job, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/chat/jobs/{job_id}")
def get_chat_job(job_id: str) -> dict[str, Any]:
    _prune_jobs()
    job = STATE_STORE.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/v1/conversations")
def list_conversations(limit: int = 50) -> dict[str, Any]:
    bounded_limit = max(1, min(limit, 100))
    conversations = _engine().memory.list_conversations(limit=bounded_limit)
    return {"conversations": conversations}


@app.get("/v1/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    conversation = _engine().memory.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.post("/v1/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    started_at = time.time()
    try:
        response = _engine().chat(
            message=request.message,
            conversation_id=request.conversation_id,
            run_pipeline=request.run_pipeline,
        )
        _record_metrics(started_at, response)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        _record_metrics(started_at, None)
        _raise_http_error(exc)
