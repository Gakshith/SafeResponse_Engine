# API Reference

Base URL for local development:

```text
http://127.0.0.1:8000
```

## `GET /health`

Returns service status.

Example response:

```json
{
  "status": "ok",
  "pipeline_busy": false,
  "final_output_exists": true,
  "final_output_path": "artifacts/final_output/final_response.json"
}
```

## `GET /metrics`

Returns in-memory demo metrics.

Fields:

- `request_count`
- `decision_counts`
- `average_latency_seconds`
- `accept_rate`
- `reject_rate`
- `risk_score_stats`
- `latency_sample_size`

## `GET /v1/final-response`

Returns the latest Stage 7 final response artifact.

## `POST /v1/query`

Runs or reads a single-query response.

Request:

```json
{
  "query": "Give me a short summary about Abraham Lincoln.",
  "run_pipeline": false
}
```

When `run_pipeline` is `false`, the endpoint returns the latest cached Stage 7
artifact. When `true`, it runs Stages 2-7 for the query.

## `POST /v1/query/jobs`

Creates a background query job.

Request:

```json
{
  "query": "Give me a short summary about Abraham Lincoln."
}
```

Response:

```json
{
  "job_id": "...",
  "status": "queued"
}
```

## `GET /v1/query/jobs/{job_id}`

Returns job status and result when complete.

## `POST /v1/chat`

Runs chat mode directly.

Request:

```json
{
  "message": "hi",
  "conversation_id": "optional-id",
  "run_pipeline": false
}
```

Chat mode supports small-talk routing, conversation memory, and memory-augmented
pipeline execution.

## `POST /v1/chat/jobs`

Creates a background chat job. The UI uses this endpoint so long model runs do
not block the initial browser request.

Request:

```json
{
  "message": "Give me a short summary about Abraham Lincoln.",
  "conversation_id": "optional-id",
  "run_pipeline": true
}
```

## `GET /v1/chat/jobs/{job_id}`

Returns chat job status and result when complete.

## Common Response Fields

- `query`: original user query or message
- `decision`: `DIRECT`, `ACCEPT`, `RERANK`, `REWRITE`, or `REJECT`
- `answer`: user-facing answer
- `confidence`: `HIGH`, `MEDIUM`, `LOW`, or `NONE`
- `source`: supporting source when available
- `risk_score`: final hallucination risk score
- `warnings`: risk signals that fired
- `risk_explanation`: explanation for the risk signals
- `pipeline_summary`: model, signal, timing, and routing metadata
