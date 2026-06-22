# SafeResponse Engine

SafeResponse Engine is an LLM **"hallucination firewall"** — a retrieval-augmented safety
middleware that wraps retrieval, generation, internal-signal verification, and a decision
router around an instruction model so **unsupported answers are rejected before they reach the
user**. Supported questions are answered with a cited source and a confidence; out-of-corpus or
ungrounded answers are rejected. An optional LoRA fine-tuning workflow is included.

## Architecture

- **Generator:** `Qwen/Qwen2.5-1.5B-Instruct` (runs locally on Apple-Silicon MPS).
- **Retrieval:** dense **FAISS** over a curated **44-article** local corpus
  (`data/demo_corpus.json`), embedded with `BAAI/bge-small-en-v1.5`; a lexical backend is kept
  as a fallback.
- **Verification signals:** logprob (HalluGuard), grounding (answer↔context similarity), and
  consistency. Per-token log-probs are **captured during generation** and reused by the trace
  stage (no second forward pass).
- **Decision router:** weighted fusion of the signals → ACCEPT / RERANK / REWRITE / REJECT, with
  grounding-derived hard rejects gated on the grounding weight (so signals can be ablated
  independently).
- **Serving:** FastAPI API + a **verdict-first** web UI (ChatGPT-style monochrome, light) that
  surfaces the decision, confidence, source, risk signals, and per-stage timings. The model and
  FAISS index are warmed at startup.

```mermaid
flowchart TD
    A["Stage 1: User Query"] --> B["Stage 2: Retrieval (dense FAISS, bge-small)"]
    B -->|"no grounded chunk"| G
    B --> C["Stage 3: Generation (Qwen2.5-1.5B, logprobs captured)"]
    C --> D["Stage 4: Trace (reuses generation logprobs)"]
    D --> E["Stage 5: Verification (logprob · grounding · consistency)"]
    E --> F["Stage 6: Fusion + Decision Router"]
    F -->|"ACCEPT / RERANK"| G["Stage 7: Final Output (answer + verdict)"]
    F -->|"REWRITE"| B
    F -->|"REJECT"| G
    G --> H["Stage 8: API · verdict-first UI · evaluation · ablation"]
```

Implemented stages:

| Stage | Notes | Main Files |
|---|---|---|
| 1. User Query | Live-fact query guard | `components/user_query.py`, `pipeline/stage_01_user_query.py` |
| 2. Retrieval | Dense FAISS (bge-small) over 44-article corpus, cached index | `components/retrieval_layer.py` |
| 3. Generation | Qwen2.5-1.5B, KV-cache, captures per-token logprobs | `components/generation_layer.py` |
| 4. Trace Collection | Reuses generation logprobs (no extra forward pass) | `components/trace_collection_layer.py`, `components/trace_stats.py` |
| 5. Verification | Calibrated logprob / grounding / consistency signals | `components/verification_layer.py` |
| 6. Fusion Router | Weighted fusion; grounding hard-rejects gated on weight | `components/fusion_decision_router.py` |
| 7. Final Output | Answer + decision + confidence + source + timings | `components/final_output.py` |
| 8. API/UI/Eval | FastAPI + verdict-first UI + ablation harness | `serving/api.py`, `templates/index.html`, `static/app.js`, `scripts/run_ablation.py` |

## Setup

Use the project virtual environment. **Python 3.12 is required** (newer interpreters such as
3.14 do not yet have torch/faiss wheels). On Apple Silicon the model runs on the MPS backend.

```bash
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

Model weights load **offline by default**. For the first run, allow downloads with
`SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1` (Qwen2.5-1.5B-Instruct ≈ 3 GB, `BAAI/bge-small-en-v1.5`
≈ 130 MB).

Run tests:

```bash
venv/bin/python -m pytest
```

Run the full local pipeline:

```bash
venv/bin/python main.py
```

Run the API and UI:

```bash
venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Fine-Tuning

The generation model is `Qwen/Qwen2.5-1.5B-Instruct`. Fine-tuning is optional and uses
a small LoRA adapter instead of rewriting the whole base model.

Validate the configured training file:

```bash
venv/bin/python scripts/finetune_model.py --dry-run
```

Download and convert SQuAD v2 for grounded fine-tuning:

```bash
venv/bin/python scripts/prepare_squad_v2.py --update-config
```

By default this creates a balanced local training file with 20,000 records:
10,000 answerable examples and 10,000 unanswerable examples. The raw Parquet
files and converted SQuAD data are stored under `data/squad_v2/` and can be
regenerated.

Run LoRA fine-tuning:

```bash
SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 \
venv/bin/python scripts/finetune_model.py
```

Small local smoke run:

```bash
SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 \
venv/bin/python scripts/finetune_model.py \
  --run-name smoke \
  --max-records 16 \
  --max-steps 1 \
  --output-dir models/saferesponse-qwen-lora-smoke
```

The script writes the adapter to `models/saferesponse-qwen-lora`. To use it in
the app, set both fields in `config/config.yaml`:

```yaml
generation_layer:
  finetuned_model_path: models/saferesponse-qwen-lora

trace_collection_layer:
  finetuned_model_path: models/saferesponse-qwen-lora
```

SQuAD v2 is a strong starting point for grounded answers and abstentions. Add
project-specific examples before expecting domain-specific quality improvements.
Training run metadata is written beside the adapter and appended to
`model_registry/registry.json`.

Docker installs `requirements-docker.txt` plus a CPU PyTorch wheel so the API/UI
image can run the same model-backed pipeline that the browser UI calls without
pulling CUDA libraries into a CPU image. By default model loading is
offline-only; pre-populate the Hugging Face cache or set
`SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1` for first-run downloads. The Docker
image defaults CPU model loading to `SAFE_RESPONSE_CPU_DTYPE=float16` to reduce
memory pressure; use `float32` if you have more memory and want the default CPU
precision. Full Qwen CPU generation needs more than the 4 GB Docker Desktop VM
limit used during local testing; allocate 8 GB or use a GPU image for production
model-backed generation.

## API

Main endpoints:

```text
GET  /health
GET  /metrics
GET  /v1/final-response
POST /v1/query
POST /v1/query/jobs
GET  /v1/query/jobs/{job_id}
POST /v1/chat
POST /v1/chat/jobs
GET  /v1/chat/jobs/{job_id}
GET  /v1/conversations
GET  /v1/conversations/{conversation_id}
```

Production controls:

```bash
SAFE_RESPONSE_API_KEY=replace-me
SAFE_RESPONSE_ALLOWED_ORIGINS=https://your-domain.com
SAFE_RESPONSE_RATE_LIMIT_PER_MINUTE=60
SAFE_RESPONSE_STATE_DB=artifacts/api_state.sqlite3
SAFE_RESPONSE_CONVERSATION_STORE=artifacts/conversation_memory/conversations.json
SAFE_RESPONSE_CPU_DTYPE=float16
```

When `SAFE_RESPONSE_API_KEY` is set, send `X-API-Key` on every non-public API
request. `/` and `/health` remain public.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hi","run_pipeline":false}'
```

## Evaluation

Full evaluation examples, including retrieval/generation runs:

```bash
venv/bin/python scripts/run_evaluation.py
```

Fast smoke evaluation without model-heavy examples:

```bash
venv/bin/python scripts/run_evaluation.py --skip-model-runs
```

The evaluation set is stored in `evaluation/examples.json`. It includes
supported corpus questions, unsupported entity-overlap questions, current-fact
questions, and stale-fact traps such as live president/weather/market queries.

## Docker

Local Docker verification:

```bash
docker build -t saferesponse-engine:local .
docker run --rm -p 8000:8000 \
  -e SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 \
  -v saferesponse-artifacts:/app/artifacts \
  -v saferesponse-hf-cache:/app/.cache/huggingface \
  saferesponse-engine:local
```

EC2/AWS deployment is intentionally not part of the active project setup. The
current repository is focused on local development, Docker packaging, and a
reproducible demo path.

## Controlled Corpus

The default config uses a curated, controlled Wikipedia corpus (44 articles across history,
science, geography, technology, sports, and the arts) with **dense FAISS retrieval**:

```yaml
retrieval_layer:
  num_articles: 44
  retrieval_backend: dense
  embedding_model: BAAI/bge-small-en-v1.5
  min_score_threshold: 0.65
```

The corpus lives in `data/demo_corpus.json` and is committed for offline reproducibility;
regenerate it with `venv/bin/python scripts/build_demo_corpus.py`. Questions supported by the
corpus are answered with a source and confidence; questions outside it are rejected quickly
without running generation when no sufficiently grounded chunks are found.

## Ablation / Research Results

The core research question — *which internal signals actually catch hallucinations?* — is
measured by an ablation that isolates each verification signal (logprob / grounding /
consistency) and compares it to all-signals-combined, reporting false-accept rate (FAR),
false-reject rate (FRR), and accuracy:

```bash
SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 venv/bin/python scripts/run_ablation.py
```

The harness keeps verification fully enabled and varies the **fusion weights** so each signal is
measured in isolation without the system failing closed (see `docs/ablation-findings.md`).
Results are written to `artifacts/ablation-report.json` (committed copy at
`docs/ablation-report.json`).

**Finding (default mode):** with abstention enabled, every isolated signal scores
FAR=0/FRR=0/acc=1.0 — not because each signal is strong, but because **retrieval gating** and
**model abstention** separate the eval set before the statistical signals matter.

**Finding (stress test, `--force-answer`):** suppressing abstention forces the model to answer
even when context does not support it, and the signals finally diverge:

| config | FAR | FRR | accuracy |
|---|---|---:|---|
| logprob_only | 0.100 | 0.000 | 0.933 |
| grounding_only | 0.000 | 0.000 | 1.000 |
| consistency_only | 0.100 | 0.000 | 0.933 |
| all_on | 0.000 | 0.000 | 1.000 |

On the trap "What company did Abraham Lincoln found?" the model confidently fabricates a company:
**logprob** and single-sample **consistency** accept it (a fluent model looks confident whether
right or wrong), while **grounding** rejects it because the answer does not match the retrieved
text. **Grounding against retrieved evidence is the signal that catches confident hallucinations.**
Full discussion: `docs/ablation-findings.md`.

## Web UI

The browser UI (`/`) is verdict-first: every answer shows the decision (ACCEPT / REVIEW /
REJECT), a confidence meter, the source citation, a per-signal risk breakdown, and live
8-stage pipeline progress while a query runs. It supports light and dark themes.

## Limitations

- This is a safety middleware prototype, not a production hallucination detector.
- The retrieval corpus is intentionally controlled (44 curated articles).
- Verification thresholds are calibrated on a small evaluation set; broader calibration is future work.
- Logprob, grounding, and consistency are the core demo signals.
- NTK, Jacobian, and spectral conditioning code paths are experimental unless
  they are enabled, benchmarked, and documented for a specific run.
- Model-heavy stages require local Hugging Face model availability or network
  access for first-time downloads.

## More Documentation

- Completion plan: `docs/project_completion_plan.md`
- Demo guide: `docs/demo_guide.md`
- API reference: `docs/api_reference.md`
- Evaluation guide: `docs/evaluation.md`
- Verification strategy: `docs/verification_strategy.md`
- Deployment runbook: `docs/deployment_runbook.md`
- Stage 6 design: `docs/stage6_documentation.md`
