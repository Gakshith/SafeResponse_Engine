# SafeResponse Engine Completion Plan

This document lists the remaining project work before the project should be
called complete. It is intentionally split into documentation tasks,
implementation tasks, validation tasks, and execution phases so the work can be
finished in a controlled order.

## Current Status

The project already has an end-to-end local pipeline through Stage 7:

```text
Stage 1: User Query
Stage 2: Retrieval + Context Layer
Stage 3: Generator Layer
Stage 4: Trace Collection Layer
Stage 5: Multi-Signal Verification Layer
Stage 6: Fusion + Decision Router
Stage 7: Final Output
```

Stage 8 is demo-ready through FastAPI, the browser UI, async chat jobs,
metrics, and Docker packaging. Remaining work beyond this
baseline is future broadening: larger retrieval, broader calibration, CI, and
research-grade validation.

## Completion Targets

### Demo-Complete Target

The project is demo-complete when:

- The local API and UI run from documented commands.
- A supported factual query returns an answer with source and confidence.
- An unsupported query is rejected safely.
- Small talk is handled without model inference.
- The test suite covers core routing, final output, and API smoke paths.
- The README explains the full Stage 1-8 architecture and limitations.

### Final-Project Target

The project is final-project complete when:

- All demo-complete requirements are satisfied.
- Stage 6 rewrite flow reruns the needed upstream stages.
- Stage 5 thresholds are calibrated on a small evaluation set.
- Evaluation examples and expected outcomes are committed.
- Metrics or evaluation summaries show accept/reject behavior.
- Generated artifacts and cache files are cleaned from version control.

### Research-Grade Target

The project is research-grade only after:

- HalluGuard-style internal signals are validated with documented experiments.
- NTK, Jacobian, and spectral signals are either enabled with evidence or
  clearly labeled experimental.
- A broader evaluation set measures false accepts, false rejects, latency, and
  grounding quality.
- Deployment has CI smoke tests and reproducible setup instructions.

## Documentation Tasks

| ID | Task | Priority | Status | Definition of Done |
|---|---|---:|---|---|
| DOC-1 | Rewrite README for full project | High | Completed | README explains stages 1-8, setup, run commands, API usage, UI usage, and known limitations. |
| DOC-2 | Add architecture diagram | High | Completed | A clear pipeline diagram shows data flow from query to final response. |
| DOC-3 | Add demo guide | High | Completed | Guide includes exact commands and example supported/unsupported questions. |
| DOC-4 | Add API documentation | Medium | Completed | Endpoints, request bodies, response fields, and error behavior are documented. |
| DOC-5 | Add evaluation documentation | Medium | Completed | Evaluation examples, expected decisions, and scoring fields are explained. |
| DOC-6 | Add deployment runbook | Medium | Completed | Local and Docker deployment steps are testable and match the current repo. |
| DOC-7 | Document limitations honestly | High | Completed | README or a dedicated limitations doc explains 10-article corpus, disabled advanced signals, and non-production status. |
| DOC-8 | Update Stage 6 docs after hardening | Medium | Completed | Stage 6 docs include rewrite loop behavior, config validation, and routing tests. |

## Implementation Tasks

| ID | Task | Priority | Status | Definition of Done |
|---|---|---:|---|---|
| CLEAN-1 | Add/update `.gitignore` | High | Completed | `__pycache__/`, `*.pyc`, logs, cache files, and generated artifacts are ignored. |
| CLEAN-2 | Remove generated files from tracking | High | Completed | Cache/log/generated files are no longer part of normal source changes. |
| CLEAN-3 | Decide demo artifacts to keep | High | Completed | Only small, useful fixture/demo artifacts remain committed. |
| TEST-1 | Add Stage 6 routing tests | High | Completed | Tests cover `ACCEPT`, `RERANK`, `REWRITE`, `REJECT`, missing judge score, and malformed score values. |
| TEST-2 | Add Stage 7 final-output tests | High | Completed | Tests cover confidence mapping, rejection output, source citation, and trimming. |
| TEST-3 | Add API smoke tests | High | Completed | Tests cover `/health`, `/v1/final-response`, `/v1/query`, and `/v1/chat`. |
| TEST-4 | Add config validation tests | Medium | Completed | Invalid thresholds, weights, and output settings fail clearly. |
| FUSION-1 | Finish Stage 6 rewrite loop | High | Completed | `REWRITE` reruns Stages 2-5 with `rewrite_query` until accept/reject/max attempts. |
| FUSION-2 | Add Stage 6 config validation | High | Completed | Fusion weights are non-negative, at least one signal is enabled, and thresholds are ordered. |
| FUSION-3 | Add risk contribution fields | Medium | Completed | Fusion artifact shows each weighted signal contribution. |
| VERIFY-1 | Calibrate grounding threshold | High | Completed | Threshold is tested on supported and unsupported examples with documented results. |
| VERIFY-2 | Choose verification embedding backend | High | Completed | Default backend is documented as lexical or HuggingFace, with fallback behavior. |
| VERIFY-3 | Validate HalluGuard signals | High | Completed | Claims are limited to signals that are enabled and tested. |
| VERIFY-4 | Decide advanced signal positioning | High | Completed | NTK, Jacobian, and spectral modules are either enabled with evidence or labeled experimental. |
| PKG-1 | Fix dependency list | High | Completed | `requirements.txt` includes all runtime and test dependencies, including `PyYAML` and `pytest`. |
| PKG-2 | Fill package setup | Medium | Completed | Packaging metadata exists or project explicitly documents that setup is venv-only. |
| PKG-3 | Add venv run commands | High | Completed | README includes create, activate, install, test, API, and UI commands. |
| RETR-1 | Decide retrieval corpus strategy | High | Completed | Project clearly chooses controlled 10-article demo or larger corpus mode. |
| RETR-2 | Tune retrieval score threshold | High | Completed | Good in-corpus answers are not rejected due to overly strict retrieval filtering. |
| UI-1 | Improve frontend loading state | Medium | Completed | UI shows long-running pipeline status clearly. |
| UI-2 | Use async job endpoint from UI | Medium | Completed | UI can submit a job and poll for completion. |
| UI-3 | Improve decision display | Medium | Completed | Decision, confidence, source, warnings, and risk are easier to read. |
| EVAL-1 | Add evaluation examples | High | Completed | Supported, unsupported, ambiguous, and misspelled examples are stored with expected decisions. |
| EVAL-2 | Add evaluation runner | Medium | Completed | A command runs the examples and summarizes pass/fail behavior. |
| METRICS-1 | Add latency tracking | Medium | Completed | Stage timing is recorded in logs or output artifacts. |
| METRICS-2 | Add decision metrics | Medium | Completed | Accept/reject/rewrite counts and score distributions are available. |
| DEPLOY-1 | Test Docker build/run | Medium | Completed | Docker image builds and serves `/health`. |
| DEPLOY-2 | Document cloud deployment boundary | Low | Completed | Cloud deployment is documented as out of scope unless credentials and infrastructure are provided. |

## Execution Plan

### Phase 1: Stabilize The Repo

Goal: make the repository clean enough to safely test and iterate.

Tasks:

1. Complete `CLEAN-1`.
2. Complete `CLEAN-2`.
3. Complete `CLEAN-3`.
4. Complete `PKG-1`.
5. Complete `PKG-3`.

Verification:

```bash
venv/bin/python -m compileall -q src app.py main.py
venv/bin/python -c "from app import app; print(app.title)"
```

### Phase 2: Add Tests Around Existing Behavior

Goal: protect the already implemented pipeline before changing routing logic.

Tasks:

1. Complete `TEST-1`.
2. Complete `TEST-2`.
3. Complete `TEST-3`.
4. Complete `TEST-4`.
5. Complete `FUSION-2`.

Verification:

```bash
venv/bin/python -m pytest
```

### Phase 3: Finish Decision Flow

Goal: make Stage 6 rewrite behavior real instead of only logged.

Tasks:

1. Complete `FUSION-1`.
2. Complete `FUSION-3`.
3. Update tests from Phase 2 for rewrite attempts.
4. Complete `DOC-8`.

Verification:

```bash
venv/bin/python -m pytest
venv/bin/python main.py
```

### Phase 4: Calibrate Verification And Retrieval

Goal: reduce false rejects and make safety claims accurate.

Tasks:

1. Complete `RETR-1`.
2. Complete `RETR-2`.
3. Complete `VERIFY-1`.
4. Complete `VERIFY-2`.
5. Complete `VERIFY-3`.
6. Complete `VERIFY-4`.

Verification:

```bash
venv/bin/python -m pytest
```

Manual calibration should use a small set of supported and unsupported queries
until the evaluation runner is added in Phase 5.

### Phase 5: Evaluation, Metrics, And Demo Quality

Goal: make the project convincing as a final demo.

Tasks:

1. Complete `EVAL-1`.
2. Complete `EVAL-2`.
3. Complete `METRICS-1`.
4. Complete `METRICS-2`.
5. Complete `UI-1`.
6. Complete `UI-2`.
7. Complete `UI-3`.

Verification:

```bash
venv/bin/python -m pytest
venv/bin/python scripts/run_evaluation.py
venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Manual checks:

- Open `http://127.0.0.1:8000`.
- Send a greeting and confirm direct response.
- Send a supported factual question and confirm answer/source/confidence.
- Send an unsupported question and confirm safe rejection.

### Phase 6: Final Documentation And Packaging

Goal: make the project understandable and reproducible for reviewers.

Tasks:

1. Complete `DOC-1`.
2. Complete `DOC-2`.
3. Complete `DOC-3`.
4. Complete `DOC-4`.
5. Complete `DOC-5`.
6. Complete `DOC-6`.
7. Complete `DOC-7`.
8. Complete `DEPLOY-1`.
9. Complete `DEPLOY-2`.

Verification:

```bash
docker build -t saferesponse-engine .
docker run --rm -p 8000:8000 saferesponse-engine
```

Manual check:

```text
GET http://127.0.0.1:8000/health
```

## Key Decisions To Make Before Implementation

### Decision 1: What Artifacts Stay In Git?

Recommended:

- Keep small JSON fixture artifacts needed for tests.
- Do not keep FAISS indexes, hidden state tensors, logs, or pycache files.
- Put large/generated runtime artifacts under ignored paths.

### Decision 2: Controlled Demo Or Larger Retrieval System?

Recommended for final project:

- Keep the controlled 10-article corpus for demo reliability.
- Document it clearly as a controlled retrieval corpus.
- Add a future-work note for larger corpus or query-time Wikipedia retrieval.

### Decision 3: Lexical Or HuggingFace Verification Backend?

Recommended:

- Use lexical backend for default tests and offline reliability.
- Allow HuggingFace backend as an optional higher-quality mode.
- Document fallback behavior.

### Decision 4: How To Present Advanced HalluGuard Signals?

Recommended:

- Treat logprob, grounding, and consistency as core demo signals.
- Treat NTK, Jacobian, and spectral conditioning as experimental unless they
  are enabled, calibrated, and evaluated.

## Suggested Immediate Next Task

No completion-plan task is still required for the current demo baseline.
Recommended future work is to broaden the retrieval corpus, add CI, and run a
larger calibration set before making research-grade claims.

## Completion Update

The plan has been executed into a final-project-ready demo baseline.

Completed:

- `.gitignore` added for generated artifacts, logs, pycache, model caches, and
  local environment files.
- Generated artifacts, logs, and pycache files were removed from git tracking
  while local copies were preserved.
- `requirements.txt` now includes runtime/test dependencies needed by the
  documented commands.
- `setup.py` now contains minimal package metadata.
- Stage 6 validates fusion weights, threshold ordering, and rewrite settings.
- Stage 6 writes explicit `risk_contributions` in the fusion artifact.
- Stage 6 pipeline wrapper reruns Stages 2-5 when `REWRITE` is selected.
- Stage 7 has fixture tests for accept, reject, confidence, source citation,
  trimming, and invalid config.
- API smoke tests cover health, final response, query, chat, metrics, and chat
  job endpoints.
- Evaluation examples were added in `evaluation/examples.json`.
- Evaluation runner was added in `scripts/run_evaluation.py`.
- API metrics endpoint was added at `/metrics`.
- Chat UI now uses async chat jobs and polls for completion.
- UI inspector now shows route, confidence, source, risk, warnings, and
  explanation.
- README now documents the full Stage 1-8 architecture, setup, commands, API,
  evaluation, controlled-corpus strategy, and limitations.
- Demo, API, evaluation, verification, deployment, and Stage 6 docs were added
  or updated.

Verified:

```bash
venv/bin/python -m pytest
venv/bin/python -m compileall -q src app.py main.py scripts
venv/bin/python scripts/run_evaluation.py
venv/bin/python -c "from app import app; print(app.title)"
```

Docker:

- The Docker image builds with `requirements-docker.txt` and CPU PyTorch.
- Container health check passed on `/health`.
- Full model-backed generation needs more Docker memory than a 4 GB local VM;
  use 8 GB or more for reliable local model runs.
