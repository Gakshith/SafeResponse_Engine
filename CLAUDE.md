# SafeResponse Engine - Project Context

## Project Purpose

SafeResponse Engine is a middleware system for reducing hallucinations in LLM responses. The intended system combines retrieval grounding, multi-candidate generation, internal model trace analysis, HalluGuard-style instability features, consistency checking, optional judge/classifier baselines, and a final decision router that can accept, rerank, rewrite, regenerate, or reject an answer.

The project is currently implemented as a staged Python pipeline. Stages 1 through 4 exist in code. Stages 5 through 8 are the next major implementation work.

## Current Repository Status

Current working directory:

```text
/Users/gojuruakshith/PycharmProjects/SafeResponse_Engine
```

Implemented code lives mostly under:

```text
src/saferesponse_engine
```

Generated runtime artifacts are currently stored under:

```text
artifacts
```

The main pipeline entry point is:

```text
main.py
```

Running `main.py` executes the currently implemented stages in order:

1. User Query
2. Retrieval Layer
3. Generation Layer
4. Trace Collection Layer

## Current Project Structure

```text
SafeResponse_Engine/
├── main.py
├── app.py
├── config/
│   └── config.yaml
├── artifacts/
│   ├── user_query/
│   ├── retrieval/
│   ├── generation/
│   └── traces/
├── logs/
├── research/
│   ├── trials.ipynb
│   ├── retrieval_layer.ipynb
│   ├── generation_layer.ipynb
│   └── trace_collection_layer.ipynb
├── src/
│   └── saferesponse_engine/
│       ├── __init__.py
│       ├── components/
│       │   ├── user_query.py
│       │   ├── retrieval_layer.py
│       │   ├── generation_layer.py
│       │   └── trace_collection_layer.py
│       ├── config/
│       │   └── configuration.py
│       ├── constants/
│       │   └── __init__.py
│       ├── entity/
│       │   └── config_entity.py
│       ├── models/
│       │   └── __init__.py
│       ├── pipeline/
│       │   ├── stage_01_user_query.py
│       │   ├── stage_02_retrieval_layer.py
│       │   ├── stage_03_generation_layer.py
│       │   └── stage_04_trace_collection_layer.py
│       └── utils/
│           └── common.py
├── requirements.txt
├── params.yaml
├── schema.yaml
├── Dockerfile
├── setup.py
├── templates/
│   └── index.html
└── README.md
```

## Implemented Stages

### Stage 1: User Query

Files:

```text
src/saferesponse_engine/components/user_query.py
src/saferesponse_engine/pipeline/stage_01_user_query.py
```

Current behavior:

- Reads the query source URL from `config/config.yaml`.
- Converts GitHub `blob` URLs to raw GitHub URLs.
- Downloads the query text.
- Saves it to `artifacts/user_query/data.txt`.
- Skips download if the local file already exists.

Current config:

```yaml
user_query:
  root_dir: artifacts/user_query
  source_url: https://github.com/Gakshith/CSC317/blob/main/user_query.txt
  local_data_file: artifacts/user_query/data.txt
```

### Stage 2: Retrieval + Context Layer

Files:

```text
src/saferesponse_engine/components/retrieval_layer.py
src/saferesponse_engine/pipeline/stage_02_retrieval_layer.py
```

Current behavior:

- Loads `BAAI/bge-m3` embeddings through `langchain_huggingface.HuggingFaceEmbeddings`.
- Loads a small slice of the Hugging Face Wikipedia dataset.
- Splits articles into chunks with `RecursiveCharacterTextSplitter`.
- Builds a local FAISS vector index.
- Saves the FAISS index under `artifacts/retrieval/faiss_index`.
- Retrieves top-k chunks for the user query.
- Saves retrieval output to `artifacts/retrieval/retrieved_chunks.json`.

Current config:

```yaml
retrieval_layer:
  root_dir: artifacts/retrieval
  query_artifact_path: artifacts/user_query/data.txt
  faiss_index_path: artifacts/retrieval/faiss_index
  retrieval_output_path: artifacts/retrieval/retrieved_chunks.json
  embedding_model: BAAI/bge-m3
  top_k: 5
  chunk_size: 500
  chunk_overlap: 100
  num_articles: 15
  min_score_threshold: 0.5
```

Notes:

- `min_score_threshold` exists in config but is not currently applied in retrieval filtering.
- Current vector DB is local FAISS. Pinecone is not implemented.

### Stage 3: Generator Layer

Files:

```text
src/saferesponse_engine/components/generation_layer.py
src/saferesponse_engine/pipeline/stage_03_generation_layer.py
```

Current behavior:

- Loads a Hugging Face causal language model and tokenizer.
- Builds a grounded prompt from retrieved chunks.
- Generates `N` candidate responses.
- Uses temperature `0.0` for the primary candidate and a sampling temperature for alternate candidates.
- Saves generated candidates to `artifacts/generation/candidates.json`.

Current config:

```yaml
generation_layer:
  root_dir: artifacts/generation
  retrieval_artifact_path: artifacts/retrieval/retrieved_chunks.json
  generation_output_path: artifacts/generation/candidates.json
  model_name: Qwen/Qwen2.5-0.5B-Instruct
  finetuned_model_path: null
  num_candidates: 3
  primary_temperature: 0.0
  sample_temperature: 0.7
  max_new_tokens: 96
  max_context_length: 1024
```

Notes:

- Frozen backbone generation exists.
- Optional fine-tuned generator comparison is represented in config but not implemented.
- The in-memory model cache is local to this module.

### Stage 4: Trace Collection Layer

Files:

```text
src/saferesponse_engine/components/trace_collection_layer.py
src/saferesponse_engine/pipeline/stage_04_trace_collection_layer.py
```

Current behavior:

- Loads a Hugging Face causal language model and tokenizer.
- Rebuilds the prompt from Stage 3 query/context.
- For each candidate, collects:
  - decoded tokens
  - per-token logprobs
  - mean logprob
  - minimum logprob
  - sequence score
  - hidden states, if enabled
  - prompt token count
  - response token count
- Saves traces to `artifacts/traces/traces.json`.
- Saves hidden states under `artifacts/traces/hidden_states`.

Current config:

```yaml
trace_collection_layer:
  root_dir: artifacts/traces
  generation_artifact_path: artifacts/generation/candidates.json
  trace_output_path: artifacts/traces/traces.json
  hidden_states_dir: artifacts/traces/hidden_states
  model_name: Qwen/Qwen2.5-1.5B-Instruct
  max_context_length: 2048
  collect_hidden_states: true
  num_hidden_layers_to_save: -1
```

Notes:

- Trace collection model currently differs from the generation model.
- Beam/sample metadata is only partially represented through temperature and sampling settings.
- True beam search metadata is not implemented.

## Target End-to-End Architecture

The intended full project is:

```text
Stage 1: User Query
    ↓
Stage 2: Retrieval + Context Layer
    - Embedding model
    - Vector DB / FAISS / Pinecone
    - retrieve top-k chunks
    ↓
Stage 3: Generator Layer
    - Frozen backbone LLMs
    - optional fine-tuned generator as comparison model
    - generate N candidate responses
    ↓
Stage 4: Trace Collection Layer
    For each candidate collect:
    - tokens
    - token logprobs
    - hidden states per decoding step
    - beam/sample metadata
    ↓
Stage 5: Multi-Signal Verification Layer
    A. HalluGuard Core
       - NTK feature builder
       - Jacobian / instability module
       - spectral conditioning
       - final HalluGuard score

    B. Retrieval Grounding Score
       - semantic similarity of answer vs retrieved chunks

    C. Consistency Score
       - SelfCheck / sample divergence across N outputs

    D. Optional Judge / Classifier
       - Judge LLM or fine-tuned classifier as baseline
    ↓
Stage 6: Fusion + Decision Router
    - combine HalluGuard + grounding + consistency + judge scores
    - accept / rerank / rewrite / regenerate / reject
    ↓
Stage 7: Final Output
    - best verified answer to user
    - confidence / hallucination-risk tag
    ↓
Stage 8: Deployment + Evaluation
    - FastAPI middleware
    - vLLM / HF inference server
    - dashboard
    - CI regression tests
    - latency / hallucination tracking
```

## Recommended Final Repository Structure

Keep the existing style, but add explicit modules for verification, fusion, output, API serving, and evaluation.

Recommended structure:

```text
src/saferesponse_engine/
├── components/
│   ├── user_query.py
│   ├── retrieval_layer.py
│   ├── generation_layer.py
│   ├── trace_collection_layer.py
│   ├── verification_layer.py
│   ├── fusion_decision_router.py
│   └── final_output.py
├── verification/
│   ├── halluguard.py
│   ├── ntk_features.py
│   ├── jacobian_instability.py
│   ├── spectral_conditioning.py
│   ├── grounding_score.py
│   ├── consistency_score.py
│   └── judge_baseline.py
├── serving/
│   ├── api.py
│   ├── schemas.py
│   └── middleware.py
├── evaluation/
│   ├── datasets.py
│   ├── metrics.py
│   ├── regression_suite.py
│   └── latency_tracking.py
├── pipeline/
│   ├── stage_01_user_query.py
│   ├── stage_02_retrieval_layer.py
│   ├── stage_03_generation_layer.py
│   ├── stage_04_trace_collection_layer.py
│   ├── stage_05_verification_layer.py
│   ├── stage_06_fusion_decision_router.py
│   └── stage_07_final_output.py
├── entity/
│   ├── config_entity.py
│   └── artifact_entity.py
├── config/
│   └── configuration.py
├── constants/
│   └── __init__.py
└── utils/
    ├── common.py
    ├── text.py
    └── torch_utils.py
```

Recommended tests:

```text
tests/
├── unit/
│   ├── test_retrieval_layer.py
│   ├── test_generation_layer.py
│   ├── test_trace_collection_layer.py
│   ├── test_grounding_score.py
│   ├── test_consistency_score.py
│   └── test_fusion_decision_router.py
├── integration/
│   ├── test_pipeline_smoke.py
│   └── test_api_middleware.py
└── fixtures/
    ├── sample_query.txt
    ├── sample_retrieval.json
    ├── sample_candidates.json
    └── sample_traces.json
```

## Stage 5 Implementation Plan

Stage 5 should consume:

```text
artifacts/retrieval/retrieved_chunks.json
artifacts/generation/candidates.json
artifacts/traces/traces.json
```

It should produce:

```text
artifacts/verification/verification_scores.json
```

Recommended output shape:

```json
{
  "query": "...",
  "candidates": [
    {
      "response_id": 0,
      "text": "...",
      "halluguard_score": 0.0,
      "grounding_score": 0.0,
      "consistency_score": 0.0,
      "judge_score": null,
      "risk_signals": {
        "low_mean_logprob": false,
        "low_min_logprob": false,
        "weak_grounding": false,
        "high_sample_divergence": false,
        "hidden_state_instability": false
      }
    }
  ]
}
```

### Stage 5A: HalluGuard Core

The HalluGuard implementation should start with a pragmatic baseline before adding expensive full Jacobian/NTK computation.

Recommended first version:

- Use token logprob statistics from Stage 4.
- Use hidden-state norm statistics.
- Use hidden-state variance across decoding steps.
- Use inter-layer hidden-state drift.
- Produce a normalized `halluguard_score` in `[0, 1]`, where higher means higher hallucination risk.

Then extend toward:

- NTK feature builder.
- Approximate Jacobian instability.
- Spectral conditioning features.
- Calibration against labeled hallucination datasets.

Important engineering note:

Full Jacobian and NTK computation can be very expensive for transformer LLMs. Implement approximations first, validate usefulness, then optimize.

### Stage 5B: Retrieval Grounding Score

Recommended first version:

- Embed each candidate answer with the same embedding model used in Stage 2.
- Embed retrieved chunks or reuse stored vector DB data where possible.
- Compute answer-to-context semantic similarity.
- Optionally compute sentence-level support by splitting the answer into claims.
- Produce `grounding_score` in `[0, 1]`, where higher means better grounding.

Suggested risk flag:

```text
weak_grounding = grounding_score < threshold
```

### Stage 5C: Consistency Score

Recommended first version:

- Compare all generated candidate responses.
- Use embedding similarity between responses.
- Optionally add lexical overlap or contradiction detection later.
- Produce `consistency_score` in `[0, 1]`, where higher means candidates are more consistent.

Suggested risk flag:

```text
high_sample_divergence = consistency_score < threshold
```

### Stage 5D: Optional Judge / Classifier

Recommended first version:

- Keep this optional and disabled by default.
- Add an interface that supports either:
  - local classifier
  - external judge LLM
  - simple heuristic baseline

Do not make the project dependent on a paid external API for core operation.

## Stage 6 Implementation Plan

Stage 6 should consume:

```text
artifacts/verification/verification_scores.json
```

It should produce:

```text
artifacts/decision/decision.json
```

Recommended decision actions:

```text
accept
rerank
rewrite
regenerate
reject
```

Recommended first routing logic:

```text
accept:
  grounding_score >= 0.75
  consistency_score >= 0.70
  halluguard_score <= 0.35

rerank:
  at least one non-primary candidate has materially better fused score

rewrite:
  answer is partially grounded but low quality or too uncertain

regenerate:
  all candidates are weak but retrieval context is strong

reject:
  retrieval context is weak and all candidates have high hallucination risk
```

Recommended fused score:

```text
safe_response_score =
  0.40 * grounding_score
  + 0.25 * consistency_score
  + 0.20 * (1 - halluguard_score)
  + 0.15 * judge_score_or_default
```

The weights should be configurable in `config/config.yaml`.

## Stage 7 Implementation Plan

Stage 7 should produce the final user-facing output.

Recommended output artifact:

```text
artifacts/final/final_response.json
```

Recommended output shape:

```json
{
  "query": "...",
  "answer": "...",
  "decision": "accept",
  "confidence": 0.82,
  "hallucination_risk": "low",
  "selected_response_id": 0,
  "supporting_sources": [
    {
      "source": "...",
      "chunk_id": 1,
      "content_hash": "..."
    }
  ],
  "scores": {
    "safe_response_score": 0.82,
    "halluguard_score": 0.21,
    "grounding_score": 0.86,
    "consistency_score": 0.78,
    "judge_score": null
  }
}
```

Risk tags:

```text
low
medium
high
rejected
```

## Stage 8 Implementation Plan

Stage 8 is deployment and evaluation.

Recommended first deployment target:

- FastAPI service in `src/saferesponse_engine/serving/api.py`.
- Request/response schemas in `src/saferesponse_engine/serving/schemas.py`.
- Middleware wrapper in `src/saferesponse_engine/serving/middleware.py`.

Recommended API endpoints:

```text
GET  /health
POST /verify
POST /generate-safe-response
GET  /metrics
```

Recommended evaluation features:

- Small labeled hallucination regression set.
- Grounded QA examples.
- Ungrounded/unanswerable examples.
- Latency tracking by stage.
- Score distribution tracking.
- CI smoke test that does not require downloading large models.

Recommended dashboard:

- Start with a simple FastAPI metrics endpoint or lightweight Streamlit dashboard.
- Track request count, average latency, decision distribution, hallucination-risk distribution, and score histograms.

## Configuration Additions Needed

Add these sections to `config/config.yaml` as stages are implemented:

```yaml
verification_layer:
  root_dir: artifacts/verification
  retrieval_artifact_path: artifacts/retrieval/retrieved_chunks.json
  generation_artifact_path: artifacts/generation/candidates.json
  trace_artifact_path: artifacts/traces/traces.json
  verification_output_path: artifacts/verification/verification_scores.json
  embedding_model: BAAI/bge-m3
  enable_halluguard: true
  enable_grounding_score: true
  enable_consistency_score: true
  enable_judge: false
  halluguard_threshold: 0.35
  grounding_threshold: 0.75
  consistency_threshold: 0.70

fusion_decision_router:
  root_dir: artifacts/decision
  verification_artifact_path: artifacts/verification/verification_scores.json
  decision_output_path: artifacts/decision/decision.json
  grounding_weight: 0.40
  consistency_weight: 0.25
  inverse_halluguard_weight: 0.20
  judge_weight: 0.15
  accept_threshold: 0.75
  reject_threshold: 0.40

final_output:
  root_dir: artifacts/final
  decision_artifact_path: artifacts/decision/decision.json
  final_output_path: artifacts/final/final_response.json

serving:
  host: 0.0.0.0
  port: 8000
  reload: false
```

## Current Gaps and Cleanup Items

These are known structural issues in the current repo:

- `app.py` is empty.
- `setup.py` is empty.
- `Dockerfile` is empty.
- `templates/index.html` is empty.
- `params.yaml` and `schema.yaml` contain placeholder values only.
- There is an accidental-looking file at:

```text
src/saferesponse_engine/models/__init__.pyconfig/config.yaml
```

- `artifacts/` and `logs/` are generated outputs. They may not belong in git long term.
- `__pycache__/` files exist in the workspace and should usually be ignored.
- Stage 2 imports some unused modules.
- Stage 2 has `min_score_threshold` in config but does not use it.
- Stage 3 has `finetuned_model_path` in config but does not use it.
- Stage 4 uses `dtype=` in `AutoModelForCausalLM.from_pretrained`; many Transformers examples use `torch_dtype=`.
- Stage 4 currently re-generates tokens to collect traces, which may not exactly match Stage 3 sampled candidates unless generation is made deterministic or traces are collected during Stage 3.

## Estimated Implementation Timeline

For one developer, from the current state to an end-to-end working prototype:

```text
10 to 15 focused working days
```

For a stronger research-grade version with evaluation, calibration, API hardening, and cleaner deployment:

```text
4 to 6 weeks
```

Detailed estimate:

| Work Area | Estimate |
|---|---:|
| Repo cleanup, packaging, configs, artifact schemas | 1-2 days |
| Stage 5 grounding score | 1-2 days |
| Stage 5 consistency score | 1-2 days |
| Stage 5 HalluGuard baseline from logprobs and hidden states | 2-4 days |
| Stage 5 NTK/Jacobian/spectral research implementation | 5-10 days |
| Stage 6 fusion and decision router | 1-2 days |
| Stage 7 final output formatting and risk tags | 0.5-1 day |
| Stage 8 FastAPI middleware | 2-3 days |
| Evaluation datasets and regression tests | 3-5 days |
| Dashboard and latency/hallucination tracking | 2-4 days |
| Documentation and cleanup | 1-2 days |

Practical milestone plan:

```text
Milestone 1: Working local pipeline through Stage 7
Estimate: 5-8 days

Milestone 2: Verification quality upgrade with HalluGuard baseline
Estimate: 8-12 days total

Milestone 3: FastAPI deployment and evaluation suite
Estimate: 10-15 days total

Milestone 4: Research-grade HalluGuard with NTK/Jacobian/spectral modules
Estimate: 4-6 weeks total
```

The main uncertainty is Stage 5A. A practical HalluGuard-inspired score can be built quickly using logprobs and hidden-state statistics. A rigorous NTK/Jacobian/spectral implementation is substantially more research-heavy and will take longer.

## Development Guidance

Follow the existing stage pattern:

1. Add a config dataclass in `entity/config_entity.py`.
2. Add config construction in `config/configuration.py`.
3. Add implementation under `components/` or a focused subpackage.
4. Add a thin stage wrapper under `pipeline/`.
5. Add the stage to `main.py`.
6. Save outputs as JSON artifacts under `artifacts/`.

Prefer small, inspectable JSON artifacts between stages. This makes research debugging easier and allows each stage to be tested independently.

For model-heavy tests, add lightweight fixture-based tests that do not require downloading Hugging Face models. Keep full model integration tests optional.

## Suggested Next Implementation Order

1. Clean up generated/cache files and placeholder project files.
2. Add artifact dataclasses or Pydantic schemas for Stage 2 through Stage 7 outputs.
3. Implement Stage 5 grounding score.
4. Implement Stage 5 consistency score.
5. Implement HalluGuard baseline using Stage 4 traces.
6. Implement Stage 6 fusion decision router.
7. Implement Stage 7 final output.
8. Add FastAPI endpoint around the full pipeline.
9. Add smoke tests and evaluation fixtures.
10. Improve HalluGuard with NTK/Jacobian/spectral features.

## Current Mental Model

The project should be treated as a safety-oriented LLM middleware pipeline, not just a generator. The generator creates candidate answers, but the core value of the system is the verification and routing layer that decides whether any answer is safe enough to return.

The highest-value next step is Stage 5. Once verification scores exist, Stages 6 and 7 become straightforward engineering work.
