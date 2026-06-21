# SafeResponse Engine — Research Upgrade Design

- **Date:** 2026-06-21
- **Status:** Approved (proceeding to implementation plan)
- **Author:** Claude (Superpowers brainstorming) with Gakshith

## 1. Problem & Goal

SafeResponse Engine is an 8-stage RAG safety pipeline whose real contribution is its
**verification layer that rejects unsupported / hallucinated answers**. In its current
state the project undermines its own thesis and behaves poorly:

- **Nonsensical answers:** the corpus is only 5 hand-pasted Wikipedia stubs, the generator
  is the weak `Qwen2.5-0.5B-Instruct`, and answers are truncated at `max_new_tokens=48`.
- **Slow:** `use_cache=False` in generation makes decoding ~quadratic; the model runs two
  full forward passes per query; nothing is warmed at startup; 7 stages serialize JSON to
  disk; may run on CPU instead of MPS.
- **Retrieval is lexical-only** with a very strict threshold (`min_score_threshold=0.95`),
  even though dense-retrieval deps (`sentence-transformers`, `faiss-cpu`) are installed but
  unused.
- **The UI discards the safety story:** it renders only `data.answer`, ignoring the
  `decision`, `confidence`, `source`, `risk_score`, and `warnings` that the API returns.

**Goal:** turn this into a credible, measurable **"hallucination firewall"** research demo:
coherent answers on supported questions, safe rejection on unsupported ones, fast responses,
a reproducible ablation/metrics harness, and a premium UI that makes the safety verdict the
hero.

## 2. Approved Decisions

| Area | Decision |
|---|---|
| Generator | `Qwen/Qwen2.5-1.5B-Instruct`, run locally on MPS |
| Corpus | Curated **~40 Wikipedia articles**, chunked |
| Retrieval | **Dense FAISS** with `BAAI/bge-small-en-v1.5`; lexical kept as fallback |
| Scope | **Full**: perf + model + corpus + ablation/eval harness + UI redesign |
| Advanced signals | NTK / Jacobian / spectral stay **off and labeled experimental** |
| Runtime | Python **3.12** venv on arm64 (system python 3.14 is too new for torch/faiss wheels) |

## 3. Architecture & Components

The existing stage architecture is sound and is kept. Changes are scoped per component.

### 3.1 Environment (`requirements.txt`, venv, project `CLAUDE.md`)
- Create `venv` with `python3.12`; install pinned requirements; confirm torch sees MPS.
- Document the working setup + first-run model-download note in the project `CLAUDE.md`.

### 3.2 Generation (`components/generation_layer.py`, `config.yaml`)
- Model → `Qwen/Qwen2.5-1.5B-Instruct` (generation, trace, verification trace_model).
- `use_cache=True`; `max_new_tokens: 160`; add `repetition_penalty` (~1.1); greedy primary.
- Keep substring stop handling; ensure clean decode of only-new tokens.

### 3.3 Startup warm-up (`serving/api.py`)
- On app startup, instantiate the engine and **pre-load the model + FAISS index** so the
  first user query does not pay the cold-load cost. Health endpoint reports "warm".

### 3.4 Retrieval (`components/retrieval_layer.py`, `data/demo_corpus.json`, `config.yaml`)
- Replace corpus with ~40 curated articles across varied domains (history, science,
  geography, technology, sports, arts) — enough breadth for a clean supported/unsupported
  split.
- Implement a **dense backend**: embed chunks with `BAAI/bge-small-en-v1.5`, build a FAISS
  index, **persist + cache** it (rebuild only when corpus/config changes). `retrieval_backend:
  dense`, `top_k: 4`. Keep the lexical backend selectable as fallback.
- Recalibrate thresholds so in-corpus questions are answered and out-of-corpus reject.

### 3.5 Verification + ablation (`components/verification_layer.py`, new `scripts/run_ablation.py`)
- Keep and calibrate the three demo signals: **logprob, grounding, consistency** on
  `evaluation/examples.json`.
- New `scripts/run_ablation.py`: run the eval set with verification **ON vs OFF** and with
  each signal individually toggled; emit a metrics report (**false-accept rate, false-reject
  rate, accept/reject/rewrite counts, mean latency**) to `Artifacts/` and append a summary to
  `model_registry/`.
- NTK/Jacobian/spectral remain disabled and documented as experimental.

### 3.6 UI (`templates/index.html`, `static/styles.css`, plus JS)
- Premium, **light + dark** via CSS variables; distinct professional palette (trust
  teal/emerald + amber/red risk accents — not the default purple/navy).
- Surface the safety verdict as the hero: **decision badge** (ACCEPT / REVIEW / REJECT),
  **confidence meter**, **source-citation card**, **per-signal risk breakdown**
  (logprob / grounding / consistency bars), **warnings**, and **latency + per-stage timings**.
- **Live 8-stage pipeline progress** while a query runs. Keep the chat-history sidebar.
- Built with the `ui-ux-pro-max` / `ui-styling` skills.

## 4. Data Flow (unchanged shape, faster + richer)

```
Query → Retrieval (dense FAISS, cached) → Generation (Qwen2.5-1.5B, KV-cache)
      → Trace (logprobs) → Verification (logprob+grounding+consistency)
      → Fusion router (ACCEPT/RERANK/REWRITE/REJECT) → Final output
      → API response (answer + decision + confidence + source + risk + timings)
      → UI renders verdict-first
```

## 5. Error Handling
- Missing local model / index → actionable message (set `SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1`).
- Out-of-corpus query → fast safe rejection without forcing a model generation.
- FAISS/embedding load failure → fall back to lexical retrieval with a logged warning.
- Pipeline busy / job failure paths preserved (existing lock + async-job behavior).

## 6. Testing & Verification
- **TDD** for pure logic: retrieval scoring/threshold, each verification signal, fusion
  routing, final-output formatting, ablation metric computation. **Existing tests stay green.**
- Verify by **running**: `pytest`, `scripts/run_evaluation.py`, `scripts/run_ablation.py`,
  then boot the API and send one supported + one unsupported question — confirm coherent
  answer, correct decision, and improved latency. Capture **before/after latency** as evidence
  in `Artifacts/`.

## 7. Execution Approach (Superpowers + Ruflo)
brainstorm (done) → using-git-worktrees → writing-plans → subagent-driven-development /
dispatching-parallel-agents (four largely independent workstreams: corpus/retrieval ·
generator/perf · verification/ablation · UI) → test-driven-development → requesting-code-review
→ verification-before-completion → finishing-a-development-branch. Ruflo swarm/memory used for
the parallel workstreams and cross-stream context.

## 8. Out of Scope (this pass)
- Turning on NTK / Jacobian / spectral signals (left experimental).
- Live web/Wikipedia retrieval at query time.
- Cloud/EC2 deployment.
- LoRA fine-tuning runs (kept available, not required for this upgrade).
