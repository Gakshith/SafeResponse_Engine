# SafeResponse Engine Research Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn SafeResponse Engine into a fast, coherent, measurable "hallucination firewall" research demo with a verdict-first UI.

**Architecture:** Keep the 8-stage pipeline. Swap the generator to Qwen2.5-1.5B-Instruct, fix the KV-cache/perf bugs, point the existing (but unused) dense FAISS retrieval at a curated 40-article local corpus, calibrate verification, add an ON-vs-OFF ablation harness, and redesign the web UI to surface the safety verdict.

**Tech Stack:** Python 3.12, PyTorch (MPS), transformers, sentence-transformers/HuggingFaceEmbeddings, FAISS (langchain_community), FastAPI, vanilla JS + CSS.

**Working dir:** `.worktrees/research-upgrade` (branch `feat/research-upgrade-20260621`). All commands use `venv/bin/python`. Tests: `venv/bin/python -m pytest`.

---

## Phase 0 — Environment & Baseline

### Task 0: Verified clean baseline
**Files:** none (setup only)

- [ ] **Step 1:** Confirm venv + deps installed: `venv/bin/python -c "import torch, transformers, faiss, sentence_transformers; print(torch.backends.mps.is_available())"` — expect `True`.
- [ ] **Step 2:** Run baseline: `venv/bin/python -m pytest -q`. Record pass/fail count. Some tests may require model downloads — note which, mark them, do NOT "fix" pre-existing failures yet.
- [ ] **Step 3:** Capture baseline latency evidence (best-effort): start API with `SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 venv/bin/uvicorn app:app --port 8000`, send one supported question via `/v1/chat`, save timing JSON to `../../../../Artifacts/before-latency.json` (container `Artifacts/`). Stop server.
- [ ] **Step 4:** Commit nothing (baseline only).

---

## Phase 1 — Generator Swap + Performance

### Task 1: Switch model to Qwen2.5-1.5B-Instruct
**Files:** Modify `config/config.yaml`

- [ ] **Step 1:** In `config/config.yaml`, set `generation_layer.model_name`, `trace_collection_layer.model_name`, and `verification_layer.trace_model_name` to `Qwen/Qwen2.5-1.5B-Instruct`. (Leave `training.base_model_name` as-is.)
- [ ] **Step 2:** Set `generation_layer.max_new_tokens: 160`.
- [ ] **Step 3:** Commit: `git commit -am "feat(gen): use Qwen2.5-1.5B-Instruct, raise max_new_tokens to 160"`.

### Task 2: Fix generation perf (KV cache + repetition penalty)
**Files:** Modify `src/saferesponse_engine/components/generation_layer.py`, Test `tests/test_generation_params.py`

- [ ] **Step 1: Write failing test** `tests/test_generation_params.py`:
```python
import inspect
from saferesponse_engine.components.generation_layer import GenerationLayer

def test_generation_uses_kv_cache_and_repetition_penalty():
    src = inspect.getsource(GenerationLayer._generate_single)
    assert "use_cache=False" not in src, "KV cache must be enabled for speed"
    assert "repetition_penalty" in src, "repetition_penalty should be set"
```
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_generation_params.py -v` — expect FAIL.
- [ ] **Step 3: Implement** in `generation_layer.py` `_generate_single`: change `use_cache=False` → `use_cache=True`, and add `repetition_penalty=1.1` to the `self.model.generate(...)` call.
- [ ] **Step 4: Run** the test — expect PASS. Then `venv/bin/python -m pytest -q` to confirm no regressions.
- [ ] **Step 5: Commit** `git commit -am "perf(gen): enable KV cache, add repetition_penalty"`.

### Task 3: Warm model + retrieval index at API startup
**Files:** Modify `src/saferesponse_engine/serving/api.py`, `src/saferesponse_engine/components/chat_engine.py`, Test `tests/test_warmup.py`

- [ ] **Step 1: Write failing test** `tests/test_warmup.py`:
```python
from saferesponse_engine.components.chat_engine import SafeResponseChatEngine

def test_engine_has_warmup_method():
    assert hasattr(SafeResponseChatEngine, "warmup")
```
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_warmup.py -v` — expect FAIL.
- [ ] **Step 3: Implement** `SafeResponseChatEngine.warmup(self)` that loads the generation model and builds/loads the FAISS index (when `retrieval_backend == "dense"`) so the first query is warm. Wrap in try/except logging a warning on failure (must not crash startup). In `api.py`, add a FastAPI startup handler (`@app.on_event("startup")` or lifespan) that calls `_engine().warmup()` in a background thread; update `status()` to include `"warm": <bool>`.
- [ ] **Step 4: Run** `venv/bin/python -m pytest tests/test_warmup.py tests/test_api.py -v` — expect PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(serving): warm model + index at startup"`.

---

## Phase 2 — Curated Corpus + Dense Retrieval

### Task 4: Build the 40-article local corpus
**Files:** Create `scripts/build_demo_corpus.py`, Create (generated, committed) `data/demo_corpus.json`, Test `tests/test_demo_corpus.py`

- [ ] **Step 1: Write failing test** `tests/test_demo_corpus.py`:
```python
import json
from pathlib import Path

def test_demo_corpus_has_breadth():
    data = json.loads(Path("data/demo_corpus.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 40, "corpus should have >= 40 articles"
    for item in data:
        assert item["title"] and len(item["text"]) > 400
```
- [ ] **Step 2: Run** — expect FAIL (only 5 articles today).
- [ ] **Step 3: Implement** `scripts/build_demo_corpus.py`: a curated list of >=40 Wikipedia article titles across domains (history, science, geography, technology, sports, arts, biology, space). Fetch each via the Wikipedia REST summary+extract API (`https://en.wikipedia.org/api/rest_v1/page/summary/<title>` plus the plain-text extract endpoint) OR reuse `RetrievalLayer._load_wikipedia_documents` against the HF dataset. Write `[{"title","text"}, ...]` to `data/demo_corpus.json`. Note: `data/demo_corpus.json` is NOT git-ignored (only `data/squad_v2/` is), so commit the generated file for offline reproducibility.
- [ ] **Step 4: Run** `venv/bin/python scripts/build_demo_corpus.py` then `venv/bin/python -m pytest tests/test_demo_corpus.py -v` — expect PASS.
- [ ] **Step 5: Commit** `git add scripts/build_demo_corpus.py data/demo_corpus.json tests/test_demo_corpus.py && git commit -m "feat(corpus): curated 40-article demo corpus + builder"`.

### Task 5: Point dense FAISS indexing at the local corpus
**Files:** Modify `src/saferesponse_engine/components/retrieval_layer.py`, Test `tests/test_retrieval_dense_local.py`

Today `build_index()` calls `_load_wikipedia_documents()` (HF download) and hardcodes bge-m3 metadata. Change it to use the local corpus and config-driven metadata.

- [ ] **Step 1: Write failing test** `tests/test_retrieval_dense_local.py`:
```python
import inspect
from saferesponse_engine.components.retrieval_layer import RetrievalLayer

def test_build_index_uses_local_corpus():
    src = inspect.getsource(RetrievalLayer.build_index)
    assert "_load_corpus_documents" in src, "dense index must use the local corpus loader"
```
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** in `build_index()`: replace the `raw_docs = self._load_wikipedia_documents()` call with `raw_docs = self._load_corpus_documents()`; change `expected_metadata` to derive from config (`"embedding": self.config.embedding_model`, `"corpus": str(self.config.local_corpus_path)`, `"num_articles": self.config.num_articles`) so a corpus/model change invalidates the cached index.
- [ ] **Step 4: Run** the new test — expect PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(retrieval): build dense FAISS index from local corpus"`.

### Task 6: Switch retrieval to dense + lighter embeddings + calibrate
**Files:** Modify `config/config.yaml`

- [ ] **Step 1:** Set `retrieval_layer.retrieval_backend: dense`; `retrieval_layer.embedding_model: BAAI/bge-small-en-v1.5` (also set `verification_layer.embedding_model` the same for grounding consistency); keep `top_k: 4` (it is currently 3 — set to 4).
- [ ] **Step 2: Calibrate** `min_score_threshold`. FAISS here returns L2 distance on normalized embeddings (lower = more similar; the code drops `score > threshold`). Run a calibration probe:
```bash
SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 venv/bin/python - <<'PY'
from saferesponse_engine.config.configuration import ConfigurationManager
from saferesponse_engine.components.retrieval_layer import RetrievalLayer
cfg = ConfigurationManager().get_retrieval_layer_config()
r = RetrievalLayer(cfg); vs = r.build_index()
for q in ["Who was Abraham Lincoln?", "What is the capital of France?", "Explain quantum entanglement", "Who won the 2050 world cup?"]:
    print(q, [round(float(s),3) for _,s in vs.similarity_search_with_score(q, k=4)])
PY
```
Pick a `min_score_threshold` that keeps in-corpus hits and drops out-of-corpus (expect somewhere ~0.8–1.2 for bge-small L2). Set it in config.
- [ ] **Step 3: Verify** end-to-end retrieval: write a supported query to the query artifact and run Stage 2; confirm chunks are returned. Run `venv/bin/python -m pytest tests/test_retrieval_layer.py -v`.
- [ ] **Step 4: Commit** `git commit -am "feat(retrieval): dense backend + bge-small + calibrated threshold"`.

---

## Phase 3 — Verification Calibration + Ablation Harness

### Task 7: Ablation metric helpers (pure logic, TDD)
**Files:** Create `src/saferesponse_engine/components/ablation_metrics.py`, Test `tests/test_ablation_metrics.py`

- [ ] **Step 1: Write failing test** `tests/test_ablation_metrics.py`:
```python
from saferesponse_engine.components.ablation_metrics import confusion

def test_confusion_counts():
    # records: (expected_supported: bool, decided_accept: bool)
    records = [(True, True), (True, False), (False, True), (False, False), (False, False)]
    m = confusion(records)
    assert m["true_accept"] == 1
    assert m["false_reject"] == 1   # supported but rejected
    assert m["false_accept"] == 1   # unsupported but accepted
    assert m["true_reject"] == 2
    assert round(m["false_accept_rate"], 3) == round(1/3, 3)  # 1 of 3 unsupported
    assert round(m["false_reject_rate"], 3) == round(1/2, 3)  # 1 of 2 supported
```
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** `confusion(records)` returning the counts and rates above (guard divide-by-zero → 0.0).
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(ablation): confusion-matrix metric helpers"`.

### Task 8: Ablation runner script
**Files:** Create `scripts/run_ablation.py`, Test `tests/test_run_ablation.py`

The eval set lives in `evaluation/examples.json`. Each example has an expected decision/support label (inspect the file; reuse `scripts/run_evaluation.py` patterns). The runner toggles `verification_layer.enable_*` flags and re-runs decisions.

- [ ] **Step 1: Write failing test** `tests/test_run_ablation.py`: import `scripts/run_ablation.py` as a module (it must expose `build_configs()` returning a dict of named flag-sets including `"all_off"`, `"all_on"`, `"logprob_only"`, `"grounding_only"`, `"consistency_only"`), and assert those keys exist. Use `--skip-model-runs`-style guard so the test does not load models.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** `scripts/run_ablation.py` with `build_configs()`, a `--skip-model-runs` flag for tests, and a `main()` that (when not skipping) runs each config over the eval examples, computes `ablation_metrics.confusion(...)`, prints a table, and writes `Artifacts/ablation-report.json` + appends a summary line to `model_registry/registry.json`.
- [ ] **Step 4: Run** `venv/bin/python -m pytest tests/test_run_ablation.py -v` — expect PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(ablation): ON-vs-OFF ablation runner"`.

### Task 9: Calibrate verification thresholds + run the real ablation
**Files:** Modify `config/config.yaml`; produces `Artifacts/ablation-report.json`

- [ ] **Step 1:** Run `SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 venv/bin/python scripts/run_ablation.py`. Inspect false-accept / false-reject per config.
- [ ] **Step 2:** Adjust `verification_layer.grounding_threshold` / `consistency_threshold` / `halluguard_threshold` and `fusion_router` thresholds to minimize false-accept while keeping false-reject reasonable. Re-run until stable.
- [ ] **Step 3:** Save the final `Artifacts/ablation-report.json`. This is the research result.
- [ ] **Step 4: Commit** `git commit -am "chore(verify): calibrate thresholds from ablation results"` and `git -C ../../.. add Artifacts/ablation-report.json` is NOT needed (Artifacts is in the container, outside the repo) — copy the report into `docs/` if it should be tracked: `cp Artifacts/ablation-report.json docs/ablation-report.json && git add docs/ablation-report.json && git commit -m "docs: commit ablation report"`.

---

## Phase 4 — Website Redesign (verdict-first, light + dark)

> Use the `ui-ux-pro-max` and `ui-styling` skills. Palette: trust teal/emerald primary + amber/red risk accents (NOT purple/navy). Full light + dark via CSS variables (`:root` + `:root[data-theme="light"]`). Do not hardcode colors.

### Task 10: Response renderer shows the safety verdict
**Files:** Modify `templates/index.html` (the inline JS + markup)

The API already returns `decision`, `confidence`, `confidence_color`, `source`, `risk_score`, `warnings`, `risk_explanation`, `pipeline_summary.stage_timings_seconds`. The current UI ignores all but `answer`.

- [ ] **Step 1:** In the assistant message render path, after writing the answer text, render a **verdict block**: a decision badge (ACCEPT=green, REVIEW/RERANK/REWRITE=amber, REJECT=red, DIRECT=neutral) using `decision`; a confidence chip from `confidence`/`confidence_color`; a source-citation line from `source`; a collapsible "Why" panel showing `risk_explanation` + `warnings` as chips; and a per-signal risk breakdown (logprob/grounding/consistency) if present in `pipeline_summary`/risk fields.
- [ ] **Step 2:** Show **latency**: read `pipeline_summary.total_stage_time_seconds` and `stage_timings_seconds`; render a small "answered in Xs" with an expandable per-stage bar.
- [ ] **Step 3: Verify** with preview tools (see Phase 5): send supported + unsupported questions, confirm ACCEPT vs REJECT badges render. Screenshot.
- [ ] **Step 4: Commit** `git commit -am "feat(ui): verdict-first response rendering"`.

### Task 11: Premium restyle + light/dark theme
**Files:** Rewrite `static/styles.css`; add a theme toggle to `templates/index.html`

- [ ] **Step 1:** Define tokens in `:root` (dark default) and `:root[data-theme="light"]` overrides: backgrounds, text, border, `--accent` (teal/emerald), `--accept`, `--review` (amber), `--reject` (red), `--muted`. Replace existing hardcoded palette.
- [ ] **Step 2:** Restyle shell, sidebar, composer, message bubbles, verdict block, badges, meters; add subtle motion (badge fade-in, meter fill transition). Keep it responsive ≥320px.
- [ ] **Step 3:** Add a header theme toggle button that flips `document.documentElement.dataset.theme` and persists to `localStorage`.
- [ ] **Step 4: Verify** with `preview_resize` (light + dark, narrow + wide) and `preview_screenshot`.
- [ ] **Step 5: Commit** `git commit -am "feat(ui): premium light/dark theme"`.

### Task 12: Live pipeline-stage progress
**Files:** Modify `templates/index.html` JS (the `waitForJob` poller)

- [ ] **Step 1:** While polling a chat job, show an 8-stage progress strip (Query → Retrieval → Generation → Trace → Verification → Fusion → Output) that advances on an estimated cadence and resolves to "done" with real `stage_timings_seconds` when the job completes.
- [ ] **Step 2: Verify** the strip appears during a live query and is replaced by the verdict block on completion. Screenshot.
- [ ] **Step 3: Commit** `git commit -am "feat(ui): live pipeline-stage progress"`.

---

## Phase 5 — Integration Verification + Docs

### Task 13: Full integration verification (evidence)
**Files:** none (verification); writes `Artifacts/after-latency.json`

- [ ] **Step 1:** `venv/bin/python -m pytest -q` — all tests green (note any model-gated skips).
- [ ] **Step 2:** `SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 venv/bin/python scripts/run_evaluation.py` — passes.
- [ ] **Step 3:** Boot API (preview tools). Send a **supported** question (e.g. about Lincoln) → expect coherent answer + ACCEPT + source. Send an **unsupported** question (e.g. "Who won the 2050 World Cup?") → expect safe REJECT. Capture both as evidence. Record latency to `Artifacts/after-latency.json` and compare to `before-latency.json`.
- [ ] **Step 4:** Screenshot the UI in light + dark with a verdict shown.

### Task 14: Update docs
**Files:** Modify `README.md`, `docs/project_completion_plan.md`, project `CLAUDE.md` (container), `docs/verification_strategy.md`

- [ ] **Step 1:** Update `README.md`: new model, dense retrieval + 40-article corpus, ablation harness command (`scripts/run_ablation.py`), UI features, and corrected setup (python3.12 venv, MPS).
- [ ] **Step 2:** Add an "Ablation Results" section to README/docs referencing `docs/ablation-report.json`.
- [ ] **Step 3:** Update the container `CLAUDE.md` runtime notes (python3.12, MPS, first-run model download env var).
- [ ] **Step 4: Commit** `git commit -am "docs: document research upgrade (model, retrieval, ablation, UI)"`.

---

## Self-Review Notes
- **Spec coverage:** env (Task 0,1) · generator+perf (1,2,3) · corpus+dense FAISS (4,5,6) · verification calibration + ablation (7,8,9) · UI verdict/theme/progress (10,11,12) · verification+docs (13,14). All spec sections mapped.
- **Parallelism:** Phases 1, 2, 3, 4 are largely independent and may be dispatched to parallel agents; Phase 0 precedes all, Phase 5 follows all. Phase 3 calibration (Task 9) depends on Phase 2 retrieval being live.
- **Artifacts location:** container `Artifacts/` is outside the git repo; tracked reports are copied into `docs/`.
