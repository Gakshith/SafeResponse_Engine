# Evaluation Guide

Evaluation examples live in:

```text
evaluation/examples.json
```

Each example includes:

- `id`
- `category`
- `query`
- `expected_decision`
- `run_pipeline`
- optional `expected_warnings_any`
- optional `expected_answer_contains`
- optional `expected_answer_not_contains`
- `notes`

Categories currently covered:

- Small talk
- Supported factual query
- Unsupported query
- Unsupported current/live facts
- Ambiguous query
- Misspelled supported query
- Entity-overlap traps

## Fast Evaluation

Run only examples that do not require model inference:

```bash
venv/bin/python scripts/run_evaluation.py --skip-model-runs
```

Model-heavy examples are skipped in this mode.

## Full Evaluation

Run all examples:

```bash
venv/bin/python scripts/run_evaluation.py
```

This can download or load Hugging Face models and rebuild retrieval artifacts.

Write a persisted evaluation artifact:

```bash
venv/bin/python scripts/run_evaluation.py \
  --output artifacts/evaluation/latest.json
```

## Interpreting Results

The runner reports:

- `passed`
- `failed`
- `skipped`
- expected decision
- actual decision
- confidence
- risk score
- source
- warnings
- failure reasons
- pass rate
- coverage rate
- category summaries

Failures should be reviewed by checking:

1. Retrieval chunks in `artifacts/retrieval/retrieved_chunks.json`.
2. Candidate answers in `artifacts/generation/candidates.json`.
3. Verification scores in `artifacts/verification/verification_scores.json`.
4. Fusion result in `artifacts/fusion/fusion_result.json`.
5. Final response in `artifacts/final_output/final_response.json`.

## Calibration Notes

Grounding and verification thresholds should be tuned against this evaluation
set before adding broader examples. The default corpus is small, so rejected
answers can be correct behavior when the corpus does not contain support.
