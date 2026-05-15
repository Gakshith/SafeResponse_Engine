# Verification Strategy

Stage 5 combines practical demo signals with optional experimental internal
model signals.

## Default Demo Signals

The default project claim should be based on these signals:

- generation log probability features
- retrieval grounding score
- candidate consistency score

These are enabled by default and can run in the local demo configuration.

## Grounding Threshold

The default grounding threshold is:

```yaml
grounding_threshold: 0.70
retrieval_layer:
  min_score_threshold: 0.95
  min_lexical_matches: 2
```

This is tuned for the controlled 10-article corpus and the lexical fallback
verification backend. It is intentionally not presented as a universal
production threshold. The `min_lexical_matches` guard prevents a single
accidental query-token overlap from sending an unsupported question into the
slower generation path.

## Embedding Backend

The default verification backend is:

```yaml
embedding_backend: lexical
```

Reason:

- It is deterministic and works offline.
- It keeps tests and demos reproducible.
- The HuggingFace embedding backend remains available for higher-quality runs
  when local model files or network access are available.

## Experimental Signals

The following modules are implemented as research paths but disabled by default:

- NTK features
- Jacobian instability
- spectral conditioning

They should be described as experimental unless a specific run enables them,
calibrates them, and reports evaluation results.

## Calibration Workflow

1. Add examples to `evaluation/examples.json`.
2. Run:

   ```bash
   venv/bin/python scripts/run_evaluation.py --include-model-runs
   ```

3. Inspect false accepts and false rejects.
4. Adjust `grounding_threshold`, `halluguard_threshold`, and retrieval
   `min_score_threshold` / `min_lexical_matches`.
5. Rerun the evaluation set and document the result.
