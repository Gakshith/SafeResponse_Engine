# SafeResponse Engine Stage 6: Fusion + Decision Router

## Overview

Stage 6 is the fusion and routing layer for SafeResponse Engine. It consumes
verification outputs from Stage 5, combines the available hallucination-risk
signals into one risk score per candidate, ranks candidates, and routes the
pipeline to one of four decisions:

- `ACCEPT`
- `RERANK`
- `REWRITE`
- `REJECT`

Stage 6 is the final decision point before Stage 7 formats a user-facing
answer.

## Pipeline Position

Reads:

```text
artifacts/verification/verification_scores.json
artifacts/traces/traces.json
```

Writes:

```text
artifacts/fusion/fusion_result.json
```

## Inputs

For each candidate, Stage 6 uses:

- `halluguard_score`: internal model-signal hallucination risk from Stage 5.
- `grounding_score`: similarity between the answer and retrieved context.
- `consistency_score`: agreement across generated candidates.
- `judge_score`: optional external or classifier-based risk score.
- `risk_signals`: binary risk flags from Stage 5.
- `supporting_source`: best grounding source metadata.
- `sequence_score`: joint response log probability from Stage 4 traces.

## Fusion Formula

When all four signals are available:

```text
combined_risk =
  0.35 * halluguard_score
+ 0.30 * (1 - grounding_score)
+ 0.20 * (1 - consistency_score)
+ 0.15 * judge_score
```

Lower `combined_risk` is safer. If `judge_score` is missing, Stage 6 excludes
the judge term and normalizes the remaining configured weights.

## Candidate Ranking

Candidates are sorted by ascending `combined_risk`. If two candidates have the
same risk score, the candidate with the higher `sequence_score` wins.

```python
ranked = sorted(candidates, key=lambda c: (c["combined_risk"], -c["sequence_score"]))
best = ranked[0]
```

## Routing Logic

Default thresholds:

| Decision | Rule | Action |
|---|---|---|
| `ACCEPT` | `risk < 0.30` | Return the best candidate to Stage 7. |
| `RERANK` | `0.30 <= risk < 0.50` and best candidate is not primary | Return safer non-primary candidate. |
| `REWRITE` | `0.50 <= risk < 0.75` and rewrite attempts remain | Regenerate with more focused context. |
| `REJECT` | `risk >= 0.75` or rewrite attempts exhausted | Decline to answer reliably. |

Current routing pseudocode:

```python
if best.combined_risk < accept_threshold:
    decision = "ACCEPT"
elif best.combined_risk < rewrite_threshold:
    if best.response_id != 0:
        decision = "RERANK"
    else:
        decision = "ACCEPT"
elif best.combined_risk < reject_threshold:
    if rewrite_attempt < max_rewrite_attempts:
        decision = "REWRITE"
    else:
        decision = "REJECT"
else:
    decision = "REJECT"
```

## Rewrite Behavior

For a `REWRITE` decision, Stage 6 emits a `rewrite_query` field:

```text
original_query

Focus specifically on: <supporting_source.source>
```

The Stage 6 pipeline wrapper now performs the rewrite loop. When `REWRITE` is
returned, it writes the `rewrite_query` to the query artifact, reruns Stages 2-5,
and then routes again. The loop stops when the router returns `ACCEPT`,
`RERANK`, `REJECT`, or when rewrite attempts are exhausted.

The component still remains pure: `FusionDecisionRouter.route(...)` only reads
artifacts and writes the fusion artifact. The pipeline wrapper owns rerunning
upstream stages.

## Output Artifact

Stage 6 writes:

```text
artifacts/fusion/fusion_result.json
```

Example shape:

```json
{
  "query": "Who was Alexander the Great?",
  "decision": "ACCEPT",
  "selected_response_id": 0,
  "rewrite_attempt": 0,
  "decision_reason": "Candidate 0 combined_risk 0.220 < accept_threshold 0.300.",
  "fusion_scores": [
    {
      "response_id": 0,
      "combined_risk": 0.22,
      "rank": 1,
      "risk_terms": {
        "halluguard_risk": 0.28,
        "grounding_risk": 0.13,
        "consistency_risk": 0.09,
        "judge_risk": 0.15
      },
      "effective_weights": {
        "halluguard": 0.35,
        "grounding": 0.3,
        "consistency": 0.2,
        "judge": 0.15
      },
      "risk_contributions": {
        "halluguard": 0.098,
        "grounding": 0.039,
        "consistency": 0.018,
        "judge": 0.0225
      },
      "halluguard_score": 0.28,
      "grounding_score": 0.87,
      "consistency_score": 0.91,
      "judge_score": 0.15
    }
  ],
  "selected_candidate": {
    "response_id": 0,
    "text": "Alexander the Great was king of Macedon...",
    "is_primary": true,
    "combined_risk": 0.22,
    "rank": 1,
    "supporting_source": {
      "source": "Alexander the Great",
      "chunk_id": 423,
      "content_hash": "a3f9b2c1d4e5f678"
    }
  }
}
```

## Configuration

Configured in:

```text
config/config.yaml
```

```yaml
fusion_router:
  root_dir: artifacts/fusion
  verification_artifact_path: artifacts/verification/verification_scores.json
  traces_artifact_path: artifacts/traces/traces.json
  fusion_output_path: artifacts/fusion/fusion_result.json

  weight_halluguard: 0.35
  weight_grounding: 0.30
  weight_consistency: 0.20
  weight_judge: 0.15

  accept_threshold: 0.30
  rewrite_threshold: 0.50
  reject_threshold: 0.75

  max_rewrite_attempts: 3
```

Stage 6 validates this config at startup:

- all weights must be non-negative
- at least one weight must be positive
- thresholds must satisfy
  `0 <= accept_threshold < rewrite_threshold < reject_threshold <= 1`
- `max_rewrite_attempts` must be non-negative

## Tests

Stage 6 routing tests are in:

```text
tests/test_fusion_decision_router.py
```

They cover:

- `ACCEPT`
- `RERANK`
- `REWRITE`
- `REJECT`
- missing `judge_score`
- malformed score values
- invalid fusion config

## Implemented Files

```text
src/saferesponse_engine/components/fusion_decision_router.py
src/saferesponse_engine/pipeline/stage_06_fusion_decision_router.py
```

Related updates:

```text
config/config.yaml
src/saferesponse_engine/entity/config_entity.py
src/saferesponse_engine/config/configuration.py
main.py
```
