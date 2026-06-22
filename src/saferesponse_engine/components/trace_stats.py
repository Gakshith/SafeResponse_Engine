"""Pure helpers for summarizing per-token log-probabilities.

Used to compute the confidence-trace statistics (mean / min / sequence log-prob)
from log-probs that are captured during generation, so the pipeline does not have
to run a second forward pass over the prompt+answer just to score it.
"""

from __future__ import annotations

from typing import Sequence


def summarize_logprobs(logprobs: Sequence[float]) -> dict[str, float]:
    """Return mean, min, and sequence (sum) of a list of token log-probs."""
    if not logprobs:
        return {"mean_logprob": 0.0, "min_logprob": 0.0, "sequence_score": 0.0}
    return {
        "mean_logprob": round(sum(logprobs) / len(logprobs), 6),
        "min_logprob": round(min(logprobs), 6),
        "sequence_score": round(sum(logprobs), 6),
    }
