"""Pure metric helpers for the verification ablation harness.

A "hallucination firewall" is judged by how well it accepts grounded answers and
rejects unsupported ones. Given per-example records of (expected_supported,
decided_accept), :func:`confusion` returns the confusion-matrix counts plus the
two rates that matter for safety:

- ``false_accept_rate``: fraction of UNSUPPORTED questions that were accepted
  (hallucinations that slipped through — lower is safer).
- ``false_reject_rate``: fraction of SUPPORTED questions that were rejected
  (good answers wrongly blocked — lower is more useful).
"""

from __future__ import annotations

from typing import Iterable


def confusion(records: Iterable[tuple[bool, bool]]) -> dict[str, float]:
    """Compute confusion-matrix counts and safety rates.

    Args:
        records: iterable of ``(expected_supported, decided_accept)`` booleans.

    Returns:
        Dict with counts (``true_accept``, ``false_reject``, ``false_accept``,
        ``true_reject``, ``total``) and rates (``false_accept_rate``,
        ``false_reject_rate``, ``accuracy``).
    """
    true_accept = false_reject = false_accept = true_reject = 0
    for expected_supported, decided_accept in records:
        if expected_supported and decided_accept:
            true_accept += 1
        elif expected_supported and not decided_accept:
            false_reject += 1
        elif not expected_supported and decided_accept:
            false_accept += 1
        else:
            true_reject += 1

    supported = true_accept + false_reject
    unsupported = false_accept + true_reject
    total = supported + unsupported

    return {
        "true_accept": true_accept,
        "false_reject": false_reject,
        "false_accept": false_accept,
        "true_reject": true_reject,
        "total": total,
        "false_accept_rate": (false_accept / unsupported) if unsupported else 0.0,
        "false_reject_rate": (false_reject / supported) if supported else 0.0,
        "accuracy": ((true_accept + true_reject) / total) if total else 0.0,
    }
