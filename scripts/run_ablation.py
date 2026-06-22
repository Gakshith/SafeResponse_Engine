"""Verification ablation harness for SafeResponse Engine.

Runs the evaluation set with each verification signal isolated (logprob-only,
grounding-only, consistency-only) and with all signals combined, reporting the
safety metrics (false-accept rate, false-reject rate, accuracy) for each.
Verification stays fully enabled; the harness varies the fusion *weights* so a
signal's individual contribution is measured cleanly (the system no longer fails
closed when grounding is removed). This is the core research artifact: it shows
which internal signals actually catch hallucinations.

Usage:
    venv/bin/python scripts/run_ablation.py                 # full run (loads model)
    venv/bin/python scripts/run_ablation.py --skip-model-runs   # config-only (tests)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_EXAMPLES = PROJECT_ROOT / "evaluation" / "examples.json"
ACCEPT_DECISIONS = {"ACCEPT", "RERANK"}


def build_configs() -> dict[str, dict[str, float]]:
    """Named fusion-weight sets that isolate each verification signal.

    Verification stays fully enabled, so every signal score AND the supporting
    source are always computed. The ablation varies which signal drives the
    fusion decision by zeroing the other weights. This isolates a signal's true
    contribution instead of failing closed when grounding is removed.
    """
    zero = {
        "weight_halluguard": 0.0,
        "weight_grounding": 0.0,
        "weight_consistency": 0.0,
        "weight_judge": 0.0,
    }
    return {
        "logprob_only": {**zero, "weight_halluguard": 1.0},
        "grounding_only": {**zero, "weight_grounding": 1.0},
        "consistency_only": {**zero, "weight_consistency": 1.0},
        "all_on": {
            "weight_halluguard": 0.35,
            "weight_grounding": 0.30,
            "weight_consistency": 0.20,
            "weight_judge": 0.15,
        },
    }


def classify_example(example: dict[str, Any]) -> bool | None:
    """Map an example's expected decision to a support label.

    Returns True if the answer should be supported/accepted, False if it should
    be rejected (unsupported), and None for non-hallucination cases (DIRECT
    small-talk) that are excluded from the confusion matrix.
    """
    expected = example.get("expected_decision")
    expected_set = {expected} if isinstance(expected, str) else set(expected or [])
    if expected_set & ACCEPT_DECISIONS:
        return True
    if "REJECT" in expected_set:
        return False
    return None


def decided_accept(decision: str) -> bool:
    return str(decision).upper() in ACCEPT_DECISIONS


@contextmanager
def patched_fusion_weights(weights: dict[str, float], force_answer: bool = False):
    """Temporarily set fusion_router.weight_* in config.yaml (verification stays on).

    When ``force_answer`` is set, the generator is told to never abstain, so the
    verification signals must catch confidently-wrong answers (the stress test).
    """
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(original_text)
        data["fusion_router"].update(weights)
        data["generation_layer"]["force_answer"] = force_answer
        # Keep all verification signals enabled so every score + the supporting
        # source are computed regardless of which signal drives the decision.
        for flag in (
            "enable_halluguard",
            "enable_grounding_score",
            "enable_consistency_score",
        ):
            data["verification_layer"][flag] = True
        CONFIG_PATH.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        yield
    finally:
        CONFIG_PATH.write_text(original_text, encoding="utf-8")


def _run_config(
    name: str,
    weights: dict[str, float],
    examples: list[dict[str, Any]],
    force_answer: bool = False,
) -> dict[str, Any]:
    from saferesponse_engine.components.ablation_metrics import confusion
    from saferesponse_engine.components.chat_engine import SafeResponseChatEngine

    records: list[tuple[bool, bool]] = []
    per_example: list[dict[str, Any]] = []
    started = time.perf_counter()
    with patched_fusion_weights(weights, force_answer=force_answer):
        engine = SafeResponseChatEngine()
        for example in examples:
            expected_supported = classify_example(example)
            if expected_supported is None:
                continue
            response = engine.query(query=example["query"], run_pipeline=True)
            decision = str(response.get("decision"))
            accepted = decided_accept(decision)
            records.append((expected_supported, accepted))
            per_example.append({
                "id": example.get("id"),
                "query": example["query"],
                "expected_supported": expected_supported,
                "decision": decision,
                "accepted": accepted,
            })
    metrics = confusion(records)
    metrics["latency_seconds"] = round(time.perf_counter() - started, 3)
    return {"config": name, "weights": weights, "metrics": metrics, "examples": per_example}


def _print_table(results: list[dict[str, Any]]) -> None:
    header = f"{'config':<18}{'FAR':>8}{'FRR':>8}{'acc':>8}{'n':>5}"
    print(header)
    print("-" * len(header))
    for result in results:
        m = result["metrics"]
        print(
            f"{result['config']:<18}"
            f"{m['false_accept_rate']:>8.3f}"
            f"{m['false_reject_rate']:>8.3f}"
            f"{m['accuracy']:>8.3f}"
            f"{m['total']:>5}"
        )
    print("\nFAR=false-accept rate (hallucinations passed), FRR=false-reject rate (good answers blocked)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the verification ablation harness.")
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--output", default="artifacts/ablation-report.json")
    parser.add_argument(
        "--skip-model-runs",
        action="store_true",
        help="Validate configs without loading the model (for tests/CI).",
    )
    parser.add_argument(
        "--force-answer",
        action="store_true",
        help="Stress test: forbid abstention so the signals must catch hallucinations.",
    )
    args = parser.parse_args()

    configs = build_configs()
    examples = json.loads(Path(args.examples).read_text(encoding="utf-8"))

    if args.skip_model_runs:
        print("Configs:", ", ".join(configs))
        n = sum(1 for ex in examples if classify_example(ex) is not None)
        print(f"{n} hallucination examples would run across {len(configs)} configs.")
        return

    if args.force_answer:
        print("[stress test] force_answer=ON — abstention suppressed.\n")
    results = [
        _run_config(name, weights, examples, force_answer=args.force_answer)
        for name, weights in configs.items()
    ]
    _print_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.time(),
        "force_answer": args.force_answer,
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
