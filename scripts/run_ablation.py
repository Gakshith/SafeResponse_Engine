"""Verification ablation harness for SafeResponse Engine.

Runs the evaluation set under several verification configurations — verification
fully OFF, fully ON, and each signal in isolation — and reports the safety
metrics (false-accept rate, false-reject rate, accuracy) for each. This is the
core research artifact: it shows whether the verification layer actually reduces
hallucinations, and which signals contribute.

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


def build_configs() -> dict[str, dict[str, bool]]:
    """Named verification flag-sets to compare.

    Each maps the three demo signals (logprob via halluguard, grounding,
    consistency). ``all_off`` is the no-verification baseline.
    """
    return {
        "all_off": {
            "enable_halluguard": False,
            "enable_grounding_score": False,
            "enable_consistency_score": False,
        },
        "logprob_only": {
            "enable_halluguard": True,
            "enable_grounding_score": False,
            "enable_consistency_score": False,
        },
        "grounding_only": {
            "enable_halluguard": False,
            "enable_grounding_score": True,
            "enable_consistency_score": False,
        },
        "consistency_only": {
            "enable_halluguard": False,
            "enable_grounding_score": False,
            "enable_consistency_score": True,
        },
        "all_on": {
            "enable_halluguard": True,
            "enable_grounding_score": True,
            "enable_consistency_score": True,
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
def patched_verification_flags(flags: dict[str, bool]):
    """Temporarily set verification_layer.enable_* flags in config.yaml."""
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(original_text)
        data["verification_layer"].update(flags)
        CONFIG_PATH.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        yield
    finally:
        CONFIG_PATH.write_text(original_text, encoding="utf-8")


def _run_config(name: str, flags: dict[str, bool], examples: list[dict[str, Any]]) -> dict[str, Any]:
    from saferesponse_engine.components.ablation_metrics import confusion
    from saferesponse_engine.components.chat_engine import SafeResponseChatEngine

    records: list[tuple[bool, bool]] = []
    per_example: list[dict[str, Any]] = []
    started = time.perf_counter()
    with patched_verification_flags(flags):
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
    return {"config": name, "flags": flags, "metrics": metrics, "examples": per_example}


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
    args = parser.parse_args()

    configs = build_configs()
    examples = json.loads(Path(args.examples).read_text(encoding="utf-8"))

    if args.skip_model_runs:
        print("Configs:", ", ".join(configs))
        n = sum(1 for ex in examples if classify_example(ex) is not None)
        print(f"{n} hallucination examples would run across {len(configs)} configs.")
        return

    results = [_run_config(name, flags, examples) for name, flags in configs.items()]
    _print_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": time.time(), "results": results}
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
