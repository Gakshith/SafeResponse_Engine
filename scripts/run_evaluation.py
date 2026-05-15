import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from saferesponse_engine.components.chat_engine import SafeResponseChatEngine


def _load_examples(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matches_expected(decision: str, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return decision in expected
    return decision == expected


def _contains_any(values: list[str], expected: list[str]) -> bool:
    normalized_values = [value.lower() for value in values]
    return any(
        expected_value.lower() in value
        for expected_value in expected
        for value in normalized_values
    )


def _answer_text(response: dict[str, Any]) -> str:
    answer = response.get("answer") or response.get("formatted_response") or ""
    return str(answer)


def _evaluate_response(
    example: dict[str, Any],
    response: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures = []
    decision = str(response.get("decision"))
    if not _matches_expected(decision, example["expected_decision"]):
        failures.append(
            f"expected decision {example['expected_decision']}, got {decision}"
        )

    warnings = [str(warning) for warning in response.get("warnings", [])]
    expected_warnings = example.get("expected_warnings_any") or []
    if expected_warnings and not _contains_any(warnings, expected_warnings):
        failures.append(f"expected one warning in {expected_warnings}, got {warnings}")

    answer = _answer_text(response).lower()
    for expected_text in example.get("expected_answer_contains", []):
        if expected_text.lower() not in answer:
            failures.append(f"answer did not contain {expected_text!r}")
    for blocked_text in example.get("expected_answer_not_contains", []):
        if blocked_text.lower() in answer:
            failures.append(f"answer unexpectedly contained {blocked_text!r}")

    return not failures, failures


def run_evaluation(
    examples_path: Path,
    include_model_runs: bool,
) -> dict[str, Any]:
    engine = SafeResponseChatEngine()
    examples = _load_examples(examples_path)
    results = []

    for example in examples:
        run_pipeline = bool(example.get("run_pipeline", False))
        if run_pipeline and not include_model_runs:
            results.append({
                "id": example["id"],
                "category": example.get("category"),
                "status": "skipped",
                "reason": "model run skipped because --skip-model-runs was used",
                "expected_decision": example["expected_decision"],
            })
            continue

        response = engine.query(
            query=example["query"],
            run_pipeline=run_pipeline,
        )
        decision = str(response.get("decision"))
        passed, failures = _evaluate_response(example, response)
        results.append({
            "id": example["id"],
            "category": example.get("category"),
            "status": "passed" if passed else "failed",
            "failures": failures,
            "query": example["query"],
            "expected_decision": example["expected_decision"],
            "actual_decision": decision,
            "confidence": response.get("confidence"),
            "risk_score": response.get("risk_score"),
            "source": response.get("source"),
            "warnings": response.get("warnings", []),
        })

    passed_count = sum(1 for result in results if result["status"] == "passed")
    failed_count = sum(1 for result in results if result["status"] == "failed")
    skipped_count = sum(1 for result in results if result["status"] == "skipped")
    evaluated_count = passed_count + failed_count
    categories: dict[str, dict[str, int]] = {}
    for result in results:
        category = str(result.get("category") or "uncategorized")
        bucket = categories.setdefault(
            category,
            {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
        )
        bucket["total"] += 1
        bucket[str(result["status"])] += 1
    return {
        "examples_path": str(examples_path),
        "include_model_runs": include_model_runs,
        "total": len(results),
        "evaluated": evaluated_count,
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "coverage_rate": round(evaluated_count / len(results), 6) if results else 0.0,
        "pass_rate": round(passed_count / evaluated_count, 6) if evaluated_count else 0.0,
        "categories": categories,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SafeResponse evaluation examples.",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("evaluation/examples.json"),
        help="Path to evaluation examples JSON.",
    )
    parser.add_argument(
        "--include-model-runs",
        action="store_true",
        help="Run examples that require the full model pipeline. Full runs are now the default.",
    )
    parser.add_argument(
        "--skip-model-runs",
        action="store_true",
        help="Skip examples that require model inference for a fast smoke check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON evaluation summary.",
    )
    args = parser.parse_args()

    include_model_runs = not args.skip_model_runs or args.include_model_runs
    summary = run_evaluation(
        examples_path=args.examples,
        include_model_runs=include_model_runs,
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
