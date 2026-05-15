import json
from pathlib import Path

from scripts.run_evaluation import _evaluate_response


def test_evaluate_response_checks_decision_warnings_and_answer_text():
    example = {
        "expected_decision": "REJECT",
        "expected_warnings_any": ["unsupported"],
        "expected_answer_not_contains": ["Abraham Lincoln"],
    }
    response = {
        "decision": "REJECT",
        "answer": "I cannot provide a reliable answer.",
        "warnings": ["unsupported_claim"],
    }

    passed, failures = _evaluate_response(example, response)

    assert passed
    assert failures == []


def test_evaluate_response_reports_answer_leak():
    example = {
        "expected_decision": "REJECT",
        "expected_answer_not_contains": ["Abraham Lincoln"],
    }
    response = {
        "decision": "REJECT",
        "answer": "Abraham Lincoln",
        "warnings": [],
    }

    passed, failures = _evaluate_response(example, response)

    assert not passed
    assert "unexpectedly contained" in failures[0]


def test_evaluation_examples_have_unique_ids_and_live_fact_coverage():
    examples = json.loads(Path("evaluation/examples.json").read_text(encoding="utf-8"))
    ids = [example["id"] for example in examples]
    categories = {example.get("category") for example in examples}

    assert len(ids) == len(set(ids))
    assert "unsupported_live_fact" in categories
    assert any("2026" in example["query"] for example in examples)
    assert any(example["expected_decision"] == "ACCEPT" for example in examples)
