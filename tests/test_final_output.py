import json
from pathlib import Path

import pytest

from saferesponse_engine.components.final_output import FinalOutputLayer
from saferesponse_engine.entity.config_entity import FinalOutputConfig


def _config(tmp_path: Path, **overrides) -> FinalOutputConfig:
    values = {
        "root_dir": tmp_path / "final",
        "fusion_artifact_path": tmp_path / "fusion.json",
        "verification_artifact_path": tmp_path / "verification.json",
        "final_output_path": tmp_path / "final" / "final_response.json",
        "high_confidence_threshold": 0.20,
        "medium_confidence_threshold": 0.40,
        "low_confidence_threshold": 0.75,
        "include_risk_explanation": True,
        "include_pipeline_summary": True,
        "include_formatted_response": True,
        "max_answer_length": 512,
        "max_answer_words": 100,
    }
    values.update(overrides)
    return FinalOutputConfig(**values)


def _write_verification(path: Path) -> None:
    path.write_text(
        json.dumps({
            "verification_model": {
                "embedding_model": "BAAI/bge-m3",
                "embedding_backend": "lexical",
                "trace_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "nli_enabled": False,
                "judge_enabled": False,
                "halluguard_modules": {
                    "ntk": False,
                    "jacobian_instability": False,
                    "spectral_conditioning": False,
                },
            }
        }),
        encoding="utf-8",
    )


def test_accept_output_contains_answer_source_and_confidence(tmp_path):
    config = _config(tmp_path)
    config.fusion_artifact_path.write_text(
        json.dumps({
            "query": "Who was Abraham Lincoln?",
            "decision": "ACCEPT",
            "decision_reason": "low risk",
            "rewrite_attempt": 0,
            "fusion_scores": [
                {
                    "effective_weights": {
                        "halluguard": 0.4,
                        "grounding": 0.4,
                        "consistency": 0.2,
                    }
                }
            ],
            "selected_candidate": {
                "text": "Abraham Lincoln was the 16th president of the United States.",
                "combined_risk": 0.15,
                "risk_signals": {},
                "supporting_source": {
                    "source": "Abraham Lincoln",
                    "chunk_id": 12,
                    "content_hash": "abc",
                },
            },
        }),
        encoding="utf-8",
    )
    _write_verification(config.verification_artifact_path)

    output = FinalOutputLayer(config).generate()

    final = output["final_response"]
    assert output["decision"] == "ACCEPT"
    assert final["confidence_tag"] == "HIGH"
    assert final["source_citation"]["title"] == "Abraham Lincoln"
    assert "Hallucination risk score" in final["formatted_response"]


def test_reject_output_uses_safe_refusal(tmp_path):
    config = _config(tmp_path)
    config.fusion_artifact_path.write_text(
        json.dumps({
            "query": "Unsupported question",
            "decision": "REJECT",
            "decision_reason": "unsupported",
            "rewrite_attempt": 0,
            "fusion_scores": [],
            "selected_candidate": {
                "text": "unsupported answer",
                "combined_risk": 0.10,
                "risk_signals": {"weak_grounding": True},
                "supporting_source": {},
            },
        }),
        encoding="utf-8",
    )
    _write_verification(config.verification_artifact_path)

    output = FinalOutputLayer(config).generate()

    final = output["final_response"]
    assert final["confidence_tag"] == "NONE"
    assert final["confidence_score"] == 1.0
    assert "cannot provide a reliable answer" in final["answer"]
    assert final["source_citation"] is None


def test_answer_trimming_respects_word_limit(tmp_path):
    config = _config(tmp_path, max_answer_words=5)
    long_answer = "one two three four five six seven"
    layer = FinalOutputLayer(config)

    assert layer._trim_answer(long_answer) == "one two three four five..."


@pytest.mark.parametrize(
    "overrides",
    [
        {"high_confidence_threshold": -0.1},
        {
            "high_confidence_threshold": 0.5,
            "medium_confidence_threshold": 0.4,
        },
        {"max_answer_length": 0},
        {"max_answer_words": 0},
    ],
)
def test_rejects_invalid_final_output_config(tmp_path, overrides):
    with pytest.raises(ValueError):
        FinalOutputLayer(_config(tmp_path, **overrides))
