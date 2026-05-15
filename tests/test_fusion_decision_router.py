import json
from pathlib import Path

import pytest

from saferesponse_engine.components.fusion_decision_router import (
    FusionDecisionRouter,
)
from saferesponse_engine.entity.config_entity import FusionRouterConfig


def _config(tmp_path: Path, **overrides) -> FusionRouterConfig:
    values = {
        "root_dir": tmp_path / "fusion",
        "verification_artifact_path": tmp_path / "verification.json",
        "traces_artifact_path": tmp_path / "traces.json",
        "fusion_output_path": tmp_path / "fusion" / "fusion_result.json",
        "weight_halluguard": 0.35,
        "weight_grounding": 0.30,
        "weight_consistency": 0.20,
        "weight_judge": 0.15,
        "accept_threshold": 0.30,
        "rewrite_threshold": 0.50,
        "reject_threshold": 0.75,
        "max_rewrite_attempts": 2,
    }
    values.update(overrides)
    return FusionRouterConfig(**values)


def _candidate(
    response_id: int,
    *,
    halluguard_score=0.1,
    grounding_score=0.9,
    consistency_score=1.0,
    judge_score=None,
    is_primary=None,
    weak_grounding=False,
    unsupported_claim=False,
    unsupported_live_fact=False,
    answer_abstained=False,
    source="Demo Source",
):
    return {
        "response_id": response_id,
        "text": f"candidate {response_id}",
        "is_primary": response_id == 0 if is_primary is None else is_primary,
        "halluguard_score": halluguard_score,
        "grounding_score": grounding_score,
        "consistency_score": consistency_score,
        "judge_score": judge_score,
        "risk_signals": {
            "weak_grounding": weak_grounding,
            "high_sample_divergence": False,
            "hidden_state_instability": False,
            "low_mean_logprob": False,
            "low_min_logprob": False,
            "unsupported_claim": unsupported_claim,
            "unsupported_live_fact": unsupported_live_fact,
            "answer_abstained": answer_abstained,
        },
        "supporting_source": {
            "source": source,
            "chunk_id": response_id,
            "content_hash": f"hash-{response_id}",
        }
        if source
        else {},
    }


def _write_inputs(config: FusionRouterConfig, candidates: list[dict]) -> None:
    config.verification_artifact_path.write_text(
        json.dumps({"query": "demo query", "candidates": candidates}),
        encoding="utf-8",
    )
    config.traces_artifact_path.write_text(
        json.dumps({
            "traces": [
                {
                    "response_id": candidate["response_id"],
                    "sequence_score": -candidate["response_id"],
                }
                for candidate in candidates
            ]
        }),
        encoding="utf-8",
    )


def test_accepts_low_risk_supported_candidate(tmp_path):
    config = _config(tmp_path)
    _write_inputs(config, [_candidate(0)])

    result = FusionDecisionRouter(config).route()

    assert result["decision"] == "ACCEPT"
    assert result["selected_response_id"] == 0
    assert result["fusion_scores"][0]["combined_risk"] < config.accept_threshold
    assert "risk_contributions" in result["fusion_scores"][0]


def test_reranks_when_non_primary_candidate_is_safer(tmp_path):
    config = _config(tmp_path)
    _write_inputs(
        config,
        [
            _candidate(
                0,
                halluguard_score=0.7,
                grounding_score=0.65,
                consistency_score=0.8,
            ),
            _candidate(
                1,
                halluguard_score=0.55,
                grounding_score=0.65,
                consistency_score=0.9,
                is_primary=False,
            ),
        ],
    )

    result = FusionDecisionRouter(config).route()

    assert result["decision"] == "RERANK"
    assert result["selected_response_id"] == 1


def test_rewrite_when_best_candidate_is_medium_risk(tmp_path):
    config = _config(tmp_path)
    _write_inputs(
        config,
        [
            _candidate(
                0,
                halluguard_score=0.8,
                grounding_score=0.2,
                consistency_score=1.0,
            )
        ],
    )

    result = FusionDecisionRouter(config).route(rewrite_attempt=0)

    assert result["decision"] == "REWRITE"
    assert result["rewrite_query"] == "demo query\n\nFocus specifically on: Demo Source"


def test_rejects_when_grounding_hard_guard_fires(tmp_path):
    config = _config(tmp_path)
    _write_inputs(config, [_candidate(0, weak_grounding=True)])

    result = FusionDecisionRouter(config).route()

    assert result["decision"] == "REJECT"
    assert "hard grounding guard" in result["decision_reason"]


def test_rejects_when_unsupported_claim_hard_guard_fires(tmp_path):
    config = _config(tmp_path)
    _write_inputs(config, [_candidate(0, unsupported_claim=True)])

    result = FusionDecisionRouter(config).route()

    assert result["decision"] == "REJECT"
    assert "not supported by the retrieved sources" in result["decision_reason"]


def test_missing_judge_score_renormalizes_weights(tmp_path):
    config = _config(tmp_path)
    _write_inputs(config, [_candidate(0, judge_score=None)])

    result = FusionDecisionRouter(config).route()

    weights = result["fusion_scores"][0]["effective_weights"]
    assert "judge" not in weights
    assert round(sum(weights.values()), 6) == 1.0


def test_malformed_scores_do_not_crash_router(tmp_path):
    config = _config(tmp_path)
    _write_inputs(
        config,
        [
            _candidate(
                0,
                halluguard_score="bad",
                grounding_score="bad",
                consistency_score="bad",
            )
        ],
    )

    result = FusionDecisionRouter(config).route()

    assert result["decision"] in {"REWRITE", "REJECT"}
    assert result["fusion_scores"][0]["risk_terms"]["halluguard_risk"] == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_grounding": -0.1},
        {
            "weight_halluguard": 0.0,
            "weight_grounding": 0.0,
            "weight_consistency": 0.0,
            "weight_judge": 0.0,
        },
        {"accept_threshold": 0.6, "rewrite_threshold": 0.5},
        {"max_rewrite_attempts": -1},
    ],
)
def test_rejects_invalid_config(tmp_path, overrides):
    with pytest.raises(ValueError):
        FusionDecisionRouter(_config(tmp_path, **overrides))
