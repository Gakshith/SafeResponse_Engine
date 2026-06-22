from pathlib import Path

from saferesponse_engine.components.fusion_decision_router import FusionDecisionRouter
from saferesponse_engine.entity.config_entity import FusionRouterConfig


def _cfg(tmp_path: Path, **overrides) -> FusionRouterConfig:
    values = dict(
        root_dir=tmp_path,
        verification_artifact_path=tmp_path / "v.json",
        traces_artifact_path=tmp_path / "t.json",
        fusion_output_path=tmp_path / "f.json",
        weight_halluguard=0.35,
        weight_grounding=0.30,
        weight_consistency=0.20,
        weight_judge=0.15,
        accept_threshold=0.30,
        rewrite_threshold=0.50,
        reject_threshold=0.75,
        max_rewrite_attempts=2,
    )
    values.update(overrides)
    return FusionRouterConfig(**values)


def _best(combined_risk, **signals):
    risk_signals = {
        "weak_grounding": False,
        "unsupported_claim": False,
        "unsupported_live_fact": False,
        "answer_abstained": False,
    }
    risk_signals.update(signals)
    return {
        "response_id": 0,
        "is_primary": True,
        "combined_risk": combined_risk,
        "risk_signals": risk_signals,
        "supporting_source": {"source": "S"},
    }


def test_grounding_hard_reject_fires_when_grounding_weighted(tmp_path):
    router = FusionDecisionRouter(_cfg(tmp_path))  # grounding weight 0.30
    decision, _ = router._route(_best(0.1, weak_grounding=True), 0)
    assert decision == "REJECT"


def test_grounding_hard_reject_suppressed_when_grounding_unweighted(tmp_path):
    # With grounding not used in fusion (weight 0), its hard-reject guard must not
    # fire — this lets the ablation isolate the other signals.
    router = FusionDecisionRouter(_cfg(tmp_path, weight_grounding=0.0))
    decision, _ = router._route(_best(0.1, weak_grounding=True, unsupported_claim=True), 0)
    assert decision == "ACCEPT"


def test_abstention_always_hard_rejects_regardless_of_weights(tmp_path):
    router = FusionDecisionRouter(_cfg(tmp_path, weight_grounding=0.0))
    decision, _ = router._route(_best(0.1, answer_abstained=True), 0)
    assert decision == "REJECT"
