import json
from pathlib import Path

from saferesponse_engine.components.verification_layer import VerificationLayer
from saferesponse_engine.entity.config_entity import VerificationConfig


def _config(tmp_path: Path) -> VerificationConfig:
    return VerificationConfig(
        root_dir=tmp_path / "verification",
        retrieval_artifact_path=tmp_path / "retrieval.json",
        generation_artifact_path=tmp_path / "generation.json",
        trace_artifact_path=tmp_path / "traces.json",
        verification_output_path=tmp_path / "verification" / "scores.json",
        embedding_model="BAAI/bge-m3",
        embedding_backend="lexical",
        enable_halluguard=False,
        enable_ntk=False,
        enable_jacobian_instability=False,
        enable_spectral_conditioning=False,
        enable_grounding_score=True,
        enable_consistency_score=True,
        enable_nli_consistency=False,
        enable_judge=False,
        trace_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        nli_model_name="cross-encoder/nli-deberta-v3-small",
        judge_model="gpt-4o-mini",
        halluguard_threshold=0.45,
        grounding_threshold=0.70,
        consistency_threshold=0.70,
    )


def _write_inputs(config: VerificationConfig, query: str, candidate_text: str) -> None:
    chunk = {
        "content": (
            "Abraham Lincoln was an American lawyer and statesman who served "
            "as the 16th president of the United States from 1861 until his "
            "assassination in 1865."
        ),
        "source": "Abraham Lincoln",
        "chunk_id": 0,
        "content_hash": "lincoln",
    }
    config.retrieval_artifact_path.write_text(
        json.dumps({"query": query, "chunks": [chunk]}),
        encoding="utf-8",
    )
    config.generation_artifact_path.write_text(
        json.dumps({
            "query": query,
            "context": chunk["content"],
            "candidates": [
                {
                    "response_id": 0,
                    "text": candidate_text,
                    "is_primary": True,
                    "temperature": 0.0,
                }
            ],
        }),
        encoding="utf-8",
    )
    config.trace_artifact_path.write_text(
        json.dumps({
            "traces": [
                {
                    "response_id": 0,
                    "mean_logprob": -0.1,
                    "min_logprob": -0.2,
                    "logprobs": [-0.1, -0.2],
                }
            ]
        }),
        encoding="utf-8",
    )


def test_verification_rejects_unsupported_generated_claim(tmp_path):
    config = _config(tmp_path)
    _write_inputs(
        config,
        query="What company did Abraham Lincoln found?",
        candidate_text="Abraham Lincoln founded the American Civil Liberties Union in 1884.",
    )

    output = VerificationLayer(config).verify()

    candidate = output["candidates"][0]
    assert candidate["grounding_score"] < config.grounding_threshold
    assert candidate["risk_signals"]["weak_grounding"] is True
    assert candidate["risk_signals"]["unsupported_claim"] is True
    assert "liberty" in candidate["grounding_features"]["missing_answer_terms"]


def test_verification_rejects_live_fact_query_even_when_terms_overlap(tmp_path):
    config = _config(tmp_path)
    _write_inputs(
        config,
        query="Who is the president of the United States?",
        candidate_text="The president of the United States is Abraham Lincoln.",
    )

    output = VerificationLayer(config).verify()

    candidate = output["candidates"][0]
    assert candidate["risk_signals"]["weak_grounding"] is True
    assert candidate["risk_signals"]["unsupported_live_fact"] is True
