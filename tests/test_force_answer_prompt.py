from pathlib import Path

from saferesponse_engine.components.generation_layer import GenerationLayer
from saferesponse_engine.entity.config_entity import GenerationConfig


def _gen_config(tmp_path: Path, force_answer: bool) -> GenerationConfig:
    return GenerationConfig(
        root_dir=tmp_path,
        retrieval_artifact_path=tmp_path / "r.json",
        generation_output_path=tmp_path / "g.json",
        model_name="dummy",
        finetuned_model_path=None,
        num_candidates=1,
        primary_temperature=0.0,
        sample_temperature=0.7,
        max_new_tokens=48,
        max_context_length=1024,
        force_answer=force_answer,
    )


def test_force_answer_prompt_forbids_abstention(tmp_path):
    layer = GenerationLayer(_gen_config(tmp_path, force_answer=True))
    prompt = layer._build_prompt("Who founded X?", "some context").lower()
    assert "never say you do not know" in prompt
    assert "i don't know based on the provided context" not in prompt


def test_normal_prompt_allows_abstention(tmp_path):
    layer = GenerationLayer(_gen_config(tmp_path, force_answer=False))
    prompt = layer._build_prompt("Who founded X?", "some context").lower()
    assert "i don't know based on the provided context" in prompt
