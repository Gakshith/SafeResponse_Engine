import json
from pathlib import Path
from types import SimpleNamespace

from saferesponse_engine.components.generation_layer import GenerationLayer
from saferesponse_engine.components.trace_collection_layer import TraceCollectionLayer
from saferesponse_engine.entity.config_entity import GenerationConfig, TraceCollectionConfig


def _generation_config(tmp_path: Path) -> GenerationConfig:
    return GenerationConfig(
        root_dir=tmp_path,
        retrieval_artifact_path=tmp_path / "retrieved_chunks.json",
        generation_output_path=tmp_path / "candidates.json",
        model_name="test-model",
        finetuned_model_path=None,
        num_candidates=1,
        primary_temperature=0.0,
        sample_temperature=0.7,
        max_new_tokens=32,
        max_context_length=128,
    )


def _trace_config(tmp_path: Path) -> TraceCollectionConfig:
    return TraceCollectionConfig(
        root_dir=tmp_path,
        generation_artifact_path=tmp_path / "candidates.json",
        trace_output_path=tmp_path / "traces.json",
        hidden_states_dir=tmp_path / "hidden",
        model_name="test-model",
        max_context_length=128,
        collect_hidden_states=False,
        num_hidden_layers_to_save=-1,
        finetuned_model_path=None,
    )


def test_generation_reads_memory_context_from_retrieval_artifact(tmp_path, monkeypatch):
    config = _generation_config(tmp_path)
    config.retrieval_artifact_path.write_text(
        json.dumps(
            {
                "query": "What did I ask before?",
                "chunks": [
                    {
                        "source": "Memory",
                        "content": "Static context.",
                    }
                ],
                "memory_context": "Turn 2 assistant: Abraham Lincoln was discussed.",
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_ensure_model(self):
        self.tokenizer = SimpleNamespace(chat_template=None)

    def fake_generate_single(self, prompt, temperature, response_id, is_primary):
        captured["prompt"] = prompt
        return {
            "response_id": response_id,
            "text": "memory-aware answer",
            "is_primary": is_primary,
            "temperature": temperature,
            "num_tokens": 3,
            "requires_model_trace": False,
        }

    monkeypatch.setattr(GenerationLayer, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr(GenerationLayer, "_generate_single", fake_generate_single)

    output = GenerationLayer(config).generate()

    assert output["memory_context"] == "Turn 2 assistant: Abraham Lincoln was discussed."
    assert "Conversation memory" in captured["prompt"]
    assert "Abraham Lincoln was discussed" in captured["prompt"]


def test_summary_query_uses_extractive_candidate_without_model(tmp_path, monkeypatch):
    config = _generation_config(tmp_path)
    config.retrieval_artifact_path.write_text(
        json.dumps(
            {
                "query": "Give me a short summary about Abraham Lincoln.",
                "chunks": [
                    {
                        "source": "Abraham Lincoln",
                        "content": (
                            "Abraham Lincoln was an American lawyer and statesman "
                            "who served as the 16th president of the United States. "
                            "He preserved the Union."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fail_if_model_loads(self):
        raise AssertionError("summary extraction should not load the model")

    monkeypatch.setattr(GenerationLayer, "_ensure_model", fail_if_model_loads)

    output = GenerationLayer(config).generate()

    assert output["runtime_model_source"] == "extractive_summary"
    assert output["candidates"][0]["requires_model_trace"] is False
    assert output["candidates"][0]["text"].startswith("Abraham Lincoln was")


def test_trace_reads_memory_context_from_generation_artifact(tmp_path, monkeypatch):
    config = _trace_config(tmp_path)
    config.generation_artifact_path.write_text(
        json.dumps(
            {
                "query": "What did I ask before?",
                "context": "Static context.",
                "memory_context": "Turn 2 assistant: Abraham Lincoln was discussed.",
                "candidates": [
                    {
                        "response_id": 0,
                        "text": "memory-aware answer",
                        "is_primary": True,
                        "temperature": 0.0,
                        "requires_model_trace": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_ensure_model(self):
        self.tokenizer = SimpleNamespace(chat_template=None)

    def fake_collect_trace(self, prompt, candidate):
        captured["prompt"] = prompt
        return {
            "response_id": candidate["response_id"],
            "text": candidate["text"],
            "tokens": [],
            "logprobs": [],
        }

    monkeypatch.setattr(TraceCollectionLayer, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr(TraceCollectionLayer, "_collect_trace", fake_collect_trace)

    TraceCollectionLayer(config).collect()

    assert "Conversation memory" in captured["prompt"]
    assert "Abraham Lincoln was discussed" in captured["prompt"]
