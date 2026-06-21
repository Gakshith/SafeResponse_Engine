import inspect

from saferesponse_engine.components.generation_layer import GenerationLayer


def test_generation_uses_kv_cache_and_repetition_penalty():
    src = inspect.getsource(GenerationLayer._generate_single)
    assert "use_cache=False" not in src, "KV cache must be enabled for speed"
    assert "repetition_penalty" in src, "repetition_penalty should be set"
