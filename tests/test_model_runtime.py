from saferesponse_engine.components.model_runtime import (
    is_peft_adapter_path,
    resolve_model_source,
    select_runtime,
)


def test_resolve_model_source_uses_base_model_when_no_finetuned_path():
    assert resolve_model_source("base-model", None) == "base-model"
    assert resolve_model_source("base-model", "") == "base-model"


def test_resolve_model_source_uses_finetuned_path_when_configured():
    assert (
        resolve_model_source("base-model", "models/saferesponse-qwen-lora")
        == "models/saferesponse-qwen-lora"
    )


def test_is_peft_adapter_path_detects_adapter_config(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    assert not is_peft_adapter_path(str(adapter_dir))

    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert is_peft_adapter_path(str(adapter_dir))


def test_select_runtime_honors_cpu_dtype_override(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setenv("SAFE_RESPONSE_CPU_DTYPE", "float16")

    device, dtype = select_runtime()

    assert device == "cpu"
    assert str(dtype) == "torch.float16"
