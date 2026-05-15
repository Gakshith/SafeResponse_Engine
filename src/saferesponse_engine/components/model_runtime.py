import os
from pathlib import Path

if os.getenv("SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS", "0").strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from saferesponse_engine import logger


_TOKENIZER_CACHE = {}
_MODEL_CACHE = {}


def _normalized_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    normalized = str(path).strip()
    return normalized or None


def resolve_model_source(model_name: str, finetuned_model_path: str | None = None) -> str:
    return _normalized_optional_path(finetuned_model_path) or model_name


def is_peft_adapter_path(model_path: str | None) -> bool:
    model_path = _normalized_optional_path(model_path)
    if model_path is None:
        return False
    return (Path(model_path) / "adapter_config.json").is_file()


def _local_files_only() -> bool:
    raw = os.getenv("SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS", "0").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def _configure_model_io() -> bool:
    local_files_only = _local_files_only()
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return local_files_only


def _model_load_error(model_name: str, exc: Exception) -> RuntimeError:
    del exc
    return RuntimeError(
        f"Model '{model_name}' is not available locally. Pre-download the model "
        "into the Hugging Face cache or set SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 "
        "to permit first-run downloads."
    )


def _load_peft_adapter(base_model, adapter_path: str):
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuned LoRA adapters require the 'peft' package. "
            "Run `venv/bin/python -m pip install -r requirements.txt` first."
        ) from exc
    return PeftModel.from_pretrained(base_model, adapter_path)


def _dtype_from_env(device: str, default: torch.dtype) -> torch.dtype:
    raw = (
        os.getenv("SAFE_RESPONSE_MODEL_DTYPE")
        or (os.getenv("SAFE_RESPONSE_CPU_DTYPE") if device == "cpu" else None)
        or ""
    )
    requested = raw.strip().lower()
    if not requested or requested == "auto":
        return default

    dtype_by_name = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if requested not in dtype_by_name:
        logger.warning(
            "[Runtime] Ignoring unsupported dtype override %r; using %s",
            raw,
            default,
        )
        return default
    return dtype_by_name[requested]


def select_runtime() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.is_bf16_supported():
            return device, _dtype_from_env(device, torch.bfloat16)
        return device, _dtype_from_env(device, torch.float16)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        return device, _dtype_from_env(device, torch.float16)
    device = "cpu"
    return device, _dtype_from_env(device, torch.float32)


def get_tokenizer(model_name: str, finetuned_model_path: str | None = None):
    model_source = resolve_model_source(model_name, finetuned_model_path)
    cache_key = (model_name, model_source)
    if cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    logger.info("[Runtime] Loading tokenizer: %s", model_source)
    local_files_only = _configure_model_io()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        if not is_peft_adapter_path(finetuned_model_path):
            raise _model_load_error(model_source, exc) from exc
        logger.info(
            "[Runtime] Adapter tokenizer not found at %s; using base tokenizer %s",
            finetuned_model_path,
            model_name,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except OSError as base_exc:
            raise _model_load_error(model_name, base_exc) from base_exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def get_causal_lm(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    finetuned_model_path: str | None = None,
):
    model_source = resolve_model_source(model_name, finetuned_model_path)
    cache_key = (model_name, model_source, device, str(dtype))
    if cache_key in _MODEL_CACHE:
        logger.info("[Runtime] Loading model from cache: %s", model_source)
        return _MODEL_CACHE[cache_key]

    logger.info(
        "[Runtime] Loading model: %s | device=%s | dtype=%s",
        model_source,
        device,
        dtype,
    )
    local_files_only = _configure_model_io()
    try:
        base_source = model_name if is_peft_adapter_path(finetuned_model_path) else model_source
        model = AutoModelForCausalLM.from_pretrained(
            base_source,
            dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise _model_load_error(base_source, exc) from exc
    if is_peft_adapter_path(finetuned_model_path):
        logger.info("[Runtime] Applying LoRA adapter: %s", finetuned_model_path)
        model = _load_peft_adapter(model, str(finetuned_model_path))
    model.to(device)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model
