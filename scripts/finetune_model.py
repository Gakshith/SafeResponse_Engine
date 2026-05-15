from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SYSTEM_PROMPT = (
    "You are a factual assistant. Answer only from the provided document context "
    "and conversation memory. If both are empty, unrelated, or do not contain "
    "the answer, reply exactly: \"I don't know based on the provided context.\" "
    "Keep the answer concise and do not invent follow-up questions."
)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _load_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("training", {})


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _record_question(record: dict[str, Any]) -> str:
    question = record.get("question") or record.get("instruction")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each fine-tuning record needs a non-empty question or instruction.")
    return question.strip()


def _record_answer(record: dict[str, Any]) -> str:
    answer = record.get("answer") or record.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Each fine-tuning record needs a non-empty answer or response.")
    return answer.strip()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            _record_question(record)
            _record_answer(record)
            records.append(record)
    if not records:
        raise ValueError(f"No fine-tuning records found in {path}.")
    return records


def build_prompt(
    tokenizer,
    question: str,
    context: str,
    memory_context: str = "",
) -> str:
    memory_block = (
        f"Conversation memory:\n{memory_context.strip()}\n\n"
        if memory_context.strip()
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"{memory_block}"
                f"Document context:\n{context.strip()}\n\n"
                f"Question: {question.strip()}\n\n"
                "Answer in 100 words or fewer."
            ),
        },
    ]
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"User: {memory_block}"
        f"Document context:\n{context.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer in 100 words or fewer.\nAssistant:"
    )


class SupervisedChatDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_seq_length: int):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        for record in records:
            prompt = build_prompt(
                tokenizer=tokenizer,
                question=_record_question(record),
                context=str(record.get("context", "")),
                memory_context=str(record.get("memory_context", "")),
            )
            answer = _record_answer(record)
            full_text = f"{prompt}{answer}{tokenizer.eos_token or ''}"

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )["input_ids"]
            tokenized = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = tokenized["input_ids"]
            labels = list(input_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            if all(label == -100 for label in labels):
                raise ValueError(
                    "A training example was truncated before the answer. "
                    "Increase training.max_seq_length."
                )
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def make_collator(tokenizer):
    pad_token_id = tokenizer.pad_token_id

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }

    return collate


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config/config.yaml")
    config_args, _ = base_parser.parse_known_args()
    config_path = _resolve_path(config_args.config)
    training_config = _load_training_config(config_path)

    parser = argparse.ArgumentParser(
        description="Fine-tune the SafeResponse generation model with LoRA."
    )
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument(
        "--base-model",
        default=training_config.get("base_model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument(
        "--train-file",
        default=training_config.get("train_file", "data/squad_v2/finetune_train.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=training_config.get("output_dir", "models/saferesponse-qwen-lora"),
    )
    parser.add_argument(
        "--registry-path",
        default=training_config.get("registry_path", "model_registry/registry.json"),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional human-readable name for this fine-tuning run.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Use only the first N converted records. Use 0 for the full training file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap optimizer steps for smoke runs. Use 0 for epoch-based training.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=int(training_config.get("max_seq_length", 768)),
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=float(training_config.get("num_train_epochs", 1)),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(training_config.get("per_device_train_batch_size", 1)),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(training_config.get("gradient_accumulation_steps", 4)),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(training_config.get("learning_rate", 2e-4)),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=float(training_config.get("warmup_ratio", 0.03)),
    )
    parser.add_argument("--lora-r", type=int, default=int(training_config.get("lora_r", 8)))
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=int(training_config.get("lora_alpha", 16)),
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=float(training_config.get("lora_dropout", 0.05)),
    )
    parser.add_argument(
        "--allow-model-downloads",
        action="store_true",
        help="Allow first-run Hugging Face model downloads for training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview the training records without loading the model.",
    )
    return parser.parse_args()


def register_training_run(registry_path: Path, metadata: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"models": []}
    registry.setdefault("models", []).append(metadata)
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> None:
    if args.allow_model_downloads:
        os.environ["SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS"] = "1"
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_file = _resolve_path(args.train_file)
    output_dir = _resolve_path(args.output_dir)
    registry_path = _resolve_path(args.registry_path)
    records = load_records(train_file)
    if args.max_records > 0:
        records = records[: args.max_records]

    local_files_only = not _truthy_env("SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SupervisedChatDataset(
        records=records,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collator(tokenizer),
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "run_name": args.run_name or output_dir.name,
        "base_model": args.base_model,
        "train_file": _display_path(train_file),
        "num_records": len(records),
        "output_dir": _display_path(output_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter_type": "lora",
        "max_steps": args.max_steps if args.max_steps > 0 else None,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
    }
    (output_dir / "saferesponse_training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    register_training_run(registry_path=registry_path, metadata=metadata)
    print(f"Saved LoRA adapter to {output_dir}")
    print(f"Registered training run in {registry_path}")
    print(
        "Set generation_layer.finetuned_model_path and "
        "trace_collection_layer.finetuned_model_path to "
        f"{_display_path(output_dir)} in config/config.yaml."
    )


def main() -> None:
    args = parse_args()
    train_file = _resolve_path(args.train_file)
    records = load_records(train_file)
    if args.max_records > 0:
        records = records[: args.max_records]
    if args.dry_run:
        first = records[0]
        print(f"Loaded {len(records)} fine-tuning records from {train_file}")
        print("\nPrompt preview:\n")
        print(
            build_prompt(
                tokenizer=None,
                question=_record_question(first),
                context=str(first.get("context", "")),
                memory_context=str(first.get("memory_context", "")),
            )
        )
        print("\nExpected answer:\n")
        print(_record_answer(first))
        return
    train(args)


if __name__ == "__main__":
    main()
