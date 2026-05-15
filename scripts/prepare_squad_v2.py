from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = "rajpurkar/squad_v2"
SPLITS = {
    "train": "squad_v2/train-00000-of-00001.parquet",
    "validation": "squad_v2/validation-00000-of-00001.parquet",
}
ABSTENTION_ANSWER = "I don't know based on the provided context."


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _answers_texts(answers: Any) -> list[str]:
    if answers is None:
        return []
    if isinstance(answers, dict):
        texts = answers.get("text", [])
    else:
        texts = getattr(answers, "get", lambda _key, _default=None: None)("text", [])
    if texts is None:
        return []
    return [str(text).strip() for text in texts if str(text).strip()]


def _row_to_record(row: Any) -> dict[str, Any]:
    answers = _answers_texts(row.answers)
    is_answerable = bool(answers)
    return {
        "question": str(row.question).strip(),
        "context": str(row.context).strip(),
        "answer": answers[0] if is_answerable else ABSTENTION_ANSWER,
        "source": "squad_v2",
        "title": str(row.title).strip(),
        "squad_id": str(row.id),
        "is_answerable": is_answerable,
    }


def _balanced_sample(
    records: list[dict[str, Any]],
    max_records: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_records <= 0 or len(records) <= max_records:
        return records

    rng = random.Random(seed)
    answerable = [record for record in records if record["is_answerable"]]
    unanswerable = [record for record in records if not record["is_answerable"]]
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)

    target_answerable = max_records // 2
    target_unanswerable = max_records - target_answerable
    selected = answerable[:target_answerable] + unanswerable[:target_unanswerable]

    if len(selected) < max_records:
        selected_ids = {record["squad_id"] for record in selected}
        remaining = [
            record
            for record in answerable[target_answerable:] + unanswerable[target_unanswerable:]
            if record["squad_id"] not in selected_ids
        ]
        selected.extend(remaining[: max_records - len(selected)])

    rng.shuffle(selected)
    return selected


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_split(split: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=SPLITS[split],
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )


def convert_split(
    split: str,
    raw_dir: Path,
    output_dir: Path,
    max_records: int,
    seed: int,
) -> tuple[Path, dict[str, int]]:
    parquet_path = download_split(split=split, raw_dir=raw_dir)
    frame = pd.read_parquet(parquet_path)
    records = [_row_to_record(row) for row in frame.itertuples(index=False)]
    records = _balanced_sample(records=records, max_records=max_records, seed=seed)
    output_path = output_dir / f"finetune_{split}.jsonl"
    _write_jsonl(output_path, records)
    summary = {
        "total": len(records),
        "answerable": sum(1 for record in records if record["is_answerable"]),
        "unanswerable": sum(1 for record in records if not record["is_answerable"]),
    }
    return output_path, summary


def update_training_config(config_path: Path, train_file: Path) -> None:
    relative_train_file = str(train_file.relative_to(ROOT))
    lines = config_path.read_text(encoding="utf-8").splitlines()
    in_training = False
    replaced = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "training:":
            in_training = True
            continue
        if in_training and line and not line.startswith((" ", "\t")):
            in_training = False
        if in_training and stripped.startswith("train_file:"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}train_file: {relative_train_file}"
            replaced = True
            break

    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["training:", f"  train_file: {relative_train_file}"])

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SQuAD v2 and convert it for SafeResponse LoRA fine-tuning."
    )
    parser.add_argument("--raw-dir", default="data/squad_v2/raw")
    parser.add_argument("--output-dir", default="data/squad_v2")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--max-train-records",
        type=int,
        default=20000,
        help="Use 0 for the full training split.",
    )
    parser.add_argument(
        "--max-validation-records",
        type=int,
        default=2000,
        help="Use 0 for the full validation split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Point training.train_file in config/config.yaml at the converted train file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = _resolve_path(args.raw_dir)
    output_dir = _resolve_path(args.output_dir)
    config_path = _resolve_path(args.config)

    train_path, train_summary = convert_split(
        split="train",
        raw_dir=raw_dir,
        output_dir=output_dir,
        max_records=args.max_train_records,
        seed=args.seed,
    )
    validation_path, validation_summary = convert_split(
        split="validation",
        raw_dir=raw_dir,
        output_dir=output_dir,
        max_records=args.max_validation_records,
        seed=args.seed,
    )
    if args.update_config:
        update_training_config(config_path=config_path, train_file=train_path)

    print("SQuAD v2 prepared for SafeResponse fine-tuning")
    print(f"Train: {train_path} {train_summary}")
    print(f"Validation: {validation_path} {validation_summary}")
    if args.update_config:
        print(f"Updated {config_path}: training.train_file -> {train_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
