# Fine-Tuning Runbook

## Dataset

The current training file is:

```text
data/squad_v2/finetune_train.jsonl
```

It is generated from SQuAD v2 with a balanced mix of answerable and
unanswerable examples:

```bash
venv/bin/python scripts/prepare_squad_v2.py --update-config
```

Use SQuAD v2 for grounded QA behavior. Add project-specific records when the
application needs domain-specific tone, vocabulary, refusal style, or supported
facts.

## Record Format

Each JSONL record should have:

```json
{
  "question": "Who tutored Alexander the Great?",
  "context": "Aristotle later tutored Alexander the Great.",
  "answer": "Aristotle tutored Alexander the Great."
}
```

For unsupported examples, use exactly:

```text
I don't know based on the provided context.
```

## Smoke Run

Run a one-step local smoke training pass before a full job:

```bash
SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 \
venv/bin/python scripts/finetune_model.py \
  --run-name smoke \
  --max-records 16 \
  --max-steps 1 \
  --output-dir models/saferesponse-qwen-lora-smoke
```

This verifies PEFT, tokenizer/model loading, adapter saving, and model registry
writing. It is not a quality-improving fine-tune.

## Full Run

Use GPU hardware for the full 20,000-record run:

```bash
SAFE_RESPONSE_ALLOW_MODEL_DOWNLOADS=1 \
venv/bin/python scripts/finetune_model.py \
  --run-name squad-v2-grounded-qa-v1
```

The adapter is saved to:

```text
models/saferesponse-qwen-lora
```

The run metadata is written to:

```text
models/saferesponse-qwen-lora/saferesponse_training_metadata.json
model_registry/registry.json
```

## Activation

After a trained adapter passes evaluation, set:

```yaml
generation_layer:
  finetuned_model_path: models/saferesponse-qwen-lora

trace_collection_layer:
  finetuned_model_path: models/saferesponse-qwen-lora
```

Restart the API server after changing the adapter path.

## Promotion Gate

Do not activate an adapter unless these pass:

```bash
venv/bin/python -m pytest -q
venv/bin/python scripts/run_evaluation.py \
  --output artifacts/evaluation/latest.json
```

Compare the new run against the base model on supported answers, unsupported
claims, current/live facts, and entity-overlap traps.
