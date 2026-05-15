# Demo Guide

This guide describes the fastest way to demo SafeResponse Engine locally.

## 1. Install

```bash
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

## 2. Run Tests

```bash
venv/bin/python -m pytest
```

## 3. Start The API And UI

```bash
venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## 4. Demo Queries

Small talk:

```text
hi
```

Expected behavior:

```text
DIRECT response without model inference.
```

Supported factual query:

```text
Give me a short summary about Abraham Lincoln.
```

Expected behavior:

```text
ACCEPT when the controlled corpus retrieves supporting Abraham Lincoln context.
```

Unsupported query:

```text
Who is the current CEO of a company that is not in the corpus?
```

Expected behavior:

```text
REJECT unless retrieved context supports the answer.
```

Misspelled query:

```text
Give me about the Abram Lilncoln.
```

Expected behavior:

```text
ACCEPT if retrieval still finds the Abraham Lincoln source.
```

## 5. What To Show

- The chat UI returns direct responses for small talk.
- The inspector shows route, confidence, source, risk, warnings, and explanation.
- The API exposes `/metrics` for request counts, latency, decisions, and risk
  score summaries.
- The evaluation runner stores expected decisions in `evaluation/examples.json`.

## 6. Important Demo Caveat

The default corpus is controlled and small. This makes rejection behavior easy to
show, but it also means many normal factual questions will be rejected because
the answer is not in the local retrieval corpus.
