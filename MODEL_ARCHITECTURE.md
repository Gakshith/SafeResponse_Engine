# SafeResponse Model Architecture

This project does not train a new base LLM. The model architecture is a
middleware/orchestration model that controls when to use retrieval, memory, and
verification before returning an answer.

## Runtime Modes

```text
Small talk
-> direct safe response
-> no model inference

Single factual question
-> 10-article document retrieval
-> candidate generation
-> trace collection
-> verification
-> fusion router
-> final safe response

Long chat / multi-turn question
-> conversation memory retrieval
-> 10-article document retrieval
-> candidate generation with memory + document context
-> verification
-> fusion router
-> final safe response
```

## Core Component

The main backend model is:

```text
src/saferesponse_engine/components/chat_engine.py
```

`SafeResponseChatEngine` owns:

1. Intent routing
2. Conversation mode detection
3. Conversation memory lookup
4. SafeResponse pipeline execution
5. Final API response formatting

FastAPI should stay thin. It should call the engine and return JSON.

## Why This Works Better

Running the full LLM pipeline for `hi` is wasteful and slow. The chat engine now
routes greetings and help messages directly.

For real questions, the engine uses the existing safety pipeline. For long chats,
it injects:

```text
Conversation summary
+ relevant original prior turns
+ current user question
+ retrieved document chunks
```

This avoids sending the full chat history while still grounding the response in
the conversation.

## Controlled Corpus

The current project intentionally uses only 10 Wikipedia articles. That is a
controlled retrieval corpus, not model training data.

The expected behavior is:

```text
Question supported by the 10 articles -> answer.
Question outside the 10 articles -> reject safely.
```

This makes hallucination prevention visible during demos.

## Production Upgrade Path

For a stronger version:

1. Replace first-10-article indexing with curated article indexing.
2. Add query-based Wikipedia page fetching.
3. Add persistent vector memory for conversation turns.
4. Add contradiction scoring between new answers and old chat facts.
5. Move long-running pipeline jobs to a worker queue.
