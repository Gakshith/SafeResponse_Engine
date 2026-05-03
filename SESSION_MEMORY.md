# SafeResponse Engine Session Memory

## Project Goal

SafeResponse Engine is an LLM safety middleware pipeline intended to reduce hallucinations by combining retrieval grounding, multi-candidate generation, internal trace analysis, HalluGuard-style model-signal features, consistency checks, optional judge/classifier scoring, and a final decision router.

Target architecture:

```text
Stage 1: User Query
Stage 2: Retrieval + Context Layer
Stage 3: Generator Layer
Stage 4: Trace Collection Layer
Stage 5: Multi-Signal Verification Layer
Stage 6: Fusion + Decision Router
Stage 7: Final Output
Stage 8: Deployment + Evaluation
```

## Current Implemented Stages

### Stage 1: User Query

Implemented files:

```text
src/saferesponse_engine/components/user_query.py
src/saferesponse_engine/pipeline/stage_01_user_query.py
```

Behavior:

- Downloads or reuses the user query artifact.
- Converts GitHub `blob` URLs to raw GitHub URLs.
- Writes query text to `artifacts/user_query/data.txt`.

### Stage 2: Retrieval + Context Layer

Implemented files:

```text
src/saferesponse_engine/components/retrieval_layer.py
src/saferesponse_engine/pipeline/stage_02_retrieval_layer.py
```

Behavior:

- Uses `BAAI/bge-m3` embeddings through LangChain HuggingFace embeddings.
- Loads Wikipedia data from Hugging Face datasets.
- Chunks articles and builds/loads local FAISS index.
- Writes top-k retrieved chunks to `artifacts/retrieval/retrieved_chunks.json`.

Important current config:

```yaml
retrieval_layer:
  num_articles: 50000
```

Note:

- The existing checked artifact may still reflect the earlier small corpus.
- Rerun Stage 2 to rebuild the index with `num_articles: 50000`.
- This can be heavy and may require Hugging Face network/model access.

### Stage 3: Generator Layer

Implemented files:

```text
src/saferesponse_engine/components/generation_layer.py
src/saferesponse_engine/pipeline/stage_03_generation_layer.py
research/generation_layer.ipynb
```

Important fix made in this session:

- Stage 3 originally had prompt bleeding: responses continued into fake `Human:` prompts or unrelated instructions.
- Fixed by using `tokenizer.apply_chat_template(...)` when available.
- Added stop substring criteria and response cleanup for markers such as:
  - `Human:`
  - `User:`
  - `System:`
  - `Assistant:`
  - `Write a`

Current clean candidate examples:

```text
Alexander the Great was a king of Macedonia who ruled from 336 to 323 BCE.
Alexander the Great was a Greek king who ruled from 336 to 323 BCE.
Alexander the Great was the king of Macedonia who ruled from 336-323 BCE.
```

### Stage 4: Trace Collection Layer

Implemented files:

```text
src/saferesponse_engine/components/trace_collection_layer.py
src/saferesponse_engine/pipeline/stage_04_trace_collection_layer.py
research/trace_collection_layer.ipynb
```

Important fix made in this session:

- Stage 4 now rebuilds the same chat-template style prompt as Stage 3.
- This keeps trace scoring aligned with the generation prompt format.

Behavior:

- Collects candidate tokens.
- Collects token logprobs.
- Saves hidden states to `artifacts/traces/hidden_states/`.
- Writes trace metadata to `artifacts/traces/traces.json`.

### Stage 5: Multi-Signal Verification Layer

Implemented files:

```text
src/saferesponse_engine/components/verification_layer.py
src/saferesponse_engine/pipeline/stage_05_verification_layer.py
research/verification_layer.ipynb
```

Stage 5 is wired into:

```text
main.py
```

after Stage 4.

Behavior:

- Reads retrieval, generation, and trace artifacts.
- Writes `artifacts/verification/verification_scores.json`.

Implemented signals:

- HalluGuard logprob features:
  - uncertainty score
  - tail risk score
  - variance score
- Hidden-state statistics:
  - norm mean
  - norm std
  - step drift mean
- NTK gram matrix:
  - `H_norm @ H_norm.T`
  - off-diagonal mean/std
  - NTK score
- Spectral conditioning:
  - `torch.linalg.svdvals(...)`
  - `s_max`
  - `s_min`
  - condition number
  - spectral score
- Jacobian instability:
  - optional autograd input-embedding gradient
  - runs over full context plus candidate response
  - current verification model: `Qwen/Qwen2.5-0.5B-Instruct`
- Retrieval grounding:
  - candidate answer vs retrieved chunks
  - uses BGE-M3 when available
  - lexical fallback exists
- Consistency:
  - pairwise candidate similarity
  - optional NLI contradiction scoring
- Judge baseline:
  - optional OpenAI judge path
  - disabled by default

Important Stage 5 fixes made:

- Added `torch.load(..., weights_only=True)` for hidden-state tensors.
- Jacobian now uses:

```python
full_text = f"{context}\n\n{candidate_text}".strip()
```

instead of candidate text only.

- `halluguard_threshold` raised from `0.35` to `0.45`.
- `embedding_backend` changed from `lexical` to `huggingface`.
- `enable_jacobian_instability` changed to `true`.
- Jacobian model changed from 1.5B to 0.5B because the 1.5B autograd run was too heavy and was killed.

Current Stage 5 config:

```yaml
verification_layer:
  embedding_model: BAAI/bge-m3
  embedding_backend: huggingface
  enable_halluguard: true
  enable_ntk: true
  enable_jacobian_instability: true
  enable_spectral_conditioning: true
  enable_grounding_score: true
  enable_consistency_score: true
  enable_nli_consistency: false
  enable_judge: false
  trace_model_name: Qwen/Qwen2.5-0.5B-Instruct
  nli_model_name: cross-encoder/nli-deberta-v3-small
  judge_model: gpt-4o-mini
  halluguard_threshold: 0.45
  grounding_threshold: 0.75
  consistency_threshold: 0.70
```

Current observed Stage 5 result after clean Stage 3/4 reruns:

- `embedding_backend`: `huggingface`
- `halluguard_threshold`: `0.45`
- `jacobian_instability`: enabled
- `hidden_state_instability`: false for all candidates
- `consistency_score`: around `0.96-0.98`
- grounding is borderline for some candidates because the retrieval artifact may still be from the old small corpus.

## API Key Notes

No API key is required with current config because:

```yaml
enable_judge: false
```

OpenAI API key is only needed if:

```yaml
enable_judge: true
```

Then set:

```bash
export OPENAI_API_KEY="..."
```

## Environment Notes

Project venv:

```text
./venv
```

Dependencies added during this session:

```text
openai
protobuf
sentencepiece
```

These were installed into the project venv.

Hugging Face network/model access was needed to refresh:

- Qwen tokenizer/model for Stage 3.
- Qwen trace model for Stage 4.
- BGE-M3 backend for Stage 5.

## Commands Verified

Syntax checks:

```bash
python3 -m py_compile \
  src/saferesponse_engine/components/generation_layer.py \
  src/saferesponse_engine/components/trace_collection_layer.py \
  src/saferesponse_engine/components/verification_layer.py \
  src/saferesponse_engine/config/configuration.py \
  src/saferesponse_engine/entity/config_entity.py \
  main.py
```

Notebook JSON checks:

```bash
python3 -m json.tool research/generation_layer.ipynb
python3 -m json.tool research/verification_layer.ipynb
```

Stage reruns completed in this session:

```bash
./venv/bin/python -c "from src.saferesponse_engine.pipeline.stage_03_generation_layer import GenerationLayerTrainingPipeline; GenerationLayerTrainingPipeline().main()"

./venv/bin/python -c "from src.saferesponse_engine.pipeline.stage_04_trace_collection_layer import TraceCollectionLayerTrainingPipeline; TraceCollectionLayerTrainingPipeline().main()"

./venv/bin/python -c "from src.saferesponse_engine.pipeline.stage_05_verification_layer import VerificationLayerTrainingPipeline; VerificationLayerTrainingPipeline().main()"
```

## Important Remaining Work

1. Rerun Stage 2 with `num_articles: 50000`.
   - Needed so retrieval can find a stronger Alexander the Great source.
   - Existing retrieval artifact may still be from the old small corpus.

2. Implement Stage 6: Fusion + Decision Router.
   - Consume `artifacts/verification/verification_scores.json`.
   - Combine HalluGuard, grounding, consistency, and judge scores.
   - Actions: accept, rerank, rewrite, regenerate, reject.

3. Implement Stage 7: Final Output.
   - Produce best verified answer.
   - Include confidence and hallucination-risk tag.

4. Implement Stage 8: Deployment + Evaluation.
   - FastAPI middleware.
   - Evaluation fixtures.
   - CI regression tests.
   - Latency and hallucination tracking.

## Practical Cautions

- Full `main.py` can be heavy because Stage 2 now uses `num_articles: 50000`.
- Jacobian is enabled but capped to the lighter 0.5B model for practicality.
- BGE-M3 is configured, but if model/cache loading fails, the code has a lexical fallback.
- `enable_judge` should remain false unless an OpenAI API key is configured.
- Generated artifacts and `__pycache__` files are currently present in the workspace.

## Best Next Step

Rerun Stage 2 with the larger corpus, then rerun Stages 3-5:

```bash
./venv/bin/python -c "from src.saferesponse_engine.pipeline.stage_02_retrieval_layer import RetrievalLayerTrainingPipeline; RetrievalLayerTrainingPipeline().main()"
./venv/bin/python -c "from src.saferesponse_engine.pipeline.stage_03_generation_layer import GenerationLayerTrainingPipeline; GenerationLayerTrainingPipeline().main()"
./venv/bin/python -c "from src.saferesponse_engine.pipeline.stage_04_trace_collection_layer import TraceCollectionLayerTrainingPipeline; TraceCollectionLayerTrainingPipeline().main()"
./venv/bin/python -c "from src.saferesponse_engine.pipeline.stage_05_verification_layer import VerificationLayerTrainingPipeline; VerificationLayerTrainingPipeline().main()"
```

Then inspect:

```text
artifacts/verification/verification_scores.json
```

If retrieval improves, grounding scores should become more meaningful and `supporting_source` should no longer always point to the Achilles article.
