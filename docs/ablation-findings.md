# Ablation Findings — Independent Signal Isolation

**Date:** 2026-06-21 · Generator: Qwen2.5-1.5B-Instruct · Corpus: 44 articles (dense FAISS)
**Eval set:** `evaluation/examples.json` (15 hallucination examples: 5 supported, 10 unsupported)

## Method

The verification layer exposes three internal signals — **logprob** (halluguard),
**grounding** (answer↔context similarity), and **consistency** (multi-sample agreement).
To measure each signal's individual contribution we keep verification fully enabled (so every
score and the supporting-source attribution are always computed) and vary the **fusion
weights**, isolating one signal at a time. Grounding-derived hard rejects only fire when
grounding is weighted, so the system no longer "fails closed" when grounding is removed.

`scripts/run_ablation.py` runs the eval set under `logprob_only`, `grounding_only`,
`consistency_only`, and `all_on`, reporting false-accept rate (FAR), false-reject rate (FRR),
and accuracy.

## Result

| config | FAR | FRR | accuracy |
|---|---|---|---|
| logprob_only | 0.000 | 0.000 | 1.000 |
| grounding_only | 0.000 | 0.000 | 1.000 |
| consistency_only | 0.000 | 0.000 | 1.000 |
| all_on | 0.000 | 0.000 | 1.000 |

Every configuration achieves a perfect split on this eval set.

## Interpretation (the honest finding)

The perfect scores are **not** evidence that each signal is independently a strong hallucination
detector. They are evidence that, on this eval set, **two cheaper mechanisms do all the work
before the statistical signals matter**:

1. **Retrieval gating.** Most "unsupported" questions are out-of-corpus, so retrieval returns no
   sufficiently-grounded chunk and the query is rejected before generation.
2. **Model abstention.** For in-corpus *entity-overlap traps* (e.g. "What company did Abraham
   Lincoln found?"), retrieval *does* return a chunk, but the instruction-tuned model **abstains**
   ("I cannot provide a reliable answer…"). This abstention is caught by an always-on guard,
   independent of logprob/grounding/consistency.

So with a well-behaved model that abstains appropriately, the elaborate internal signals are
**largely redundant** on straightforward unsupported queries. They would earn their keep against
a model that **confidently hallucinates** instead of abstaining — which this eval set does not
contain.

## Stress test: suppress abstention (the differentiating experiment)

To measure what each signal actually catches, we add a `force_answer` mode that forbids the
model from abstaining, so it produces confident answers even when the context does not support
them. Re-running the ablation under this mode (`scripts/run_ablation.py --force-answer`,
report: `docs/ablation-report-forceanswer.json`) makes the signals do the discriminating:

| config | FAR | FRR | accuracy |
|---|---|---|---|
| logprob_only | **0.100** | 0.000 | 0.933 |
| grounding_only | **0.000** | 0.000 | 1.000 |
| consistency_only | **0.100** | 0.000 | 0.933 |
| all_on | **0.000** | 0.000 | 1.000 |

The signals now diverge. The separating case is the in-corpus entity-overlap trap **"What
company did Abraham Lincoln found?"**: forced to answer, the model **confidently fabricates** a
company.

- **logprob_only → ACCEPT** (false accept): the model is confident, so its token
  log-probabilities look normal — logprob cannot detect a fluent, confident hallucination.
- **consistency_only → ACCEPT** (false accept): with a single sample there is nothing to
  disagree with; consistency needs multiple samples to contribute.
- **grounding_only → REJECT** (correct): the fabricated answer does not match the retrieved
  Lincoln passage, so the answer↔context similarity is low.
- **all_on → REJECT** (correct).

### Conclusion

**Grounding (answer↔context similarity) is the signal that actually catches confident
hallucinations.** Logprob and single-sample consistency do not — a fluent model is fluent
whether it is right or wrong, and one sample cannot vote against itself. This is the core,
honest contribution: the verification value comes from grounding against retrieved evidence, not
from the model's own confidence. Future work: multi-sample consistency (num_candidates > 1) to
give the consistency signal a fair test, and a larger adversarial eval set.
