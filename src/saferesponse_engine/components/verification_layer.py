import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from langchain_huggingface import HuggingFaceEmbeddings

from src.saferesponse_engine import logger
from src.saferesponse_engine.entity.config_entity import VerificationConfig
from src.saferesponse_engine.utils.common import load_json


class VerificationLayer:
    def __init__(self, config: VerificationConfig):
        self.config = config
        self.embeddings = None
        self.nli_model = None
        self.trace_tokenizer = None
        self.trace_model = None
        self.embedding_backend = config.embedding_backend.lower()
        if config.enable_grounding_score or config.enable_consistency_score:
            if self.embedding_backend == "lexical":
                logger.info("[Stage 5] Using lexical verification backend")
            else:
                logger.info(
                    "[Stage 5] Loading verification embedding model: %s",
                    config.embedding_model,
                )
                try:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=config.embedding_model,
                        model_kwargs={"local_files_only": True},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                    self.embedding_backend = "huggingface"
                except Exception as exc:
                    self.embedding_backend = "lexical_fallback"
                    logger.warning(
                        "[Stage 5] Embedding model unavailable; using lexical fallback. Error: %s",
                        exc,
                    )

        if config.enable_nli_consistency:
            try:
                from sentence_transformers import CrossEncoder

                self.nli_model = CrossEncoder(
                    config.nli_model_name,
                    local_files_only=True,
                )
                logger.info("[Stage 5] Loaded NLI model: %s", config.nli_model_name)
            except Exception as exc:
                logger.warning(
                    "[Stage 5] NLI model unavailable; using embedding-only consistency. Error: %s",
                    exc,
                )

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, list):
            return [VerificationLayer._to_plain(item) for item in value]
        if isinstance(value, dict):
            return {
                key: VerificationLayer._to_plain(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.embeddings is not None:
            return self.embeddings.embed_documents(texts)
        return self._embed_documents_lexical(texts)

    @staticmethod
    def _embed_documents_lexical(texts: list[str]) -> list[list[float]]:
        tokenized_texts = [
            re.findall(r"[a-z0-9]+", text.lower())
            for text in texts
        ]
        vocabulary = sorted({
            token
            for tokens in tokenized_texts
            for token in tokens
        })
        if not vocabulary:
            return [[0.0] for _ in texts]

        token_to_index = {
            token: index
            for index, token in enumerate(vocabulary)
        }
        vectors = []
        for tokens in tokenized_texts:
            counts = Counter(tokens)
            vector = [0.0] * len(vocabulary)
            for token, count in counts.items():
                vector[token_to_index[token]] = float(count)
            vectors.append(vector)
        return vectors

    def _compute_grounding_scores(
        self,
        candidates: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        if not chunks:
            return {
                candidate["response_id"]: {
                    "score": 0.0,
                    "best_source": None,
                    "best_chunk_id": None,
                    "best_content_hash": None,
                }
                for candidate in candidates
            }

        candidate_texts = [candidate.get("text", "") for candidate in candidates]
        chunk_texts = [chunk.get("content", "") for chunk in chunks]
        embeddings = self._embed_documents(candidate_texts + chunk_texts)
        candidate_embeddings = embeddings[: len(candidate_texts)]
        chunk_embeddings = embeddings[len(candidate_texts):]

        scores = {}
        for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
            similarities = [
                self._cosine_similarity(candidate_embedding, chunk_embedding)
                for chunk_embedding in chunk_embeddings
            ]
            best_index = max(range(len(similarities)), key=similarities.__getitem__)
            best_similarity = self._clip((similarities[best_index] + 1.0) / 2.0)
            best_chunk = chunks[best_index]
            scores[candidate["response_id"]] = {
                "score": round(best_similarity, 6),
                "best_source": best_chunk.get("source"),
                "best_chunk_id": best_chunk.get("chunk_id"),
                "best_content_hash": best_chunk.get("content_hash"),
            }
        return scores

    def _compute_consistency_scores(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        if len(candidates) <= 1:
            return {
                candidate["response_id"]: {
                    "score": 1.0,
                    "embedding_consistency": 1.0,
                    "nli_contradiction_score": None,
                }
                for candidate in candidates
            }

        candidate_texts = [candidate.get("text", "") for candidate in candidates]
        candidate_embeddings = self._embed_documents(candidate_texts)
        embedding_scores = {}
        for idx, candidate in enumerate(candidates):
            similarities = []
            for other_idx, other_embedding in enumerate(candidate_embeddings):
                if idx == other_idx:
                    continue
                similarity = self._cosine_similarity(
                    candidate_embeddings[idx],
                    other_embedding,
                )
                similarities.append(self._clip((similarity + 1.0) / 2.0))
            embedding_scores[candidate["response_id"]] = mean(similarities)

        contradiction_scores = self._compute_nli_contradiction_scores(candidates)
        scores = {}
        for candidate in candidates:
            response_id = candidate["response_id"]
            embedding_consistency = embedding_scores[response_id]
            contradiction_score = contradiction_scores.get(response_id)
            if contradiction_score is None:
                final_score = embedding_consistency
            else:
                final_score = embedding_consistency * (1.0 - contradiction_score)
            scores[response_id] = {
                "score": round(self._clip(final_score), 6),
                "embedding_consistency": round(self._clip(embedding_consistency), 6),
                "nli_contradiction_score": (
                    round(self._clip(contradiction_score), 6)
                    if contradiction_score is not None
                    else None
                ),
            }
        return scores

    def _compute_nli_contradiction_scores(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[int, float | None]:
        if self.nli_model is None or len(candidates) <= 1:
            return {
                candidate["response_id"]: None
                for candidate in candidates
            }

        scores = {}
        for idx, candidate in enumerate(candidates):
            pairs = []
            for other_idx, other_candidate in enumerate(candidates):
                if idx == other_idx:
                    continue
                pairs.append((candidate.get("text", ""), other_candidate.get("text", "")))
            predictions = self.nli_model.predict(pairs)
            tensor_predictions = torch.as_tensor(predictions, dtype=torch.float32)
            if tensor_predictions.ndim == 1:
                contradiction_probability = torch.sigmoid(tensor_predictions).mean()
            else:
                probabilities = torch.softmax(tensor_predictions, dim=-1)
                contradiction_probability = probabilities[:, 0].mean()
            scores[candidate["response_id"]] = float(contradiction_probability.item())
        return scores

    def _compute_ntk_features(self, hidden_tensor: torch.Tensor) -> dict[str, Any]:
        h_last = hidden_tensor[-1].float()
        if h_last.shape[0] < 2:
            return {
                "available": False,
                "off_diagonal_mean": None,
                "off_diagonal_std": None,
                "score": 0.0,
            }

        h_norm = F.normalize(h_last, p=2, dim=-1)
        gram_matrix = h_norm @ h_norm.T
        mask = ~torch.eye(
            gram_matrix.shape[0],
            dtype=torch.bool,
            device=gram_matrix.device,
        )
        off_diagonal = gram_matrix[mask]
        off_diagonal_mean = float(off_diagonal.mean().item())
        off_diagonal_std = float(off_diagonal.std(unbiased=False).item())

        return {
            "available": True,
            "off_diagonal_mean": round(off_diagonal_mean, 6),
            "off_diagonal_std": round(off_diagonal_std, 6),
            "score": round(self._clip((off_diagonal_mean + 1.0) / 2.0), 6),
        }

    def _compute_spectral_features(self, hidden_tensor: torch.Tensor) -> dict[str, Any]:
        h_last = hidden_tensor[-1].float()
        if min(h_last.shape) == 0:
            return {
                "available": False,
                "s_max": None,
                "s_min": None,
                "condition_number": None,
                "score": 0.0,
            }

        singular_values = torch.linalg.svdvals(h_last)
        s_max = float(singular_values[0].item())
        s_min = float(singular_values[-1].item())
        condition_number = s_max / max(s_min, 1e-10)
        spectral_score = self._clip(math.log1p(condition_number) / 10.0)

        return {
            "available": True,
            "s_max": round(s_max, 6),
            "s_min": round(s_min, 6),
            "condition_number": round(condition_number, 6),
            "score": round(spectral_score, 6),
        }

    def _load_trace_model(self):
        if self.trace_model is not None and self.trace_tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.trace_tokenizer = AutoTokenizer.from_pretrained(
            self.config.trace_model_name,
            local_files_only=True,
        )
        if self.trace_tokenizer.pad_token_id is None:
            self.trace_tokenizer.pad_token = self.trace_tokenizer.eos_token
        self.trace_model = AutoModelForCausalLM.from_pretrained(
            self.config.trace_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.trace_model.to("cpu")
        self.trace_model.eval()

    def _compute_jacobian_features(self, text: str) -> dict[str, Any]:
        if not self.config.enable_jacobian_instability:
            return {
                "available": False,
                "method": "disabled",
                "grad_norm_mean": None,
                "grad_norm_max": None,
                "score": 0.0,
            }

        try:
            self._load_trace_model()
            inputs = self.trace_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            input_embeddings = self.trace_model.get_input_embeddings()(input_ids)
            input_embeddings = input_embeddings.detach().requires_grad_(True)
            outputs = self.trace_model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            final_hidden = outputs.hidden_states[-1][0, -1, :]
            target = torch.linalg.vector_norm(final_hidden)
            gradients = torch.autograd.grad(
                target,
                input_embeddings,
                retain_graph=False,
                create_graph=False,
            )[0]
            grad_norms = torch.linalg.vector_norm(gradients[0], dim=-1)
            grad_norm_mean = float(grad_norms.mean().item())
            grad_norm_max = float(grad_norms.max().item())
            jacobian_score = self._clip(math.log1p(grad_norm_mean) / 5.0)
            return {
                "available": True,
                "method": "autograd_input_embedding_gradient",
                "grad_norm_mean": round(grad_norm_mean, 6),
                "grad_norm_max": round(grad_norm_max, 6),
                "score": round(jacobian_score, 6),
            }
        except Exception as exc:
            logger.warning(
                "[Stage 5] Jacobian instability unavailable for this candidate. Error: %s",
                exc,
            )
            return {
                "available": False,
                "method": "autograd_input_embedding_gradient",
                "grad_norm_mean": None,
                "grad_norm_max": None,
                "score": 0.0,
                "error": str(exc),
            }

    def _compute_halluguard_score(
        self,
        trace: dict[str, Any],
        candidate_text: str,
        context: str,
    ) -> dict[str, Any]:
        logprobs = trace.get("logprobs", [])
        mean_logprob = float(trace.get("mean_logprob", 0.0))
        min_logprob = float(trace.get("min_logprob", 0.0))

        uncertainty_score = self._clip(abs(mean_logprob) / 3.0)
        tail_risk_score = self._clip(abs(min_logprob) / 8.0)
        variance_score = 0.0
        if len(logprobs) > 1:
            logprob_mean = sum(logprobs) / len(logprobs)
            variance = sum((value - logprob_mean) ** 2 for value in logprobs)
            variance /= len(logprobs)
            variance_score = self._clip(math.sqrt(variance) / 3.0)

        hidden_state_score = 0.0
        hidden_state_stats = {
            "available": False,
            "norm_mean": None,
            "norm_std": None,
            "step_drift_mean": None,
        }
        ntk_features = {
            "available": False,
            "off_diagonal_mean": None,
            "off_diagonal_std": None,
            "score": 0.0,
        }
        spectral_features = {
            "available": False,
            "s_max": None,
            "s_min": None,
            "condition_number": None,
            "score": 0.0,
        }
        full_text = f"{context}\n\n{candidate_text}".strip()
        jacobian_features = self._compute_jacobian_features(full_text)
        hidden_states_path = trace.get("hidden_states_path")
        if hidden_states_path and Path(hidden_states_path).exists():
            hidden_tensor = torch.load(
                hidden_states_path,
                map_location="cpu",
                weights_only=True,
            )
            token_vectors = hidden_tensor.mean(dim=0)
            norms = torch.linalg.vector_norm(token_vectors, dim=1)
            norm_mean = float(norms.mean().item())
            norm_std = float(norms.std(unbiased=False).item())
            step_drift_mean = 0.0
            if token_vectors.shape[0] > 1:
                drift = torch.linalg.vector_norm(
                    token_vectors[1:] - token_vectors[:-1],
                    dim=1,
                )
                step_drift_mean = float(drift.mean().item())
            relative_norm_std = norm_std / max(norm_mean, 1e-6)
            relative_step_drift = step_drift_mean / max(norm_mean, 1e-6)
            hidden_state_score = self._clip(
                0.6 * relative_norm_std + 0.4 * relative_step_drift
            )
            hidden_state_stats = {
                "available": True,
                "norm_mean": round(norm_mean, 6),
                "norm_std": round(norm_std, 6),
                "step_drift_mean": round(step_drift_mean, 6),
            }
            if self.config.enable_ntk:
                ntk_features = self._compute_ntk_features(hidden_tensor)
            if self.config.enable_spectral_conditioning:
                spectral_features = self._compute_spectral_features(hidden_tensor)

        score = (
            0.25 * uncertainty_score
            + 0.15 * tail_risk_score
            + 0.10 * variance_score
            + 0.10 * hidden_state_score
            + 0.20 * ntk_features["score"]
            + 0.10 * spectral_features["score"]
            + 0.10 * jacobian_features["score"]
        )
        return {
            "score": round(self._clip(score), 6),
            "features": {
                "uncertainty_score": round(uncertainty_score, 6),
                "tail_risk_score": round(tail_risk_score, 6),
                "variance_score": round(variance_score, 6),
                "hidden_state_score": round(hidden_state_score, 6),
                "hidden_state_stats": hidden_state_stats,
                "ntk": ntk_features,
                "spectral_conditioning": spectral_features,
                "jacobian_instability": jacobian_features,
            },
        }

    def _compute_judge_score(
        self,
        chunks: list[dict[str, Any]],
        candidate_text: str,
    ) -> float | None:
        if not self.config.enable_judge:
            return None

        try:
            from openai import OpenAI

            context = "\n\n".join(
                chunk.get("content", "")
                for chunk in chunks[:5]
            )
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.config.judge_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Rate hallucination risk from 0 to 1. "
                        "Return only one number.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Response:\n{candidate_text}"
                    ),
                }],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            match = re.search(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", content)
            if match is None:
                return None
            return round(self._clip(float(match.group())), 6)
        except Exception as exc:
            logger.warning("[Stage 5] Judge score unavailable. Error: %s", exc)
            return None

    def verify(self) -> dict[str, Any]:
        retrieval_data = self._to_plain(load_json(self.config.retrieval_artifact_path))
        generation_data = self._to_plain(load_json(self.config.generation_artifact_path))
        trace_data = self._to_plain(load_json(self.config.trace_artifact_path))

        query = generation_data["query"]
        context = generation_data.get("context", "")
        chunks = retrieval_data.get("chunks", [])
        candidates = generation_data.get("candidates", [])
        traces = {
            trace["response_id"]: trace
            for trace in trace_data.get("traces", [])
        }

        logger.info("[Stage 5] Verifying %s candidates", len(candidates))

        grounding_scores = {}
        if self.config.enable_grounding_score:
            grounding_scores = self._compute_grounding_scores(candidates, chunks)

        consistency_scores = {}
        if self.config.enable_consistency_score:
            consistency_scores = self._compute_consistency_scores(candidates)

        verified_candidates = []
        for candidate in candidates:
            response_id = candidate["response_id"]
            trace = traces.get(response_id, {})

            halluguard_result = {"score": None, "features": {}}
            if self.config.enable_halluguard:
                halluguard_result = self._compute_halluguard_score(
                    trace=trace,
                    candidate_text=candidate.get("text", ""),
                    context=context,
                )

            grounding_result = grounding_scores.get(response_id, {})
            grounding_score = grounding_result.get("score")
            consistency_result = consistency_scores.get(response_id, {})
            consistency_score = consistency_result.get("score")
            judge_score = self._compute_judge_score(
                chunks=chunks,
                candidate_text=candidate.get("text", ""),
            )

            risk_signals = {
                "low_mean_logprob": (
                    trace.get("mean_logprob") is not None
                    and float(trace.get("mean_logprob")) < -2.0
                ),
                "low_min_logprob": (
                    trace.get("min_logprob") is not None
                    and float(trace.get("min_logprob")) < -8.0
                ),
                "weak_grounding": (
                    grounding_score is not None
                    and grounding_score < self.config.grounding_threshold
                ),
                "high_sample_divergence": (
                    consistency_score is not None
                    and consistency_score < self.config.consistency_threshold
                ),
                "hidden_state_instability": (
                    halluguard_result["score"] is not None
                    and halluguard_result["score"] > self.config.halluguard_threshold
                ),
            }

            verified_candidates.append({
                "response_id": response_id,
                "text": candidate.get("text", ""),
                "is_primary": candidate.get("is_primary", False),
                "temperature": candidate.get("temperature"),
                "halluguard_score": halluguard_result["score"],
                "halluguard_features": halluguard_result["features"],
                "grounding_score": grounding_score,
                "consistency_score": consistency_score,
                "consistency_features": {
                    "embedding_consistency": consistency_result.get(
                        "embedding_consistency"
                    ),
                    "nli_contradiction_score": consistency_result.get(
                        "nli_contradiction_score"
                    ),
                },
                "judge_score": judge_score,
                "supporting_source": {
                    "source": grounding_result.get("best_source"),
                    "chunk_id": grounding_result.get("best_chunk_id"),
                    "content_hash": grounding_result.get("best_content_hash"),
                },
                "risk_signals": risk_signals,
            })

        output = {
            "query": query,
            "verification_model": {
                "embedding_model": self.config.embedding_model,
                "embedding_backend": self.embedding_backend,
                "nli_model": self.config.nli_model_name,
                "nli_enabled": self.config.enable_nli_consistency,
                "trace_model": self.config.trace_model_name,
                "judge_enabled": self.config.enable_judge,
                "judge_model": self.config.judge_model,
                "halluguard_modules": {
                    "ntk": self.config.enable_ntk,
                    "jacobian_instability": self.config.enable_jacobian_instability,
                    "spectral_conditioning": self.config.enable_spectral_conditioning,
                },
            },
            "thresholds": {
                "halluguard_threshold": self.config.halluguard_threshold,
                "grounding_threshold": self.config.grounding_threshold,
                "consistency_threshold": self.config.consistency_threshold,
            },
            "candidates": verified_candidates,
        }

        output_path = Path(self.config.verification_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[Stage 5] Verification artifact saved: %s", output_path)
        return output
