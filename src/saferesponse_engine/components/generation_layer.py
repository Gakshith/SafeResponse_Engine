import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from saferesponse_engine import logger
from saferesponse_engine.components.model_runtime import (
    get_causal_lm,
    get_tokenizer,
    resolve_model_source,
    select_runtime,
)
from saferesponse_engine.entity.config_entity import GenerationConfig
from saferesponse_engine.utils.common import load_json


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: list[str], prompt_length: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_length:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return any(stop_string in generated_text for stop_string in self.stop_strings)


class GenerationLayer:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device, self.dtype = select_runtime()
        self.tokenizer = None
        self.model = None

    def _ensure_model(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer(
                self.config.model_name,
                self.config.finetuned_model_path,
            )
        if self.model is None:
            self.model = get_causal_lm(
                self.config.model_name,
                self.device,
                self.dtype,
                self.config.finetuned_model_path,
            )

    # build_context — joins retrieved chunks into one context string
    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")
            parts.append(f"[Source {i} - {source}]:\n{content}")
        return "\n\n".join(parts)

    # build_prompt — chat-formatted system prompt + context + query
    def _build_prompt(
        self,
        user_query: str,
        context: str,
        memory_context: str = "",
    ) -> str:
        memory_block = (
            f"Conversation memory:\n{memory_context}\n\n"
            if memory_context.strip()
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a factual assistant. Answer only from the provided "
                    "document context and conversation memory. If both are empty, "
                    "unrelated, or do not contain the answer, reply exactly: "
                    "\"I don't know based on the provided context.\" Keep the "
                    "answer concise. Do not add background, explanations, or "
                    "details that are not explicitly stated in the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{memory_block}"
                    f"Document context:\n{context}\n\n"
                    f"Question: {user_query}\n\n"
                    "Answer in one short sentence, 40 words or fewer."
                ),
            },
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            "System: You are a factual assistant. Answer only from the provided "
            "document context and conversation memory. If both are empty, unrelated, "
            "or do not contain the answer, reply exactly: \"I don't know based on "
            "the provided context.\" Keep the answer concise. Do not add background, "
            "explanations, or details that are not explicitly stated in the "
            "context.\n\n"
            f"User: {memory_block}"
            f"Document context:\n{context}\n\n"
            f"Question: {user_query}\n\n"
            "Answer in one short sentence, 40 words or fewer.\nAssistant:"
        )

    @staticmethod
    def _stop_strings() -> list[str]:
        return [
            "\nHuman:",
            "\nUser:",
            "\nSystem:",
            "\nAssistant:",
            "Human:",
            "User:",
            "System:",
            "Write a",
            "write a",
        ]

    def _clean_response(self, response_text: str) -> str:
        cleaned = response_text.strip()
        for stop_string in self._stop_strings():
            stop_index = cleaned.find(stop_string)
            if stop_index != -1:
                cleaned = cleaned[:stop_index].strip()
        return cleaned

    @staticmethod
    def _is_summary_query(query: str) -> bool:
        normalized = re.sub(r"\s+", " ", query.lower()).strip()
        return any(
            phrase in normalized
            for phrase in (
                "summary",
                "summarize",
                "tell me about",
                "give me about",
                "give me a short",
            )
        )

    @staticmethod
    def _trim_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).rstrip(",;:") + "."

    def _extractive_summary_candidate(self, chunks: list[dict]) -> dict[str, Any]:
        content = str(chunks[0].get("content", "")).strip()
        first_sentence = re.split(r"(?<=[.!?])\s+", content, maxsplit=1)[0].strip()
        answer = self._trim_words(first_sentence or content, 40)
        return {
            "response_id": 0,
            "text": answer,
            "is_primary": True,
            "temperature": 0.0,
            "num_tokens": len(answer.split()),
            "requires_model_trace": False,
        }

    # generate_single — generate one candidate at a given temperature
    def _generate_single(
        self,
        prompt: str,
        temperature: float,
        response_id: int,
        is_primary: bool
    ) -> dict[str, Any]:
        self._ensure_model()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList([
            StopOnSubstrings(
                tokenizer=self.tokenizer,
                stop_strings=self._stop_strings(),
                prompt_length=prompt_length,
            )
        ])

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                use_cache=True,
            )

        # decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        response_ids = output_ids[0][input_len:]
        response_text = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True
        )
        response_text = self._clean_response(response_text)

        logger.info(
            "[Stage 3] Candidate %s generated (%s tokens)",
            response_id, len(response_ids)
        )

        return {
            "response_id": response_id,
            "text": response_text,
            "is_primary": is_primary,
            "temperature": temperature,
            "num_tokens": len(response_ids),
            "requires_model_trace": True,
        }

    @staticmethod
    def _optional_field(data: Any, key: str, default: Any = "") -> Any:
        if isinstance(data, dict):
            return data.get(key, default)
        return getattr(data, key, default)

    @staticmethod
    def _context_guard_candidate() -> dict[str, Any]:
        return {
            "response_id": 0,
            "text": "I don't know based on the provided context.",
            "is_primary": True,
            "temperature": 0.0,
            "num_tokens": 0,
            "requires_model_trace": False,
        }

    # generate — main method, reads Stage 2 artifact, saves Stage 3 artifact
    def generate(self) -> dict[str, Any]:

        # Step 1: read Stage 2 artifact
        retrieval_data = load_json(self.config.retrieval_artifact_path)
        user_query = retrieval_data["query"]
        chunks = retrieval_data["chunks"]
        memory_context = self._optional_field(retrieval_data, "memory_context", "")

        logger.info("[Stage 3] Query: '%s'", user_query)
        logger.info("[Stage 3] Using %s retrieved chunks", len(chunks))

        if not chunks and not memory_context.strip():
            logger.info(
                "[Stage 3] No retrieved context found; using direct context guard."
            )
            candidates = [self._context_guard_candidate()]
            output = {
                "query": user_query,
                "context": "",
                "memory_context": memory_context,
                "model_name": "context_guard",
                "finetuned_model_path": self.config.finetuned_model_path,
                "runtime_model_source": "context_guard",
                "num_candidates": 1,
                "candidates": candidates,
            }
            output_path = Path(self.config.generation_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(output, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            logger.info("[Stage 3] Artifact saved: %s", output_path)
            return output

        context = self._build_context(chunks)
        if chunks and self._is_summary_query(user_query):
            logger.info("[Stage 3] Using extractive summary for summary-style query.")
            candidates = [self._extractive_summary_candidate(chunks)]
            output = {
                "query": user_query,
                "context": context,
                "memory_context": memory_context,
                "model_name": "extractive_summary",
                "finetuned_model_path": self.config.finetuned_model_path,
                "runtime_model_source": "extractive_summary",
                "num_candidates": 1,
                "candidates": candidates,
            }
            output_path = Path(self.config.generation_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(output, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("[Stage 3] Artifact saved: %s", output_path)
            return output

        self._ensure_model()

        # Step 2: build context + prompt
        prompt = self._build_prompt(user_query, context, memory_context)

        # Step 3: generate N candidates
        candidates = []

        for i in range(self.config.num_candidates):
            is_primary = (i == 0)
            temperature = (
                self.config.primary_temperature if is_primary
                else self.config.sample_temperature
            )
            logger.info(
                "[Stage 3] Generating candidate %s (temp=%.1f)...", i, temperature
            )
            candidate = self._generate_single(
                prompt=prompt,
                temperature=temperature,
                response_id=i,
                is_primary=is_primary
            )
            candidates.append(candidate)

        # Step 4: save Stage 3 artifact
        output = {
            "query": user_query,
            "context": context,
            "memory_context": memory_context,
            "model_name": self.config.model_name,
            "finetuned_model_path": self.config.finetuned_model_path,
            "runtime_model_source": resolve_model_source(
                self.config.model_name,
                self.config.finetuned_model_path,
            ),
            "num_candidates": self.config.num_candidates,
            "candidates": candidates
        }

        output_path = Path(self.config.generation_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(
            "[Stage 3] Artifact saved: %s", output_path
        )
        return output
