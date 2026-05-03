import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from src.saferesponse_engine import logger
from src.saferesponse_engine.entity.config_entity import GenerationConfig
from src.saferesponse_engine.utils.common import load_json

_MODEL_CACHE: dict = {}


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
        self.device, self.dtype = self._select_runtime()

        # load tokenizer
        logger.info("[Stage 3] Loading tokenizer: %s", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # check cache first — skip loading if already in memory
        if config.model_name in _MODEL_CACHE:
            logger.info("[Stage 3] Loading model from cache — skipping disk read")
            self.model = _MODEL_CACHE[config.model_name]
        else:
            logger.info(
                "[Stage 3] Loading model from disk: %s | device=%s | dtype=%s",
                config.model_name,
                self.device,
                self.dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            self.model.eval()
            _MODEL_CACHE[config.model_name] = self.model
            logger.info("[Stage 3] Model loaded and cached in memory")

    @staticmethod
    def _select_runtime() -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "cuda", torch.bfloat16
            return "cuda", torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    # build_context — joins retrieved chunks into one context string
    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")
            parts.append(f"[Source {i} - {source}]:\n{content}")
        return "\n\n".join(parts)

    # build_prompt — chat-formatted system prompt + context + query
    def _build_prompt(self, user_query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a factual assistant. Answer only from the provided "
                    "context. If the context does not contain the answer, say "
                    "\"I don't know.\" Keep the answer concise and do not invent "
                    "follow-up questions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {user_query}\n\n"
                    "Answer in one short paragraph."
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
            "context. If the context does not contain the answer, say \"I don't know.\" "
            "Keep the answer concise and do not invent follow-up questions.\n\n"
            f"User: Context:\n{context}\n\n"
            f"Question: {user_query}\n\n"
            "Answer in one short paragraph.\nAssistant:"
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

    # generate_single — generate one candidate at a given temperature
    def _generate_single(
        self,
        prompt: str,
        temperature: float,
        response_id: int,
        is_primary: bool
    ) -> dict[str, Any]:

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
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
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
            "num_tokens": len(response_ids)
        }

    # generate — main method, reads Stage 2 artifact, saves Stage 3 artifact
    def generate(self) -> dict[str, Any]:

        # Step 1: read Stage 2 artifact
        retrieval_data = load_json(self.config.retrieval_artifact_path)
        user_query = retrieval_data["query"]
        chunks = retrieval_data["chunks"]

        logger.info("[Stage 3] Query: '%s'", user_query)
        logger.info("[Stage 3] Using %s retrieved chunks", len(chunks))

        # Step 2: build context + prompt
        context = self._build_context(chunks)
        prompt = self._build_prompt(user_query, context)

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
            "model_name": self.config.model_name,
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
