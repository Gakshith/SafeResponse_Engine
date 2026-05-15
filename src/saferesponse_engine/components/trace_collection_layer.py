import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from saferesponse_engine import logger
from saferesponse_engine.components.model_runtime import (
    get_causal_lm,
    get_tokenizer,
    resolve_model_source,
    select_runtime,
)
from saferesponse_engine.entity.config_entity import TraceCollectionConfig
from saferesponse_engine.utils.common import load_json


class TraceCollectionLayer:
    def __init__(self, config: TraceCollectionConfig):
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

    @staticmethod
    def _optional_field(data: Any, key: str, default: Any = "") -> Any:
        if isinstance(data, dict):
            return data.get(key, default)
        return getattr(data, key, default)

    def _build_prompt(
        self,
        query: str,
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
                    f"Question: {query}\n\n"
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
            f"Question: {query}\n\n"
            "Answer in one short sentence, 40 words or fewer.\nAssistant:"
        )

    def _collect_trace(self, prompt: str, candidate: dict[str, Any]) -> dict[str, Any]:
        response_id = candidate["response_id"]
        temperature = candidate["temperature"]
        is_primary = candidate["is_primary"]
        candidate_text = candidate.get("text", "")

        if not candidate.get("requires_model_trace", True):
            logger.info(
                "[Stage 4] Skipping model trace for context-guard candidate %s",
                response_id,
            )
            return {
                "response_id": response_id,
                "text": candidate_text,
                "is_primary": is_primary,
                "temperature": temperature,
                "tokens": [],
                "logprobs": [],
                "mean_logprob": 0.0,
                "min_logprob": 0.0,
                "sequence_score": 0.0,
                "hidden_states_path": None,
                "num_layers": 0,
                "hidden_dim": 0,
                "num_tokens": 0,
                "num_tokens_prompt": 0,
                "trace_skipped": True,
                "skip_reason": "context_guard",
            }

        self._ensure_model()

        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length,
        )
        input_len = prompt_inputs["input_ids"].shape[1]
        full_inputs = self.tokenizer(
            prompt + candidate_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length,
        )
        full_inputs = {
            key: value.to(self.device)
            for key, value in full_inputs.items()
        }
        full_ids = full_inputs["input_ids"][0]
        response_ids = full_ids[input_len:]
        if len(response_ids) == 0 and candidate_text:
            response_ids = self.tokenizer(
                candidate_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(self.device)

        with torch.inference_mode():
            output = self.model(
                **full_inputs,
                output_hidden_states=self.config.collect_hidden_states,
                use_cache=False,
            )

        tokens = [self.tokenizer.decode([token_id]) for token_id in response_ids]

        logprobs = []
        logits = output.logits[0]
        for token_pos in range(input_len, min(len(full_ids), logits.shape[0])):
            if token_pos == 0:
                continue
            token_id = full_ids[token_pos].item()
            log_prob = F.log_softmax(logits[token_pos - 1], dim=-1)[token_id].item()
            logprobs.append(round(log_prob, 6))

        mean_logprob = round(sum(logprobs) / len(logprobs), 6) if logprobs else 0.0
        min_logprob = round(min(logprobs), 6) if logprobs else 0.0
        sequence_score = round(sum(logprobs), 6)

        hidden_states_path = None
        num_layers = 0
        hidden_dim = 0
        if self.config.collect_hidden_states and output.hidden_states:
            hidden_states_path, num_layers, hidden_dim = self._save_hidden_states(
                hidden_states=output.hidden_states,
                response_id=response_id,
                input_len=input_len,
            )

        logger.info(
            "[Stage 4] Trace collected for candidate %s | tokens=%s | mean_logprob=%.4f | min_logprob=%.4f",
            response_id,
            len(tokens),
            mean_logprob,
            min_logprob,
        )

        return {
            "response_id": response_id,
            "text": candidate["text"],
            "is_primary": is_primary,
            "temperature": temperature,
            "tokens": tokens,
            "logprobs": logprobs,
            "mean_logprob": mean_logprob,
            "min_logprob": min_logprob,
            "sequence_score": sequence_score,
            "hidden_states_path": hidden_states_path,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_tokens": len(tokens),
            "num_tokens_prompt": input_len,
        }

    def _save_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        response_id: int,
        input_len: int,
    ) -> tuple[str, int, int]:
        num_layers_total = len(hidden_states)
        if self.config.num_hidden_layers_to_save == -1:
            layers_to_save = list(range(num_layers_total))
        else:
            start = max(0, num_layers_total - self.config.num_hidden_layers_to_save)
            layers_to_save = list(range(start, num_layers_total))

        stacked = []
        for layer_idx in layers_to_save:
            layer_states = hidden_states[layer_idx][0, input_len:, :]
            if layer_states.shape[0] == 0:
                layer_states = hidden_states[layer_idx][0, -1:, :]
            stacked.append(layer_states)

        hidden_tensor = torch.stack(stacked, dim=0).cpu()
        num_layers = hidden_tensor.shape[0]
        hidden_dim = hidden_tensor.shape[2]

        save_path = self.config.hidden_states_dir / f"candidate_{response_id}_hidden.pt"
        torch.save(hidden_tensor, save_path)
        logger.info(
            "[Stage 4] Hidden states saved: %s | shape: %s",
            save_path,
            list(hidden_tensor.shape),
        )
        return str(save_path), num_layers, hidden_dim

    def collect(self) -> dict[str, Any]:
        generation_data = load_json(self.config.generation_artifact_path)
        query = generation_data["query"]
        context = generation_data["context"]
        memory_context = self._optional_field(generation_data, "memory_context", "")
        candidates = generation_data["candidates"]

        logger.info("[Stage 4] Collecting traces for %s candidates", len(candidates))
        needs_model_trace = any(
            candidate.get("requires_model_trace", True)
            for candidate in candidates
        )
        if needs_model_trace:
            self._ensure_model()
            prompt = self._build_prompt(
                query=query,
                context=context,
                memory_context=memory_context,
            )
        else:
            prompt = ""

        traces = []
        for candidate in candidates:
            logger.info(
                "[Stage 4] Collecting trace for candidate %s...",
                candidate["response_id"],
            )
            traces.append(self._collect_trace(prompt=prompt, candidate=candidate))

        output = {
            "query": query,
            "model_name": self.config.model_name,
            "finetuned_model_path": self.config.finetuned_model_path,
            "runtime_model_source": resolve_model_source(
                self.config.model_name,
                self.config.finetuned_model_path,
            ),
            "traces": traces,
        }

        output_path = Path(self.config.trace_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[Stage 4] Trace artifact saved: %s", output_path)
        return output
