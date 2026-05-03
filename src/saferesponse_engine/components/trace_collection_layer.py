import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.saferesponse_engine import logger
from src.saferesponse_engine.entity.config_entity import TraceCollectionConfig
from src.saferesponse_engine.utils.common import load_json

_MODEL_CACHE: dict[str, AutoModelForCausalLM] = {}


class TraceCollectionLayer:
    def __init__(self, config: TraceCollectionConfig):
        self.config = config
        self.device, self.dtype = self._select_runtime()

        logger.info("[Stage 4] Loading tokenizer: %s", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if config.model_name in _MODEL_CACHE:
            logger.info("[Stage 4] Model loaded from cache")
            self.model = _MODEL_CACHE[config.model_name]
        else:
            logger.info(
                "[Stage 4] Loading model from disk: %s | device=%s | dtype=%s",
                config.model_name,
                self.device,
                self.dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            self.model.eval()
            _MODEL_CACHE[config.model_name] = self.model
            logger.info("[Stage 4] Model loaded and cached")

    @staticmethod
    def _select_runtime() -> tuple[str, torch.dtype]:
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "cuda", torch.bfloat16
            return "cuda", torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _build_prompt(self, query: str, context: str) -> str:
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
                    f"Question: {query}\n\n"
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
            "User: "
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer in one short paragraph.\nAssistant:"
        )

    def _collect_trace(self, prompt: str, candidate: dict[str, Any]) -> dict[str, Any]:
        response_id = candidate["response_id"]
        temperature = candidate["temperature"]
        is_primary = candidate["is_primary"]

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=candidate["num_tokens"],
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                output_hidden_states=self.config.collect_hidden_states,
                return_dict_in_generate=True,
            )

        response_ids = output.sequences[0][input_len:]
        tokens = [self.tokenizer.decode([token_id]) for token_id in response_ids]

        logprobs = []
        for step, score_tensor in enumerate(output.scores):
            if step >= len(response_ids):
                break
            token_id = response_ids[step].item()
            log_prob = F.log_softmax(score_tensor[0], dim=-1)[token_id].item()
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
        hidden_states: tuple[tuple[torch.Tensor, ...], ...],
        response_id: int,
    ) -> tuple[str, int, int]:
        num_layers_total = len(hidden_states[0])
        if self.config.num_hidden_layers_to_save == -1:
            layers_to_save = list(range(num_layers_total))
        else:
            start = max(0, num_layers_total - self.config.num_hidden_layers_to_save)
            layers_to_save = list(range(start, num_layers_total))

        stacked = []
        for layer_idx in layers_to_save:
            layer_states = []
            for token_step in hidden_states:
                token_hidden = token_step[layer_idx][0, -1, :]
                layer_states.append(token_hidden)
            stacked.append(torch.stack(layer_states, dim=0))

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
        candidates = generation_data["candidates"]

        logger.info("[Stage 4] Collecting traces for %s candidates", len(candidates))
        prompt = self._build_prompt(query=query, context=context)

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
