import json
from pathlib import Path
from typing import Any

from saferesponse_engine import logger
from saferesponse_engine.entity.config_entity import FinalOutputConfig
from saferesponse_engine.utils.common import load_json


RISK_SIGNAL_MESSAGES = {
    "weak_grounding": "The response may not be fully supported by the retrieved sources.",
    "unsupported_claim": "The response contains claims that were not supported by the retrieved sources.",
    "unsupported_live_fact": "The question asks for current or live state outside the static corpus.",
    "answer_abstained": "The model could not produce a supported answer from the provided context.",
    "high_sample_divergence": "Multiple response candidates disagreed; the model showed uncertainty on this question.",
    "hidden_state_instability": "Internal model representations showed instability during generation.",
    "low_mean_logprob": "The model showed low overall confidence in its token predictions.",
    "low_min_logprob": "At least one token in the response had very low generation confidence.",
}

TERMINAL_PUNCTUATION = ".!?"

DECISION_NOTES = {
    "ACCEPT": None,
    "RERANK": (
        "The primary response was flagged as higher risk. This answer was "
        "selected from an alternative candidate."
    ),
    "REWRITE": "The initial response required refinement and was regenerated.",
    "REJECT": (
        "The retrieved sources did not sufficiently support a reliable answer."
    ),
}


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _to_plain(value.to_dict())
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _as_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class FinalOutputLayer:
    def __init__(self, config: FinalOutputConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        thresholds = (
            self.config.high_confidence_threshold,
            self.config.medium_confidence_threshold,
            self.config.low_confidence_threshold,
        )
        if not all(0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError("Final output confidence thresholds must be between 0 and 1.")
        if not (
            self.config.high_confidence_threshold
            <= self.config.medium_confidence_threshold
            <= self.config.low_confidence_threshold
        ):
            raise ValueError(
                "Final output confidence thresholds must be ordered high <= medium <= low."
            )
        if self.config.max_answer_length <= 0:
            raise ValueError("Final output max_answer_length must be positive.")
        if self.config.max_answer_words <= 0:
            raise ValueError("Final output max_answer_words must be positive.")

    def _map_confidence(self, combined_risk: float, decision: str) -> tuple[str, str]:
        if decision == "REJECT":
            return "NONE", "dark_red"
        if combined_risk < self.config.high_confidence_threshold:
            return "HIGH", "green"
        if combined_risk < self.config.medium_confidence_threshold:
            return "MEDIUM", "amber"
        return "LOW", "red"

    def _build_risk_explanation(
        self,
        risk_signals: dict[str, Any],
    ) -> tuple[str, list[str]]:
        fired = [
            signal
            for signal in RISK_SIGNAL_MESSAGES
            if risk_signals.get(signal) is True
        ]
        fired.extend(
            signal
            for signal, value in risk_signals.items()
            if value is True and signal not in RISK_SIGNAL_MESSAGES
        )

        if not fired:
            return "No risk signals detected.", []

        messages = [
            RISK_SIGNAL_MESSAGES.get(signal, signal)
            for signal in fired
        ]
        return " ".join(messages), fired

    def _format_response(
        self,
        answer: str,
        confidence_tag: str,
        combined_risk: float,
        source_title: str | None,
        decision_note: str | None,
        rewrite_attempt: int,
    ) -> str:
        lines = [answer]

        if decision_note:
            lines.extend(["", f"Note: {decision_note}"])

        if rewrite_attempt > 0:
            lines.append(f"Rewrite attempts: {rewrite_attempt}")

        lines.extend(["", f"Confidence: {confidence_tag}"])
        if source_title:
            lines.append(f"Source: {source_title} (Wikipedia)")
        else:
            lines.append("Source: unavailable")
        lines.append(f"Hallucination risk score: {combined_risk:.3f}")

        return "\n".join(lines)

    def _trim_answer(self, answer: str) -> str:
        trimmed = answer.strip()
        if len(trimmed) > self.config.max_answer_length:
            trimmed = trimmed[: self.config.max_answer_length].rstrip() + "..."

        words = trimmed.split()
        if len(words) > self.config.max_answer_words:
            trimmed = " ".join(words[: self.config.max_answer_words]).rstrip() + "..."

        trimmed = self._trim_to_complete_sentence(trimmed)
        return trimmed

    def _trim_to_complete_sentence(self, answer: str) -> str:
        trimmed = answer.rstrip()
        if not trimmed or trimmed[-1] in TERMINAL_PUNCTUATION:
            return trimmed

        last_sentence_end = max(trimmed.rfind(mark) for mark in TERMINAL_PUNCTUATION)
        if last_sentence_end >= 40:
            return trimmed[: last_sentence_end + 1].rstrip()

        return trimmed.rstrip(",;:") + "."

    def _build_pipeline_summary(
        self,
        fusion_data: dict[str, Any],
        verification_data: dict[str, Any],
    ) -> dict[str, Any]:
        verification_model = verification_data.get("verification_model", {})
        fusion_scores = fusion_data.get("fusion_scores", [])
        effective_weights = {}
        if fusion_scores:
            effective_weights = fusion_scores[0].get("effective_weights", {}) or {}

        signals_used = [
            signal
            for signal, weight in effective_weights.items()
            if _as_float(weight, 0.0) > 0
        ]
        if not signals_used:
            configured_weights = fusion_data.get("configured_weights", {})
            judge_enabled = bool(verification_model.get("judge_enabled", False))
            signals_used = [
                signal
                for signal, weight in configured_weights.items()
                if _as_float(weight, 0.0) > 0 and (signal != "judge" or judge_enabled)
            ]

        return {
            "total_candidates": len(fusion_scores),
            "rewrite_attempts": int(fusion_data.get("rewrite_attempt", 0)),
            "signals_used": signals_used,
            "embedding_model": verification_model.get("embedding_model"),
            "embedding_backend": verification_model.get("embedding_backend"),
            "generation_model": (
                verification_model.get("generation_model")
                or verification_model.get("trace_model")
            ),
            "trace_model": verification_model.get("trace_model"),
            "nli_enabled": bool(verification_model.get("nli_enabled", False)),
            "judge_enabled": bool(verification_model.get("judge_enabled", False)),
            "halluguard_modules": verification_model.get("halluguard_modules", {}),
        }

    def generate(self) -> dict[str, Any]:
        fusion_data = _to_plain(load_json(self.config.fusion_artifact_path))
        verification_data = _to_plain(load_json(self.config.verification_artifact_path))

        query = fusion_data.get("query") or verification_data.get("query", "")
        decision = str(fusion_data.get("decision", "REJECT"))
        decision_reason = fusion_data.get("decision_reason")
        rewrite_attempt = int(fusion_data.get("rewrite_attempt", 0))
        selected = fusion_data.get("selected_candidate", {}) or {}
        combined_risk = max(
            0.0,
            min(1.0, _as_float(selected.get("combined_risk"), 1.0)),
        )
        risk_signals = selected.get("risk_signals", {}) or {}
        supporting_source = selected.get("supporting_source", {}) or {}

        risk_explanation, fired_signals = self._build_risk_explanation(risk_signals)
        confidence_tag, confidence_color = self._map_confidence(
            combined_risk=combined_risk,
            decision=decision,
        )

        if decision == "REJECT":
            answer = (
                "I cannot provide a reliable answer to this question because the "
                "retrieved sources do not sufficiently support it."
            )
        else:
            answer = self._trim_answer(str(selected.get("text", "")))

        source_title = supporting_source.get("source")
        source_citation = None
        if source_title and decision != "REJECT":
            source_citation = {
                "title": source_title,
                "chunk_id": supporting_source.get("chunk_id"),
                "content_hash": supporting_source.get("content_hash"),
                "database": "Wikipedia 20220301",
            }

        decision_note = DECISION_NOTES.get(decision)
        display_risk = 1.0 if decision == "REJECT" else combined_risk
        final_response = {
            "answer": answer,
            "confidence_tag": confidence_tag,
            "confidence_color": confidence_color,
            "confidence_score": round(display_risk, 6),
            "selected_candidate_risk": round(combined_risk, 6),
            "source_citation": source_citation,
            "risk_explanation": (
                risk_explanation if self.config.include_risk_explanation else None
            ),
            "risk_signals_fired": fired_signals,
            "decision_note": decision_note,
        }

        if self.config.include_formatted_response:
            final_response["formatted_response"] = self._format_response(
                answer=answer,
                confidence_tag=confidence_tag,
                combined_risk=display_risk,
                source_title=source_title if decision != "REJECT" else None,
                decision_note=decision_note,
                rewrite_attempt=rewrite_attempt,
            )

        output = {
            "query": query,
            "decision": decision,
            "decision_reason": decision_reason,
            "rewrite_attempt": rewrite_attempt,
            "final_response": final_response,
        }
        if self.config.include_pipeline_summary:
            output["pipeline_summary"] = self._build_pipeline_summary(
                fusion_data=fusion_data,
                verification_data=verification_data,
            )

        output_path = Path(self.config.final_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "[Stage 7] Query: %r | Decision: %s | Risk: %.3f",
            query,
            decision,
            combined_risk,
        )
        logger.info(
            "[Stage 7] Final response saved: %s | confidence=%s | risk=%.3f",
            output_path,
            confidence_tag,
            combined_risk,
        )
        return output
