import json
from pathlib import Path
from typing import Any

from saferesponse_engine import logger
from saferesponse_engine.entity.config_entity import FusionRouterConfig
from saferesponse_engine.utils.common import load_json


class FusionDecisionRouter:
    ACCEPT = "ACCEPT"
    RERANK = "RERANK"
    REWRITE = "REWRITE"
    REJECT = "REJECT"
    HARD_REJECT_SIGNALS = {
        "weak_grounding": "failed the hard grounding guard",
        "unsupported_claim": "contains claims that are not supported by the retrieved sources",
        "unsupported_live_fact": "asks for current or live state outside the static corpus",
        "answer_abstained": "did not produce a supported answer",
    }
    # Hard-reject signals derived from the grounding check; only applied when
    # grounding contributes to the fusion decision (weight_grounding > 0).
    GROUNDING_HARD_REJECT_SIGNALS = {"weak_grounding", "unsupported_claim"}

    def __init__(self, config: FusionRouterConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        weights = {
            "halluguard": self.config.weight_halluguard,
            "grounding": self.config.weight_grounding,
            "consistency": self.config.weight_consistency,
            "judge": self.config.weight_judge,
        }
        invalid_weights = {
            name: weight
            for name, weight in weights.items()
            if weight < 0
        }
        if invalid_weights:
            raise ValueError(
                f"Fusion weights must be non-negative: {invalid_weights}"
            )
        if sum(weights.values()) <= 0:
            raise ValueError("At least one fusion weight must be positive.")
        if not (
            0.0 <= self.config.accept_threshold
            < self.config.rewrite_threshold
            < self.config.reject_threshold
            <= 1.0
        ):
            raise ValueError(
                "Fusion thresholds must satisfy "
                "0 <= accept_threshold < rewrite_threshold < reject_threshold <= 1."
            )
        if self.config.max_rewrite_attempts < 0:
            raise ValueError("max_rewrite_attempts must be non-negative.")

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return FusionDecisionRouter._to_plain(value.to_dict())
        if isinstance(value, list):
            return [FusionDecisionRouter._to_plain(item) for item in value]
        if isinstance(value, dict):
            return {
                key: FusionDecisionRouter._to_plain(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _fuse_scores(self, candidate: dict[str, Any]) -> dict[str, Any]:
        halluguard_score = self._clip(
            self._as_float(candidate.get("halluguard_score"), 1.0)
        )
        grounding_score = self._clip(
            self._as_float(candidate.get("grounding_score"), 0.0)
        )
        consistency_score = self._clip(
            self._as_float(candidate.get("consistency_score"), 0.0)
        )
        judge_score = candidate.get("judge_score")

        risk_terms = {
            "halluguard_risk": halluguard_score,
            "grounding_risk": 1.0 - grounding_score,
            "consistency_risk": 1.0 - consistency_score,
        }
        weighted_terms = [
            ("halluguard", risk_terms["halluguard_risk"], self.config.weight_halluguard),
            ("grounding", risk_terms["grounding_risk"], self.config.weight_grounding),
            (
                "consistency",
                risk_terms["consistency_risk"],
                self.config.weight_consistency,
            ),
        ]

        if judge_score is not None:
            risk_terms["judge_risk"] = self._clip(self._as_float(judge_score, 1.0))
            weighted_terms.append(
                ("judge", risk_terms["judge_risk"], self.config.weight_judge)
            )
        else:
            risk_terms["judge_risk"] = None

        weight_sum = sum(weight for _, _, weight in weighted_terms if weight > 0)
        if weight_sum <= 0:
            return {
                "combined_risk": 1.0,
                "risk_terms": risk_terms,
                "effective_weights": {},
                "risk_contributions": {},
            }

        combined_risk = sum(
            (weight / weight_sum) * risk
            for _, risk, weight in weighted_terms
            if weight > 0
        )
        effective_weights = {
            name: round(weight / weight_sum, 6)
            for name, _, weight in weighted_terms
            if weight > 0
        }
        risk_contributions = {
            name: round((weight / weight_sum) * risk, 6)
            for name, risk, weight in weighted_terms
            if weight > 0
        }

        return {
            "combined_risk": round(self._clip(combined_risk), 6),
            "risk_terms": {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in risk_terms.items()
            },
            "effective_weights": effective_weights,
            "risk_contributions": risk_contributions,
        }

    def _route(
        self,
        best: dict[str, Any],
        rewrite_attempt: int,
    ) -> tuple[str, str]:
        risk = best["combined_risk"]
        response_id = best["response_id"]
        is_primary = bool(best.get("is_primary", response_id == 0))
        risk_signals = best.get("risk_signals", {}) or {}

        # Grounding-derived hard rejects only apply when grounding actually
        # contributes to the decision (weight > 0). This lets an ablation isolate
        # the other signals by zeroing the grounding weight without grounding
        # still vetoing every answer. Query/abstention guards always apply.
        grounding_active = self.config.weight_grounding > 0
        for signal, reason in self.HARD_REJECT_SIGNALS.items():
            if signal in self.GROUNDING_HARD_REJECT_SIGNALS and not grounding_active:
                continue
            if risk_signals.get(signal) is True:
                return (
                    self.REJECT,
                    (
                        f"Candidate {response_id} {reason}. "
                        "The system cannot provide a reliable answer."
                    ),
                )

        if grounding_active and risk_signals.get("weak_grounding") is True:
            return (
                self.REJECT,
                (
                    f"Candidate {response_id} failed the hard grounding guard. "
                    "The selected response is not sufficiently supported by the "
                    "retrieved sources."
                ),
            )

        if not best.get("supporting_source", {}).get("source"):
            return (
                self.REJECT,
                (
                    f"Candidate {response_id} has no supporting source, so the "
                    "system cannot provide a reliable answer."
                ),
            )

        if risk < self.config.accept_threshold:
            return (
                self.ACCEPT,
                (
                    f"Candidate {response_id} combined_risk {risk:.3f} "
                    f"< accept_threshold {self.config.accept_threshold:.3f}."
                ),
            )

        if risk < self.config.rewrite_threshold:
            if not is_primary:
                return (
                    self.RERANK,
                    (
                        f"Candidate {response_id} is safer than the primary "
                        f"candidate and combined_risk {risk:.3f} "
                        f"< rewrite_threshold {self.config.rewrite_threshold:.3f}."
                    ),
                )
            return (
                self.ACCEPT,
                (
                    f"Primary candidate is best available and combined_risk "
                    f"{risk:.3f} < rewrite_threshold "
                    f"{self.config.rewrite_threshold:.3f}."
                ),
            )

        if risk < self.config.reject_threshold:
            if rewrite_attempt < self.config.max_rewrite_attempts:
                return (
                    self.REWRITE,
                    (
                        f"Best candidate combined_risk {risk:.3f} requires "
                        f"rewrite attempt {rewrite_attempt + 1} of "
                        f"{self.config.max_rewrite_attempts}."
                    ),
                )
            return (
                self.REJECT,
                (
                    f"Best candidate combined_risk {risk:.3f} is below reject "
                    "threshold, but max rewrite attempts are exhausted."
                ),
            )

        return (
            self.REJECT,
            (
                f"Best candidate combined_risk {risk:.3f} >= reject_threshold "
                f"{self.config.reject_threshold:.3f}."
            ),
        )

    def _build_rewrite_query(
        self,
        query: str,
        decision: str,
        selected_candidate: dict[str, Any],
    ) -> str | None:
        if decision != self.REWRITE:
            return None

        source = selected_candidate.get("supporting_source", {}).get("source")
        if not source:
            return query
        return f"{query}\n\nFocus specifically on: {source}"

    def route(self, rewrite_attempt: int = 0) -> dict[str, Any]:
        verification_data = self._to_plain(
            load_json(self.config.verification_artifact_path)
        )
        trace_data = self._to_plain(load_json(self.config.traces_artifact_path))

        query = verification_data["query"]
        candidates = verification_data.get("candidates", [])
        if not candidates:
            raise ValueError("No verified candidates found for Stage 6 routing.")

        traces = {
            trace["response_id"]: trace
            for trace in trace_data.get("traces", [])
        }

        logger.info("[Stage 6] Fusing scores for %s candidates", len(candidates))
        fusion_scores = []
        for candidate in candidates:
            response_id = candidate["response_id"]
            trace = traces.get(response_id, {})
            fused = self._fuse_scores(candidate)

            fusion_scores.append({
                "response_id": response_id,
                "text": candidate.get("text", ""),
                "is_primary": bool(candidate.get("is_primary", response_id == 0)),
                "combined_risk": fused["combined_risk"],
                "risk_terms": fused["risk_terms"],
                "effective_weights": fused["effective_weights"],
                "risk_contributions": fused["risk_contributions"],
                "sequence_score": trace.get("sequence_score"),
                "halluguard_score": candidate.get("halluguard_score"),
                "grounding_score": candidate.get("grounding_score"),
                "consistency_score": candidate.get("consistency_score"),
                "judge_score": candidate.get("judge_score"),
                "risk_signals": candidate.get("risk_signals", {}),
                "supporting_source": candidate.get("supporting_source", {}),
            })

        ranked = sorted(
            fusion_scores,
            key=lambda item: (
                item["combined_risk"],
                -self._as_float(item.get("sequence_score"), float("-inf")),
            ),
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate["rank"] = rank

        best = ranked[0]
        decision, decision_reason = self._route(
            best=best,
            rewrite_attempt=rewrite_attempt,
        )
        rewrite_query = self._build_rewrite_query(
            query=query,
            decision=decision,
            selected_candidate=best,
        )

        output = {
            "query": query,
            "decision": decision,
            "selected_response_id": best["response_id"],
            "rewrite_attempt": rewrite_attempt,
            "decision_reason": decision_reason,
            "thresholds": {
                "accept_threshold": self.config.accept_threshold,
                "rewrite_threshold": self.config.rewrite_threshold,
                "reject_threshold": self.config.reject_threshold,
                "max_rewrite_attempts": self.config.max_rewrite_attempts,
            },
            "configured_weights": {
                "halluguard": self.config.weight_halluguard,
                "grounding": self.config.weight_grounding,
                "consistency": self.config.weight_consistency,
                "judge": self.config.weight_judge,
            },
            "source_artifacts": {
                "verification": str(self.config.verification_artifact_path),
                "traces": str(self.config.traces_artifact_path),
            },
            "fusion_scores": ranked,
            "selected_candidate": {
                "response_id": best["response_id"],
                "text": best["text"],
                "is_primary": best["is_primary"],
                "combined_risk": best["combined_risk"],
                "rank": best["rank"],
                "risk_terms": best["risk_terms"],
                "effective_weights": best["effective_weights"],
                "risk_contributions": best["risk_contributions"],
                "supporting_source": best["supporting_source"],
                "risk_signals": best["risk_signals"],
            },
            "rewrite_query": rewrite_query,
        }

        output_path = Path(self.config.fusion_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[Stage 6] Fusion artifact saved: %s", output_path)
        return output
