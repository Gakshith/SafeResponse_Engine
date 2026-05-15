import json
import re
import fcntl
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from saferesponse_engine.components.conversation_memory import (
    ConversationMemoryLayer,
)
from saferesponse_engine.config.configuration import ConfigurationManager
from saferesponse_engine.utils.common import load_json


PIPELINE_LOCK = threading.Lock()


class PipelineBusyError(RuntimeError):
    pass


class SafeResponseChatEngine:
    """
    Backend chat model/orchestrator.

    It is not a newly trained LLM. It is the decision layer that decides whether
    to answer directly, run the single-turn SafeResponse pipeline, or run the
    memory-augmented SafeResponse pipeline for longer chats.
    """

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.memory = ConversationMemoryLayer(
            config=self.config_manager.get_conversation_memory_config()
        )

    @staticmethod
    def _plain(value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return SafeResponseChatEngine._plain(value.to_dict())
        if isinstance(value, list):
            return [SafeResponseChatEngine._plain(item) for item in value]
        if isinstance(value, dict):
            return {
                key: SafeResponseChatEngine._plain(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _classify_intent(self, message: str) -> str:
        normalized = self._normalize(message)
        stripped = normalized.strip(" .!?")
        greetings = {
            "hi",
            "hii",
            "hello",
            "hey",
            "yo",
            "sup",
            "what's up",
            "whats up",
            "good morning",
            "good afternoon",
            "good evening",
        }
        help_phrases = {
            "help",
            "what can you do",
            "how does this work",
            "what is this",
        }

        if stripped in greetings:
            return "small_talk"
        if stripped in help_phrases:
            return "help"
        return "question"

    def _direct_response(
        self,
        message: str,
        conversation_id: str,
        mode: str,
        intent: str,
        memory_context: dict[str, Any],
    ) -> dict[str, Any]:
        if intent == "help":
            answer = (
                "Ask a factual question. I will retrieve from the controlled "
                "10-article corpus, use conversation memory when this is a "
                "long chat, and reject answers that are not supported."
            )
        else:
            answer = (
                "Hi. Ask a factual question and I will verify it before returning "
                "an answer."
            )

        response = {
            "query": message,
            "decision": "DIRECT",
            "answer": answer,
            "confidence": "HIGH",
            "confidence_color": "green",
            "source": None,
            "risk_score": 0.0,
            "warnings": [],
            "risk_explanation": "No retrieval was needed for this chat turn.",
            "decision_reason": f"Handled as {intent} without running model inference.",
            "formatted_response": answer,
            "pipeline_summary": {
                "mode": mode,
                "intent": intent,
                "pipeline_run": False,
            },
            "conversation_id": conversation_id,
            "mode": mode,
            "intent": intent,
            "memory_summary": memory_context["summary"],
            "relevant_turn_ids": [
                turn["turn_id"]
                for turn in memory_context["relevant_turns"]
            ],
        }
        return response

    def final_output_path(self) -> Path:
        return self.config_manager.get_final_output_config().final_output_path

    def load_final_output(self) -> dict[str, Any]:
        path = self.final_output_path()
        if not path.exists():
            raise FileNotFoundError(
                "Final response artifact not found. Run Stage 7 or run the pipeline."
            )
        return self._plain(load_json(path))

    def api_response(self, output: dict[str, Any]) -> dict[str, Any]:
        final_response = output.get("final_response", {})
        source_citation = final_response.get("source_citation") or {}
        source_title = source_citation.get("title")
        source = f"{source_title} (Wikipedia)" if source_title else None

        return {
            "query": output.get("query"),
            "decision": output.get("decision"),
            "answer": final_response.get("answer"),
            "confidence": final_response.get("confidence_tag"),
            "confidence_color": final_response.get("confidence_color"),
            "source": source,
            "risk_score": final_response.get("confidence_score"),
            "warnings": final_response.get("risk_signals_fired", []),
            "risk_explanation": final_response.get("risk_explanation"),
            "decision_reason": output.get("decision_reason"),
            "formatted_response": final_response.get("formatted_response"),
            "pipeline_summary": output.get("pipeline_summary", {}),
        }

    def _write_query_artifact(self, query: str) -> None:
        retrieval_config = self.config_manager.get_retrieval_layer_config()
        retrieval_config.query_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        retrieval_config.query_artifact_path.write_text(
            query.strip(),
            encoding="utf-8",
        )

    def _inject_memory_context(self, memory_context: str) -> None:
        if not memory_context.strip():
            return

        retrieval_config = self.config_manager.get_retrieval_layer_config()
        output_path = retrieval_config.retrieval_output_path
        retrieval_data = json.loads(output_path.read_text(encoding="utf-8"))
        retrieval_data["memory_context"] = memory_context
        output_path.write_text(
            json.dumps(retrieval_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _run_timed_stage(
        name: str,
        action,
        stage_timings: dict[str, float],
    ) -> None:
        started_at = time.perf_counter()
        action()
        stage_timings[name] = round(time.perf_counter() - started_at, 6)

    def _attach_stage_timings(
        self,
        output: dict[str, Any],
        stage_timings: dict[str, float],
    ) -> dict[str, Any]:
        pipeline_summary = output.setdefault("pipeline_summary", {})
        pipeline_summary["stage_timings_seconds"] = stage_timings
        pipeline_summary["total_stage_time_seconds"] = round(
            sum(stage_timings.values()),
            6,
        )
        self.final_output_path().write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output

    @contextmanager
    def _pipeline_process_lock(self):
        lock_path = Path(self.config_manager.config.artifacts_root) / ".pipeline.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PipelineBusyError(
                    "A SafeResponse pipeline run is already in progress in another process. "
                    "Try again later."
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def run_pipeline_for_query(
        self,
        query: str,
        memory_context: str = "",
    ) -> dict[str, Any]:
        from saferesponse_engine.pipeline.stage_02_retrieval_layer import (
            RetrievalLayerTrainingPipeline,
        )
        from saferesponse_engine.pipeline.stage_03_generation_layer import (
            GenerationLayerTrainingPipeline,
        )
        from saferesponse_engine.pipeline.stage_04_trace_collection_layer import (
            TraceCollectionLayerTrainingPipeline,
        )
        from saferesponse_engine.pipeline.stage_05_verification_layer import (
            VerificationLayerTrainingPipeline,
        )
        from saferesponse_engine.pipeline.stage_06_fusion_decision_router import (
            FusionDecisionRouterTrainingPipeline,
        )
        from saferesponse_engine.pipeline.stage_07_final_output import (
            FinalOutputPipeline,
        )

        if not PIPELINE_LOCK.acquire(blocking=False):
            raise PipelineBusyError(
                "A SafeResponse pipeline run is already in progress. Try again later."
            )

        try:
            with self._pipeline_process_lock():
                stage_timings: dict[str, float] = {}
                self._write_query_artifact(query)
                self._run_timed_stage(
                    "retrieval",
                    RetrievalLayerTrainingPipeline().main,
                    stage_timings,
                )
                self._inject_memory_context(memory_context)
                self._run_timed_stage(
                    "generation",
                    GenerationLayerTrainingPipeline().main,
                    stage_timings,
                )
                self._run_timed_stage(
                    "trace_collection",
                    TraceCollectionLayerTrainingPipeline().main,
                    stage_timings,
                )
                self._run_timed_stage(
                    "verification",
                    VerificationLayerTrainingPipeline().main,
                    stage_timings,
                )
                self._run_timed_stage(
                    "fusion_router",
                    FusionDecisionRouterTrainingPipeline().main,
                    stage_timings,
                )
                self._run_timed_stage(
                    "final_output",
                    FinalOutputPipeline().main,
                    stage_timings,
                )
                return self._attach_stage_timings(
                    output=self.load_final_output(),
                    stage_timings=stage_timings,
                )
        finally:
            PIPELINE_LOCK.release()

    def query(self, query: str, run_pipeline: bool = False) -> dict[str, Any]:
        intent = self._classify_intent(query)
        if intent != "question":
            return self._direct_response(
                message=query,
                conversation_id="single-query",
                mode="single_turn",
                intent=intent,
                memory_context={
                    "summary": "",
                    "relevant_turns": [],
                },
            )

        output = self.run_pipeline_for_query(query) if run_pipeline else self.load_final_output()
        return self.api_response(output)

    def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        run_pipeline: bool = True,
    ) -> dict[str, Any]:
        conversation_id = conversation_id or str(uuid.uuid4())
        memory_context = self.memory.build_context(
            conversation_id=conversation_id,
            query=message,
        )
        mode = memory_context["mode"]
        intent = self._classify_intent(message)

        if intent != "question":
            response = self._direct_response(
                message=message,
                conversation_id=conversation_id,
                mode=mode,
                intent=intent,
                memory_context=memory_context,
            )
        else:
            if run_pipeline:
                output = self.run_pipeline_for_query(
                    query=message,
                    memory_context=memory_context["memory_context"],
                )
            else:
                output = self.load_final_output()

            response = self.api_response(output)
            response.update({
                "conversation_id": conversation_id,
                "mode": mode,
                "intent": intent,
                "memory_summary": memory_context["summary"],
                "relevant_turn_ids": [
                    turn["turn_id"]
                    for turn in memory_context["relevant_turns"]
                ],
            })

        self.memory.append_turn(
            conversation_id=conversation_id,
            role="user",
            text=message,
            metadata={"mode": mode, "intent": intent},
        )
        self.memory.append_turn(
            conversation_id=conversation_id,
            role="assistant",
            text=str(response.get("answer", "")),
            metadata={
                "mode": mode,
                "intent": intent,
                "decision": response.get("decision"),
                "confidence": response.get("confidence"),
                "warnings": response.get("warnings", []),
            },
        )
        return response

    def status(self) -> dict[str, Any]:
        final_output_path = self.final_output_path()
        return {
            "status": "ok",
            "pipeline_busy": PIPELINE_LOCK.locked(),
            "final_output_exists": final_output_path.exists(),
            "final_output_path": str(final_output_path),
        }
