from saferesponse_engine import logger
from saferesponse_engine.components.fusion_decision_router import (
    FusionDecisionRouter,
)
from saferesponse_engine.config.configuration import ConfigurationManager

STAGE_NAME = "Fusion_Decision_Router_STAGE"

class FusionDecisionRouterTrainingPipeline:
    def __init__(self):
        pass

    def _run_rewrite_pass(self, rewrite_query: str) -> None:
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

        cm = ConfigurationManager()
        retrieval_config = cm.get_retrieval_layer_config()
        retrieval_config.query_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        retrieval_config.query_artifact_path.write_text(
            rewrite_query.strip(),
            encoding="utf-8",
        )

        logger.info("[Stage 6] Rerunning Stages 2-5 for rewrite query")
        RetrievalLayerTrainingPipeline().main()
        GenerationLayerTrainingPipeline().main()
        TraceCollectionLayerTrainingPipeline().main()
        VerificationLayerTrainingPipeline().main()

    def main(self, rewrite_attempt: int = 0):
        cm = ConfigurationManager()
        fusion_router_config = cm.get_fusion_router_config()
        fusion_router = FusionDecisionRouter(config=fusion_router_config)
        result = None

        while True:
            result = fusion_router.route(rewrite_attempt=rewrite_attempt)
            decision = result["decision"]
            logger.info("[Stage 6] Decision: %s", decision)

            if decision in (
                FusionDecisionRouter.ACCEPT,
                FusionDecisionRouter.RERANK,
                FusionDecisionRouter.REJECT,
            ):
                break

            if decision == FusionDecisionRouter.REWRITE:
                rewrite_attempt += 1
                rewrite_query = result.get("rewrite_query")
                logger.info(
                    "[Stage 6] Rewrite attempt %s - augmented query: %s",
                    rewrite_attempt,
                    rewrite_query,
                )
                if not rewrite_query:
                    logger.warning(
                        "[Stage 6] Missing rewrite_query; stopping rewrite flow."
                    )
                    break
                self._run_rewrite_pass(rewrite_query)
                continue

        return result
