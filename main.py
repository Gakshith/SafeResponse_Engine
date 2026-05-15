import sys
from pathlib import Path
from typing import Callable, Protocol


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from saferesponse_engine import logger
from saferesponse_engine.pipeline.stage_01_user_query import UserQueryTrainingPipeline
from saferesponse_engine.pipeline.stage_02_retrieval_layer import RetrievalLayerTrainingPipeline
from saferesponse_engine.pipeline.stage_03_generation_layer import GenerationLayerTrainingPipeline
from saferesponse_engine.pipeline.stage_04_trace_collection_layer import (
    TraceCollectionLayerTrainingPipeline,
)
from saferesponse_engine.pipeline.stage_05_verification_layer import (
    VerificationLayerTrainingPipeline,
)
from saferesponse_engine.pipeline.stage_06_fusion_decision_router import (
    FusionDecisionRouterTrainingPipeline,
)
from saferesponse_engine.pipeline.stage_07_final_output import FinalOutputPipeline


class PipelineStage(Protocol):
    def main(self) -> None:
        ...


PipelineFactory = Callable[[], PipelineStage]


PIPELINE_STAGES: tuple[tuple[str, PipelineFactory], ...] = (
    ("User Query", UserQueryTrainingPipeline),
    ("Retrieval Layer", RetrievalLayerTrainingPipeline),
    ("Generation Layer", GenerationLayerTrainingPipeline),
    ("Trace Collection Layer", TraceCollectionLayerTrainingPipeline),
    ("Verification Layer", VerificationLayerTrainingPipeline),
    ("Fusion Decision Router", FusionDecisionRouterTrainingPipeline),
    ("Final Output", FinalOutputPipeline),
)


def run_stage(stage_name: str, factory: PipelineFactory) -> None:
    logger.info("Starting %s stage", stage_name)
    try:
        stage = factory()
        stage.main()
    except Exception:
        logger.exception("%s stage failed", stage_name)
        raise
    logger.info("Completed %s stage", stage_name)


def main() -> None:
    for stage_name, factory in PIPELINE_STAGES:
        run_stage(stage_name, factory)


if __name__ == "__main__":
    main()
