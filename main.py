from src.saferesponse_engine import logger
from src.saferesponse_engine.pipeline.stage_01_user_query import UserQueryTrainingPipeline
from src.saferesponse_engine.pipeline.stage_02_retrieval_layer import RetrievalLayerTrainingPipeline
from src.saferesponse_engine.pipeline.stage_03_generation_layer import GenerationLayerTrainingPipeline
from src.saferesponse_engine.pipeline.stage_04_trace_collection_layer import (
    TraceCollectionLayerTrainingPipeline,
)

STAGE_NAME = "User Query stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    user_query = UserQueryTrainingPipeline()
    user_query.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Retrieval Layer stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    retrieval_layer = RetrievalLayerTrainingPipeline()
    retrieval_layer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Generation Layer stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    generation_layer = GenerationLayerTrainingPipeline()
    generation_layer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Trace Collection Layer stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    trace_collection_layer = TraceCollectionLayerTrainingPipeline()
    trace_collection_layer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
