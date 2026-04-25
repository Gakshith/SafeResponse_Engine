from src.saferesponse_engine import logger
from src.saferesponse_engine.pipeline.stage_01_user_query import UserQueryTrainingPipeline
from src.saferesponse_engine.pipeline.stage_02_retrieval_layer import RetrievalLayerTrainingPipeline

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
