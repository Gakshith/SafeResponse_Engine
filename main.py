from src.saferesponse_engine import logger
from src.saferesponse_engine.pipeline.stage_01_user_query import UserQueryTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   user_query = UserQueryTrainingPipeline()
   user_query.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

