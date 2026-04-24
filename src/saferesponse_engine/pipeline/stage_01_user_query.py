from src.saferesponse_engine import logger
from src.saferesponse_engine.config.configuration import ConfigurationManager
from src.saferesponse_engine.components.user_query import UserQuery
STAGE_NAME = "User_Query_STAGE"


class UserQueryTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        cm = ConfigurationManager()
        user_query_config = cm.get_user_query_config()
        user_query = UserQuery(config=user_query_config)
        user_query.download_file()