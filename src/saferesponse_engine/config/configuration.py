from src.saferesponse_engine.utils.common import read_yaml, create_directories
from src.saferesponse_engine.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH, SCHEMA_FILE_PATH
from src.saferesponse_engine.entity.config_entity import UserQueryConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAM_FILE_PATH)
        self.schema = read_yaml(SCHEMA_FILE_PATH)
        create_directories([Path(self.config.artifacts_root)])

    def get_user_query_config(self) -> UserQueryConfig:
        config = self.config.user_query

        root_dir = Path(config.root_dir)
        create_directories([root_dir])

        return UserQueryConfig(
            root_dir=root_dir,
            source_url=str(config.source_url),
            local_data_file=Path(config.local_data_file),
        )