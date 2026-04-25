from src.saferesponse_engine.utils.common import read_yaml, create_directories
from src.saferesponse_engine.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH, SCHEMA_FILE_PATH
from src.saferesponse_engine.entity.config_entity import UserQueryConfig, RetrievalConfig
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

    def get_retrieval_layer_config(self) -> RetrievalConfig:
        config = self.config.retrieval_layer

        root_dir = Path(config.root_dir)
        faiss_index_path = Path(config.faiss_index_path)
        retrieval_output_path = Path(config.retrieval_output_path)
        create_directories([root_dir, faiss_index_path, retrieval_output_path.parent])

        return RetrievalConfig(
            root_dir=root_dir,
            query_artifact_path=Path(config.query_artifact_path),
            faiss_index_path=faiss_index_path,
            retrieval_output_path=retrieval_output_path,
            embedding_model=str(config.embedding_model),
            top_k=int(config.top_k),
            chunk_size=int(config.chunk_size),
            chunk_overlap=int(config.chunk_overlap),
            num_articles=int(config.num_articles),
            min_score_threshold=float(config.min_score_threshold),
        )
