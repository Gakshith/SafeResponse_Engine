from src.saferesponse_engine.utils.common import read_yaml, create_directories
from src.saferesponse_engine.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH, SCHEMA_FILE_PATH
from src.saferesponse_engine.entity.config_entity import (
    UserQueryConfig,
    RetrievalConfig,
    GenerationConfig,
    TraceCollectionConfig,
    VerificationConfig,
)
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

    def get_generation_layer_config(self) -> GenerationConfig:
        config = self.config.generation_layer

        root_dir = Path(config.root_dir)
        retrieval_artifact_path = Path(config.retrieval_artifact_path)
        generation_output_path = Path(config.generation_output_path)
        create_directories([root_dir, generation_output_path.parent])

        return GenerationConfig(
            root_dir=root_dir,
            retrieval_artifact_path=retrieval_artifact_path,
            generation_output_path=generation_output_path,
            model_name=str(config.model_name),
            finetuned_model_path=(
                str(config.finetuned_model_path)
                if config.finetuned_model_path is not None
                else None
            ),
            num_candidates=int(config.num_candidates),
            primary_temperature=float(config.primary_temperature),
            sample_temperature=float(config.sample_temperature),
            max_new_tokens=int(config.max_new_tokens),
            max_context_length=int(config.max_context_length),
        )

    def get_trace_collection_config(self) -> TraceCollectionConfig:
        config = self.config.trace_collection_layer

        root_dir = Path(config.root_dir)
        trace_output_path = Path(config.trace_output_path)
        hidden_states_dir = Path(config.hidden_states_dir)
        create_directories([root_dir, hidden_states_dir, trace_output_path.parent])

        return TraceCollectionConfig(
            root_dir=root_dir,
            generation_artifact_path=Path(config.generation_artifact_path),
            trace_output_path=trace_output_path,
            hidden_states_dir=hidden_states_dir,
            model_name=str(config.model_name),
            max_context_length=int(config.max_context_length),
            collect_hidden_states=bool(config.collect_hidden_states),
            num_hidden_layers_to_save=int(config.num_hidden_layers_to_save),
        )

    def get_verification_config(self) -> VerificationConfig:
        config = self.config.verification_layer

        root_dir = Path(config.root_dir)
        verification_output_path = Path(config.verification_output_path)
        create_directories([root_dir, verification_output_path.parent])

        return VerificationConfig(
            root_dir=root_dir,
            retrieval_artifact_path=Path(config.retrieval_artifact_path),
            generation_artifact_path=Path(config.generation_artifact_path),
            trace_artifact_path=Path(config.trace_artifact_path),
            verification_output_path=verification_output_path,
            embedding_model=str(config.embedding_model),
            embedding_backend=str(getattr(config, "embedding_backend", "lexical")),
            enable_halluguard=bool(config.enable_halluguard),
            enable_ntk=bool(getattr(config, "enable_ntk", True)),
            enable_jacobian_instability=bool(
                getattr(config, "enable_jacobian_instability", False)
            ),
            enable_spectral_conditioning=bool(
                getattr(config, "enable_spectral_conditioning", True)
            ),
            enable_grounding_score=bool(config.enable_grounding_score),
            enable_consistency_score=bool(config.enable_consistency_score),
            enable_nli_consistency=bool(
                getattr(config, "enable_nli_consistency", False)
            ),
            enable_judge=bool(config.enable_judge),
            trace_model_name=str(
                getattr(config, "trace_model_name", self.config.trace_collection_layer.model_name)
            ),
            nli_model_name=str(
                getattr(config, "nli_model_name", "cross-encoder/nli-deberta-v3-small")
            ),
            judge_model=str(getattr(config, "judge_model", "gpt-4o-mini")),
            halluguard_threshold=float(config.halluguard_threshold),
            grounding_threshold=float(config.grounding_threshold),
            consistency_threshold=float(config.consistency_threshold),
        )
