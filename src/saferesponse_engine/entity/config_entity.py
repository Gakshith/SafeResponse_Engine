from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserQueryConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path


@dataclass(frozen=True)
class RetrievalConfig:
    root_dir: Path
    query_artifact_path: Path
    faiss_index_path: Path
    retrieval_output_path: Path
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    num_articles: int
    min_score_threshold: float


@dataclass(frozen=True)
class GenerationConfig:
    root_dir: Path
    retrieval_artifact_path: Path
    generation_output_path: Path
    model_name: str
    finetuned_model_path: str | None
    num_candidates: int
    primary_temperature: float
    sample_temperature: float
    max_new_tokens: int
    max_context_length: int


@dataclass(frozen=True)
class TraceCollectionConfig:
    root_dir: Path
    generation_artifact_path: Path
    trace_output_path: Path
    hidden_states_dir: Path
    model_name: str
    max_context_length: int
    collect_hidden_states: bool
    num_hidden_layers_to_save: int


@dataclass(frozen=True)
class VerificationConfig:
    root_dir: Path
    retrieval_artifact_path: Path
    generation_artifact_path: Path
    trace_artifact_path: Path
    verification_output_path: Path
    embedding_model: str
    embedding_backend: str
    enable_halluguard: bool
    enable_ntk: bool
    enable_jacobian_instability: bool
    enable_spectral_conditioning: bool
    enable_grounding_score: bool
    enable_consistency_score: bool
    enable_nli_consistency: bool
    enable_judge: bool
    trace_model_name: str
    nli_model_name: str
    judge_model: str
    halluguard_threshold: float
    grounding_threshold: float
    consistency_threshold: float
