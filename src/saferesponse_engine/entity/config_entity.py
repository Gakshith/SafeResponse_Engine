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
