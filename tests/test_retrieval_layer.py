from pathlib import Path

from saferesponse_engine.components.retrieval_layer import RetrievalLayer
from saferesponse_engine.entity.config_entity import RetrievalConfig


def _retrieval_config(tmp_path: Path) -> RetrievalConfig:
    repo_root = Path(__file__).resolve().parents[1]
    query_path = tmp_path / "query.txt"
    return RetrievalConfig(
        root_dir=tmp_path,
        query_artifact_path=query_path,
        faiss_index_path=tmp_path / "faiss_index",
        retrieval_output_path=tmp_path / "retrieved_chunks.json",
        retrieval_backend="lexical",
        local_corpus_path=repo_root / "data" / "demo_corpus.json",
        embedding_model="BAAI/bge-m3",
        top_k=3,
        chunk_size=500,
        chunk_overlap=100,
        num_articles=10,
        min_score_threshold=0.95,
        min_lexical_matches=2,
    )


def test_lexical_retrieval_rejects_single_accidental_match(tmp_path):
    config = _retrieval_config(tmp_path)
    config.query_artifact_path.write_text(
        "Who is the current CEO of a company that is not in the corpus?",
        encoding="utf-8",
    )

    chunks = RetrievalLayer(config).retrieve()

    assert chunks == []


def test_lexical_retrieval_rejects_live_office_query(tmp_path):
    config = _retrieval_config(tmp_path)
    config.query_artifact_path.write_text(
        "Who is the president of the United States?",
        encoding="utf-8",
    )

    chunks = RetrievalLayer(config).retrieve()

    assert chunks == []


def test_lexical_retrieval_keeps_supported_misspelled_entity(tmp_path):
    config = _retrieval_config(tmp_path)
    config.query_artifact_path.write_text(
        "Give me about the Abram Lilncoln.",
        encoding="utf-8",
    )

    chunks = RetrievalLayer(config).retrieve()

    assert chunks
    assert chunks[0]["source"] == "Abraham Lincoln"
    assert chunks[0]["lexical_matched_tokens"] >= 2
