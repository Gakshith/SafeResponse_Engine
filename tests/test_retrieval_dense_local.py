import inspect

from saferesponse_engine.components.retrieval_layer import RetrievalLayer


def test_build_index_uses_local_corpus():
    src = inspect.getsource(RetrievalLayer.build_index)
    assert "_load_corpus_documents" in src, (
        "dense index must build from the local corpus loader, not the HF download"
    )
    assert "_load_wikipedia_documents" not in src, (
        "dense index should no longer hardcode the HF wikipedia download path"
    )


def test_build_index_metadata_is_config_driven():
    src = inspect.getsource(RetrievalLayer.build_index)
    assert "bge_m3_1024dim" not in src, "index metadata must not hardcode bge-m3"
    assert "self.config.embedding_model" in src, (
        "index metadata should reflect the configured embedding model"
    )
