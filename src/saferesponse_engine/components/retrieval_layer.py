from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import re
from typing import Any
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from src.saferesponse_engine import logger
from src.saferesponse_engine.entity.config_entity import RetrievalConfig
from src.saferesponse_engine.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH, SCHEMA_FILE_PATH
from src.saferesponse_engine.utils.common import create_directories, read_yaml

class RetrievalLayer:

    def __init__(self, config: RetrievalConfig):
        self.config = config
        logger.info("[Stage 2] Loading BGE-M3 embedding model: %s", config.embedding_model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,          # "BAAI/bge-m3"
            encode_kwargs={"normalize_embeddings": True}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )

    def build_index(self) -> FAISS:
        index_dir     = Path(self.config.faiss_index_path)
        index_marker  = index_dir / "index.faiss"
        metadata_path = index_dir / "index_metadata.json"
        expected_metadata = {
            "corpus":        "wikipedia_20220301_en",
            "embedding":     "bge_m3_1024dim",
            "num_articles":  self.config.num_articles,
        }

        if index_marker.exists() and metadata_path.exists():
            stored_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if stored_metadata == expected_metadata:
                logger.info(
                    "[Stage 2] Loading existing FAISS index from %s", index_dir
                )
                return FAISS.load_local(
                    str(index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

        raw_docs = self._load_wikipedia_documents()
        if not raw_docs:
            raise RuntimeError("Unable to build the Wikipedia retrieval corpus.")
        chunks = self.splitter.split_documents(raw_docs)
        for chunk_id, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]     = chunk_id
            chunk.metadata["char_count"]   = len(chunk.page_content)
            chunk.metadata["content_hash"] = hashlib.sha256(
                chunk.page_content.encode()
            ).hexdigest()[:16]
        logger.info(
            "[Stage 2] Building FAISS vector store with %s chunks", len(chunks)
        )
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_dir))
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2), encoding="utf-8"
        )
        logger.info("[Stage 2] FAISS index saved to %s", index_dir)
        return vectorstore

    def retrieve(self) -> list[dict[str, Any]]:
        vectorstore = self.build_index()
        query = self._read_query()
        logger.info("[Stage 2] Query: %s", query)
        results = vectorstore.similarity_search_with_score(
            query, k=self.config.top_k
        )
        retrieved_chunks: list[dict[str, Any]] = []
        for rank, (doc, score) in enumerate(results, start=1):
            retrieved_chunks.append({
                "content":        doc.page_content,
                "source":         doc.metadata.get("source", "unknown"),
                "chunk_id":       doc.metadata.get("chunk_id"),
                "char_count":     doc.metadata.get("char_count"),
                "content_hash":   doc.metadata.get("content_hash"),
                "retrieval_rank": rank,
                "score":          round(float(score), 6),
                "metadata":       doc.metadata,
            })
        scores = [c["score"] for c in retrieved_chunks]
        score_stats = {
            "min":  round(min(scores), 6),
            "max":  round(max(scores), 6),
            "mean": round(sum(scores) / len(scores), 6),
        }
        output = {
            "query":            query,
            "embedding_model":  self.config.embedding_model,
            "top_k":            self.config.top_k,
            "score_stats":      score_stats,
            "chunks":           retrieved_chunks,
        }
        self.config.retrieval_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.retrieval_output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "[Stage 2] Saved %s chunks to %s | scores: %s",
            len(retrieved_chunks),
            self.config.retrieval_output_path,
            score_stats,
        )
        return retrieved_chunks

    def _load_wikipedia_documents(self) -> list[Document]:
        logger.info(
            "[Stage 2] Loading %s Wikipedia articles...", self.config.num_articles
        )
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split=f"train[:{self.config.num_articles}]",
            trust_remote_code=True,
        )
        raw_docs = [
            Document(
                page_content=article["text"],
                metadata={
                    "source":     article["title"],
                    "loader":     "wikipedia_huggingface",
                    "topic_area": "general",
                }
            )
            for article in tqdm(dataset, desc="Converting Wikipedia articles")
        ]
        logger.info("[Stage 2] Loaded %s Wikipedia articles.", len(raw_docs))
        return raw_docs
    def _read_query(self) -> str:
        query = self.config.query_artifact_path.read_text(encoding="utf-8").strip()
        if not query:
            raise ValueError("Query artifact is empty.")
        return query