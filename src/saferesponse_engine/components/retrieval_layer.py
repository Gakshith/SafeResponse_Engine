from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
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
from saferesponse_engine import logger
from saferesponse_engine.entity.config_entity import RetrievalConfig
from saferesponse_engine.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH, SCHEMA_FILE_PATH
from saferesponse_engine.utils.common import create_directories, read_yaml

_EMBEDDING_CACHE: dict[str, HuggingFaceEmbeddings] = {}
_VECTORSTORE_CACHE: dict[tuple[str, str, int], FAISS] = {}


class RetrievalLayer:

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.retrieval_backend = config.retrieval_backend.lower()
        self.embeddings = None
        if self.retrieval_backend == "lexical":
            logger.info("[Stage 2] Using fast lexical retrieval backend")
        elif config.embedding_model in _EMBEDDING_CACHE:
            logger.info(
                "[Stage 2] Loading embedding model from cache: %s",
                config.embedding_model,
            )
            self.embeddings = _EMBEDDING_CACHE[config.embedding_model]
        else:
            logger.info(
                "[Stage 2] Loading BGE-M3 embedding model: %s",
                config.embedding_model,
            )
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                encode_kwargs={"normalize_embeddings": True}
            )
            _EMBEDDING_CACHE[config.embedding_model] = self.embeddings
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
        cache_key = (
            str(index_dir.resolve()),
            self.config.embedding_model,
            self.config.num_articles,
        )
        expected_metadata = {
            "corpus":        str(self.config.local_corpus_path),
            "embedding":     self.config.embedding_model,
            "num_articles":  self.config.num_articles,
        }

        if cache_key in _VECTORSTORE_CACHE:
            logger.info("[Stage 2] Loading FAISS index from memory cache")
            return _VECTORSTORE_CACHE[cache_key]

        if index_marker.exists() and metadata_path.exists():
            stored_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            if stored_metadata == expected_metadata:
                logger.info(
                    "[Stage 2] Loading existing FAISS index from %s", index_dir
                )
                vectorstore = FAISS.load_local(
                    str(index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                _VECTORSTORE_CACHE[cache_key] = vectorstore
                return vectorstore

        raw_docs = self._load_corpus_documents()
        if not raw_docs:
            raise RuntimeError("Unable to build the retrieval corpus.")
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
        _VECTORSTORE_CACHE[cache_key] = vectorstore
        return vectorstore

    def retrieve(self) -> list[dict[str, Any]]:
        query = self._read_query()
        logger.info("[Stage 2] Query: %s", query)
        if self._is_live_fact_query(query):
            logger.warning(
                "[Stage 2] Rejecting live/current fact query before retrieval: %s",
                query,
            )
            self._write_output(
                query=query,
                retrieved_chunks=[],
                raw_scores=[],
                score_stats={"min": None, "max": None, "mean": None},
                query_guard={
                    "triggered": True,
                    "reason": "live_fact_query",
                    "message": (
                        "The query appears to ask for current or live state, "
                        "which is outside the controlled static corpus."
                    ),
                },
            )
            return []
        if self.retrieval_backend == "lexical":
            return self._retrieve_lexical(query)

        vectorstore = self.build_index()
        results = vectorstore.similarity_search_with_score(
            query, k=self.config.top_k
        )
        retrieved_chunks: list[dict[str, Any]] = []
        raw_scores = [round(float(score), 6) for _, score in results]
        for rank, (doc, score) in enumerate(results, start=1):
            score = float(score)
            if score > self.config.min_score_threshold:
                logger.info(
                    "[Stage 2] Dropping low-relevance chunk rank=%s source=%s score=%.6f threshold=%.6f",
                    rank,
                    doc.metadata.get("source", "unknown"),
                    score,
                    self.config.min_score_threshold,
                )
                continue
            retrieved_chunks.append({
                "content":        doc.page_content,
                "source":         doc.metadata.get("source", "unknown"),
                "chunk_id":       doc.metadata.get("chunk_id"),
                "char_count":     doc.metadata.get("char_count"),
                "content_hash":   doc.metadata.get("content_hash"),
                "retrieval_rank": rank,
                "score":          round(score, 6),
                "metadata":       doc.metadata,
            })
        scores = [c["score"] for c in retrieved_chunks]
        if scores:
            score_stats = {
                "min":  round(min(scores), 6),
                "max":  round(max(scores), 6),
                "mean": round(sum(scores) / len(scores), 6),
            }
        else:
            score_stats = {"min": None, "max": None, "mean": None}
            logger.warning(
                "[Stage 2] No retrieved chunks passed min_score_threshold=%.6f. Raw scores: %s",
                self.config.min_score_threshold,
                raw_scores,
            )
        self._write_output(
            query=query,
            retrieved_chunks=retrieved_chunks,
            raw_scores=raw_scores,
            score_stats=score_stats,
        )
        return retrieved_chunks

    def _retrieve_lexical(self, query: str) -> list[dict[str, Any]]:
        raw_docs = self._load_corpus_documents()
        chunks = self.splitter.split_documents(raw_docs)
        query_tokens = self._tokens(query)
        min_required_matches = (
            min(self.config.min_lexical_matches, len(query_tokens))
            if query_tokens
            else 0
        )

        scored_chunks = []
        for chunk_id, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["char_count"] = len(chunk.page_content)
            chunk.metadata["content_hash"] = hashlib.sha256(
                chunk.page_content.encode()
            ).hexdigest()[:16]
            chunk_tokens = self._tokens(chunk.page_content)
            matched_count = self._token_match_count(query_tokens, chunk_tokens)
            overlap = self._token_overlap(query_tokens, matched_count)
            score = round(1.0 - overlap, 6)
            scored_chunks.append((score, matched_count, chunk))

        scored_chunks.sort(key=lambda item: item[0])
        top_results = scored_chunks[: self.config.top_k]
        raw_scores = [score for score, _, _ in top_results]
        retrieved_chunks = []
        for rank, (score, matched_count, doc) in enumerate(top_results, start=1):
            if matched_count < min_required_matches:
                logger.info(
                    "[Stage 2] Dropping weak lexical chunk rank=%s source=%s matches=%s required=%s",
                    rank,
                    doc.metadata.get("source", "unknown"),
                    matched_count,
                    min_required_matches,
                )
                continue
            if score > self.config.min_score_threshold:
                logger.info(
                    "[Stage 2] Dropping low-relevance lexical chunk rank=%s source=%s score=%.6f threshold=%.6f",
                    rank,
                    doc.metadata.get("source", "unknown"),
                    score,
                    self.config.min_score_threshold,
                )
                continue
            retrieved_chunks.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "char_count": doc.metadata.get("char_count"),
                "content_hash": doc.metadata.get("content_hash"),
                "retrieval_rank": rank,
                "score": score,
                "lexical_matched_tokens": matched_count,
                "lexical_required_matches": min_required_matches,
                "metadata": doc.metadata,
            })

        scores = [chunk["score"] for chunk in retrieved_chunks]
        if scores:
            score_stats = {
                "min": round(min(scores), 6),
                "max": round(max(scores), 6),
                "mean": round(sum(scores) / len(scores), 6),
            }
        else:
            score_stats = {"min": None, "max": None, "mean": None}
            logger.warning(
                "[Stage 2] No lexical chunks passed min_score_threshold=%.6f. Raw scores: %s",
                self.config.min_score_threshold,
                raw_scores,
            )

        self._write_output(
            query=query,
            retrieved_chunks=retrieved_chunks,
            raw_scores=raw_scores,
            score_stats=score_stats,
        )
        return retrieved_chunks

    def _write_output(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        raw_scores: list[float],
        score_stats: dict[str, Any],
        query_guard: dict[str, Any] | None = None,
    ) -> None:
        output = {
            "query":            query,
            "retrieval_backend": self.retrieval_backend,
            "embedding_model":  self.config.embedding_model,
            "top_k":            self.config.top_k,
            "min_score_threshold": self.config.min_score_threshold,
            "raw_scores":       raw_scores,
            "score_stats":      score_stats,
            "chunks":           retrieved_chunks,
        }
        if query_guard:
            output["query_guard"] = query_guard
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

    def _load_corpus_documents(self) -> list[Document]:
        if self.config.local_corpus_path and self.config.local_corpus_path.exists():
            corpus = json.loads(
                self.config.local_corpus_path.read_text(encoding="utf-8")
            )
            return [
                Document(
                    page_content=item["text"],
                    metadata={
                        "source": item["title"],
                        "loader": "local_demo_corpus",
                        "topic_area": "controlled_demo",
                    },
                )
                for item in corpus
            ]
        return self._load_wikipedia_documents()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "briefly", "did", "do", "does", "give", "how", "i", "in", "is", "it", "me", "of",
            "on", "or", "that", "the", "this", "to", "us", "was", "we",
            "what", "who", "why", "with", "you", "about", "tell", "short",
            "summary",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in stopwords and len(token) > 1
        }

    @staticmethod
    def _is_live_fact_query(query: str) -> bool:
        normalized = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
        if re.search(
            r"\b(current|currently|latest|now|today|present|incumbent|"
            r"right now|as of)\b",
            normalized,
        ):
            return True

        historical_markers = re.search(
            r"\b(1st|2nd|3rd|[0-9]+th|former|past|served|in [0-9]{3,4}|"
            r"during|between|from [0-9]{3,4})\b",
            normalized,
        )
        live_role_question = re.search(
            r"\bwho is the (president|ceo|chief executive|prime minister|"
            r"governor|mayor|chair|director|leader|head)\b",
            normalized,
        )
        return bool(live_role_question and historical_markers is None)

    @classmethod
    def _token_match_count(
        cls,
        query_tokens: set[str],
        chunk_tokens: set[str],
    ) -> int:
        matched = 0
        for query_token in query_tokens:
            if query_token in chunk_tokens:
                matched += 1
                continue
            if cls._has_fuzzy_match(query_token, chunk_tokens):
                matched += 1
        return matched

    @staticmethod
    def _token_overlap(
        query_tokens: set[str],
        matched_count: int,
    ) -> float:
        if not query_tokens:
            return 0.0
        return matched_count / len(query_tokens)

    @staticmethod
    def _has_fuzzy_match(query_token: str, chunk_tokens: set[str]) -> bool:
        if len(query_token) < 5:
            return False
        for chunk_token in chunk_tokens:
            if abs(len(query_token) - len(chunk_token)) > 3:
                continue
            if SequenceMatcher(None, query_token, chunk_token).ratio() >= 0.78:
                return True
        return False

    def _read_query(self) -> str:
        query = self.config.query_artifact_path.read_text(encoding="utf-8").strip()
        if not query:
            raise ValueError("Query artifact is empty.")
        return query
