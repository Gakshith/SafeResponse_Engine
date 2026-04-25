import json
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.saferesponse_engine import logger
from src.saferesponse_engine.entity.config_entity import RetrievalConfig


class RetrievalLayer:
    PHYSICS_TOPICS = [
        "Physics",
        "Classical mechanics",
        "Newton's laws of motion",
        "Momentum",
        "Impulse (physics)",
        "Force",
        "Work (physics)",
        "Energy",
        "Quantum mechanics",
        "Relativity",
    ]

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._vectorstore: FAISS | None = None

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )

    def build_index(self) -> FAISS:
        index_dir = self.config.faiss_index_path
        index_marker = index_dir / "index.faiss"
        metadata_marker = index_dir / "corpus_meta.json"

        expected_topics = self.PHYSICS_TOPICS[: min(self.config.num_articles, len(self.PHYSICS_TOPICS))]
        expected_meta = {
            "topics": expected_topics,
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }

        if index_marker.exists() and metadata_marker.exists():
            try:
                saved_meta = json.loads(metadata_marker.read_text(encoding="utf-8"))
                if saved_meta == expected_meta:
                    logger.info("[Stage 2] Loading cached FAISS index from %s", index_dir)
                    self._vectorstore = FAISS.load_local(
                        str(index_dir),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    return self._vectorstore
            except Exception as exc:
                logger.warning("[Stage 2] Failed to read cached metadata: %s", exc)

        logger.info("[Stage 2] Building new FAISS index")
        raw_docs = self._load_wikipedia_documents(expected_topics)
        if not raw_docs:
            raise RuntimeError("No Wikipedia documents could be loaded.")

        logger.info("[Stage 2] Loaded %d source documents", len(raw_docs))

        chunks = self.splitter.split_documents(raw_docs)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx

        logger.info("[Stage 2] Split into %d chunks", len(chunks))
        logger.info("[Stage 2] Creating FAISS vector store")

        self._vectorstore = FAISS.from_documents(chunks, self.embeddings)

        index_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(index_dir))
        metadata_marker.write_text(
            json.dumps(expected_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("[Stage 2] FAISS index saved to %s", index_dir)
        return self._vectorstore

    def retrieve(self) -> list[dict[str, Any]]:
        vectorstore = self.build_index()
        query = self._read_query()

        logger.info("[Stage 2] Query: %s", query)

        results = vectorstore.similarity_search_with_score(
            query=query,
            k=self.config.top_k,
        )

        retrieved_chunks: list[dict[str, Any]] = []
        for rank, (doc, score) in enumerate(results, start=1):
            retrieved_chunks.append(
                {
                    "query": query,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "topic_area": doc.metadata.get("topic_area", "physics"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "retrieval_rank": rank,
                    "score": float(score),
                    "metadata": doc.metadata,
                }
            )

        self.config.retrieval_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.retrieval_output_path.write_text(
            json.dumps(retrieved_chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "[Stage 2] Saved %d retrieved chunks to %s",
            len(retrieved_chunks),
            self.config.retrieval_output_path,
        )

        return retrieved_chunks

    def _load_wikipedia_documents(self, topics: list[str]) -> list[Document]:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "SafeResponseEngine/1.0 (educational project retrieval pipeline)"
            }
        )

        raw_docs: list[Document] = []

        logger.info("[Stage 2] Fetching %d physics topics from Wikipedia", len(topics))

        for title in tqdm(topics, desc="Fetching Wikipedia topics"):
            try:
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                summary_resp = session.get(summary_url, timeout=20)
                summary_resp.raise_for_status()
                summary_data = summary_resp.json()

                title_text = (summary_data.get("title") or title).strip()
                description_text = (summary_data.get("description") or "").strip()
                summary_text = (summary_data.get("extract") or "").strip()

                extract_resp = session.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "format": "json",
                        "prop": "extracts",
                        "explaintext": 1,
                        "titles": title,
                    },
                    timeout=20,
                )
                extract_resp.raise_for_status()
                extract_data = extract_resp.json()

                pages = extract_data.get("query", {}).get("pages", {})
                full_extract = ""
                for page in pages.values():
                    full_extract = (page.get("extract") or "").strip()
                    if full_extract:
                        break

                page_content_parts = [
                    title_text,
                    description_text,
                    summary_text,
                    full_extract,
                ]
                page_content = "\n\n".join(part for part in page_content_parts if part)

                if not page_content.strip():
                    logger.warning("[Stage 2] Empty content for topic: %s", title)
                    continue

                raw_docs.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "source": title_text,
                            "loader": "wikipedia_api",
                            "topic_area": "physics",
                            "topic_title": title,
                        },
                    )
                )

            except Exception as exc:
                logger.warning("[Stage 2] Failed to fetch '%s': %s", title, exc)

        return raw_docs

    def _read_query(self) -> str:
        raw_text = self.config.query_artifact_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            raise ValueError("Query artifact is empty.")

        try:
            data = json.loads(raw_text)
            if isinstance(data, dict) and "query" in data:
                query = str(data["query"]).strip()
                if not query:
                    raise ValueError("Query field is empty.")
                return query
        except json.JSONDecodeError:
            pass

        return raw_text
