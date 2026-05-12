"""Retrieve-then-rerank pipeline.

Two-stage retrieval is standard in modern RAG:

    [query]
        │
        ▼
    bi-encoder embedding model  →  vector store  →  top-K candidates (fast, recall-oriented)
        │
        ▼
    cross-encoder reranker      →  top-N final docs        (slow, precision-oriented)

The bi-encoder is fast because query and document are embedded independently.
The cross-encoder is slow because it re-reads the query against every
candidate document, but it's far more accurate. K=8 → N=3 is a sensible
default for short corpora like this one.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

import config


@dataclass
class RetrievedDoc:
    id: str
    text: str
    title: str
    category: str
    score: float  # cross-encoder relevance score


class GuidelineRetriever:
    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(config.INDEX_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        self.collection = client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embed_fn,
        )
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        # Stage 1: bi-encoder candidate retrieval
        raw = self.collection.query(
            query_texts=[query],
            n_results=config.RETRIEVE_K,
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]

        if not docs:
            return []

        # Stage 2: cross-encoder reranking
        pairs = [(query, d) for d in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True
        )[: config.RERANK_K]

        return [
            RetrievedDoc(
                id=i, text=d, title=m["title"], category=m["category"], score=float(s)
            )
            for i, d, m, s in ranked
        ]


if __name__ == "__main__":
    r = GuidelineRetriever()
    query = "Can we say our product cures injuries faster than other brands?"
    print(f"Q: {query}\n")
    for hit in r.retrieve(query):
        print(f"[{hit.score:+.2f}] {hit.title}  ({hit.category})")
        print(f"   {hit.text[:120]}...\n")
