"""Build a Chroma vector index over the brand-guidelines corpus.

Pipeline:
    1. Load documents (here: in-process; in prod: from S3 / SharePoint / CMS).
    2. Embed with a sentence-transformer.
    3. Persist to a local Chroma collection.

Re-ranking happens at query time (see ``retriever.py``); we don't bake it
into the index because the cross-encoder is too slow to run on the full
corpus.

Run:
    python build_index.py
"""
from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

import config
from brand_guidelines import as_documents


def build_index() -> None:
    client = chromadb.PersistentClient(path=str(config.INDEX_DIR))

    # Drop and recreate so re-runs are idempotent.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    docs = as_documents()
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    print(f"✅ Indexed {len(docs)} guidelines into '{config.COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()
