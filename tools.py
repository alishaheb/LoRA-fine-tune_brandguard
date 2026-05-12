"""Tools the agent can call.

LangGraph's prebuilt ReAct agent dispatches to tools based on the LLM's
tool-calling output. Each tool is a plain Python function decorated with
``@tool`` and a clear docstring — the docstring becomes the tool description
the LLM sees, so write them like API docs.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool

from inference import BrandSafetyClassifier
from retriever import GuidelineRetriever


# Lazy singletons — these models are expensive to load. We only load them
# once per process, on first use.

@lru_cache(maxsize=1)
def _classifier() -> BrandSafetyClassifier:
    return BrandSafetyClassifier()


@lru_cache(maxsize=1)
def _retriever() -> GuidelineRetriever:
    return GuidelineRetriever()


@tool
def classify_brand_safety(copy: str) -> dict:
    """Classify a piece of marketing copy as SAFE or UNSAFE for a mainstream
    consumer brand. Returns a label and the model's confidence.

    Use this tool on every piece of copy you review. It catches offensive,
    aggressive, profane, or hateful language.

    Args:
        copy: The marketing copy to classify (a short string, usually <500 chars).
    """
    v = _classifier().classify(copy)
    return {"label": v.label, "confidence": round(v.confidence, 3)}


@tool
def retrieve_brand_guidelines(query: str) -> list[dict]:
    """Retrieve the most relevant brand guidelines for a query.

    Use this whenever the copy mentions performance claims, prices,
    competitors, health/medical effects, sustainability, imagery, minors,
    or anything where a written brand rule might apply. Call it multiple
    times with different queries to cover different aspects of the copy.

    Args:
        query: A natural-language description of what to look up
               (e.g. 'rules about competitor mentions', 'health claims').

    Returns: A list of guideline dicts with title, category and content.
    """
    hits = _retriever().retrieve(query)
    return [
        {
            "id": h.id,
            "title": h.title,
            "category": h.category,
            "content": h.text,
            "relevance": round(h.score, 3),
        }
        for h in hits
    ]


ALL_TOOLS = [classify_brand_safety, retrieve_brand_guidelines]
