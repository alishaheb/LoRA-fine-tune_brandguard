"""Lightweight RAG evaluation: hit-rate and MRR on a tiny relevance set.

For a corpus this small (15 guidelines) full RAGAS is overkill. Instead we
hand-label a few queries with the guideline IDs we *expect* to be retrieved,
and measure:

    - Hit@K: did we retrieve at least one relevant doc in the top-K?
    - MRR:   how high up the first relevant doc was (1/rank), averaged.

In a real project you'd scale this to ~100-500 queries built from real
analyst questions, and add RAGAS faithfulness / answer-relevance for the
generation stage.
"""
from __future__ import annotations

from retriever import GuidelineRetriever


# (query, set of relevant guideline IDs)
EVAL_SET: list[tuple[str, set[str]]] = [
    ("Can we say our product cures injuries?", {"claims-002"}),
    ("Are we allowed to name Nike in our ads?", {"comp-001"}),
    ("What words should we avoid using?", {"voice-002", "voice-003"}),
    ("How should we depict cyclists in photos?", {"img-001"}),
    ("Are we allowed to say 'eco-friendly'?", {"claims-003"}),
    ("Can children appear in the campaign?", {"legal-002"}),
    ("Rules about discount language?", {"legal-001"}),
    ("How should we frame mental health benefits?", {"incl-003"}),
]


def evaluate(k: int = 3) -> None:
    r = GuidelineRetriever()
    hits, mrrs = [], []
    for query, relevant in EVAL_SET:
        retrieved = [doc.id for doc in r.retrieve(query)]
        hit = any(rid in relevant for rid in retrieved[:k])
        hits.append(int(hit))

        rr = 0.0
        for rank, rid in enumerate(retrieved[:k], start=1):
            if rid in relevant:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        mark = "✓" if hit else "✗"
        print(f"{mark} {query!r}")
        print(f"   retrieved: {retrieved[:k]}  relevant: {relevant}")

    print(f"\nHit@{k}: {sum(hits) / len(hits):.2%}")
    print(f"MRR@{k}: {sum(mrrs) / len(mrrs):.3f}")


if __name__ == "__main__":
    evaluate()
