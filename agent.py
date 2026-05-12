"""BrandGuard agent: a LangGraph ReAct loop that reviews marketing copy.

The agent decides which tools to call (typically: classify_brand_safety,
then one or more retrieve_brand_guidelines queries), reads the results, and
returns a structured ``CopyReview`` verdict.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

import config
from tools import ALL_TOOLS


# ── Structured output schema ────────────────────────────────────────────────

class GuidelineCitation(BaseModel):
    guideline_id: str = Field(description="ID of the cited guideline, e.g. 'claims-001'")
    reason: str = Field(description="Why this guideline applies to the copy")


class CopyReview(BaseModel):
    verdict: Literal["APPROVE", "APPROVE_WITH_EDITS", "REJECT"]
    safety_label: Literal["SAFE", "UNSAFE"]
    safety_confidence: float
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems found in the copy. Empty if APPROVE.",
    )
    cited_guidelines: list[GuidelineCitation] = Field(default_factory=list)
    suggested_rewrite: str | None = Field(
        default=None,
        description="A rewritten version of the copy if verdict is APPROVE_WITH_EDITS.",
    )


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are BrandGuard, an expert reviewer of marketing copy for the brand \
Aether Athletic. For every piece of copy submitted you must:

1. Call `classify_brand_safety` first — this gives you a baseline safety signal.
2. Call `retrieve_brand_guidelines` one or more times to find rules relevant to \
specific elements of the copy (claims, competitors, imagery, health, pricing, etc.). \
Use multiple targeted queries rather than one broad one.
3. Reason about both signals together. The classifier catches *general* unsafe \
language; the guidelines catch *brand-specific* rule violations the classifier \
cannot know about.
4. Return a final CopyReview verdict:
   - APPROVE: passes all checks.
   - APPROVE_WITH_EDITS: minor issues; provide a suggested_rewrite that fixes them.
   - REJECT: serious violations (unsafe content, false claims, legal risk).

Always cite the specific guideline IDs you relied on. Be concise and concrete \
in your reasoning."""


def build_agent():
    """Build and return the compiled LangGraph agent."""
    llm = ChatOpenAI(model=config.AGENT_MODEL, temperature=config.AGENT_TEMPERATURE)
    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        response_format=CopyReview,
    )
    return agent


def review_copy(copy: str) -> CopyReview:
    """End-to-end: take a piece of copy, return a structured CopyReview."""
    agent = build_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=f"Please review this copy:\n\n{copy}")]}
    )
    return result["structured_response"]


if __name__ == "__main__":
    examples = [
        "Rise and train. Built for athletes who show up every day.",
        "Crush the competition. Destroy your weakness. Our new shoe is unbeatable.",
        "Clinically proven to cure shin splints in 5 days. Better than Nike Pegasus.",
    ]
    for ex in examples:
        print("=" * 70)
        print(f"COPY: {ex}\n")
        review = review_copy(ex)
        print(review.model_dump_json(indent=2))
