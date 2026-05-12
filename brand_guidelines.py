"""Synthetic brand guidelines for a fictional brand 'Aether Athletic'.

In production this module is replaced by a loader that pulls real brand books,
campaign briefs, legal-approved messaging, and style guides from a document
store (SharePoint, S3, a CMS, etc.). Keeping it as Python data here makes the
project reproducible without external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Guideline:
    id: str
    title: str
    category: str  # voice | claims | legal | inclusivity | competitor | imagery
    content: str


BRAND_GUIDELINES: list[Guideline] = [
    Guideline(
        id="voice-001",
        title="Brand voice — tone",
        category="voice",
        content=(
            "Aether Athletic speaks with confident, motivating energy. Use active "
            "voice and short, punchy sentences. Never use aggressive, sarcastic, "
            "condescending, or fear-based language. Aspirational, not shaming."
        ),
    ),
    Guideline(
        id="voice-002",
        title="Brand voice — vocabulary",
        category="voice",
        content=(
            "Preferred verbs: move, build, push, rise, train. Avoid: crush, "
            "destroy, kill, dominate, war, attack — any combat or violence "
            "metaphors are off-brand."
        ),
    ),
    Guideline(
        id="claims-001",
        title="Performance claims require evidence",
        category="claims",
        content=(
            "Any quantitative performance claim ('30% faster recovery', "
            "'doubles endurance') must reference a peer-reviewed study or an "
            "approved internal test. Unsubstantiated superlatives ('the best', "
            "'unbeatable') are prohibited under ASA CAP code rule 3.7."
        ),
    ),
    Guideline(
        id="claims-002",
        title="Health and medical claims",
        category="claims",
        content=(
            "Do not imply the product treats, cures, or prevents any medical "
            "condition. Wording such as 'heals injuries', 'cures soreness', or "
            "'medical-grade' is prohibited unless legal pre-approval is on file."
        ),
    ),
    Guideline(
        id="legal-001",
        title="Pricing and promotion language",
        category="legal",
        content=(
            "All discount claims must include the reference price and the "
            "duration of the offer. 'Up to X% off' requires that at least 10% of "
            "the range was discounted by X%."
        ),
    ),
    Guideline(
        id="legal-002",
        title="Children and minors",
        category="legal",
        content=(
            "Marketing must not directly exhort minors to purchase, nor depict "
            "minors using products marketed at adult athletes. Casting for "
            "imagery featuring under-18s requires legal sign-off."
        ),
    ),
    Guideline(
        id="incl-001",
        title="Inclusive body representation",
        category="inclusivity",
        content=(
            "Visuals and copy must represent a range of body types, ages, "
            "ethnicities, and abilities. Avoid framing any body type as a "
            "before/after target. Never imply that the product is required to "
            "achieve a 'real' or 'true' version of the self."
        ),
    ),
    Guideline(
        id="incl-002",
        title="Gender-neutral defaults",
        category="inclusivity",
        content=(
            "Use 'athletes', 'people', 'you' as defaults rather than gendered "
            "collective nouns. Gendered fit references ('men's cut', 'women's "
            "cut') are acceptable when describing product fit specifically."
        ),
    ),
    Guideline(
        id="comp-001",
        title="Competitor references",
        category="competitor",
        content=(
            "Do not name competitors directly in any consumer-facing copy. "
            "Comparative claims must be generic ('other leading brands') and "
            "backed by approved comparison data on file with legal."
        ),
    ),
    Guideline(
        id="img-001",
        title="Imagery — context and safety",
        category="imagery",
        content=(
            "Athletes in imagery must wear appropriate safety equipment for the "
            "depicted activity (helmets for cycling, harnesses for climbing). "
            "No depictions of training to the point of injury, vomiting, or "
            "collapse."
        ),
    ),
    Guideline(
        id="img-002",
        title="Imagery — brand colour usage",
        category="imagery",
        content=(
            "Primary palette is graphite (#222), signal-orange (#FF5A1F), and "
            "off-white (#F5F2EC). Signal-orange should occupy no more than 15% "
            "of the visual field in any single asset."
        ),
    ),
    Guideline(
        id="voice-003",
        title="Profanity and slang",
        category="voice",
        content=(
            "No profanity, including mild ('damn', 'hell'). Slang specific to "
            "particular subcultures may be used only with prior brand-team "
            "approval to avoid appropriation."
        ),
    ),
    Guideline(
        id="claims-003",
        title="Environmental and sustainability claims",
        category="claims",
        content=(
            "Sustainability claims ('eco-friendly', 'carbon-neutral', "
            "'recycled') must reference a specific certification or measurable "
            "metric. Vague greenwashing language is prohibited under CMA "
            "Green Claims Code."
        ),
    ),
    Guideline(
        id="incl-003",
        title="Mental-health framing",
        category="inclusivity",
        content=(
            "When referencing mental health, never frame exercise as a "
            "substitute for therapy or medication. Wellness claims must "
            "position movement as supportive, not curative."
        ),
    ),
    Guideline(
        id="legal-003",
        title="Influencer and testimonial disclosure",
        category="legal",
        content=(
            "Any paid endorsement must be clearly marked '#ad' or 'Paid "
            "partnership'. Testimonials require signed release forms and must "
            "reflect typical, not exceptional, results."
        ),
    ),
]


def as_documents() -> list[dict]:
    """Format guidelines as documents for ingestion into a vector store."""
    return [
        {
            "id": g.id,
            "text": f"{g.title}\n\n{g.content}",
            "metadata": {"title": g.title, "category": g.category},
        }
        for g in BRAND_GUIDELINES
    ]


if __name__ == "__main__":
    docs = as_documents()
    print(f"Loaded {len(docs)} brand guidelines.")
    for d in docs[:3]:
        print(f"  [{d['id']}] {d['metadata']['title']}")
