"""Prepare brand-safety classification data in chat-template format.

We turn each labelled example into a (system, user, assistant) conversation so
the base instruction-tuned model only has to learn the *task-specific output
distribution*, not a new format. This is the standard recipe for
instruction-style LoRA fine-tunes.
"""
from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

import config

SYSTEM_PROMPT = (
    "You are a brand-safety classifier for marketing copy. "
    "Read the user's text and respond with exactly one token: "
    "SAFE if the copy is appropriate for a mainstream consumer brand, "
    "or UNSAFE if it contains offensive, hateful, profane, aggressive, "
    "or otherwise brand-damaging language. Respond with the label only."
)


def _format_example(text: str, label: int, tokenizer) -> dict:
    """Render one example using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Marketing copy:\n{text}"},
        {"role": "assistant", "content": config.LABEL_MAP[label]},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": rendered}


def build_datasets(tokenizer) -> tuple[Dataset, Dataset]:
    """Return (train, eval) datasets formatted for SFTTrainer."""
    raw = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)

    train = raw["train"].map(
        lambda x: _format_example(x["text"], x["label"], tokenizer),
        remove_columns=raw["train"].column_names,
    )
    val = raw["validation"].map(
        lambda x: _format_example(x["text"], x["label"], tokenizer),
        remove_columns=raw["validation"].column_names,
    )
    return train, val


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    train, val = build_datasets(tok)
    print(f"Train: {len(train):,} examples | Val: {len(val):,} examples")
    print("\n── sample ──")
    print(train[0]["text"])
