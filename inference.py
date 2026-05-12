"""Inference wrapper around the LoRA-fine-tuned brand-safety classifier.

Exposes a single ``BrandSafetyClassifier`` class with a ``.classify(text)``
method that returns a structured verdict. This is the unit the agent will
later call as a tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from prepare_data import SYSTEM_PROMPT


@dataclass
class SafetyVerdict:
    label: str          # 'SAFE' | 'UNSAFE'
    confidence: float   # 0..1 — prob assigned to the chosen label
    raw_output: str


class BrandSafetyClassifier:
    """Loads the base model + LoRA adapter once, classifies on demand."""

    def __init__(
        self,
        adapter_path: str | Path = config.LORA_OUTPUT_DIR,
        device: str | None = None,
    ) -> None:
        adapter_path = Path(adapter_path)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No LoRA adapter found at {adapter_path}.\n"
                f"Run `python train.py` first to fine-tune the model, "
                f"then re-run this script."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

        base = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        self.model.eval()

        # Pre-compute the token ids for SAFE / UNSAFE so we can score them
        # directly from the next-token logits — much faster and more robust
        # than free-form generation + string parsing.
        self.safe_id = self.tokenizer.encode("SAFE", add_special_tokens=False)[0]
        self.unsafe_id = self.tokenizer.encode("UNSAFE", add_special_tokens=False)[0]

    @torch.no_grad()
    def classify(self, text: str) -> SafetyVerdict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Marketing copy:\n{text}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        logits = self.model(**inputs).logits[0, -1]  # last-token logits
        # Compare only SAFE vs UNSAFE — constrained classification
        pair = torch.tensor([logits[self.safe_id], logits[self.unsafe_id]])
        probs = torch.softmax(pair, dim=0)
        idx = int(probs.argmax().item())
        label = ("SAFE", "UNSAFE")[idx]
        conf = float(probs[idx].item())

        return SafetyVerdict(label=label, confidence=conf, raw_output=label)


if __name__ == "__main__":
    clf = BrandSafetyClassifier()
    examples = [
        "Rise and train. Built for athletes who show up every day.",
        "Crush your enemies. Destroy the competition. Be unstoppable.",
        "Cures back pain in 7 days — guaranteed or your money back.",
    ]
    for ex in examples:
        v = clf.classify(ex)
        print(f"[{v.label} | {v.confidence:.2%}]  {ex}")
