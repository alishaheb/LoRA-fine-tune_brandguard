"""Evaluate the LoRA classifier on the held-out test split.

Reports accuracy, precision, recall, F1 and a confusion matrix. Run this
*before* you put the model into the agent — a junior-level mistake is to
ship a model whose offline metrics you never measured.
"""
from __future__ import annotations

from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

import config
from inference import BrandSafetyClassifier


def evaluate(max_examples: int = 500) -> None:
    test = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split="test")
    if max_examples:
        test = test.select(range(min(max_examples, len(test))))

    clf = BrandSafetyClassifier()
    y_true, y_pred = [], []
    for ex in tqdm(test, desc="Scoring"):
        v = clf.classify(ex["text"])
        y_true.append(config.LABEL_MAP[ex["label"]])
        y_pred.append(v.label)

    print("\n── Classification report ──")
    print(classification_report(y_true, y_pred, digits=3))

    print("── Confusion matrix (rows=true, cols=pred) ──")
    labels = ["SAFE", "UNSAFE"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"{'':>8} {labels[0]:>8} {labels[1]:>8}")
    for lbl, row in zip(labels, cm):
        print(f"{lbl:>8} {row[0]:>8} {row[1]:>8}")

    print(f"\nMacro-F1: {f1_score(y_true, y_pred, average='macro'):.3f}")


if __name__ == "__main__":
    evaluate()
