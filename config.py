"""Central configuration for BrandGuard.

Keeping all paths, model names and hyperparameters here makes the rest of the
codebase read like prose and makes experiments reproducible.
"""
from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
INDEX_DIR = ROOT / "vector_store"
LOG_DIR = ROOT / "logs"

for _p in (MODELS_DIR, INDEX_DIR, LOG_DIR):
    _p.mkdir(exist_ok=True)

# ── LoRA fine-tune ───────────────────────────────────────────────────────────
# Qwen2.5-0.5B is small enough to QLoRA-train on a free Colab T4 (~5 min/epoch).
# Swap for "meta-llama/Llama-3.2-1B-Instruct" if you have a stronger GPU.
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_OUTPUT_DIR = MODELS_DIR / "qwen-brandsafety-lora"

# Public stand-in for proprietary brand-safety data.
# In a real WPP setting this would be replaced by labelled brand-safety
# incidents and approved-copy corpora.
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "offensive"  # 0 = safe, 1 = unsafe
LABEL_MAP = {0: "SAFE", 1: "UNSAFE"}

# LoRA hyperparameters — sensible defaults for a small classifier.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

TRAIN_EPOCHS = 2
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 256

# ── RAG ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, fast
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "brand_guidelines"
RETRIEVE_K = 8       # candidates from vector store
RERANK_K = 3         # final docs after cross-encoder

# ── Agent ────────────────────────────────────────────────────────────────────
# Cheap, fast tool-calling model. Replace with claude-3-5-haiku or a local
# model served via vLLM if you'd rather not depend on OpenAI.
AGENT_MODEL = "gpt-4o-mini"
AGENT_TEMPERATURE = 0.0
