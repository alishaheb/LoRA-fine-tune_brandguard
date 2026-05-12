# BrandGuard

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> An agentic system for reviewing marketing copy against brand-safety rules and
> brand-specific guidelines. Combines a **LoRA-fine-tuned classifier**, a
> **retrieval-augmented generation** pipeline with cross-encoder reranking, and
> a **LangGraph ReAct agent** that orchestrates both.

Built as a portfolio project to demonstrate end-to-end ownership of the three
patterns that dominate modern applied-LLM work.

## Architecture

```
         ┌──────────────────────────────┐
         │   marketing copy (input)     │
         └───────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   LangGraph ReAct agent     │
          │   (gpt-4o-mini, tool-call)  │
          └──┬───────────────────────┬──┘
             │                       │
┌────────────▼────────────┐  ┌───────▼────────────────────┐
│ classify_brand_safety   │  │ retrieve_brand_guidelines  │
│                         │  │                            │
│ Qwen2.5-0.5B + LoRA     │  │ MiniLM bi-encoder          │
│ (QLoRA, 4-bit)          │  │  → Chroma vector store     │
│ binary SAFE / UNSAFE    │  │  → MS-MARCO cross-encoder  │
│ + confidence            │  │     reranker               │
└─────────────────────────┘  └────────────────────────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
           ┌──────────────────────────────┐
           │ structured CopyReview verdict│
           │ (Pydantic schema)            │
           └──────────────────────────────┘
```

## Components

| Layer | What it does | Key files |
|---|---|---|
| **Data** | Synthetic brand guidelines for fictional brand "Aether Athletic" | `brand_guidelines.py` |
| **LoRA fine-tune** | QLoRA adapter on Qwen2.5-0.5B-Instruct for SAFE/UNSAFE classification, trained on `cardiffnlp/tweet_eval` (offensive) | `prepare_data.py`, `train.py`, `inference.py` |
| **RAG** | Sentence-transformer embeddings → Chroma → MS-MARCO cross-encoder rerank | `build_index.py`, `retriever.py` |
| **Agent** | LangGraph prebuilt ReAct agent, two tools, Pydantic structured output | `tools.py`, `agent.py` |
| **Eval** | Classification report on held-out test set + RAG Hit@K/MRR | `eval_classifier.py`, `eval_rag.py` |
| **CLI** | Reviews one copy snippet end-to-end | `app.py` |

## Setup

```bash
git clone https://github.com/alishaheb/LoRA-fine-tune_brandguard.git
cd LoRA-fine-tune_brandguard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# OpenAI key for the agent's reasoning LLM
cp .env.example .env
# then edit .env and paste your real key
```

A free Colab T4 is enough; locally, any GPU with ≥6 GB VRAM works for the
QLoRA training. CPU also works for everything *except* training, just slowly.

## Running, end to end

```bash
# 1. Build the RAG vector index  (a few seconds, no GPU needed)
python build_index.py

# 2. Evaluate the RAG pipeline
python eval_rag.py

# 3. Train the LoRA brand-safety classifier  (~5–10 min on a T4)
python train.py

# 4. Evaluate the trained classifier
python eval_classifier.py

# 5. Run the agent on a piece of marketing copy
python app.py "Crush the competition. Clinically proven to cure shin splints."
```

Expected output for that last command (illustrative):

```
╭── BrandGuard verdict ──────────────────────────────────╮
│ REJECT   safety = UNSAFE (0.87)                        │
╰────────────────────────────────────────────────────────╯
Issues:
  • Uses combat language ("crush") banned under voice rules.
  • Makes a medical claim ("cure shin splints") without evidence.
                  Cited guidelines
┃ ID            ┃ Reason                                ┃
┃ voice-002     ┃ "crush" is on the prohibited verb list│
┃ claims-002    ┃ "cure" is a medical claim             │
```

## Results

> Replace these placeholders with your actual numbers after running
> `python eval_classifier.py` and `python eval_rag.py`.

**Classifier (LoRA on Qwen2.5-0.5B, `tweet_eval/offensive` test split):**
- Macro-F1: _TBD_
- Accuracy: _TBD_

**RAG (8 hand-labelled queries, top-3 retrieval):**
- Hit@3: _TBD_
- MRR@3: _TBD_

## Design choices worth talking about in an interview

- **QLoRA over full fine-tune.** 4-bit base + ~1% trainable params means the
  whole training run fits on a T4 and the adapter is <10 MB. Catastrophic
  forgetting is avoided because the base weights are frozen.
- **Logit-constrained classification.** Inference reads the SAFE/UNSAFE token
  logits directly rather than free-form generating and string-parsing. This is
  faster, deterministic, and gives a calibrated confidence score.
- **Two-stage retrieve-then-rerank.** Bi-encoders are fast but only ~70%
  accurate at the top-1; a cross-encoder re-scores the candidates and lifts
  precision substantially. Standard pattern in production RAG.
- **Structured output via Pydantic.** The agent returns a typed `CopyReview`,
  not free text — this is what makes the system actually usable as a
  component of a larger pipeline (e.g. a campaign-approval workflow).
- **Tool docstrings as agent contracts.** The LLM sees each tool's docstring
  as its description; writing them like API docs is what makes the agent
  reliably call the right tool with the right args.

## What's not here (and where you'd take it next)

- Real brand data — swap `brand_guidelines.py` for a loader pulling from
  your document store.
- LLM-as-judge or RAGAS evaluation of the agent's *reasoning*, not just the
  retrieval and classification components.
- Streaming output, async tool calls, and a proper web UI.
- A monitoring layer (LangSmith, Langfuse, or a custom OpenTelemetry exporter)
  for production traffic.

## Tech stack

`transformers` · `peft` · `trl` · `bitsandbytes` · `sentence-transformers` ·
`chromadb` · `langgraph` · `pydantic` · `scikit-learn` · `rich`

## License

MIT
