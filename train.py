"""QLoRA fine-tune Qwen2.5-0.5B-Instruct for brand-safety classification.

Why QLoRA?
    - Load the base model in 4-bit (bitsandbytes) → fits in <2 GB VRAM.
    - Train only the LoRA adapters (~1% of params) → fast, cheap, no
      catastrophic forgetting of the base model's general capabilities.

Run:
    python train.py

Outputs a LoRA adapter to MODELS_DIR / 'qwen-brandsafety-lora/'.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config
from prepare_data import build_datasets


def main() -> None:
    # 1. Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Base model in 4-bit ──────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing

    # 3. LoRA config ──────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Data ─────────────────────────────────────────────────────────────────
    train_ds, eval_ds = build_datasets(tokenizer)
    print(f"Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # 5. Training config ──────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=str(config.LORA_OUTPUT_DIR),
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        max_seq_length=config.MAX_SEQ_LEN,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",  # swap to 'wandb' to track properly
        dataset_text_field="text",
        packing=False,
    )

    # 6. Train ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(config.LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(config.LORA_OUTPUT_DIR))
    print(f"\n✅ LoRA adapter saved to {config.LORA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
