"""Kaggle GPU QLoRA training entry point for Qwen2.5-7B-Instruct.

Input is the local-prepared JSONL with a `messages` field. This script trains an
adapter only, then saves both the LoRA adapter and an Ollama-friendly GGUF export
when the installed Unsloth build supports direct GGUF saving.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_dataset(path: Path):
    from datasets import load_dataset

    return load_dataset("json", data_files=str(path), split="train")


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-7B-Instruct on Kaggle")
    parser.add_argument("--train", type=Path, default=Path("/kaggle/input/pre-cab/qwen_qlora_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/pre_cab_qwen"))
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer

    if not args.train.exists():
        raise SystemExit(f"Training JSONL not found: {args.train}")
    dataset = _load_dataset(args.train)
    if len(dataset) == 0:
        raise SystemExit("Training dataset is empty")

    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = dataset.map(format_chat, num_proc=1)
    use_bf16 = is_bfloat16_supported()
    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        seed=args.seed,
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    print(f"Training records: {len(dataset)}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    stats = trainer.train()
    print(stats, flush=True)

    adapter_dir = args.output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_dir = args.output / "merged_16bit"
    try:
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except Exception as exc:
        print(f"Merged 16-bit export skipped: {type(exc).__name__}: {exc}", flush=True)

    gguf_dir = args.output / "gguf"
    try:
        gguf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=os.getenv("GGUF_QUANT", "q4_k_m"),
        )
        print(f"GGUF export written under {gguf_dir}", flush=True)
    except Exception as exc:
        print(f"Direct GGUF export skipped: {type(exc).__name__}: {exc}", flush=True)

    manifest = {
        "base_model": args.model,
        "records": len(dataset),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "adapter": str(adapter_dir),
        "merged_16bit": str(merged_dir),
        "gguf": str(gguf_dir),
        "train_only": True,
        "validation_and_holdout_used_for_training": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
