from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a GPT-OSS 20B QLoRA adapter on Kaggle")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="unsloth/gpt-oss-20b")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. On Kaggle run: "
            "pip install -U unsloth trl peft datasets accelerate bitsandbytes"
        ) from exc

    train_file = args.dataset_dir / "train_reasoning.jsonl"
    valid_file = args.dataset_dir / "validation_reasoning.jsonl"
    if not train_file.exists() or not valid_file.exists():
        raise SystemExit("Expected train_reasoning.jsonl and validation_reasoning.jsonl in dataset-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset("json", data_files=str(train_file), split="train")
    valid_ds = load_dataset("json", data_files=str(valid_file), split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=7,
    )

    def format_example(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_ds = train_ds.map(format_example, desc="Formatting training examples")
    valid_ds = valid_ds.map(format_example, desc="Formatting validation examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=SFTConfig(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
            bf16=False,
            report_to="none",
            max_length=args.max_seq_length,
            seed=7,
            gradient_checkpointing=True,
        ),
    )

    train_kwargs = {}
    if args.resume_from:
        train_kwargs["resume_from_checkpoint"] = args.resume_from

    trainer.train(**train_kwargs)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)

    print(f"Adapter saved to {args.output_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
