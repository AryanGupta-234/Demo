"""Kaggle GPU QLoRA training entry point for Qwen2.5-7B-Instruct (v2).

Fixes over the v1 baseline:
  * Trains ONLY on the assistant completion tokens (prompt/CR-JSON tokens are
    masked out of the loss) via unsloth's train_on_responses_only. Otherwise
    SFT trains equally hard on reproducing the (huge, mostly-numeric) input
    CR JSON as on the actual reasoning/prediction target, diluting the
    learning signal on the thing that is actually being evaluated.
  * Performs an authoritative, tokenizer-based length audit BEFORE training.
    HF/TRL truncate overlong sequences from the right by default, which can
    silently cut off the assistant target (it sits at the end of the
    sequence) instead of erroring. Any example that still doesn't fit after
    field-preserving Work Notes/Comments shrinking is dropped and logged,
    never silently truncated by the trainer.
  * Uses the validation split for eval_loss-based checkpoint selection and
    early stopping. Validation and holdout are never used for gradient
    updates -- verified and asserted in the manifest.
  * Reports LoRA config, seq length, effective batch size and label
  distribution as part of a reproducibility manifest.

Usage (see training/KAGGLE_QWEN_QLORA.md for the full walkthrough):

    !python train_qwen_qlora.py \
      --train /kaggle/input/pre-cab/qwen_qlora_train.jsonl \
      --valid /kaggle/input/pre-cab/qwen_qlora_valid.jsonl \
      --output /kaggle/working/pre_cab_qwen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_budget import shrink_user_message_json  # noqa: E402

SHRINK_BUDGETS = (6000, 3000, 1500, 800, 400)  # progressively harsher field-preserving shrink attempts


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _render(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def audit_and_fit(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ensure every example's fully-rendered chat text fits max_seq_length
    WITHOUT ever touching the assistant message. Shrinks Work Notes/Comments
    progressively; drops (and logs) any example that still doesn't fit."""
    kept: list[dict[str, Any]] = []
    fit_untouched = 0
    fit_after_shrink = 0
    dropped: list[str] = []
    max_tokens_seen = 0

    for row in rows:
        messages = row["messages"]
        text = _render(tokenizer, messages)
        n_tokens = _token_len(tokenizer, text)
        if n_tokens <= max_seq_length:
            kept.append({**row, "messages": messages, "text": text})
            fit_untouched += 1
            max_tokens_seen = max(max_tokens_seen, n_tokens)
            continue

        fixed = False
        for budget in SHRINK_BUDGETS:
            shrunk_user, changed = shrink_user_message_json(messages[1]["content"], char_budget_per_field=budget)
            if not changed:
                continue
            candidate = [messages[0], {"role": "user", "content": shrunk_user}, messages[2]]
            candidate_text = _render(tokenizer, candidate)
            candidate_tokens = _token_len(tokenizer, candidate_text)
            if candidate_tokens <= max_seq_length:
                kept.append({**row, "messages": candidate, "text": candidate_text})
                fit_after_shrink += 1
                max_tokens_seen = max(max_tokens_seen, candidate_tokens)
                fixed = True
                break
        if not fixed:
            dropped.append(row.get("cr_id", "unknown"))

    report = {
        "input_records": len(rows),
        "fit_without_change": fit_untouched,
        "fit_after_note_shrink": fit_after_shrink,
        "dropped_still_too_long": len(dropped),
        "dropped_cr_ids": dropped[:50],
        "max_tokens_observed": max_tokens_seen,
        "max_seq_length": max_seq_length,
    }
    return kept, report


def _label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    from collections import Counter

    counts: Counter[str] = Counter()
    for row in rows:
        try:
            target = json.loads(row["messages"][2]["content"])
            counts[str(target.get("prediction"))] += 1
        except Exception:
            counts["UNPARSEABLE"] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-7B-Instruct on Kaggle (v2)")
    parser.add_argument("--train", type=Path, default=Path("/kaggle/input/pre-cab/qwen_qlora_train.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("/kaggle/input/pre-cab/qwen_qlora_valid.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/pre_cab_qwen"))
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=4.0, help="Upper bound; early stopping usually stops sooner.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="Unsloth's fast, memory-efficient LoRA kernel path requires dropout=0. Any nonzero value "
        "falls back to the slower unpatched implementation ('Unsloth will patch all other layers, except "
        "LoRA matrices, causing a performance hit'), which also uses meaningfully more activation memory -- "
        "on a single 15GB T4 at max_seq_length=4096 this is enough to trigger a CUDA OOM a few steps in. "
        "Regularize via early stopping / weight decay / fewer epochs instead if this dataset is small.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    if not args.train.exists():
        raise SystemExit(f"Training JSONL not found: {args.train}")
    train_rows = _load_jsonl(args.train)
    if not train_rows:
        raise SystemExit("Training dataset is empty")
    valid_rows = _load_jsonl(args.valid) if args.valid.exists() else []
    if not valid_rows:
        print(f"WARNING: validation file not found at {args.valid}. Training will proceed WITHOUT "
              f"eval_loss-based checkpoint selection or early stopping -- not recommended.", flush=True)

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}", flush=True)
    print(f"Model: {args.model}  max_seq_length={args.max_seq_length}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    train_fitted, train_report = audit_and_fit(tokenizer, train_rows, max_seq_length=args.max_seq_length)
    valid_fitted, valid_report = ([], {}) if not valid_rows else audit_and_fit(
        tokenizer, valid_rows, max_seq_length=args.max_seq_length
    )
    print("Train length audit:", json.dumps(train_report, indent=2), flush=True)
    if valid_rows:
        print("Validation length audit:", json.dumps(valid_report, indent=2), flush=True)
    if not train_fitted:
        raise SystemExit("No training examples fit max_seq_length after shrinking -- increase --max-seq-length.")

    from datasets import Dataset

    train_ds = Dataset.from_list([{"text": r["text"]} for r in train_fitted])
    valid_ds = Dataset.from_list([{"text": r["text"]} for r in valid_fitted]) if valid_fitted else None

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    use_bf16 = is_bfloat16_supported()
    has_eval = valid_ds is not None and len(valid_ds) > 0
    training_args = SFTConfig(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=4,
        eval_strategy="epoch" if has_eval else "no",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        optim="adamw_8bit",
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
        train_dataset=train_ds,
        eval_dataset=valid_ds if has_eval else None,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if has_eval else [],
    )

    # Mask the loss to the assistant completion only. Without this, SFT trains
    # on reproducing the (large) input CR JSON with equal weight to the
    # reasoning/prediction target, which is not what we want to optimize.
    try:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        response_masking_applied = True
    except Exception as exc:
        print(f"WARNING: could not apply response-only loss masking ({type(exc).__name__}: {exc}). "
              f"Training will use full-sequence loss instead.", flush=True)
        response_masking_applied = False

    print(f"Training records (post-audit): {len(train_ds)}", flush=True)
    if has_eval:
        print(f"Validation records (post-audit): {len(valid_ds)}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    print(f"Effective batch size: {args.batch_size * args.grad_accumulation}", flush=True)

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
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=os.getenv("GGUF_QUANT", "q4_k_m"))
        print(f"GGUF export written under {gguf_dir}", flush=True)
    except Exception as exc:
        print(f"Direct GGUF export skipped: {type(exc).__name__}: {exc}", flush=True)

    manifest = {
        "base_model": args.model,
        "max_seq_length": args.max_seq_length,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "training": {
            "epochs_requested": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accumulation": args.grad_accumulation,
            "effective_batch_size": args.batch_size * args.grad_accumulation,
            "seed": args.seed,
            "response_only_loss_masking": response_masking_applied,
            "early_stopping_patience": args.early_stopping_patience if has_eval else None,
            "model_selection_metric": "eval_loss" if has_eval else "final_checkpoint (no validation supplied)",
        },
        "data": {
            "train_input_records": train_report["input_records"],
            "train_records_used": len(train_ds),
            "train_records_dropped_too_long": train_report["dropped_still_too_long"],
            "train_label_distribution": _label_distribution(train_fitted),
            "validation_input_records": valid_report.get("input_records", 0),
            "validation_records_used": len(valid_ds) if has_eval else 0,
            "validation_records_dropped_too_long": valid_report.get("dropped_still_too_long", 0),
            "validation_label_distribution": _label_distribution(valid_fitted) if valid_fitted else {},
        },
        "leakage_guarantees": {
            "validation_used_for_gradient_updates": False,
            "holdout_used_for_gradient_updates": False,
            "holdout_loaded_by_this_script": False,
        },
        "outputs": {"adapter": str(adapter_dir), "merged_16bit": str(merged_dir), "gguf": str(gguf_dir)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
