from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # GPT-OSS may emit a small amount of surrounding text. Recover the first
    # complete JSON object rather than relying on a greedy regex.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def load_model(base_model: str, adapter: Path):
    """Load the exact PEFT adapter used by the Kaggle QLoRA run."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(adapter), use_fast=True)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    # Prefer adapter tokenizer files, but fall back to the base tokenizer when
    # the adapter artifact does not contain a complete tokenizer.
    if len(tokenizer) == 0:
        tokenizer = base_tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    # Evaluation records are expected to contain the system/user messages and
    # a held-out assistant target. Never feed that target to the model.
    prompt_messages = messages[:2]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test or benchmark a trained Pre-CAB GPT-OSS PEFT adapter"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--base-model", default="unsloth/gpt-oss-20b")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of holdout records; use 0 for the full holdout set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL path for per-record model outputs",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if not args.adapter.exists():
        raise SystemExit(f"Adapter not found: {args.adapter}")
    if not args.holdout.exists():
        raise SystemExit(f"Holdout not found: {args.holdout}")

    model, tokenizer = load_model(args.base_model, args.adapter)
    rows = load_rows(args.holdout, args.limit)
    if not rows:
        raise SystemExit("Holdout contains no usable records")

    actual: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    correct = 0
    scored = 0
    false_pass = 0
    false_fail = 0
    unparsed = 0
    output_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        messages = row.get("messages") or []
        if len(messages) < 2:
            unparsed += 1
            print(f"[{idx}/{len(rows)}] INVALID_INPUT missing messages")
            continue

        prompt = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        # device_map='auto' can shard the model; putting inputs on the first
        # model device is the standard Transformers path for generation.
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as exc:
            unparsed += 1
            print(f"[{idx}/{len(rows)}] GENERATION_ERROR {type(exc).__name__}: {exc}")
            continue

        result = extract_json(decoded)
        pred = result.get("prediction") if result else None
        gold = row.get("metadata", {}).get("historical_outcome")
        cr_id = row.get("metadata", {}).get("cr_id", f"row-{idx}")

        if pred:
            predicted[str(pred)] += 1
        else:
            unparsed += 1
        if gold:
            actual[str(gold)] += 1
            if pred:
                scored += 1
                if pred == gold:
                    correct += 1
                elif pred == "PASS":
                    false_pass += 1
                elif gold == "PASS":
                    false_fail += 1

        output_rows.append(
            {
                "cr_id": cr_id,
                "prediction": pred,
                "historical_outcome": gold,
                "parsed": result is not None,
                "raw_output": decoded,
                "result": result,
            }
        )
        print(f"[{idx}/{len(rows)}] {cr_id} -> {pred or 'UNPARSED'}")

    report = {
        "records": len(rows),
        "scored": scored,
        "parsed": len(rows) - unparsed,
        "unparsed_or_failed": unparsed,
        "accuracy": correct / scored if scored else 0.0,
        "false_passes": false_pass,
        "false_pass_rate": false_pass / scored if scored else 0.0,
        "false_fails": false_fail,
        "false_fail_rate": false_fail / scored if scored else 0.0,
        "actual_distribution": dict(actual),
        "predicted_distribution": dict(predicted),
        "adapter": str(args.adapter),
        "base_model": args.base_model,
        "max_new_tokens": args.max_new_tokens,
    }

    output_path = args.output or (args.adapter / "holdout_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions_path = output_path.with_suffix(".jsonl")
    with predictions_path.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== PRE-CAB GPT-OSS EVALUATION ===")
    print(json.dumps(report, indent=2))
    print(f"Report: {output_path}")
    print(f"Predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
