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
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Pre-CAB adapter on untouched holdout records")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--base-model", default="unsloth/gpt-oss-20b")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows: list[dict[str, Any]] = []
    with args.holdout.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]

    actual: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    correct = 0
    scored = 0
    false_pass = 0
    false_fail = 0

    for idx, row in enumerate(rows, 1):
        messages = row.get("messages") or []
        prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=700, do_sample=False)
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        text = decoded if isinstance(decoded, str) else " ".join(decoded)
        result = extract_json(text) or {}
        pred = result.get("prediction")
        gold = row.get("metadata", {}).get("historical_outcome")
        if pred:
            predicted[str(pred)] += 1
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
        print(f"[{idx}/{len(rows)}] {row.get('metadata', {}).get('cr_id', 'unknown')} -> {pred or 'UNPARSED'}")

    report = {
        "records": len(rows),
        "scored": scored,
        "accuracy": correct / scored if scored else 0.0,
        "false_passes": false_pass,
        "false_pass_rate": false_pass / scored if scored else 0.0,
        "false_fails": false_fail,
        "false_fail_rate": false_fail / scored if scored else 0.0,
        "actual_distribution": dict(actual),
        "predicted_distribution": dict(predicted),
        "adapter": str(args.adapter),
        "base_model": args.base_model,
    }
    out = args.adapter / "holdout_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
