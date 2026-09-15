"""Prepare the leakage-safe training split for Qwen2.5-7B-Instruct QLoRA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("training/output/train_reasoning.jsonl")
DEFAULT_OUTPUT = Path("training/output/qwen_qlora_train.jsonl")


def _messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = example.get("messages") or []
    return [
        {"role": str(item.get("role")), "content": str(item.get("content") or "")}
        for item in messages
        if isinstance(item, dict) and item.get("role") in {"system", "user", "assistant"}
    ]


def _valid(example: dict[str, Any]) -> bool:
    messages = _messages(example)
    if len(messages) < 3:
        return False
    if [m["role"] for m in messages[:3]] != ["system", "user", "assistant"]:
        return False
    try:
        user = json.loads(messages[1]["content"])
        target = json.loads(messages[2]["content"])
    except json.JSONDecodeError:
        return False
    return isinstance(user, dict) and isinstance(target, dict) and bool(target.get("prediction"))


def prepare(input_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            example = json.loads(line)
            if not isinstance(example, dict) or not _valid(example):
                skipped += 1
                continue
            metadata = example.get("metadata") or {}
            messages = _messages(example)
            user = json.loads(messages[1]["content"])
            cr = user.get("cr") if isinstance(user.get("cr"), dict) else {}
            cr_id = str(metadata.get("cr_id") or cr.get("Number") or f"row-{line_number}")
            if cr_id in seen:
                skipped += 1
                continue
            seen.add(cr_id)
            dst.write(json.dumps({"messages": messages, "cr_id": cr_id}, ensure_ascii=False) + "\n")
            kept += 1
    return {"records": kept, "skipped": skipped, "output": str(output_path), "source": str(input_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare training split for Qwen2.5-7B QLoRA")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = prepare(args.input, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
