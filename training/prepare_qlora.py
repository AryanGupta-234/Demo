"""Prepare leakage-safe Qwen2.5-7B QLoRA splits from training/output/*_reasoning.jsonl.

Reads train_reasoning.jsonl, validation_reasoning.jsonl and holdout_reasoning.jsonl
(as written by training/build_dataset.py) and writes:

    training/output/qwen_qlora_train.jsonl
    training/output/qwen_qlora_valid.jsonl
    training/output/qwen_qlora_holdout.jsonl
    training/output/qwen_qlora_prep_report.json

Only qwen_qlora_train.jsonl is ever used for gradient updates. The validation
file is for eval_loss / model selection during training; the holdout file is
for final unbiased evaluation only. Upload all three to the private Kaggle
dataset so the training and evaluation scripts can consume them directly.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from text_budget import shrink_user_message_json

DEFAULT_INPUT_DIR = Path("training/output")
DEFAULT_OUTPUT_DIR = Path("training/output")

# Generous per-field character budget applied here as a first-pass size/safety
# control (keeps the uploaded JSONL sane and removes pathological outliers).
# The Kaggle training script applies the authoritative, tokenizer-based pass
# using the real Qwen tokenizer and max_seq_length before any example reaches
# the trainer -- this local pass only prevents absurd outliers from bloating
# the dataset upload.
DEFAULT_CHAR_BUDGET_PER_FIELD = 9000

SPLITS = ("train", "validation", "holdout")
SPLIT_TO_OUTPUT_NAME = {"train": "qwen_qlora_train.jsonl", "validation": "qwen_qlora_valid.jsonl", "holdout": "qwen_qlora_holdout.jsonl"}


def _messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = example.get("messages") or []
    return [
        {"role": str(item.get("role")), "content": str(item.get("content") or "")}
        for item in messages
        if isinstance(item, dict) and item.get("role") in {"system", "user", "assistant"}
    ]


def _valid(example: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
    messages = _messages(example)
    if len(messages) < 3:
        return False, None, None
    if [m["role"] for m in messages[:3]] != ["system", "user", "assistant"]:
        return False, None, None
    try:
        user = json.loads(messages[1]["content"])
        target = json.loads(messages[2]["content"])
    except json.JSONDecodeError:
        return False, None, None
    if not (isinstance(user, dict) and isinstance(target, dict) and bool(target.get("prediction"))):
        return False, None, None
    return True, user, target


def _prepare_split(
    input_path: Path,
    *,
    char_budget_per_field: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    shrunk = 0
    label_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            example = json.loads(line)
            if not isinstance(example, dict):
                skipped += 1
                continue
            ok, user, target = _valid(example)
            if not ok:
                skipped += 1
                continue
            metadata = example.get("metadata") or {}
            cr = user.get("cr") if isinstance(user.get("cr"), dict) else {}
            cr_id = str(metadata.get("cr_id") or cr.get("Number") or f"row-{line_number}")
            if cr_id in seen:
                skipped += 1
                continue
            seen.add(cr_id)

            messages = _messages(example)
            new_user_content, changed = shrink_user_message_json(
                messages[1]["content"], char_budget_per_field=char_budget_per_field
            )
            if changed:
                messages[1]["content"] = new_user_content
                shrunk += 1

            label_counts[str(target.get("prediction"))] += 1
            kept.append({"messages": messages, "cr_id": cr_id, "label": target.get("prediction")})

    report = {
        "source": str(input_path),
        "records": len(kept),
        "skipped": skipped,
        "notes_shrunk": shrunk,
        "label_distribution": dict(label_counts),
    }
    return kept, report


def _oversample_minority(rows: list[dict[str, Any]], *, max_ratio: float, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Duplicate minority-class rows (train split only) so no class falls below
    max_ratio of the majority class count. Capped and logged -- this is a
    coarse imbalance mitigation for a small dataset, not a substitute for
    macro-averaged evaluation metrics, and duplicated rows are exact copies so
    keep epochs modest to avoid memorizing them."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("label")), []).append(row)
    if not by_label:
        return rows, {"applied": False}
    majority = max(len(v) for v in by_label.values())
    rng = random.Random(seed)
    out = list(rows)
    added: dict[str, int] = {}
    for label, subset in by_label.items():
        target_count = int(majority * max_ratio)
        if len(subset) >= target_count or not subset:
            continue
        need = target_count - len(subset)
        extra = [rng.choice(subset) for _ in range(need)]
        out.extend(extra)
        added[label] = need
    rng.shuffle(out)
    return out, {"applied": bool(added), "majority_count": majority, "added_per_label": added}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"messages": row["messages"], "cr_id": row["cr_id"]}, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare train/validation/holdout splits for Qwen2.5-7B QLoRA")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--char-budget-per-field", type=int, default=DEFAULT_CHAR_BUDGET_PER_FIELD)
    parser.add_argument(
        "--oversample-minority",
        action="store_true",
        help="Duplicate minority-class TRAIN rows toward a max_ratio of the majority class. Off by default.",
    )
    parser.add_argument("--oversample-max-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    prepared: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, Any] = {}
    for split in SPLITS:
        input_path = args.input_dir / f"{split}_reasoning.jsonl"
        if not input_path.exists():
            raise SystemExit(f"Missing {input_path}. Run training/build_dataset.py first.")
        rows, report = _prepare_split(input_path, char_budget_per_field=args.char_budget_per_field)
        prepared[split] = rows
        reports[split] = report

    # Defense-in-depth: build_dataset.py already splits by source CR id, but
    # verify no CR id leaked across splits before anything is uploaded/trained.
    id_sets = {split: {row["cr_id"] for row in rows} for split, rows in prepared.items()}
    overlaps = {
        f"{a}&{b}": sorted(id_sets[a] & id_sets[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
        if id_sets[a] & id_sets[b]
    }
    if overlaps:
        raise SystemExit(f"Cross-split CR id leakage detected, aborting: {json.dumps(overlaps)[:2000]}")

    oversample_report: dict[str, Any] = {"applied": False}
    if args.oversample_minority:
        prepared["train"], oversample_report = _oversample_minority(
            prepared["train"], max_ratio=args.oversample_max_ratio, seed=args.seed
        )

    for split in SPLITS:
        write_jsonl(args.output_dir / SPLIT_TO_OUTPUT_NAME[split], prepared[split])

    full_report = {
        "splits": reports,
        "cross_split_leakage": overlaps or None,
        "oversample_minority": oversample_report,
        "char_budget_per_field": args.char_budget_per_field,
        "outputs": {split: str(args.output_dir / SPLIT_TO_OUTPUT_NAME[split]) for split in SPLITS},
        "note": "Only qwen_qlora_train.jsonl is used for gradient updates. Validation is for eval_loss/model "
        "selection; holdout is for final evaluation only.",
    }
    (args.output_dir / "qwen_qlora_prep_report.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(json.dumps(full_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
