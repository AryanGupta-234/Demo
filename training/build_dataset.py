from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_cab.cab_outcomes import normalize_cab_recommendation
from pre_cab.input_loader import load_cr_records, normalize_cr_record, record_type, source_id
from pre_cab.benchmark_leakage import strip_post_decision_fields


DESCRIPTIVE_FIELDS = [
    "Short description", "Description", "Justification", "Implementation plan",
    "Change plan", "Backout plan", "Test plan", "Configuration item", "Risk", "Priority",
    "Risk and impact analysis", "Environment", "Change Class",
]
SIGNOFF_FIELDS = [
    "UAT signoff", "Customer Approval", "TCS QA signoff",
    "Test Results Evidence", "Lower Environment Reference CR/SR",
]
CONTEXT_FIELDS = [
    "Number", "Type", "Category", "Sub Category", "Environment", "Change Class",
    "Conflict status", "Planned start", "Planned end",
]
NOTE_FIELDS = ["Comments and Work notes", "Work notes", "Notes"]

NOTE_HEADER_RE = re.compile(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} - [^()]+?\([^)]*\)\s*", re.MULTILINE)


def present(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return False
    return str(value).strip().lower() not in {"none", "null", "nan", "n/a", "na", "not applicable"}


def note_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in NOTE_FIELDS:
        value = row.get(field)
        if present(value):
            parts.append(str(value))
    return "\n".join(parts).strip()


def clean_note_text(value: str) -> str:
    return NOTE_HEADER_RE.sub(" ", value).strip()


def profile_fields(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = sorted({key for row in rows for key in row})
    result: dict[str, dict[str, Any]] = {}
    for field in fields:
        values = [row.get(field) for row in rows if present(row.get(field))]
        result[field] = {
            "rows": len(rows),
            "populated": len(values),
            "fill_rate": (len(values) / len(rows)) if rows else 0.0,
            "learned_role": (
                "NOT_OBSERVED" if not values else
                "HIGH_FREQUENCY" if len(values) / len(rows) >= 0.90 else
                "CONTEXTUAL" if len(values) / len(rows) >= 0.40 else
                "LOW_FREQUENCY"
            ) if rows else "NOT_OBSERVED",
        }
    return result


def selection_profile(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return profile_fields(rows)


def select_fields(row: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for field in DESCRIPTIVE_FIELDS:
        value = row.get(field)
        if present(value) and profiles.get(field, {}).get("learned_role") != "NOT_OBSERVED":
            selected[field] = value
    for field in SIGNOFF_FIELDS:
        if present(row.get(field)):
            selected[field] = row.get(field)
    for field in CONTEXT_FIELDS:
        if present(row.get(field)):
            selected[field] = row.get(field)
    return selected


def outcome(row: dict[str, Any]):
    raw = row.get("CAB Outcome") or row.get("CAB recommendation") or row.get("CAB Recommendation")
    return normalize_cab_recommendation(raw)


def note_signals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    corpus: Counter[str] = Counter()
    outcome_by_phrase: dict[str, Counter[str]] = {}
    for row in rows:
        text = clean_note_text(note_text(row)).lower()
        if not text:
            continue
        label = outcome(row)
        if label is None:
            continue
        tokens = sorted(set(re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", text)))
        for token in tokens:
            corpus[token] += 1
            outcome_by_phrase.setdefault(token, Counter())[label.value] += 1
    return {
        "note_records_with_outcomes": sum(bool(note_text(r)) and outcome(r) is not None for r in rows),
        "top_tokens": [
            {"token": token, "support": count, "outcomes": dict(outcome_by_phrase[token])}
            for token, count in corpus.most_common(100)
        ],
    }


def technical_chain(row: dict[str, Any]) -> list[str]:
    implementation = str(row.get("Implementation plan") or "").strip()
    backout = str(row.get("Backout plan") or "").strip()
    ci = str(row.get("Configuration item") or "").strip()
    lower = str(row.get("Lower Environment Reference CR/SR") or "").strip()
    chain = [
        "implementation is present" if implementation else "implementation is not described",
        "configuration item identified" if ci else "configuration item not identified",
        "rollback text is present" if backout else "rollback mechanism not described",
        "lower-environment reference present" if lower else "no explicit lower-environment reference",
    ]
    return chain


def testing_chain(row: dict[str, Any]) -> list[str]:
    test_plan = str(row.get("Test plan") or "").strip()
    evidence = str(row.get("Test Results Evidence") or "").strip()
    uat = str(row.get("UAT signoff") or "").strip()
    return [
        "test plan present" if test_plan else "test plan missing",
        "test execution evidence present" if evidence else "no test execution evidence",
        f"UAT disposition = {uat}" if uat else "UAT disposition not recorded",
    ]


def build_example(row: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = select_fields(row, profiles)
    notes = note_text(row)
    target = {
        "facts": [f"{k} is populated" for k, v in selected.items() if present(v)],
        "technical_reasoning": technical_chain(row),
        "testing_reasoning": testing_chain(row),
        "uncertainties": [],
        "contradictions": [],
        "cab_questions": [],
        "recommendations": [],
    }
    label = outcome(row)
    if label is not None:
        target["prediction"] = label.value
    user_context = {
        "task": "PRE_CAB_ANALYSIS",
        "cr": selected,
        "note_signal": {
            "available": bool(notes),
            "note_fields_present": [f for f in NOTE_FIELDS if present(row.get(f))],
        },
    }
    return {
        "messages": [
            {"role": "system", "content": "You are a Pre-CAB reasoning specialist. Separate facts, inferences, uncertainty, contradictions, and recommendations. UAT is contextual. Never invent evidence."},
            {"role": "user", "content": json.dumps(user_context, ensure_ascii=False, default=str)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, default=str)},
        ],
        "metadata": {
            "cr_id": source_id(row),
            "historical_outcome": label.value if label else None,
            "has_notes": bool(notes),
            "selected_field_count": len(selected),
        },
    }


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list, list, list]:
    ids: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ids.setdefault(source_id(row), []).append(row)
    keys = list(ids)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    a = int(n * 0.70)
    b = a + int(n * 0.15)
    return (
        [r for k in keys[:a] for r in ids[k]],
        [r for k in keys[a:b] for r in ids[k]],
        [r for k in keys[b:] for r in ids[k]],
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build adaptive Pre-CAB training datasets from historical CR JSON")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/output"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = [normalize_cr_record(r) for r in load_cr_records(args.input)]
    normal = [r for r in rows if record_type(r) == "normal"]
    train_rows, valid_rows, holdout_rows = split_rows(normal, args.seed)
    profiles = selection_profile(train_rows)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for name, subset in (("train", train_rows), ("validation", valid_rows), ("holdout", holdout_rows)):
        examples = [build_example(row, profiles) for row in subset]
        write_jsonl(out / f"{name}_reasoning.jsonl", examples)

    raw_notes = []
    for row in normal:
        notes = note_text(row)
        if notes:
            raw_notes.append({"cr_id": source_id(row), "notes": notes})
    write_jsonl(out / "notes_raw.jsonl", raw_notes)

    # Note-derived signals are stored separately and never used as decision labels.
    (out / "notes_signals.json").write_text(json.dumps(note_signals(train_rows), indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "field_profile.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")

    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()[:16]
    manifest = {
        "dataset_version": f"pre-cab-{digest}",
        "source_records": len(rows),
        "normal_records": len(normal),
        "train_records": len(train_rows),
        "validation_records": len(valid_rows),
        "holdout_records": len(holdout_rows),
        "train_examples": len(train_rows),
        "historical_outcome_counts": dict(Counter((outcome(r).value if outcome(r) else "UNSCORABLE") for r in normal)),
        "notes_records": len(raw_notes),
        "note_fields": NOTE_FIELDS,
        "seed": args.seed,
        "leakage_policy": "known post-decision fields and raw notes are excluded from decision labels; notes are retained as a separate auxiliary signal stream",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
