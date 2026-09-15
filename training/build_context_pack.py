"""Build a compact, outcome-aware but prediction-safe historical context pack."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DESCRIPTIVE_FIELDS = (
    "Short description", "Description", "Justification", "Implementation plan",
    "Change plan", "Backout plan", "Work notes", "Comments", "Test plan",
    "Configuration item", "Risk", "Priority",
)
SIGNOFF_FIELDS = (
    "UAT signoff", "Customer Approval", "TCS QA signoff",
    "Test Results Evidence", "Lower Environment Reference CR/SR",
)


def _user_context(example: dict[str, Any]) -> dict[str, Any]:
    for message in example.get("messages") or []:
        if message.get("role") == "user":
            try:
                value = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def _target(example: dict[str, Any]) -> dict[str, Any]:
    for message in example.get("messages") or []:
        if message.get("role") == "assistant":
            try:
                value = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def _present(value: Any) -> bool:
    return value not in (None, "", [], {}) and str(value).strip().lower() not in {"none", "null", "nan"}


def _compact(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit // 2] + " ... " + text[-(limit // 2) :]


def _compact_cr(cr: dict[str, Any]) -> dict[str, Any]:
    """Keep all mapped field presence while retaining useful text for historical retrieval."""
    card: dict[str, Any] = {}
    for field, limit in (
        ("Number", 16), ("Short description", 120), ("Description", 500), ("Justification", 350),
        ("Implementation plan", 600), ("Change plan", 350), ("Backout plan", 500), ("Test plan", 500),
        ("Configuration item", 50), ("Risk", 30), ("Priority", 30), ("Category", 40),
        ("Sub Category", 40), ("Change Class", 40), ("Environment", 20), ("Conflict status", 30),
    ):
        if field in cr:
            value = cr.get(field)
            card[field] = _compact(value, limit) if value is not None else None
    card["descriptive_field_presence"] = {field: _present(cr.get(field)) for field in DESCRIPTIVE_FIELDS}
    card["signoff_dispositions"] = {
        field: str(cr.get(field)).strip() if _present(cr.get(field)) else None
        for field in SIGNOFF_FIELDS
    }
    return card


def _compact_requirements(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    gaps = []
    for item in value.get("gaps", [])[:12]:
        if isinstance(item, dict):
            gaps.append(f"{item.get('field')}:{item.get('requirement_level')}")
    notes = []
    for item in value.get("note_signals", [])[:6]:
        if isinstance(item, dict) and item.get("phrase"):
            notes.append(str(item["phrase"])[:60])
    return {"scope": value.get("resolved_scope"), "gaps": gaps, "note_signals": notes}


def _compact_notes(value: str, limit: int = 800) -> str:
    return _compact(value, limit)


def build(train_jsonl: Path, notes_jsonl: Path | None) -> dict[str, Any]:
    fallback: dict[str, dict[str, str]] = {}
    if notes_jsonl and notes_jsonl.exists():
        with notes_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    fallback[str(row.get("cr_id"))] = {
                        key: str(value or "") for key, value in row.items() if key != "cr_id"
                    }

    records: list[dict[str, Any]] = []
    distributions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    field_requirements: defaultdict[str, Counter[str]] = defaultdict(Counter)
    note_phrases: Counter[str] = Counter()
    embedded_journals = 0

    with train_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            context = _user_context(example)
            target = _target(example)
            cr = context.get("cr") if isinstance(context.get("cr"), dict) else {}
            meta = example.get("metadata") or {}
            cr_id = str(meta.get("cr_id") or cr.get("Number") or "")
            key = "::".join(str(cr.get(k) or "Unknown") for k in ("Category", "Sub Category", "Change Class"))

            distributions[key][str(target.get("prediction") or "UNSCORABLE")] += 1
            fr = context.get("field_requirements") or {}
            for gap in fr.get("gaps", [])[:12]:
                if isinstance(gap, dict):
                    field_requirements[key][f"{gap.get('field')}={gap.get('requirement_level')}"] += 1
            for signal in fr.get("note_signals", [])[:6]:
                if isinstance(signal, dict) and signal.get("phrase"):
                    note_phrases[str(signal["phrase"])] += 1

            work_notes = str(context.get("work_notes") or "").strip()
            comments = str(context.get("comments") or "").strip()
            legacy = str(context.get("legacy_comments_and_work_notes") or "").strip()
            fallback_row = fallback.get(cr_id, {})
            if not work_notes:
                work_notes = fallback_row.get("Work notes", "")
            if not comments:
                comments = fallback_row.get("Comments", "")
            if not legacy:
                legacy = fallback_row.get("Comments and Work notes", "")
            if work_notes or comments or legacy:
                embedded_journals += 1

            records.append({
                "cr": _compact_cr(cr),
                "requirements": _compact_requirements(fr),
                "work_notes": _compact_notes(work_notes),
                "comments": _compact_notes(comments),
                "legacy_comments_and_work_notes": _compact_notes(legacy),
                "journal_present": bool(work_notes or comments or legacy),
            })

    buckets = [
        {
            "context": key,
            "records": sum(counts.values()),
            "outcomes": dict(counts),
            "common_requirement_gaps": field_requirements[key].most_common(6),
        }
        for key, counts in sorted(distributions.items(), key=lambda item: -sum(item[1].values()))
    ]
    return {
        "version": 6,
        "purpose": "full_mapped_training_corpus_as_compact_reference_without_per_record_prediction_leakage",
        "schema": {
            "descriptive_fields": list(DESCRIPTIVE_FIELDS),
            "signoff_fields": list(SIGNOFF_FIELDS),
            "work_notes_and_comments_are_per_cr": True,
        },
        "records_mapped": len(records),
        "journal_embedded_records": embedded_journals,
        "context_buckets": buckets,
        "historical_note_phrases": note_phrases.most_common(40),
        "records": records,
        "prediction_policy": {
            "individual_historical_outcomes_hidden_from_reference_records": True,
            "bucket_outcomes_are_aggregate_context_only": True,
            "current_cr_prediction_must_be_derived_from_current_cr_evidence": True,
            "work_notes_and_comments_never_become_labels": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact full-corpus context for the Ollama reasoning brain")
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("--notes-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("training/output/pre_cab_context_pack.json"))
    args = parser.parse_args()
    pack = build(args.train_jsonl, args.notes_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {pack['records_mapped']} mapped records to {args.output} ({pack['journal_embedded_records']} with CR journal context)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
