"""Build a compact, outcome-aware but prediction-safe historical context pack."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
    return value not in (None, "", [], {}) and str(value).strip().lower() not in {"none", "null", "nan", "n/a", "na", "not applicable"}


def _compact_cr(cr: dict[str, Any]) -> dict[str, Any]:
    """Encode the mapped CR in a context-efficient card (~200-300 chars typical)."""
    card: dict[str, Any] = {}
    for field, limit in (
        ("Number", 16), ("Short description", 90), ("Category", 24), ("Sub Category", 24),
        ("Change Class", 24), ("Environment", 12), ("Risk", 12), ("Configuration item", 35),
    ):
        value = cr.get(field)
        if _present(value):
            card[field] = str(value).strip()[:limit]
    # Presence maps preserve the training field requirements without spending
    # hundreds of tokens repeating long implementation/test narratives.
    card["evidence_presence"] = {
        "implementation": _present(cr.get("Implementation plan")),
        "backout": _present(cr.get("Backout plan")),
        "test_plan": _present(cr.get("Test plan")),
        "risk_impact": _present(cr.get("Risk and impact analysis")),
        "customer_approval": _present(cr.get("Customer Approval")),
        "uat_signoff": _present(cr.get("UAT signoff")),
        "test_results": _present(cr.get("Test Results Evidence")),
        "lower_env": _present(cr.get("Lower Environment Reference CR/SR")),
        "conflict": _present(cr.get("Conflict status")),
    }
    return card


def _compact_requirements(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    gaps = []
    for item in value.get("gaps", [])[:8]:
        if isinstance(item, dict):
            gaps.append(f"{item.get('field')}:{item.get('requirement_level')}")
    notes = []
    for item in value.get("note_signals", [])[:4]:
        if isinstance(item, dict) and item.get("phrase"):
            notes.append(str(item["phrase"])[:45])
    return {"scope": value.get("resolved_scope"), "gaps": gaps, "note_signals": notes}


def _compact_notes(value: str) -> str:
    value = str(value or "").strip()
    if len(value) <= 160:
        return value
    return value[:80] + " ... " + value[-75:]


def build(train_jsonl: Path, notes_jsonl: Path | None) -> dict[str, Any]:
    notes: dict[str, str] = {}
    if notes_jsonl and notes_jsonl.exists():
        with notes_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    notes[str(row.get("cr_id"))] = str(row.get("notes") or "")

    records: list[dict[str, Any]] = []
    distributions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    field_requirements: defaultdict[str, Counter[str]] = defaultdict(Counter)
    note_phrases: Counter[str] = Counter()

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
            for signal in fr.get("note_signals", [])[:5]:
                if isinstance(signal, dict) and signal.get("phrase"):
                    note_phrases[str(signal["phrase"])] += 1
            records.append({
                "cr": _compact_cr(cr),
                "requirements": _compact_requirements(fr),
                "work_notes": _compact_notes(notes.get(cr_id, "")),
            })

    buckets = [
        {"context": key, "records": sum(counts.values()), "outcomes": dict(counts), "common_requirement_gaps": field_requirements[key].most_common(6)}
        for key, counts in sorted(distributions.items(), key=lambda item: -sum(item[1].values()))
    ]
    return {
        "version": 4,
        "purpose": "full_mapped_training_corpus_as_compact_reference_without_per_record_prediction_leakage",
        "records_mapped": len(records),
        "context_buckets": buckets,
        "historical_note_phrases": note_phrases.most_common(40),
        "records": records,
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
    print(f"Wrote {pack['records_mapped']} mapped records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
