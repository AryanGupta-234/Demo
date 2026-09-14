"""Build an outcome-aware but prediction-safe historical context pack.

The pack is derived from the already-mapped reasoning JSONL plus the separate
work-notes stream. It is deliberately compact so a long-context model can see a
broad slice of the historical corpus without receiving the original 141-column
export.

It does NOT copy assistant predictions into the context pack. Historical outcomes
are retained only as aggregate distributions by category/sub-category/change-class
so the model can learn organizational patterns without being handed a matching
CR's answer verbatim.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _user_context(example: dict[str, Any]) -> dict[str, Any]:
    messages = example.get("messages") or []
    for message in messages:
        if message.get("role") == "user":
            try:
                value = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def _target(example: dict[str, Any]) -> dict[str, Any]:
    messages = example.get("messages") or []
    for message in messages:
        if message.get("role") == "assistant":
            try:
                value = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def _compact_cr(cr: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "Number", "Type", "Short description", "Description", "Justification",
        "Implementation plan", "Backout plan", "Test plan", "Risk",
        "Risk and impact analysis", "Change Class", "Configuration item",
        "Environment", "Category", "Sub Category", "Conflict status",
    )
    limits = {
        "Description": 420,
        "Justification": 220,
        "Implementation plan": 520,
        "Backout plan": 360,
        "Test plan": 360,
        "Risk and impact analysis": 360,
        "Short description": 220,
    }
    result: dict[str, Any] = {}
    for field in keep:
        value = cr.get(field)
        if value in (None, "", [], {}):
            continue
        text = str(value).strip()
        result[field] = text[: limits.get(field, 180)]
    return result


def _compact_requirements(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    gaps = []
    for item in value.get("gaps", [])[:8]:
        if isinstance(item, dict):
            gaps.append({
                "field": item.get("field"),
                "level": item.get("requirement_level"),
                "evidence": str(item.get("evidence") or "")[:220],
            })
    notes = []
    for item in value.get("note_signals", [])[:5]:
        if isinstance(item, dict):
            notes.append({"phrase": item.get("phrase"), "evidence": str(item.get("evidence") or "")[:180]})
    return {"scope": value.get("resolved_scope"), "gaps": gaps, "note_signals": notes}


def build(train_jsonl: Path, notes_jsonl: Path | None) -> dict[str, Any]:
    notes: dict[str, str] = {}
    if notes_jsonl and notes_jsonl.exists():
        with notes_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                notes[str(row.get("cr_id"))] = str(row.get("notes") or "")

    examples: list[dict[str, Any]] = []
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
            label = str(target.get("prediction") or "UNSCORABLE")
            distributions[key][label] += 1
            for gap in context.get("field_requirements", {}).get("gaps", [])[:12]:
                if isinstance(gap, dict):
                    field_requirements[key][f"{gap.get('field')}={gap.get('requirement_level')}"] += 1
            for signal in context.get("field_requirements", {}).get("note_signals", [])[:5]:
                if isinstance(signal, dict) and signal.get("phrase"):
                    note_phrases[str(signal["phrase"])] += 1
            raw_notes = notes.get(cr_id, "")
            # Keep the note trail as a separate evidence channel. Do not mix it into
            # the CR fields because a work note can describe a later state/change.
            examples.append({
                "cr": _compact_cr(cr),
                "requirements": _compact_requirements(context.get("field_requirements")),
                "work_notes": raw_notes[:500],
            })

    buckets = []
    for key, counts in sorted(distributions.items(), key=lambda item: -sum(item[1].values())):
        buckets.append({
            "context": key,
            "records": sum(counts.values()),
            "outcomes": dict(counts),
            "common_requirement_gaps": field_requirements[key].most_common(8),
        })

    return {
        "version": 2,
        "purpose": "historical_pre_cab_context_without_verbatim_prediction_leakage",
        "records_mapped": len(examples),
        "context_buckets": buckets,
        "historical_note_phrases": note_phrases.most_common(40),
        "records": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact historical context for the Ollama reasoning brain")
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
