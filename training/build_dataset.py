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
from pre_cab.field_requirement_engine import FieldRequirementEngine, RequirementLevel


# Exact model-facing descriptive field set requested for the CR understanding stage.
# Unlike signoffs, these are represented by both value and presence so absence itself
# remains observable to the model. Change plan is retained even when historically sparse.
DESCRIPTIVE_FIELDS = [
    "Short description",
    "Description",
    "Justification",
    "Implementation plan",
    "Change plan",
    "Backout plan",
    "Work notes",
    "Comments",
    "Test plan",
    "Configuration item",
    "Risk",
    "Priority",
]
SIGNOFF_FIELDS = [
    "UAT signoff",
    "Customer Approval",
    "TCS QA signoff",
    "Test Results Evidence",
    "Lower Environment Reference CR/SR",
]
CONTEXT_FIELDS = [
    "Number", "Type", "Category", "Sub Category",
    "Conflict status", "Planned start", "Planned end", "Change Class", "Environment",
]

# Some ServiceNow exports expose a combined journal column. Preserve it distinctly
# rather than pretending it is a clean Comments-only field. The canonical Work notes
# and Comments fields win when they are present.
LEGACY_COMBINED_NOTES_FIELD = "Comments and Work notes"
NOTE_COMPAT_FIELDS = ("Work notes", "Comments", LEGACY_COMBINED_NOTES_FIELD, "Notes")
NOTE_HEADER_RE = re.compile(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} - [^()]+?\([^)]*\)\s*", re.MULTILINE)
FIELD_REQUIREMENT_ENGINE = FieldRequirementEngine()

SYSTEM_PROMPT = (
    "You are the Pre-CAB CR analysis specialist. First understand the CR and map its "
    "requirements before making any CAB prediction. The input contains an optimized, "
    "fixed descriptive field set, disposition-aware signoff fields, contextual fields, "
    "and the actual Work Notes/Comments belonging to this same CR when available.\n\n"
    "Descriptive fields are evaluated by value and presence; Change plan is retained in "
    "the schema even when historical fill is low. Signoff fields must be interpreted by "
    "their disposition (Yes/No/Not Applicable/etc.), not merely by non-empty status.\n\n"
    "Work Notes are chronological auxiliary evidence from this CR. Use them to identify "
    "status, testing, approval, rollback, incident, rework, or other operational claims, "
    "but do not silently overwrite current field state. Comments are separate from Work "
    "Notes. If only the legacy combined journal field exists, identify that limitation.\n\n"
    "Requirements are deterministic historical signals for this CR's Category/Sub Category, "
    "not absolute policy truth. UAT is contextual. Never invent evidence, approvals, testing, "
    "historical outcomes, or requirements. First produce grounded analysis; CAB prediction "
    "is a separate later step."
)


def present(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return False
    return str(value).strip().lower() not in {"none", "null", "nan", "n/a", "na", "not applicable"}


def raw_present(value: Any) -> bool:
    """Presence for descriptive mapping: retain explicit N/A-like values as values."""
    return value is not None and value != "" and value != [] and value != {}


def _canonical_notes(row: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in ("Work notes", "Comments"):
        value = row.get(field)
        if raw_present(value):
            result[field] = str(value).strip()
    combined = row.get(LEGACY_COMBINED_NOTES_FIELD)
    if raw_present(combined):
        result[LEGACY_COMBINED_NOTES_FIELD] = str(combined).strip()
    notes = row.get("Notes")
    if raw_present(notes):
        result["Notes"] = str(notes).strip()
    return result


def note_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    notes = _canonical_notes(row)
    for field in ("Work notes", "Comments"):
        if field in notes:
            parts.append(f"[{field}]\n{notes[field]}")
    if LEGACY_COMBINED_NOTES_FIELD in notes:
        parts.append(f"[{LEGACY_COMBINED_NOTES_FIELD}]\n{notes[LEGACY_COMBINED_NOTES_FIELD]}")
    if "Notes" in notes:
        parts.append(f"[Notes]\n{notes['Notes']}")
    return "\n\n".join(parts).strip()


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
    # Keep every requested descriptive field in the schema. This makes field presence
    # learnable and prevents a missing field from being indistinguishable from omission.
    for field in DESCRIPTIVE_FIELDS:
        selected[field] = row.get(field) if raw_present(row.get(field)) else None
    for field in SIGNOFF_FIELDS:
        value = row.get(field)
        selected[field] = value if raw_present(value) else None
    for field in CONTEXT_FIELDS:
        value = row.get(field)
        if raw_present(value):
            selected[field] = value
    return strip_post_decision_fields(selected)


def descriptive_presence(row: dict[str, Any]) -> dict[str, bool]:
    return {field: raw_present(row.get(field)) for field in DESCRIPTIVE_FIELDS}


def signoff_dispositions(row: dict[str, Any]) -> dict[str, str | None]:
    return {
        field: str(row.get(field)).strip() if raw_present(row.get(field)) else None
        for field in SIGNOFF_FIELDS
    }


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


def technical_chain(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    from pre_cab.agents import TechnicalAgent
    from pre_cab.schemas import AgentContext
    result = TechnicalAgent().run(AgentContext(cr=row))
    return result.notes.get("chain", []), [f"technical uncertainty: {result.notes.get('technical_uncertainty', 'unknown')}"]


def testing_chain(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    from pre_cab.agents import TestingAgent
    from pre_cab.schemas import AgentContext
    result = TestingAgent().run(AgentContext(cr=row))
    uncertainties = [] if result.notes.get("functional_coverage_ok") else ["functional test coverage is not fully confirmed"]
    return result.notes.get("chain", []), uncertainties


def risk_chain(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    from pre_cab.agents import RiskAgent
    from pre_cab.schemas import AgentContext
    result = RiskAgent().run(AgentContext(cr=row))
    contradictions = [f"declared risk ({row.get('Risk') or 'unknown'}) conflicts with the described impact"] if result.notes.get("contradiction") else []
    return result.notes.get("chain", []), contradictions


def field_requirement_context(row: dict[str, Any]) -> dict[str, Any]:
    report = FIELD_REQUIREMENT_ENGINE.evaluate(row)
    return {
        "resolved_scope": report.resolved_scope,
        "gaps": [
            {"field": f.field, "requirement_level": f.requirement_level.value, "evidence": f.evidence}
            for f in report.findings
            if not f.satisfied and f.requirement_level in (RequirementLevel.REQUIRED, RequirementLevel.CONDITIONAL)
        ],
        "note_signals": [{"phrase": s.phrase, "evidence": s.evidence} for s in report.note_signals],
    }


def _cab_questions(fr_context: dict[str, Any], test_uncertainties: list[str], notes: str) -> list[str]:
    questions: list[str] = []
    if test_uncertainties:
        questions.append("What functional scenarios were tested and what execution evidence supports the claim?")
    if fr_context["gaps"]:
        questions.append(f"Can {fr_context['gaps'][0]['field']} be confirmed or explicitly justified as not applicable?")
    if notes:
        questions.append("Do the chronological Work Notes or Comments contain newer status, rework, approval, or rollback information that should be reconciled with the current CR fields?")
    return questions[:4] or ["Are there any residual concerns not captured in the CR fields, Work Notes, or Comments?"]


def _recommendations(fr_context: dict[str, Any], notes: str) -> list[str]:
    recs = [f"Populate or justify {gap['field']} before CAB review." for gap in fr_context["gaps"][:3]]
    if notes:
        recs.append("Reconcile any material Work Note or Comment claims with the current CR fields before approval.")
    recs.append("Confirm final CAB evidence before approval.")
    return recs[:4]


def build_example(row: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = select_fields(row, profiles)
    notes = note_text(row)
    fr_context = field_requirement_context(row)
    tech_reasoning, tech_uncertainties = technical_chain(row)
    test_reasoning, test_uncertainties = testing_chain(row)
    risk_reasoning, risk_contradictions = risk_chain(row)

    target = {
        "facts": [f"{k} is {'present' if raw_present(v) else 'missing'}" for k, v in selected.items() if k in DESCRIPTIVE_FIELDS],
        "descriptive_field_presence": descriptive_presence(row),
        "signoff_dispositions": signoff_dispositions(row),
        "technical_reasoning": tech_reasoning,
        "testing_reasoning": test_reasoning,
        "risk_reasoning": risk_reasoning,
        "uncertainties": tech_uncertainties + test_uncertainties,
        "contradictions": risk_contradictions,
        "cab_questions": _cab_questions(fr_context, test_uncertainties, notes),
        "recommendations": _recommendations(fr_context, notes),
    }
    label = outcome(row)
    if label is not None:
        target["prediction"] = label.value

    canonical_notes = _canonical_notes(row)
    user_context = {
        "task": "PRE_CAB_CR_UNDERSTANDING",
        "cr": selected,
        "field_requirements": fr_context,
        "descriptive_fields": DESCRIPTIVE_FIELDS,
        "signoff_fields": SIGNOFF_FIELDS,
        "context_fields": CONTEXT_FIELDS,
        "descriptive_field_presence": descriptive_presence(row),
        "signoff_dispositions": signoff_dispositions(row),
        "work_notes": canonical_notes.get("Work notes"),
        "comments": canonical_notes.get("Comments"),
        "legacy_comments_and_work_notes": canonical_notes.get(LEGACY_COMBINED_NOTES_FIELD),
        "other_notes": canonical_notes.get("Notes"),
        "work_notes_policy": {
            "same_cr_only": True,
            "role": "chronological auxiliary evidence",
            "use_for_context": True,
            "do_not_use_as_label": True,
            "do_not_silently_overwrite_current_fields": True,
            "distinguish_claims_from_verified_evidence": True,
        },
        "prediction_stage": "LATER_SEPARATE_STEP",
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_context, ensure_ascii=False, default=str)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, default=str)},
        ],
        "metadata": {
            "cr_id": source_id(row),
            "historical_outcome": label.value if label else None,
            "has_work_notes": bool(canonical_notes.get("Work notes")),
            "has_comments": bool(canonical_notes.get("Comments")),
            "has_legacy_combined_journal": bool(canonical_notes.get(LEGACY_COMBINED_NOTES_FIELD)),
            "notes_chars": len(note_text(row)),
            "selected_field_count": len(selected),
            "descriptive_field_count": len(DESCRIPTIVE_FIELDS),
            "signoff_field_count": len(SIGNOFF_FIELDS),
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
    return ([r for k in keys[:a] for r in ids[k]], [r for k in keys[a:b] for r in ids[k]], [r for k in keys[b:] for r in ids[k]])


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
        write_jsonl(out / f"{name}_reasoning.jsonl", [build_example(row, profiles) for row in subset])

    raw_notes = []
    for row in normal:
        canonical = _canonical_notes(row)
        if canonical:
            raw_notes.append({"cr_id": source_id(row), **canonical})
    write_jsonl(out / "notes_raw.jsonl", raw_notes)
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
        "note_fields": NOTE_COMPAT_FIELDS,
        "descriptive_fields": DESCRIPTIVE_FIELDS,
        "signoff_fields": SIGNOFF_FIELDS,
        "context_fields": CONTEXT_FIELDS,
        "notes_embedded_in_training_messages": True,
        "prediction_stage": "later_separate_step",
        "seed": args.seed,
        "leakage_policy": "post-decision fields are excluded from model-visible CR fields; Work Notes and Comments belong to the same CR and are included as auxiliary chronological evidence, never as decision labels",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
