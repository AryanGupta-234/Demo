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
from pre_cab.rules import context_flags


# Ordered REQUIRED -> CONDITIONAL -> OPTIONAL, per the real 2,013-record historical
# export mined into config/field_requirements.generated.json (global bucket). Most
# decision-relevant fields come first so a compact/truncated context still carries
# the highest-value fields. "Change plan" is deliberately excluded: it is
# NOT_OBSERVED (0.00 fill rate) across every real Normal CR in the dataset - this
# org has never once used it, so it is pure noise to a model learning this org's
# actual patterns, not this org's actual behavior.
DESCRIPTIVE_FIELDS = [
    "Short description", "Description", "Justification", "Implementation plan", "Priority",
    "Test plan", "Backout plan", "Risk", "Risk and impact analysis", "Change Class",
    "Configuration item", "Environment",
]
SIGNOFF_FIELDS = [
    "Customer Approval", "Test Results Evidence", "UAT signoff",
    "TCS QA signoff", "Lower Environment Reference CR/SR",
]
CONTEXT_FIELDS = [
    "Number", "Type", "Category", "Sub Category",
    "Conflict status", "Planned start", "Planned end",
]
NOTE_FIELDS = ["Comments and Work notes", "Work notes", "Notes"]

NOTE_HEADER_RE = re.compile(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} - [^()]+?\([^)]*\)\s*", re.MULTILINE)

FIELD_REQUIREMENT_ENGINE = FieldRequirementEngine()

SYSTEM_PROMPT = (
    "You are the Pre-CAB reasoning specialist for a real ServiceNow Change Advisory "
    "Board pipeline. You are given: (1) the CR's own fields, selected because the "
    "org's real historical data shows they carry decision-relevant signal, and (2) "
    "field_requirements - a deterministic, data-mined verdict per field for this "
    "CR's specific Category/Sub Category, already computed from real historical "
    "fill/disposition rates. field_requirements is evidence, not a suggestion: "
    "treat a REQUIRED-and-unsatisfied field as a real gap, and do not claim a field "
    "is satisfied when its own entry says otherwise. Where field_requirements is "
    "silent or its scope is 'global' (a category with little historical precedent), "
    "say so as an uncertainty rather than guessing organizational policy.\n\n"
    "Decision framework:\n"
    "PASS - no blocking issues and sufficient confidence/evidence.\n"
    "CONDITIONAL - no hard blocker, but specific unresolved items remain before approval.\n"
    "NOT_READY - a hard blocker, a major contradiction, or material missing mandatory evidence.\n\n"
    "Rules: UAT is contextual, not universally mandatory - infer applicability from "
    "the change type, do not assume every change needs it. A short rollback "
    "description can still be credible; judge the mechanism, not the word count. "
    "Never invent evidence, approvals, testing, or historical outcomes that are not "
    "present in the given context. A contradiction (e.g. declared risk is low but "
    "the impact narrative describes significant production/customer exposure) must "
    "be surfaced explicitly, not smoothed over.\n\n"
    "Respond with exactly these keys: facts, technical_reasoning, testing_reasoning, "
    "risk_reasoning, uncertainties, contradictions, cab_questions, recommendations, "
    "prediction. The four *_reasoning arrays are terse declarative bullet lines "
    "(e.g. 'implementation is present', 'no direct contradiction'), matching this "
    "system's existing deterministic agents so your output can sit alongside theirs."
)


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
    # Defense in depth: the allowlists above are the primary leakage control, but
    # if a post-decision field (CAB Outcome, State, Close notes, Approval history,
    # ...) is ever accidentally added to one of them later, this still keeps it
    # out of the model-visible context rather than relying solely on the allowlist
    # having been written correctly.
    return strip_post_decision_fields(selected)


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
    """Reuse the real, production TechnicalAgent so the fine-tuned model learns to
    mimic exactly what the deterministic system already produces, rather than a
    second, slightly-different reimplementation drifting out of sync over time."""
    from pre_cab.agents import TechnicalAgent
    from pre_cab.schemas import AgentContext

    result = TechnicalAgent().run(AgentContext(cr=row))
    chain = result.notes.get("chain", [])
    uncertainties = [f"technical uncertainty: {result.notes.get('technical_uncertainty', 'unknown')}"]
    return chain, uncertainties


def testing_chain(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    from pre_cab.agents import TestingAgent
    from pre_cab.schemas import AgentContext

    result = TestingAgent().run(AgentContext(cr=row))
    chain = result.notes.get("chain", [])
    uncertainties = [] if result.notes.get("functional_coverage_ok") else ["functional test coverage is not fully confirmed"]
    return chain, uncertainties


def risk_chain(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    from pre_cab.agents import RiskAgent
    from pre_cab.schemas import AgentContext

    result = RiskAgent().run(AgentContext(cr=row))
    chain = result.notes.get("chain", [])
    contradictions = (
        [f"declared risk ({row.get('Risk') or 'unknown'}) conflicts with the described impact"]
        if result.notes.get("contradiction") else []
    )
    return chain, contradictions


def field_requirement_context(row: dict[str, Any]) -> dict[str, Any]:
    """The exact same data-mined per-category verdicts the production decision
    gate uses (see field_requirement_engine.py) - this is the primary answer to
    "only feed the model the fields that actually matter": rather than a flat
    field dump, the model sees which fields THIS org's real history says are
    REQUIRED/CONDITIONAL/OPTIONAL for THIS CR's specific Category/Sub Category,
    with the real support counts behind each verdict.
    """
    report = FIELD_REQUIREMENT_ENGINE.evaluate(row)
    return {
        "resolved_scope": report.resolved_scope,
        "gaps": [
            {"field": f.field, "requirement_level": f.requirement_level.value, "evidence": f.evidence}
            for f in report.findings
            if not f.satisfied and f.requirement_level in (RequirementLevel.REQUIRED, RequirementLevel.CONDITIONAL)
        ],
        "note_signals": [
            {"phrase": s.phrase, "evidence": s.evidence} for s in report.note_signals
        ],
    }


def _cab_questions(fr_context: dict[str, Any], test_uncertainties: list[str]) -> list[str]:
    questions: list[str] = []
    if test_uncertainties:
        questions.append("What functional scenarios were tested?")
    if fr_context["gaps"]:
        gap = fr_context["gaps"][0]["field"]
        questions.append(f"Can {gap} be confirmed or explicitly justified as not applicable?")
    if fr_context["note_signals"]:
        questions.append("Do this CR's work notes indicate it may be withdrawn or reworked?")
    return questions[:4] or ["Are there any residual concerns not captured in the CR fields?"]


def _recommendations(fr_context: dict[str, Any]) -> list[str]:
    recs = [f"Populate or justify {gap['field']} before CAB review." for gap in fr_context["gaps"][:3]]
    recs.append("Confirm final CAB evidence before approval.")
    return recs


def build_example(row: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = select_fields(row, profiles)
    notes = note_text(row)
    fr_context = field_requirement_context(row)
    tech_reasoning, tech_uncertainties = technical_chain(row)
    test_reasoning, test_uncertainties = testing_chain(row)
    risk_reasoning, risk_contradictions = risk_chain(row)

    target = {
        "facts": [f"{k} is populated" for k, v in selected.items() if present(v)],
        "technical_reasoning": tech_reasoning,
        "testing_reasoning": test_reasoning,
        "risk_reasoning": risk_reasoning,
        "uncertainties": tech_uncertainties + test_uncertainties,
        "contradictions": risk_contradictions,
        "cab_questions": _cab_questions(fr_context, test_uncertainties),
        "recommendations": _recommendations(fr_context),
    }
    label = outcome(row)
    if label is not None:
        target["prediction"] = label.value
    user_context = {
        "task": "PRE_CAB_ANALYSIS",
        "cr": selected,
        "field_requirements": fr_context,
        "note_signal": {
            "available": bool(notes),
            "note_fields_present": [f for f in NOTE_FIELDS if present(row.get(f))],
        },
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
