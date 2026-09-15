"""Build persistent organization-specific Qwen knowledge from labeled historical CRs.

This is an application-level learning pass: Qwen reads historical CR examples and
summarizes patterns into a reusable JSON artifact. It does NOT update model weights.
The artifact is consumed by scripts/predict_qwen.py during the prediction pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_cab.runtime_provider import build_runtime_provider

LEARN_SYSTEM = (
    "You are the historical-learning stage of a Pre-CAB ServiceNow CR validator. "
    "Analyze the supplied labeled historical CR examples and extract organization-specific "
    "patterns that can help a later prediction stage. Learn relationships among the fixed "
    "descriptive fields, signoff dispositions, requirements, Work Notes, Comments, and the "
    "historical CAB outcome. Do not copy a single record as a rule. Prefer recurring patterns, "
    "exceptions, contradictions, and conditional behavior. Keep evidence-grounded statements. "
    "Return ONLY valid JSON."
)

SYNTHESIS_SYSTEM = (
    "You are the final synthesis stage for a Pre-CAB historical learning pass. Combine the "
    "chunk analyses into a compact, reusable organization knowledge artifact. Do not retain "
    "individual CR predictions as decision labels for future cases. Extract generalized patterns, "
    "field-level signals, signoff behavior, journal/Work Note/Comment patterns, recurring gaps, "
    "exceptions, and outcome tendencies. Make it explicit that this artifact is prior knowledge, "
    "not current-CR evidence. Return ONLY valid JSON."
)


def _json_response(provider: Any, *, system: str, payload: dict[str, Any], temperature: float = 0.0) -> dict[str, Any]:
    response = provider.generate(
        system=system,
        user=json.dumps(payload, ensure_ascii=False, default=str),
        temperature=temperature,
        response_format={"type": "json_object"},
        reasoning_effort=os.getenv("PRE_CAB_REASONING_EFFORT", "medium"),
    )
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Qwen returned non-JSON learning output: {response.text[:500]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Qwen learning output must be a JSON object")
    return parsed


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {path}") from exc
            if isinstance(value, dict):
                rows.append(value)
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def _messages(example: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    user: dict[str, Any] = {}
    target: dict[str, Any] = {}
    for message in example.get("messages") or []:
        if message.get("role") == "user":
            try:
                parsed = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                user = parsed
        elif message.get("role") == "assistant":
            try:
                parsed = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                target = parsed
    return user, target


def _training_view(example: dict[str, Any]) -> dict[str, Any]:
    user, target = _messages(example)
    cr = user.get("cr") if isinstance(user.get("cr"), dict) else {}
    metadata = example.get("metadata") or {}
    return {
        "cr_id": metadata.get("cr_id") or cr.get("Number"),
        "cr": cr,
        "field_requirements": user.get("field_requirements") or {},
        "descriptive_field_presence": user.get("descriptive_field_presence") or {},
        "signoff_dispositions": user.get("signoff_dispositions") or {},
        "work_notes": user.get("work_notes"),
        "comments": user.get("comments"),
        "legacy_comments_and_work_notes": user.get("legacy_comments_and_work_notes"),
        "historical_outcome": target.get("prediction"),
        "technical_reasoning": target.get("technical_reasoning") or [],
        "testing_reasoning": target.get("testing_reasoning") or [],
        "risk_reasoning": target.get("risk_reasoning") or [],
        "uncertainties": target.get("uncertainties") or [],
        "contradictions": target.get("contradictions") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn organization-specific Pre-CAB patterns with local Qwen via Ollama")
    parser.add_argument("--input", type=Path, default=Path("training/output/train_reasoning.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("training/output/qwen_learning.json"))
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="0 = all records")
    parser.add_argument("--provider", choices=["ollama"], default="ollama")
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")

    examples = [_training_view(e) for e in _load_examples(args.input)]
    if args.limit:
        examples = examples[: args.limit]

    provider = build_runtime_provider(args.provider)
    chunk_summaries: list[dict[str, Any]] = []
    total = (len(examples) + args.chunk_size - 1) // args.chunk_size
    for index in range(0, len(examples), args.chunk_size):
        chunk = examples[index : index + args.chunk_size]
        chunk_number = index // args.chunk_size + 1
        print(f"[LEARN {chunk_number}/{total}] {len(chunk)} historical CRs -> {provider.model_name}", flush=True)
        result = _json_response(
            provider,
            system=LEARN_SYSTEM,
            payload={
                "stage": "HISTORICAL_LEARNING_CHUNK",
                "schema": {
                    "descriptive_fields": [
                        "Short description", "Description", "Justification", "Implementation plan",
                        "Change plan", "Backout plan", "Work notes", "Comments", "Test plan",
                        "Configuration item", "Risk", "Priority",
                    ],
                    "signoff_fields": [
                        "UAT signoff", "Customer Approval", "TCS QA signoff",
                        "Test Results Evidence", "Lower Environment Reference CR/SR",
                    ],
                },
                "examples": chunk,
                "instructions": {
                    "use_outcomes_to_learn_patterns": True,
                    "use_work_notes_and_comments": True,
                    "keep_them_tied_to_same_cr": True,
                    "do_not_invent_policy": True,
                    "do_not_generalize_from_one_cr": True,
                },
            },
        )
        chunk_summaries.append({"chunk": chunk_number, "records": len(chunk), "analysis": result})

    print(f"[LEARN SYNTHESIS] {len(chunk_summaries)} chunk analyses -> persistent knowledge", flush=True)
    synthesis = _json_response(
        provider,
        system=SYNTHESIS_SYSTEM,
        payload={
            "stage": "HISTORICAL_LEARNING_SYNTHESIS",
            "records_analyzed": len(examples),
            "chunk_analyses": chunk_summaries,
            "required_output_sections": [
                "organization_patterns",
                "outcome_patterns",
                "descriptive_field_patterns",
                "signoff_patterns",
                "work_notes_patterns",
                "comments_patterns",
                "requirement_patterns",
                "contradictions_and_exceptions",
                "false_pass_risks",
                "prediction_guidance",
            ],
        },
    )

    artifact = {
        "version": 1,
        "type": "qwen_organization_learning",
        "model": provider.model_name,
        "records_analyzed": len(examples),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunk_summaries),
        "knowledge": synthesis,
        "source": {
            "input": str(args.input),
            "historical_labels_used_during_learning": True,
            "future_prediction_must_not_treat_historical_labels_as_current_evidence": True,
            "work_notes_and_comments_are_same_cr_journal_context": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote learned Qwen knowledge to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
