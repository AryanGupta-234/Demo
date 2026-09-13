"""Shared local JSON export loader used by profiling and benchmark tools."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _find_records(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    if isinstance(payload, dict):
        # Prefer common ServiceNow/export envelopes first.
        for key in ("result", "records", "changes", "data", "items", "results"):
            if key in payload:
                found = _find_records(payload[key])
                if found is not None:
                    return found
        # Some exports wrap records one or more levels deeper under arbitrary keys.
        for value in payload.values():
            found = _find_records(value)
            if found is not None:
                return found
    return None


def load_cr_records(path: str | Path) -> list[dict[str, Any]]:
    """Load CR records from bare lists or nested ServiceNow/export envelopes."""
    source = Path(path)
    # ``utf-8-sig`` also accepts exports saved by Excel/Windows tools with a BOM.
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    records = _find_records(payload)
    if records is None:
        raise ValueError("Input JSON does not contain CR records as a list or nested export envelope")
    return records


def first_value(record: dict[str, Any], *names: str) -> Any:
    """Case-insensitive field lookup for common ServiceNow/export naming variants."""
    lowered = {str(key).strip().lower(): value for key, value in record.items()}
    for name in names:
        value = lowered.get(name.strip().lower())
        if value is not None:
            return value
    return None


def normalize_cr_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with the validator's canonical CR keys populated.

    ServiceNow JSON exports commonly vary only by case, spaces, underscores, or
    their display label.  Keeping the original fields while adding these aliases
    lets the deterministic rules operate consistently without tying the local
    JSON workflow to a particular export or API format.
    """
    normalized = dict(record)
    aliases = {
        "Number": ("Number", "Effective number", "Change Number", "change_number", "sys_id"),
        "Type": ("Type", "Change type", "change_type", "Change class", "change_class"),
        "Short description": ("Short description", "short_description", "Summary", "Title"),
        "Description": ("Description", "description", "Details"),
        "Justification": ("Justification", "Business justification", "business_justification"),
        "Implementation plan": ("Implementation plan", "Implementation", "implementation_plan", "Plan"),
        "Backout plan": ("Backout plan", "Rollback plan", "backout_plan", "rollback_plan"),
        "Test plan": ("Test plan", "test_plan", "Testing plan"),
        "Test Results Evidence": ("Test Results Evidence", "Test results", "test_results_evidence"),
        "UAT signoff": ("UAT signoff", "UAT", "uat_signoff", "UAT approval"),
        "Customer Approval": ("Customer Approval", "customer_approval", "Customer signoff"),
        "Risk": ("Risk", "Risk level", "risk_level"),
        "Risk and impact analysis": ("Risk and impact analysis", "Impact analysis", "risk_impact_analysis"),
        "Configuration item": ("Configuration item", "Configuration Item", "CI", "cmdb_ci"),
        "Category": ("Category", "category"),
        "Sub Category": ("Sub Category", "Subcategory", "sub_category", "subcategory"),
        "Conflict status": ("Conflict status", "Conflict Status", "conflict_status"),
        "Planned start": ("Planned start", "Planned start date", "planned_start_date", "Start date"),
        "Planned end": ("Planned end", "Planned end date", "planned_end_date", "End date"),
        "Environment": ("Environment", "environment", "Target environment"),
        "TCS QA signoff": ("TCS QA signoff", "tcs_qa_signoff", "QA signoff"),
        "Lower Environment Reference CR/SR": (
            "Lower Environment Reference CR/SR", "Lower Environment Reference", "lower_environment_reference",
        ),
    }
    for canonical, names in aliases.items():
        value = first_value(record, *names)
        if value is not None:
            normalized[canonical] = value

    # Real ServiceNow-export-via-Excel data commonly carries date/datetime columns
    # as raw Excel serial numbers (e.g. 45872.52) once converted to JSON, since the
    # xlsx->JSON step doesn't know those columns are dates. A raw float is useless
    # (and actively misleading) to anything downstream that reads it as text -
    # human-facing reports, CAB summaries, or a training example fed to an LLM
    # would all see a meaningless number instead of a date. Convert in place.
    for canonical in ("Planned start", "Planned end"):
        value = normalized.get(canonical)
        iso = _excel_serial_to_iso(value)
        if iso is not None:
            normalized[canonical] = iso
    return normalized


def _excel_serial_to_iso(value: Any) -> str | None:
    """Convert a plausible Excel serial date/datetime number to an ISO string.

    Returns None for anything that isn't a bare int/float in a plausible date
    range (roughly 1970-2100) - a real date string, None, or other type passes
    through untouched by the caller.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # Excel's day-0 epoch is 1899-12-30 (accounts for Excel's 1900 leap-year bug).
    # Serial ~25569 = 1970-01-01; ~73050 = 2100-01-01 - outside that, it's very
    # unlikely to actually be a date serial, so leave it alone rather than guess.
    if not (25569 <= value <= 73050):
        return None
    epoch = datetime(1899, 12, 30)
    try:
        return (epoch + timedelta(days=float(value))).isoformat()
    except (OverflowError, ValueError):
        return None


def normalize_change_type(value: Any) -> str:
    """Canonicalize ServiceNow change-class variations into a single Normal/other label."""
    text = " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())
    if not text:
        return ""
    if text in {"normal", "normal change", "normalchange", "normal-change", "normal-change-request"}:
        return "normal"
    return text


def record_type(record: dict[str, Any]) -> str:
    value = first_value(record, "Type", "change type", "change_type", "type", "change_class", "change_classification")
    return normalize_change_type(value)


def source_id(record: dict[str, Any]) -> str:
    value = first_value(record, "Number", "Effective number", "number", "effective_number", "Change Number", "change_number", "sys_id")
    return str(value or "unknown")
