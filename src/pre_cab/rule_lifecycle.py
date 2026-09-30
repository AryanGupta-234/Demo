"""Governed automatic lifecycle promotion for mined field-requirement rules.

Candidate rules are evaluated against a separate holdout Normal-CR set. Promotion
is based on stability of the mined requirement class between the training table
and holdout observations. This is an automated validation gate, not an
unbounded self-modifying policy engine.
"""
from __future__ import annotations

import copy
import math
from datetime import UTC, datetime
from typing import Any


LIFECYCLE_ORDER = {
    "CANDIDATE": 0,
    "VALIDATED": 1,
    "ACTIVE": 2,
    "DEPRECATED": 3,
}

DEFAULT_VALIDATED_THRESHOLD = 0.80
DEFAULT_ACTIVE_THRESHOLD = 0.90
DEFAULT_MIN_HOLDOUT_RECORDS = 100
DEFAULT_REQUIRED_FILL_FLOOR = 0.85


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _blank(value: Any) -> bool:
    return _norm(value) in {"", "null", "none"}


def _observed_level(rule: dict[str, Any], records: list[dict[str, Any]], field: str) -> tuple[str, float]:
    if not records:
        return "NOT_OBSERVED", 0.0

    kind = str(rule.get("kind", "descriptive"))
    support = len(records)
    if kind == "signoff":
        values = [_norm(record.get(field)) for record in records if not _blank(record.get(field))]
        yes = sum(value in {"yes", "yes sr", "yes cr"} for value in values)
        no = sum(value == "no" for value in values)
        na = sum(value in {"not applicable", "na", "n/a"} for value in values)
        decided = yes + no + na
        decided_rate = decided / support if support else 0.0
        applicability_rate = ((yes + no) / decided) if decided else 0.0
        if decided_rate >= 0.30 and applicability_rate >= 0.60:
            level = "REQUIRED"
        elif decided_rate >= 0.15 and applicability_rate >= 0.35:
            level = "CONDITIONAL"
        elif applicability_rate > 0 or decided_rate > 0:
            level = "OPTIONAL"
        else:
            level = "NOT_OBSERVED"
        return level, decided_rate

    filled = sum(not _blank(record.get(field)) for record in records)
    fill_rate = filled / support if support else 0.0
    if fill_rate >= 0.90:
        level = "REQUIRED"
    elif fill_rate >= 0.50:
        level = "RECOMMENDED"
    elif fill_rate >= 0.05:
        level = "OPTIONAL"
    else:
        level = "NOT_OBSERVED"
    return level, fill_rate


def _scope_records(
    records: list[dict[str, Any]],
    *,
    scope_type: str,
    category: str | None = None,
    sub_category: str | None = None,
) -> list[dict[str, Any]]:
    normal = [record for record in records if _norm(record.get("Type")) == "normal"]
    if scope_type == "bucket":
        return [
            record for record in normal
            if _norm(record.get("Category")) == _norm(category)
            and _norm(record.get("Sub Category")) == _norm(sub_category)
        ]
    if scope_type == "category":
        return [record for record in normal if _norm(record.get("Category")) == _norm(category)]
    return normal


def _iter_rules(candidate: dict[str, Any]):
    for key, bucket in dict(candidate.get("buckets") or {}).items():
        category = str(bucket.get("category") or "")
        sub_category = str(bucket.get("sub_category") or "")
        for field, rule in dict(bucket.get("fields") or {}).items():
            yield "bucket", key, category, sub_category, field, rule

    for category, bucket in dict(candidate.get("categories") or {}).items():
        for field, rule in dict(bucket.get("fields") or {}).items():
            yield "category", str(category), str(category), None, field, rule

    for field, rule in dict((candidate.get("global") or {}).get("fields") or {}).items():
        yield "global", "global", None, None, field, rule


def validate_candidate(
    candidate: dict[str, Any],
    holdout_records: list[dict[str, Any]],
    *,
    min_holdout_records: int = DEFAULT_MIN_HOLDOUT_RECORDS,
    required_fill_floor: float = DEFAULT_REQUIRED_FILL_FLOOR,
) -> dict[str, Any]:
    """Measure requirement-level stability on a separate holdout dataset."""
    normal = [record for record in holdout_records if _norm(record.get("Type")) == "normal"]
    observations = []
    stable = 0
    evaluated = 0
    required_regressions = 0

    for scope_type, scope_key, category, sub_category, field, rule in _iter_rules(candidate):
        records = _scope_records(
            normal,
            scope_type=scope_type,
            category=category,
            sub_category=sub_category,
        )
        if len(records) < 5:
            continue
        candidate_level = str(rule.get("requirement_level", "OPTIONAL"))
        observed_level, observed_rate = _observed_level(rule, records, field)
        same = candidate_level == observed_level
        evaluated += 1
        stable += int(same)
        if candidate_level == "REQUIRED" and observed_rate < required_fill_floor:
            required_regressions += 1
        observations.append({
            "scope_type": scope_type,
            "scope": scope_key,
            "field": field,
            "candidate_level": candidate_level,
            "holdout_level": observed_level,
            "holdout_rate": round(observed_rate, 4),
            "stable": same,
            "holdout_support": len(records),
        })

    stability = stable / evaluated if evaluated else 0.0
    enough_data = len(normal) >= min_holdout_records
    passed_required_floor = required_regressions == 0
    if not enough_data:
        status = "CANDIDATE"
        reason = f"Holdout contains {len(normal)} Normal CRs; {min_holdout_records} are required."
    elif stability >= DEFAULT_ACTIVE_THRESHOLD and passed_required_floor:
        status = "ACTIVE"
        reason = "Holdout requirement classes are stable and all candidate REQUIRED rules meet the fill-rate floor."
    elif stability >= DEFAULT_VALIDATED_THRESHOLD:
        status = "VALIDATED"
        reason = "Holdout requirement classes are sufficiently stable for validation but did not meet the ACTIVE gate."
    else:
        status = "CANDIDATE"
        reason = "Holdout stability is below the VALIDATED threshold."

    return {
        "status": status,
        "reason": reason,
        "normal_holdout_records": len(normal),
        "rules_evaluated": evaluated,
        "rules_stable": stable,
        "stability_rate": round(stability, 4),
        "required_rule_regressions": required_regressions,
        "required_fill_floor": required_fill_floor,
        "validated_threshold": DEFAULT_VALIDATED_THRESHOLD,
        "active_threshold": DEFAULT_ACTIVE_THRESHOLD,
        "observations": observations,
    }


def automatically_promote(
    candidate: dict[str, Any],
    holdout_records: list[dict[str, Any]],
    *,
    min_holdout_records: int = DEFAULT_MIN_HOLDOUT_RECORDS,
    required_fill_floor: float = DEFAULT_REQUIRED_FILL_FLOOR,
) -> dict[str, Any]:
    """Validate and promote a candidate table without changing rule contents."""
    result = validate_candidate(
        candidate,
        holdout_records,
        min_holdout_records=min_holdout_records,
        required_fill_floor=required_fill_floor,
    )
    promoted = copy.deepcopy(candidate)
    prior = str(promoted.get("lifecycle_status", "CANDIDATE")).upper()
    status = str(result["status"])
    if LIFECYCLE_ORDER.get(status, 0) < LIFECYCLE_ORDER.get(prior, 0):
        status = prior

    promoted["lifecycle_status"] = status
    promoted["validation"] = {
        "method": "holdout_requirement_stability_v1",
        "validated_at": datetime.now(UTC).isoformat(),
        "source_holdout_record_count": len(holdout_records),
        "normal_holdout_record_count": result["normal_holdout_records"],
        "stability_rate": result["stability_rate"],
        "rules_evaluated": result["rules_evaluated"],
        "rules_stable": result["rules_stable"],
        "required_rule_regressions": result["required_rule_regressions"],
        "required_fill_floor": result["required_fill_floor"],
        "validated_threshold": result["validated_threshold"],
        "active_threshold": result["active_threshold"],
        "promotion_reason": result["reason"],
    }
    promoted["lifecycle_history"] = [
        *list(promoted.get("lifecycle_history") or []),
        {
            "from": prior,
            "to": status,
            "at": promoted["validation"]["validated_at"],
            "method": "automatic_holdout_validation",
            "stability_rate": result["stability_rate"],
        },
    ]
    return promoted
