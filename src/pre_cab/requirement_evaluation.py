"""Evaluate contextual requirement inference against reviewed ground truth.

Ground truth is intentionally explicit/reviewed rather than inferred from sparse ServiceNow fields.
This avoids declaring UAT/customer-approval labels based on historical field population alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .requirements import infer_requirements


@dataclass(frozen=True)
class RequirementTruth:
    cr_id: str
    requirements: dict[str, bool]


@dataclass(frozen=True)
class RequirementMetric:
    name: str
    total: int
    accuracy: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_requirements(
    crs: Iterable[dict[str, Any]],
    truth: Iterable[RequirementTruth],
) -> dict[str, RequirementMetric]:
    rows = {
        str(row.get("Number") or row.get("Effective number") or ""): row
        for row in crs
    }
    counts: dict[str, dict[str, int]] = {}

    for item in truth:
        cr = rows.get(item.cr_id)
        if not cr:
            continue
        predicted = {prediction.name: prediction.required for prediction in infer_requirements(cr)}
        for name, actual in item.requirements.items():
            if name not in predicted:
                continue
            pred = predicted[name]
            bucket = counts.setdefault(name, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
            if pred and actual:
                bucket["tp"] += 1
            elif pred and not actual:
                bucket["fp"] += 1
            elif not pred and actual:
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1

    result: dict[str, RequirementMetric] = {}
    for name, bucket in counts.items():
        tp, tn, fp, fn = bucket["tp"], bucket["tn"], bucket["fp"], bucket["fn"]
        total = tp + tn + fp + fn
        result[name] = RequirementMetric(
            name=name,
            total=total,
            accuracy=(tp + tn) / total if total else 0.0,
            precision=tp / (tp + fp) if tp + fp else 0.0,
            recall=tp / (tp + fn) if tp + fn else 0.0,
            false_positive_rate=fp / (fp + tn) if fp + tn else 0.0,
            false_negative_rate=fn / (fn + tp) if fn + tp else 0.0,
        )
    return result
