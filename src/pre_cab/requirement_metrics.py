"""Metrics for reviewed requirement-level benchmark labels."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def _binary_scores(actual: list[str], predicted: list[str], positive: str) -> dict[str, Any]:
    tp = sum(a == positive and p == positive for a, p in zip(actual, predicted))
    fp = sum(a != positive and p == positive for a, p in zip(actual, predicted))
    fn = sum(a == positive and p != positive for a, p in zip(actual, predicted))
    tn = sum(a != positive and p != positive for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "positive": positive,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def requirement_metrics(rows: Iterable[dict]) -> dict:
    """Return accuracy and binary confusion metrics for reviewed labels.

    Requirement labels are intentionally evaluated only where both reviewed ``actual`` and model
    ``predicted`` values exist. For multi-valued requirements, precision/recall/F1 are reported
    one-vs-rest for every observed label rather than inventing an ordering between labels.
    """
    scored = [row for row in rows if row.get("actual") is not None and row.get("predicted") is not None]
    correct = sum(row["actual"] == row["predicted"] for row in scored)
    by_requirement: dict[str, dict[str, object]] = {}
    for name in sorted({str(row.get("requirement", "")) for row in scored}):
        subset = [row for row in scored if str(row.get("requirement", "")) == name]
        actual = [str(row["actual"]) for row in subset]
        predicted = [str(row["predicted"]) for row in subset]
        labels = sorted(set(actual) | set(predicted))
        hits = sum(a == p for a, p in zip(actual, predicted))
        by_requirement[name] = {
            "count": len(subset),
            "accuracy": hits / len(subset) if subset else 0.0,
            "labels": {label: _binary_scores(actual, predicted, label) for label in labels},
        }
    return {
        "total_scored": len(scored),
        "accuracy": correct / len(scored) if scored else 0.0,
        "actual_distribution": dict(Counter(str(row["actual"]) for row in scored)),
        "predicted_distribution": dict(Counter(str(row["predicted"]) for row in scored)),
        "by_requirement": by_requirement,
    }
