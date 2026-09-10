"""Metrics for reviewed requirement-level benchmark labels."""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def requirement_metrics(rows: Iterable[dict]) -> dict:
    """Return accuracy and per-requirement confusion counts for reviewed labels."""
    scored = [row for row in rows if row.get("actual") is not None and row.get("predicted") is not None]
    correct = sum(row["actual"] == row["predicted"] for row in scored)
    by_requirement: dict[str, dict[str, int | float]] = {}
    for name in sorted({str(row.get("requirement", "")) for row in scored}):
        subset = [row for row in scored if str(row.get("requirement", "")) == name]
        hits = sum(row["actual"] == row["predicted"] for row in subset)
        by_requirement[name] = {"count": len(subset), "accuracy": hits / len(subset) if subset else 0.0}
    return {
        "total_scored": len(scored),
        "accuracy": correct / len(scored) if scored else 0.0,
        "actual_distribution": dict(Counter(str(row["actual"]) for row in scored)),
        "predicted_distribution": dict(Counter(str(row["predicted"]) for row in scored)),
        "by_requirement": by_requirement,
    }
