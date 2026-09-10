"""Post-run analysis helpers for finding actionable benchmark failures."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .benchmark_runner import BenchmarkPrediction


def failure_buckets(predictions: Iterable[BenchmarkPrediction]) -> dict[str, Any]:
    rows = list(predictions)
    false_pass = [row for row in rows if row.false_pass]
    false_fail = [row for row in rows if row.false_fail]
    wrong = [row for row in rows if not row.correct]
    return {
        "false_pass_count": len(false_pass),
        "false_fail_count": len(false_fail),
        "wrong_count": len(wrong),
        "false_pass_finding_codes": dict(Counter(code for row in false_pass for code in row.finding_codes).most_common()),
        "false_fail_finding_codes": dict(Counter(code for row in false_fail for code in row.finding_codes).most_common()),
        "wrong_source_ids": [row.source_id for row in wrong],
    }
