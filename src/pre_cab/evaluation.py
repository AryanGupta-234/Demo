"""Offline benchmark helpers for measuring Pre-CAB prediction quality."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .schemas import Decision


@dataclass(frozen=True)
class BenchmarkRow:
    change_id: str
    predicted: Decision
    actual: Decision | None
    false_pass: bool
    correct: bool


@dataclass(frozen=True)
class BenchmarkSummary:
    total: int
    scored: int
    correct: int
    accuracy: float
    false_passes: int
    false_pass_rate: float


def benchmark(
    records: Iterable[dict[str, Any]],
    predictor: Callable[[dict[str, Any]], Decision],
    actual_field: str = "historical_prediction_label",
) -> tuple[list[BenchmarkRow], BenchmarkSummary]:
    """Score a predictor while explicitly surfacing dangerous false-pass errors.

    The actual label must be supplied by a controlled benchmark-preparation step. The helper does
    not infer ground truth from free-form CAB text.
    """
    rows: list[BenchmarkRow] = []
    for record in records:
        predicted = predictor(record)
        raw_actual = record.get(actual_field)
        actual = Decision(str(raw_actual)) if raw_actual in {d.value for d in Decision} else None
        false_pass = predicted == Decision.PASS and actual in {Decision.CONDITIONAL, Decision.NOT_READY}
        rows.append(
            BenchmarkRow(
                change_id=str(record.get("Number") or record.get("Effective number") or "unknown"),
                predicted=predicted,
                actual=actual,
                false_pass=false_pass,
                correct=actual is not None and predicted == actual,
            )
        )

    scored = [r for r in rows if r.actual is not None]
    correct = sum(r.correct for r in scored)
    false_passes = sum(r.false_pass for r in scored)
    total = len(rows)
    return rows, BenchmarkSummary(
        total=total,
        scored=len(scored),
        correct=correct,
        accuracy=(correct / len(scored)) if scored else 0.0,
        false_passes=false_passes,
        false_pass_rate=(false_passes / len(scored)) if scored else 0.0,
    )
