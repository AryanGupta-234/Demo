"""Benchmark metrics focused on Pre-CAB safety and requirement inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import Decision


@dataclass(frozen=True)
class SafetyMetrics:
    total_scored: int
    accuracy: float
    false_passes: int
    false_pass_rate: float
    false_fails: int
    false_fail_rate: float
    pass_precision: float
    pass_recall: float
    conditional_recall: float
    not_ready_recall: float


def _recall(scored: list[tuple[Decision, Decision]], label: Decision) -> float:
    actual = sum(actual == label for _, actual in scored)
    true_positive = sum(pred == label and actual == label for pred, actual in scored)
    return true_positive / actual if actual else 0.0


def compute_safety_metrics(rows: Iterable[tuple[Decision, Decision | None]]) -> SafetyMetrics:
    scored = [(pred, actual) for pred, actual in rows if actual is not None]
    typed_scored: list[tuple[Decision, Decision]] = [(pred, actual) for pred, actual in scored if actual is not None]
    total = len(typed_scored)
    correct = sum(pred == actual for pred, actual in typed_scored)
    false_passes = sum(pred == Decision.PASS and actual != Decision.PASS for pred, actual in typed_scored)
    false_fails = sum(pred != Decision.PASS and actual == Decision.PASS for pred, actual in typed_scored)
    predicted_passes = sum(pred == Decision.PASS for pred, _ in typed_scored)
    actual_passes = sum(actual == Decision.PASS for _, actual in typed_scored)
    true_passes = sum(pred == Decision.PASS and actual == Decision.PASS for pred, actual in typed_scored)

    return SafetyMetrics(
        total_scored=total,
        accuracy=correct / total if total else 0.0,
        false_passes=false_passes,
        false_pass_rate=false_passes / total if total else 0.0,
        false_fails=false_fails,
        false_fail_rate=false_fails / total if total else 0.0,
        pass_precision=true_passes / predicted_passes if predicted_passes else 0.0,
        pass_recall=true_passes / actual_passes if actual_passes else 0.0,
        conditional_recall=_recall(typed_scored, Decision.CONDITIONAL),
        not_ready_recall=_recall(typed_scored, Decision.NOT_READY),
    )
