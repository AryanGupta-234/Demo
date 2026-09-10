"""Confidence calibration helpers for historical benchmark analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schemas import Decision


@dataclass(frozen=True)
class CalibrationPoint:
    confidence: float
    correct: bool
    predicted: Decision
    actual: Decision


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def calibration_bins(points: Iterable[CalibrationPoint], bins: int = 10) -> list[CalibrationBin]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    groups: list[list[CalibrationPoint]] = [[] for _ in range(bins)]
    for point in points:
        confidence = min(1.0, max(0.0, point.confidence))
        index = min(bins - 1, int(confidence * bins))
        groups[index].append(point)

    result: list[CalibrationBin] = []
    for index, group in enumerate(groups):
        lower = index / bins
        upper = (index + 1) / bins
        if not group:
            continue
        result.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(group),
                mean_confidence=sum(item.confidence for item in group) / len(group),
                accuracy=sum(item.correct for item in group) / len(group),
            )
        )
    return result


def expected_calibration_error(points: Iterable[CalibrationPoint], bins: int = 10) -> float:
    points_list = list(points)
    if not points_list:
        return 0.0
    grouped = calibration_bins(points_list, bins=bins)
    return sum(
        (bucket.count / len(points_list)) * abs(bucket.accuracy - bucket.mean_confidence)
        for bucket in grouped
    )
