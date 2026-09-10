"""Reusable leakage-safe benchmark engine for historical Normal CRs."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .benchmark_metrics import SafetyMetrics, compute_safety_metrics
from .benchmark_prepare import BenchmarkExample
from .calibration import CalibrationPoint, expected_calibration_error
from .orchestrator import run_stage1
from .schemas import Decision, Strictness


@dataclass(frozen=True)
class BenchmarkPrediction:
    source_id: str
    actual: Decision
    predicted: Decision
    confidence: float
    strictness: Strictness
    finding_codes: tuple[str, ...]
    deterministic_prediction: str | None = None
    model_prediction: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.actual

    @property
    def false_pass(self) -> bool:
        return self.predicted == Decision.PASS and self.actual != Decision.PASS

    @property
    def false_fail(self) -> bool:
        return self.predicted != Decision.PASS and self.actual == Decision.PASS


@dataclass(frozen=True)
class BenchmarkReport:
    strictness: Strictness
    metrics: SafetyMetrics
    predictions: tuple[BenchmarkPrediction, ...]
    confusion: dict[str, dict[str, int]]
    calibration_error: float
    failure_codes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strictness": self.strictness.value,
            "metrics": asdict(self.metrics),
            "confusion": self.confusion,
            "calibration_error": self.calibration_error,
            "failure_codes": self.failure_codes,
            "predictions": [
                {
                    **asdict(prediction),
                    "actual": prediction.actual.value,
                    "predicted": prediction.predicted.value,
                    "strictness": prediction.strictness.value,
                    "correct": prediction.correct,
                    "false_pass": prediction.false_pass,
                    "false_fail": prediction.false_fail,
                }
                for prediction in self.predictions
            ],
        }


def _confusion(predictions: Iterable[BenchmarkPrediction]) -> dict[str, dict[str, int]]:
    labels = [decision.value for decision in Decision]
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for item in predictions:
        matrix[item.actual.value][item.predicted.value] += 1
    return matrix


def run_benchmark(
    examples: Iterable[BenchmarkExample],
    *,
    strictness: Strictness = Strictness.BALANCED,
    model: Any = None,
    memory: Any = None,
) -> BenchmarkReport:
    predictions: list[BenchmarkPrediction] = []
    for example in examples:
        if example.actual is None:
            continue
        result = run_stage1(
            example.input_record,
            strictness=strictness,
            model=model,
            memory=memory,
        ).stage1
        predictions.append(
            BenchmarkPrediction(
                source_id=example.source_id,
                actual=example.actual,
                predicted=result.decision,
                confidence=result.confidence,
                strictness=strictness,
                finding_codes=tuple(finding.code for finding in result.findings),
                deterministic_prediction=result.metadata.get("deterministic_prediction"),
                model_prediction=result.metadata.get("model_prediction"),
            )
        )

    metrics = compute_safety_metrics((item.predicted, item.actual) for item in predictions)
    points = [
        CalibrationPoint(item.confidence, item.correct, item.predicted, item.actual)
        for item in predictions
    ]
    failures = Counter()
    for item in predictions:
        if item.correct:
            continue
        for code in item.finding_codes:
            failures[code] += 1

    return BenchmarkReport(
        strictness=strictness,
        metrics=metrics,
        predictions=tuple(predictions),
        confusion=_confusion(predictions),
        calibration_error=expected_calibration_error(points),
        failure_codes=dict(failures.most_common()),
    )


def strictness_sweep(
    examples: Iterable[BenchmarkExample],
    *,
    model: Any = None,
    memory: Any = None,
) -> dict[str, BenchmarkReport]:
    examples_list = list(examples)
    return {
        strictness.value: run_benchmark(
            examples_list,
            strictness=strictness,
            model=model,
            memory=memory,
        )
        for strictness in Strictness
    }
