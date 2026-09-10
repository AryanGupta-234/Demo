"""Replayable structured trace for a single Pre-CAB evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceStep:
    actor: str
    action: str
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class EvaluationTrace:
    cr_number: str
    strictness: str
    steps: list[TraceStep] = field(default_factory=list)

    def add(self, actor: str, action: str, *, input_summary=None, output_summary=None) -> None:
        self.steps.append(
            TraceStep(
                actor=actor,
                action=action,
                input_summary=input_summary or {},
                output_summary=output_summary or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
