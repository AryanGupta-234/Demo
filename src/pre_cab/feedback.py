"""Human/CAB feedback capture for episodic memory and later fine-tuning curation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from .memory import MemoryKind, MemoryRecord, UnifiedMemory
from .schemas import Decision


@dataclass(frozen=True)
class CABFeedback:
    cr_number: str
    predicted: Decision
    actual: Decision
    reviewer_notes: str = ""
    corrected_requirements: dict[str, bool] = field(default_factory=dict)
    lessons: tuple[str, ...] = ()
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now(UTC).isoformat())

    @property
    def correct(self) -> bool:
        return self.predicted == self.actual

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predicted"] = self.predicted.value
        payload["actual"] = self.actual.value
        payload["correct"] = self.correct
        return payload


def remember_feedback(memory: UnifiedMemory, feedback: CABFeedback) -> MemoryRecord:
    """Store a reviewer-confirmed outcome as an episodic memory record."""
    status = "correct prediction" if feedback.correct else "prediction correction"
    text_parts = [
        f"CAB feedback for {feedback.cr_number}: {status}.",
        f"Predicted={feedback.predicted.value}; Actual={feedback.actual.value}.",
    ]
    if feedback.reviewer_notes:
        text_parts.append(f"Reviewer notes: {feedback.reviewer_notes}")
    if feedback.corrected_requirements:
        text_parts.append(
            "Requirement corrections: "
            + ", ".join(f"{name}={required}" for name, required in feedback.corrected_requirements.items())
        )
    if feedback.lessons:
        text_parts.append("Lessons: " + "; ".join(feedback.lessons))

    record = MemoryRecord(
        memory_id=f"feedback:{feedback.cr_number}:{feedback.timestamp}",
        kind=MemoryKind.EPISODE,
        text="\n".join(text_parts),
        metadata=feedback.to_dict(),
    )
    memory.remember(record)
    return record
