"""Domain-training example contracts for future GPT-OSS 120B fine-tuning.

This deliberately separates training examples from raw ServiceNow records. A reviewed example can
contain the reasoning/evidence structure we want the model to learn without exposing raw identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import Decision


@dataclass(frozen=True)
class TrainingExample:
    example_id: str
    instruction: str
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def make_reasoning_example(
    *,
    example_id: str,
    cr: dict[str, Any],
    decision: Decision,
    facts: list[str],
    inferences: list[str],
    uncertainties: list[str],
    contradictions: list[str],
    recommendations: list[str],
    metadata: dict[str, Any] | None = None,
) -> TrainingExample:
    """Build a high-value structured example for later supervised fine-tuning."""
    output = {
        "decision": decision.value,
        "facts": facts,
        "inferences": inferences,
        "uncertainties": uncertainties,
        "contradictions": contradictions,
        "recommendations": recommendations,
    }
    return TrainingExample(
        example_id=example_id,
        instruction=(
            "Analyze this Normal ServiceNow Change Request as a Pre-CAB reviewer. "
            "Separate observable facts from inference, identify uncertainty and contradictions, "
            "and provide a defensible CAB recommendation with technical reasoning."
        ),
        input=cr,
        output=output,
        metadata=metadata or {},
    )


def to_chat_example(example: TrainingExample) -> dict[str, Any]:
    """Convert the canonical example to a simple messages-style training record."""
    return {
        "messages": [
            {"role": "user", "content": f"{example.instruction}\n\nCR={example.input}"},
            {"role": "assistant", "content": str(example.output)},
        ],
        "metadata": example.metadata | {"example_id": example.example_id},
    }
