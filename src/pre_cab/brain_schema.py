"""Machine-readable response schema for the GPT-OSS 120B reasoning brain."""
from __future__ import annotations

BRAIN_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "pre_cab_reasoning",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prediction": {"type": "string", "enum": ["PASS", "CONDITIONAL", "NOT_READY"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "facts": {"type": "array", "items": {"type": "string"}},
                "inferences": {"type": "array", "items": {"type": "string"}},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
                "contradictions": {"type": "array", "items": {"type": "string"}},
                "technical_reasoning": {"type": "string"},
                "cab_reasoning": {"type": "string"},
                "cab_questions": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "self_critique": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "prediction", "confidence", "facts", "inferences", "uncertainties",
                "contradictions", "technical_reasoning", "cab_reasoning", "cab_questions",
                "recommendations", "self_critique"
            ],
        },
    },
}
