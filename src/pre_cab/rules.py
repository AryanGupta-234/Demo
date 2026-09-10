"""Data-driven requirement evidence for Normal CRs.

This module produces advisory applicability signals from a reviewed field profile. It is deliberately
separate from the deterministic decision gate: historical frequency is evidence, not policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApplicabilitySignal:
    requirement: str
    applicable: bool
    confidence: float
    rationale: str
    source: str


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def contextual_signals(cr: dict[str, Any]) -> list[ApplicabilitySignal]:
    """Return conservative requirement signals for a single Normal CR."""
    text = " ".join(_text(cr.get(k)) for k in ("Short description", "Description", "Justification", "Category", "Sub Category"))
    customer_facing = any(term in text for term in ("customer", "sms", "payment", "transaction", "atm", "invoice", "workflow", "api"))
    infrastructure = any(term in text for term in ("os patch", "patching", "server patch", "security patch", "reboot", "infrastructure"))
    production = _text(cr.get("Environment") or cr.get("Production system")) in {"prod", "production", "production environment", "yes", "true"}

    signals = [
        ApplicabilitySignal(
            "UAT",
            customer_facing and not infrastructure,
            0.88 if customer_facing and not infrastructure else 0.84,
            "Customer-facing or functional behavior is indicated; infrastructure maintenance alone does not trigger UAT.",
            "contextual-rules-v1",
        ),
        ApplicabilitySignal(
            "Customer approval",
            customer_facing,
            0.82 if customer_facing else 0.78,
            "Customer-facing impact is indicated; approval applicability should be confirmed against the organization's process.",
            "contextual-rules-v1",
        ),
        ApplicabilitySignal(
            "Outage / maintenance-window detail",
            production,
            0.76 if production else 0.70,
            "Production changes should explain expected service impact and implementation timing.",
            "contextual-rules-v1",
        ),
        ApplicabilitySignal(
            "Rollback / recovery",
            True,
            0.99,
            "Every in-scope Normal production change needs a credible recovery path unless an approved exception applies.",
            "baseline-rule-v1",
        ),
        ApplicabilitySignal(
            "Conflict check",
            True,
            0.99,
            "A current conflict state should be known before a Normal production change proceeds.",
            "baseline-rule-v1",
        ),
    ]
    return signals
