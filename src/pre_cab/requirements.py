"""Contextual requirement inference for Normal CRs.

The engine produces hypotheses, not policy truth. Organization-specific rules can later be loaded
from policy memory and benchmarked against historical CAB outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequirementPrediction:
    name: str
    required: bool
    confidence: float
    reason: str
    signals: tuple[str, ...] = ()


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def infer_requirements(cr: dict[str, Any]) -> list[RequirementPrediction]:
    description = " ".join(_text(cr.get(k)) for k in ("Short description", "Description", "Justification"))
    category = _text(cr.get("Category"))
    subcategory = _text(cr.get("Sub Category"))

    infra = category in {"infrastructure", "infra"} or any(x in description for x in ("os patch", "patching", "server reboot"))
    functional = any(x in description for x in (
        "enhancement", "defect", "transaction", "payment", "sms", "report", "workflow",
        "customer", "user facing", "functional", "api", "interface",
    )) or subcategory in {"atm transactions", "branchchannel", "credittransfer", "api", "esb"}
    customer_impact = any(x in description for x in ("customer", "payment", "transaction", "sms", "user facing"))

    predictions = [
        RequirementPrediction(
            "UAT",
            functional and not infra,
            0.90 if functional and not infra else 0.82,
            "Functional/customer-facing change signals UAT or equivalent functional validation." if functional and not infra else "Infrastructure maintenance does not automatically require UAT; operational validation may be sufficient.",
            tuple(x for x in ("functional-change", "customer-facing", "infrastructure") if {"functional-change": functional, "customer-facing": customer_impact, "infrastructure": infra}[x]),
        ),
        RequirementPrediction(
            "Customer approval",
            customer_impact,
            0.78 if customer_impact else 0.68,
            "Customer-visible behavior or service impact suggests customer approval should be verified." if customer_impact else "No strong customer-impact signal was detected from the supplied fields.",
            tuple(x for x in ("customer", "payment", "transaction", "sms") if x in description),
        ),
        RequirementPrediction(
            "Outage/downtime detail",
            infra or any(x in description for x in ("restart", "reboot", "maintenance window", "downtime", "unavailable")),
            0.84 if infra else 0.75,
            "Operational changes and service restarts should state expected service interruption or explicitly state none." if (infra or "restart" in description or "reboot" in description) else "No strong downtime signal detected.",
            tuple(x for x in ("infrastructure", "restart", "reboot", "downtime") if x in description or (x == "infrastructure" and infra)),
        ),
        RequirementPrediction(
            "Rollback/recovery",
            True,
            0.99,
            "A production change should have a credible recovery path unless an explicitly approved exception applies.",
            ("production-change",),
        ),
        RequirementPrediction(
            "Conflict check",
            True,
            0.99,
            "The change should have a current conflict status before readiness is asserted.",
            ("production-change",),
        ),
    ]
    return predictions
