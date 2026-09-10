"""Specialized agent contracts with one shared context and one model brain interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory import MemoryKind, UnifiedMemory
from .models import ModelProvider
from .schemas import AgentContext, Finding, FindingSeverity


@dataclass
class AgentResult:
    agent: str
    findings: list[Finding]
    requirements: list[dict[str, Any]]
    notes: dict[str, Any]


class BaseAgent:
    name = "base"

    def __init__(self, model: ModelProvider | None = None, memory: UnifiedMemory | None = None) -> None:
        self.model = model
        self.memory = memory

    def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError


class FieldAgent(BaseAgent):
    name = "field"

    def run(self, context: AgentContext) -> AgentResult:
        populated = sum(value not in (None, "", [], {}) for value in context.cr.values())
        return AgentResult(
            self.name,
            [],
            [],
            {"field_count": len(context.cr), "populated_fields": populated, "validator_owned_by_orchestrator": True},
        )


class ContextAgent(BaseAgent):
    name = "context"

    def run(self, context: AgentContext) -> AgentResult:
        from .requirements import infer_requirements
        predictions = infer_requirements(context.cr)
        findings = [
            Finding(
                code="CONTEXT_REQUIREMENTS",
                title="Contextual requirements inferred",
                severity=FindingSeverity.INFO,
                message="Change-specific validation requirements were inferred from the CR context.",
                technical_detail="; ".join(
                    f"{prediction.name}={prediction.required} ({prediction.confidence:.2f})"
                    for prediction in predictions
                ),
            )
        ]
        return AgentResult(
            self.name,
            findings,
            [
                {
                    "name": prediction.name,
                    "required": prediction.required,
                    "confidence": prediction.confidence,
                    "reason": prediction.reason,
                    "signals": list(prediction.signals),
                }
                for prediction in predictions
            ],
            {},
        )


class TechnicalAgent(BaseAgent):
    name = "technical"

    def run(self, context: AgentContext) -> AgentResult:
        implementation = str(context.cr.get("Implementation plan") or "").strip()
        backout = str(context.cr.get("Backout plan") or "").strip()
        ci = str(context.cr.get("Configuration item") or "").strip()
        finding = Finding(
            code="TECH_SCOPE",
            title="Technical scope extracted",
            severity=FindingSeverity.INFO,
            message="Technical implementation scope has been captured for deeper review.",
            technical_detail=(f"CI={ci!r}; implementation={implementation[:2500]}; backout={backout[:1500]}"),
        )
        return AgentResult(
            self.name,
            [finding],
            [],
            {"implementation_length": len(implementation), "backout_length": len(backout), "ci": ci},
        )


class BusinessImpactAgent(BaseAgent):
    name = "business_impact"

    def run(self, context: AgentContext) -> AgentResult:
        cr = context.cr
        description = " ".join(
            str(cr.get(field) or "")
            for field in ("Short description", "Description", "Justification", "Risk and impact analysis")
        ).lower()
        customer_visible = any(
            term in description
            for term in ("customer", "user", "transaction", "sms", "payment", "service unavailable", "outage")
        )
        explicit_customer = bool(cr.get("Customer") or cr.get("Affected Customers"))
        message = (
            "Potential customer/business impact was detected and requires CAB consideration."
            if customer_visible or explicit_customer
            else "No strong customer/business impact signal was found in the supplied fields."
        )
        return AgentResult(
            self.name,
            [Finding("BUSINESS_IMPACT", "Business impact assessed", FindingSeverity.INFO, message, technical_detail=description[:3000])],
            [],
            {"customer_visible_signal": customer_visible, "explicit_customer_reference": explicit_customer},
        )


class MemoryAgent(BaseAgent):
    name = "memory"

    def run(self, context: AgentContext) -> AgentResult:
        if not self.memory:
            return AgentResult(self.name, [], [], {"matches": []})
        query = " ".join(str(context.cr.get(k) or "") for k in ("Short description", "Description", "Category", "Sub Category", "Configuration item"))
        matches = self.memory.search(query, kinds=[MemoryKind.CAB_HISTORY, MemoryKind.SIMILARITY, MemoryKind.POLICY], limit=8)
        return AgentResult(self.name, [], [], {"matches": [m.__dict__ for m in matches]})


class TestingAgent(BaseAgent):
    name = "testing"

    def run(self, context: AgentContext) -> AgentResult:
        from .requirements import infer_requirements
        predictions = infer_requirements(context.cr)
        uat = next((prediction for prediction in predictions if prediction.name == "UAT"), None)
        test_plan = str(context.cr.get("Test plan") or "").strip()
        evidence = str(context.cr.get("Test Results Evidence") or "").strip()
        findings: list[Finding] = []
        if not test_plan:
            findings.append(Finding(
                code="TEST_PLAN_MISSING",
                title="Testing plan missing",
                severity=FindingSeverity.WARNING,
                message="No testing plan is populated in the CR.",
                recommendation="Provide an applicable test approach or documented rationale.",
            ))
        findings.append(Finding(
            code="TESTING_CONTEXT",
            title="Testing requirement assessed",
            severity=FindingSeverity.INFO,
            message="Testing requirement was inferred from the CR context.",
            technical_detail=(
                f"UAT required={uat.required if uat else False}; "
                f"{uat.reason if uat else 'no UAT prediction'}; evidence field={evidence!r}"
            ),
        ))
        return AgentResult(
            self.name,
            findings,
            ([{"name": "UAT", "required": uat.required, "reason": uat.reason}] if uat else []),
            {},
        )


class RiskAgent(BaseAgent):
    name = "risk"

    def run(self, context: AgentContext) -> AgentResult:
        risk = str(context.cr.get("Risk") or "").strip()
        impact = str(context.cr.get("Risk and impact analysis") or "").strip()
        findings: list[Finding] = []
        if not risk:
            findings.append(Finding(
                code="RISK_MISSING",
                title="Risk classification missing",
                severity=FindingSeverity.WARNING,
                message="Risk is not populated in the CR.",
            ))
        findings.append(Finding(
            code="RISK_CONTEXT",
            title="Risk context captured",
            severity=FindingSeverity.INFO,
            message=f"Declared risk: {risk or 'unknown'}.",
            technical_detail=impact[:4000],
        ))
        return AgentResult(self.name, findings, [], {})


class EvidenceAgent(BaseAgent):
    name = "evidence"

    def run(self, context: AgentContext) -> AgentResult:
        claim = str(context.cr.get("Test Results Evidence") or "").strip()
        finding = Finding(
            code="EVIDENCE_CLAIM",
            title="Evidence claim recorded",
            severity=FindingSeverity.INFO,
            message="The CR evidence declaration is recorded; attachments require independent verification.",
            technical_detail=f"Test Results Evidence={claim!r}",
        )
        return AgentResult(self.name, [finding], [], {"requires_document_stage": True})


class CloneAgent(BaseAgent):
    name = "clone"

    def run(self, context: AgentContext) -> AgentResult:
        if not self.memory:
            return AgentResult(self.name, [], [], {"clone_candidates": []})
        from .clone import clone_analysis

        query = " ".join(
            str(context.cr.get(k) or "")
            for k in ("Short description", "Description", "Category", "Sub Category", "Configuration item")
        )
        matches = self.memory.search(query, kinds=[MemoryKind.CAB_HISTORY, MemoryKind.SIMILARITY], limit=8)
        analyses: list[dict[str, Any]] = []
        findings: list[Finding] = []
        for match in matches:
            historical = match.metadata.get("source_record") if isinstance(match.metadata, dict) else None
            if not isinstance(historical, dict):
                continue
            analysis = clone_analysis(context.cr, historical)
            item = {
                "change_id": analysis.candidate.change_id,
                "similarity": analysis.candidate.similarity,
                "historical_decision": match.metadata.get("historical_outcome") or analysis.candidate.historical_decision,
                "cab_recommendation": analysis.candidate.cab_recommendation,
                "reusable_fields": list(analysis.reusable_fields),
                "changed_fields": list(analysis.changed_fields),
                "revalidation_fields": list(analysis.revalidation_fields),
                "recommendation": analysis.recommendation,
                "retrieval_score": match.score,
            }
            analyses.append(item)
        analyses.sort(key=lambda item: (item["similarity"], item.get("retrieval_score") or 0), reverse=True)
        strong = next((item for item in analyses if item["recommendation"] == "CLONE_CANDIDATE"), None)
        if strong:
            findings.append(
                Finding(
                    code="CLONE_CANDIDATE_FOUND",
                    title="Strong historical clone candidate found",
                    severity=FindingSeverity.INFO,
                    message=f"A {strong['similarity']:.0%} similar historical Normal CR was found.",
                    technical_detail=f"Historical CR={strong['change_id']}; changed fields={strong['changed_fields']}",
                    recommendation="Reuse only the stable structure and revalidate every changed field/evidence item.",
                )
            )
        return AgentResult(self.name, findings, [], {"clone_candidates": analyses[:5]})


class DecisionAgent(BaseAgent):
    name = "decision"

    def run(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            self.name,
            [Finding(
                code="DECISION_GATED",
                title="Decision remains policy-gated",
                severity=FindingSeverity.INFO,
                message="Final readiness is determined after specialist findings, evidence verification and policy gates.",
            )],
            [],
            {"generative_brain_is_external": True},
        )


DEFAULT_AGENT_TYPES = [
    FieldAgent,
    ContextAgent,
    TechnicalAgent,
    BusinessImpactAgent,
    MemoryAgent,
    TestingAgent,
    RiskAgent,
    EvidenceAgent,
    CloneAgent,
    DecisionAgent,
]
