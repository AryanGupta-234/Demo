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
        from .decision import validate_fields
        result = validate_fields(context.cr, context.strictness)
        return AgentResult(self.name, result.findings, [r.__dict__ for r in result.requirements], {"score": result.score})


class ContextAgent(BaseAgent):
    name = "context"

    def run(self, context: AgentContext) -> AgentResult:
        from .decision import infer_uat_requirement
        required, reason = infer_uat_requirement(context.cr)
        finding = Finding(
            code="UAT_REQUIREMENT",
            title="Contextual testing requirement",
            severity=FindingSeverity.INFO,
            message=("UAT is contextually expected." if required else "UAT is not assumed mandatory."),
            technical_detail=reason,
        )
        return AgentResult(self.name, [finding], [{"name": "UAT", "required": required, "reason": reason}], {})


class TechnicalAgent(BaseAgent):
    name = "technical"

    def run(self, context: AgentContext) -> AgentResult:
        implementation = str(context.cr.get("Implementation plan") or "").strip()
        finding = Finding(
            code="TECH_SCOPE",
            title="Technical scope extracted",
            severity=FindingSeverity.INFO,
            message="Technical implementation scope has been captured for deeper review.",
            technical_detail=implementation[:4000],
        )
        return AgentResult(self.name, [finding], [], {"implementation_length": len(implementation)})


class BusinessImpactAgent(BaseAgent):
    name = "business_impact"

    def run(self, context: AgentContext) -> AgentResult:
        cr = context.cr
        fields = {
            "customer": cr.get("Customer"),
            "affected_customers": cr.get("Affected Customers"),
            "business_service": cr.get("Business service"),
            "description": cr.get("Description"),
        }
        customer_visible = any(
            term in str(fields.get("description") or "").lower()
            for term in ("customer", "user", "transaction", "sms", "payment", "service")
        )
        severity = FindingSeverity.INFO
        message = "No explicit customer/business impact finding was raised at Stage 1."
        if customer_visible or fields["customer"] or fields["affected_customers"]:
            message = "Potential customer/business impact was detected and will be reasoned over."
        return AgentResult(
            self.name,
            [Finding("BUSINESS_IMPACT", "Business impact assessed", severity, message, technical_detail=str(fields))],
            [],
            {"customer_visible_signal": customer_visible},
        )


class MemoryAgent(BaseAgent):
    name = "memory"

    def run(self, context: AgentContext) -> AgentResult:
        if not self.memory:
            return AgentResult(self.name, [], [], {"matches": []})
        query = " ".join(str(context.cr.get(k) or "") for k in ("Short description", "Description", "Category", "Sub Category"))
        matches = self.memory.search(query, kinds=[MemoryKind.CAB_HISTORY, MemoryKind.SIMILARITY, MemoryKind.POLICY], limit=8)
        return AgentResult(self.name, [], [], {"matches": [m.__dict__ for m in matches]})


class TestingAgent(BaseAgent):
    name = "testing"

    def run(self, context: AgentContext) -> AgentResult:
        from .decision import infer_uat_requirement
        required, reason = infer_uat_requirement(context.cr)
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
            technical_detail=f"UAT required={required}; {reason}; evidence field={evidence!r}",
        ))
        return AgentResult(self.name, findings, [{"name": "UAT", "required": required, "reason": reason}], {})


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
        matches = []
        if self.memory:
            query = " ".join(str(context.cr.get(k) or "") for k in ("Short description", "Description", "Category", "Sub Category"))
            matches = self.memory.search(query, kinds=[MemoryKind.CAB_HISTORY, MemoryKind.SIMILARITY], limit=5)
        return AgentResult(self.name, [], [], {"clone_candidates": [m.__dict__ for m in matches]})


class DecisionAgent(BaseAgent):
    name = "decision"

    def run(self, context: AgentContext) -> AgentResult:
        # The shared AgenticReasoningLoop is the sole generative pass. This agent exposes the
        # deterministic boundary rather than making a second, redundant model call.
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
