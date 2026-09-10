"""Specialized agent contracts with one shared context and one model brain interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory import MemoryKind, UnifiedMemory
from .models import ModelProvider
from .schemas import AgentContext, Finding, FindingSeverity, Strictness


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
        return AgentResult(self.name, result.findings, result.requirements, {"score": result.score})


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
        cr = context.cr
        implementation = str(cr.get("Implementation plan") or "").strip()
        finding = Finding(
            code="TECH_SCOPE",
            title="Technical scope extracted",
            severity=FindingSeverity.INFO,
            message="Technical implementation scope has been captured for deeper review.",
            technical_detail=implementation[:4000],
        )
        return AgentResult(self.name, [finding], [], {"implementation_length": len(implementation)})


class MemoryAgent(BaseAgent):
    name = "memory"

    def run(self, context: AgentContext) -> AgentResult:
        if not self.memory:
            return AgentResult(self.name, [], [], {"matches": []})
        query = " ".join(
            str(context.cr.get(k) or "")
            for k in ("Short description", "Description", "Category", "Sub Category")
        )
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
            findings.append(
                Finding(
                    code="TEST_PLAN_MISSING",
                    title="Testing plan missing",
                    severity=FindingSeverity.WARNING,
                    message="No testing plan is populated in the CR.",
                    recommendation="Provide an applicable test approach or documented rationale.",
                )
            )
        findings.append(
            Finding(
                code="TESTING_CONTEXT",
                title="Testing requirement assessed",
                severity=FindingSeverity.INFO,
                message="Testing requirement was inferred from the CR context.",
                technical_detail=f"UAT required={required}; {reason}; evidence field={evidence!r}",
            )
        )
        return AgentResult(self.name, findings, [{"name": "UAT", "required": required, "reason": reason}], {})


class RiskAgent(BaseAgent):
    name = "risk"

    def run(self, context: AgentContext) -> AgentResult:
        risk = str(context.cr.get("Risk") or "").strip()
        impact = str(context.cr.get("Risk and impact analysis") or "").strip()
        findings: list[Finding] = []
        if not risk:
            findings.append(
                Finding(
                    code="RISK_MISSING",
                    title="Risk classification missing",
                    severity=FindingSeverity.WARNING,
                    message="Risk is not populated in the CR.",
                )
            )
        findings.append(
            Finding(
                code="RISK_CONTEXT",
                title="Risk context captured",
                severity=FindingSeverity.INFO,
                message=f"Declared risk: {risk or 'unknown'}.",
                technical_detail=impact[:4000],
            )
        )
        return AgentResult(self.name, findings, [], {})


class EvidenceAgent(BaseAgent):
    name = "evidence"

    def run(self, context: AgentContext) -> AgentResult:
        # Attachments are deliberately verified in Stage 2. Stage 1 only records the claim.
        claim = str(context.cr.get("Test Results Evidence") or "").strip()
        finding = Finding(
            code="EVIDENCE_CLAIM",
            title="Evidence claim recorded",
            severity=FindingSeverity.INFO,
            message="The CR's evidence declaration has been recorded; attachments require independent verification.",
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
        if not self.model:
            finding = Finding(
                code="MODEL_UNAVAILABLE",
                title="Reasoning model unavailable",
                severity=FindingSeverity.WARNING,
                message="No model provider is configured; deterministic validation can still run.",
            )
            return AgentResult(self.name, [finding], [], {})

        system = (
            "You are the reasoning brain of a Pre-CAB validator. Analyze the supplied structured facts "
            "and retrieved context. Separate facts, inferences, uncertainties and recommendations. "
            "Never invent approvals, test results, evidence or outcomes. Return concise, auditable reasoning."
        )
        user = f"STRICTNESS={context.strictness.value}\nCR={context.cr}\nMEMORY={list(context.retrieved_memory)}\nEVIDENCE={list(context.evidence)}"
        response = self.model.generate(system=system, user=user, temperature=0.1)
        finding = Finding(
            code="LLM_REASONING",
            title="GPT-OSS 120B reasoning generated",
            severity=FindingSeverity.INFO,
            message="Model reasoning completed; final decision remains governed by deterministic gates.",
            technical_detail=response.text[:8000],
        )
        return AgentResult(self.name, [finding], [], {"model": response.model})


DEFAULT_AGENT_TYPES = [
    FieldAgent,
    ContextAgent,
    TechnicalAgent,
    MemoryAgent,
    TestingAgent,
    RiskAgent,
    EvidenceAgent,
    CloneAgent,
    DecisionAgent,
]
