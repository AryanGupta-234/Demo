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
    """Select the minimum decision-relevant CR fields for generative reasoning.

    The complete normalized CR remains available to deterministic validation. This
    agent is specifically a context compressor: it uses the mined field-policy
    table plus known descriptive/signoff semantics to decide what GPT-OSS needs
    to see. Fields classified NOT_OBSERVED by the historical rule table are
    excluded as noise (for example, this org's unused ``Change plan`` field).
    """

    name = "field"

    _DESCRIPTIVE_FIELDS = (
        "Short description",
        "Description",
        "Justification",
        "Implementation plan",
        "Change plan",
        "Backout plan",
        "Test plan",
        "Configuration item",
        "Risk",
        "Priority",
    )
    _SIGNOFF_FIELDS = (
        "UAT signoff",
        "Customer Approval",
        "TCS QA signoff",
        "Test Results Evidence",
        "Lower Environment Reference CR/SR",
    )
    _CONTEXT_FIELDS = (
        "Number",
        "Type",
        "Category",
        "Sub Category",
        "Environment",
        "Planned start",
        "Planned end",
        "Conflict status",
        "Change Class",
    )
    _EMPTY = {None, "", [], {}}  # only used through _present below

    @staticmethod
    def _present(value: Any) -> bool:
        if value is None or value == "" or value == [] or value == {}:
            return False
        return str(value).strip().lower() not in {"none", "null", "nan"}

    @staticmethod
    def _policy_levels(cr: dict[str, Any]) -> dict[str, str]:
        from .field_requirement_engine import FieldRequirementEngine

        report = FieldRequirementEngine().evaluate(cr)
        return {finding.field: finding.requirement_level.value for finding in report.findings}

    def run(self, context: AgentContext) -> AgentResult:
        cr = context.cr
        levels = self._policy_levels(cr)

        selected: list[str] = []
        reasons: dict[str, str] = {}

        # Always retain the high-information descriptive core when present.
        for field_name in self._DESCRIPTIVE_FIELDS:
            if not self._present(cr.get(field_name)):
                continue
            level = levels.get(field_name)
            if level == "NOT_OBSERVED":
                continue
            selected.append(field_name)
            reasons[field_name] = "descriptive-field"

        # Signoffs are disposition fields: include them when populated, or when
        # the mined policy/context says they are genuinely required/conditional.
        for field_name in self._SIGNOFF_FIELDS:
            level = levels.get(field_name)
            include = self._present(cr.get(field_name)) or level in {"REQUIRED", "CONDITIONAL"}
            if include and level != "NOT_OBSERVED":
                selected.append(field_name)
                reasons[field_name] = "signoff-disposition"

        # Compact context needed to interpret the selected business/technical text.
        for field_name in self._CONTEXT_FIELDS:
            level = levels.get(field_name)
            include = self._present(cr.get(field_name)) or level in {"REQUIRED", "CONDITIONAL"}
            if include and level != "NOT_OBSERVED":
                selected.append(field_name)
                reasons[field_name] = "decision-context"

        # Preserve order while avoiding duplicates.
        selected = list(dict.fromkeys(selected))
        compact = {field_name: cr.get(field_name) for field_name in selected}

        omitted_populated = [
            field_name
            for field_name, value in cr.items()
            if self._present(value) and field_name not in compact
        ]
        not_observed_omitted = [
            field_name for field_name in self._DESCRIPTIVE_FIELDS
            if levels.get(field_name) == "NOT_OBSERVED"
        ]

        chain = [
            f"reviewed {len(cr)} normalized fields",
            f"selected {len(selected)} decision-relevant fields for GPT-OSS",
            f"kept descriptive core: {', '.join(f for f in self._DESCRIPTIVE_FIELDS if f in compact) or 'none'}",
            f"included applicable signoff dispositions: {', '.join(f for f in self._SIGNOFF_FIELDS if f in compact) or 'none'}",
        ]
        if not_observed_omitted:
            chain.append(f"excluded NOT_OBSERVED fields: {', '.join(not_observed_omitted)}")
        chain.append(f"omitted {len(omitted_populated)} populated non-decision fields from model context")

        return AgentResult(
            self.name,
            [],
            [],
            {
                "field_count": len(cr),
                "populated_fields": sum(self._present(v) for v in cr.values()),
                "selected_field_count": len(selected),
                "selected_fields": selected,
                "selected_cr": compact,
                "selection_reason": reasons,
                "omitted_populated_fields_count": len(omitted_populated),
                "not_observed_omitted": not_observed_omitted,
                "validator_owned_by_orchestrator": True,
                "llm_context_role": "field-selected decision context",
                "chain": chain,
            },
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
        from .decision import rollback_quality

        cr = context.cr
        implementation = str(cr.get("Implementation plan") or "").strip()
        backout = str(cr.get("Backout plan") or "").strip()
        ci = str(cr.get("Configuration item") or "").strip()
        evidence = str(cr.get("Test Results Evidence") or "").strip()
        lower_env_ref = str(cr.get("Lower Environment Reference CR/SR") or "").strip()

        implementation_present = bool(implementation)
        ci_present = bool(ci)
        rollback_ok, rollback_reason, _explicit_na = rollback_quality(backout)
        rollback_aligned = rollback_ok and implementation_present
        dependency_evidence_present = bool(evidence) and evidence.lower() not in {"na", "n/a", "none"} or (
            bool(lower_env_ref) and lower_env_ref.lower() not in {"na", "n/a", "none", "not applicable"}
        )

        present_signals = sum([implementation_present, ci_present, rollback_ok, dependency_evidence_present])
        uncertainty = "low" if present_signals >= 4 else ("medium" if present_signals >= 2 else "high")

        chain = [
            "implementation is present" if implementation_present else "implementation is not described",
            "configuration item identified" if ci_present else "no configuration item identified",
            (
                "rollback mechanism aligns with implementation" if rollback_aligned
                else "rollback mechanism could not be correlated with implementation"
            ),
            (
                "dependency/lower-environment validation evidence present" if dependency_evidence_present
                else "no evidence of dependency validation"
            ),
            f"{uncertainty} technical uncertainty",
        ]

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
            {
                "implementation_length": len(implementation),
                "backout_length": len(backout),
                "ci": ci,
                "chain": chain,
                "technical_uncertainty": uncertainty,
                "rollback_aligned": rollback_aligned,
                "rollback_reason": rollback_reason,
            },
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
        cr = context.cr
        predictions = infer_requirements(cr)
        uat = next((prediction for prediction in predictions if prediction.name == "UAT"), None)
        test_plan = str(cr.get("Test plan") or "").strip()
        evidence = str(cr.get("Test Results Evidence") or "").strip()
        findings: list[Finding] = []
        if not test_plan:
            findings.append(Finding(
                code="TEST_PLAN_MISSING",
                title="Testing plan missing",
                severity=FindingSeverity.WARNING,
                message="No testing plan is populated in the CR.",
                recommendation="Provide an applicable test approach or documented rationale.",
            ))

        narrative_text = " ".join(
            str(cr.get(field) or "") for field in ("Test plan", "Description", "Lower Environment Reference CR/SR")
        ).lower()
        pre_prod_claimed = any(
            term in narrative_text for term in ("pre-prod", "pre prod", "uat", "staging", "sit environment", "test environment")
        )
        evidence_present = bool(evidence) and evidence.strip().lower() not in {"na", "n/a", "none", "not applicable"}
        functional_coverage_ok = bool(test_plan) and evidence_present

        chain = [
            "pre-PROD testing claimed" if pre_prod_claimed else "no pre-PROD testing claimed",
            "test execution evidence present" if evidence_present else "no test execution evidence",
            (
                "UAT required based on change context" if (uat and uat.required)
                else "UAT not required based on change context"
            ),
            "functional coverage appears adequate" if functional_coverage_ok else "functional coverage remains uncertain",
        ]

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
            {
                "chain": chain,
                "pre_prod_claimed": pre_prod_claimed,
                "evidence_present": evidence_present,
                "functional_coverage_ok": functional_coverage_ok,
            },
        )


class RiskAgent(BaseAgent):
    name = "risk"

    def run(self, context: AgentContext) -> AgentResult:
        from .rules import context_flags

        cr = context.cr
        risk = str(cr.get("Risk") or "").strip()
        impact = str(cr.get("Risk and impact analysis") or "").strip()
        flags = context_flags(cr)
        elevated_impact = bool(flags.get("elevated-impact"))
        declared_low = risk.lower() in {"low", "none", "minimal", "minimal risk"}
        contradiction = declared_low and elevated_impact

        confidence = 0.5
        if risk:
            confidence += 0.2
        if impact:
            confidence += 0.2
        if contradiction:
            confidence -= 0.3
        confidence = round(max(0.05, min(0.95, confidence)), 2)

        findings: list[Finding] = []
        if not risk:
            findings.append(Finding(
                code="RISK_MISSING",
                title="Risk classification missing",
                severity=FindingSeverity.WARNING,
                message="Risk is not populated in the CR.",
            ))
        if contradiction:
            findings.append(Finding(
                code="RISK_IMPACT_CONTRADICTION",
                title="Declared risk conflicts with impact narrative",
                severity=FindingSeverity.WARNING,
                message=f"Risk is declared {risk!r} but the change context suggests elevated production impact.",
                technical_detail=impact[:2000],
                recommendation="Reconcile the declared risk level with the described impact before CAB review.",
            ))
        findings.append(Finding(
            code="RISK_CONTEXT",
            title="Risk context captured",
            severity=FindingSeverity.INFO,
            message=f"Declared risk: {risk or 'unknown'}.",
            technical_detail=impact[:4000],
        ))

        impact_level = "elevated" if elevated_impact else "moderate" if impact else "unstated"
        chain = [
            f"declared risk = {risk or 'unknown'}",
            f"impact narrative suggests {impact_level} production exposure" if impact else "no impact narrative provided",
            "contradiction: declared risk conflicts with impact narrative" if contradiction else "no direct contradiction",
            f"confidence {confidence:.2f}",
        ]
        return AgentResult(self.name, findings, [], {"chain": chain, "risk_confidence": confidence, "contradiction": contradiction})


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
        if analyses:
            top = analyses[0]
            plural = "es" if len(analyses) != 1 else ""
            changed = top.get("changed_fields") or []
            chain = [
                f"{len(analyses)} historical match{plural}",
                (
                    f"strongest match was {top['historical_decision']}" if top.get("historical_decision")
                    else "strongest match has no recorded historical outcome"
                ),
                (
                    f"current CR differs in: {', '.join(changed[:3])}" if changed
                    else "no material field differences identified vs. strongest match"
                ),
                (
                    "historical outcome cannot be directly reused" if top.get("recommendation") != "CLONE_CANDIDATE"
                    else "historical outcome may inform, but does not substitute for, this decision"
                ),
            ]
        else:
            chain = ["no historical matches found"]
        return AgentResult(self.name, findings, [], {"clone_candidates": analyses[:5], "chain": chain})


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
