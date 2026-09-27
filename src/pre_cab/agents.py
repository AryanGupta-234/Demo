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
    """Select the minimum decision-relevant CR fields for generative reasoning."""

    name = "field"

    _DESCRIPTIVE_FIELDS = (
        "Short description", "Description", "Justification", "Implementation plan",
        "Change plan", "Backout plan", "Work notes", "Comments", "Test plan",
        "Configuration item", "Risk", "Priority",
    )
    _SIGNOFF_FIELDS = (
        "UAT signoff", "Customer Approval", "TCS QA signoff",
        "Test Results Evidence", "Lower Environment Reference CR/SR",
    )
    _CONTEXT_FIELDS = (
        "Number", "Type", "Category", "Sub Category", "Environment",
        "Planned start", "Planned end", "Conflict status", "Change Class",
    )

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

        for field_name in self._DESCRIPTIVE_FIELDS:
            if self._present(cr.get(field_name)) and levels.get(field_name) != "NOT_OBSERVED":
                selected.append(field_name)
                reasons[field_name] = "descriptive-field"

        for field_name in self._SIGNOFF_FIELDS:
            level = levels.get(field_name)
            if self._present(cr.get(field_name)) or level in {"REQUIRED", "CONDITIONAL"}:
                if level != "NOT_OBSERVED":
                    selected.append(field_name)
                    reasons[field_name] = "signoff-disposition"

        for field_name in self._CONTEXT_FIELDS:
            level = levels.get(field_name)
            if self._present(cr.get(field_name)) or level in {"REQUIRED", "CONDITIONAL"}:
                if level != "NOT_OBSERVED":
                    selected.append(field_name)
                    reasons[field_name] = "decision-context"

        selected = list(dict.fromkeys(selected))
        compact = {field_name: cr.get(field_name) for field_name in selected}
        omitted_populated = [
            field_name for field_name, value in cr.items()
            if self._present(value) and field_name not in compact
        ]
        not_observed_omitted = [
            field_name for field_name in self._DESCRIPTIVE_FIELDS
            if levels.get(field_name) == "NOT_OBSERVED"
        ]
        chain = [
            f"reviewed {len(cr)} normalized fields",
            f"selected {len(selected)} decision-relevant fields for reasoning",
            f"kept descriptive core: {', '.join(f for f in self._DESCRIPTIVE_FIELDS if f in compact) or 'none'}",
            f"included applicable signoff dispositions: {', '.join(f for f in self._SIGNOFF_FIELDS if f in compact) or 'none'}",
        ]
        if not_observed_omitted:
            chain.append(f"excluded NOT_OBSERVED fields: {', '.join(not_observed_omitted)}")
        chain.append(f"omitted {len(omitted_populated)} populated non-decision fields from model context")

        return AgentResult(
            self.name, [], [], {
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
        from .rules import change_profile, contradiction_signals
        predictions = infer_requirements(context.cr)
        profile = change_profile(context.cr)
        contradictions = contradiction_signals(context.cr)
        findings = [
            Finding(
                code="CONTEXT_REQUIREMENTS",
                title="Contextual requirements inferred",
                severity=FindingSeverity.INFO,
                message="Change-specific validation requirements were inferred from the CR context.",
                technical_detail="; ".join(f"{p.name}={p.required} ({p.confidence:.2f})" for p in predictions),
            )
        ]
        for signal in contradictions:
            if signal["code"] in {"RISK_IMPACT_CONTRADICTION", "FUNCTIONAL_TESTING_GAP"}:
                continue  # owned by the specialist agent for this domain
            findings.append(Finding(
                signal["code"], "Context contradiction detected", FindingSeverity.WARNING,
                signal["message"], recommendation="Reconcile the conflicting statements before CAB review.",
            ))
        return AgentResult(self.name, findings, [
            {"name": p.name, "required": p.required, "confidence": p.confidence, "reason": p.reason, "signals": list(p.signals)}
            for p in predictions
        ], {
            "change_profile": profile,
            "contradiction_signals": contradictions,
            "chain": [
                f"primary change archetype = {profile['primary_archetype']}",
                f"archetypes = {', '.join(profile['archetypes'])}",
                f"formal test evidence expected = {profile['formal_test_evidence_expected']}",
                f"non-PROD validation expected = {profile['non_prod_validation_expected']}",
                f"customer approval expected = {profile['approval_expected']}",
                f"context confidence = {profile['confidence']:.0%}",
            ],
        })


class TechnicalAgent(BaseAgent):
    name = "technical"

    def run(self, context: AgentContext) -> AgentResult:
        from .decision import rollback_quality
        from .rules import context_flags, effective_environment
        cr = context.cr
        flags = context_flags(cr)
        implementation = str(cr.get("Implementation plan") or "").strip()
        change_plan = str(cr.get("Change plan") or "").strip()
        backout = str(cr.get("Backout plan") or "").strip()
        ci = str(cr.get("Configuration item") or "").strip()
        test_plan = str(cr.get("Test plan") or "").strip()
        implementation_present = bool(implementation)
        rollback_ok, rollback_reason, _explicit_na = rollback_quality(backout)
        rollback_aligned = rollback_ok and implementation_present
        operational_context = " ".join(str(cr.get(k) or "") for k in (
            "Description", "Implementation plan", "Change plan", "Work notes", "Comments",
        )).lower()
        dependency_terms = (
            "dependency", "dependencies", "sequence", "sequencing", "prerequisite",
            "depends on", "blocked by", "coordinate with", "coordination with",
        )
        # Non-PROD testing and temporal words inside a test plan are not dependency evidence.
        dependency_evidence_present = bool(change_plan) or any(term in operational_context for term in dependency_terms)
        present_signals = sum([implementation_present, rollback_ok, bool(test_plan), dependency_evidence_present])
        uncertainty = "low" if present_signals >= 3 else ("medium" if present_signals >= 2 else "high")
        chain = [
            "implementation is present" if implementation_present else "implementation is not described",
            "configuration item identified" if ci else "configuration item not identified; traceability observation only",
            f"target environment resolved as {effective_environment(cr)}" + (" from workflow context" if not str(cr.get("Environment") or "").strip() else ""),
            "rollback mechanism aligns with implementation" if rollback_aligned else "rollback mechanism could not be correlated with implementation",
            "non-PROD validation reference found (SIT/UAT/Pre-PROD/lower environment treated equivalently)" if flags.get("non-prod-validation") else "no non-PROD validation reference stated",
            "dependency/sequencing context is present"
            if dependency_evidence_present
            else (
                "no unresolved dependency signal identified"
                if not flags.get("high-impact") and not flags.get("database-change") and not flags.get("network-change")
                else "dependency/sequencing context is not explicitly stated"
            ),
            f"{uncertainty} technical uncertainty",
        ]
        finding = Finding("TECH_SCOPE", "Technical scope extracted", FindingSeverity.INFO, "Technical implementation scope has been captured for deeper review.", technical_detail=f"CI={ci!r}; implementation={implementation[:2500]}; backout={backout[:1500]}")
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
                "dependency_evidence_present": dependency_evidence_present,
                "non_prod_validation_claimed": bool(flags.get("non-prod-validation")),
                "effective_environment": effective_environment(cr),
            },
        )


class BusinessImpactAgent(BaseAgent):
    name = "business_impact"

    def run(self, context: AgentContext) -> AgentResult:
        cr = context.cr
        description = " ".join(str(cr.get(field) or "") for field in ("Short description", "Description", "Justification", "Risk and impact analysis")).lower()
        customer_visible = any(term in description for term in ("customer", "user", "transaction", "sms", "payment", "service unavailable", "outage"))
        explicit_customer = bool(cr.get("Customer") or cr.get("Affected Customers"))
        message = "Potential customer/business impact was detected and requires CAB consideration." if customer_visible or explicit_customer else "No strong customer/business impact signal was found in the supplied fields."
        return AgentResult(self.name, [Finding("BUSINESS_IMPACT", "Business impact assessed", FindingSeverity.INFO, message, technical_detail=description[:3000])], [], {"customer_visible_signal": customer_visible, "explicit_customer_reference": explicit_customer})


class MemoryAgent(BaseAgent):
    name = "memory"

    def run(self, context: AgentContext) -> AgentResult:
        if not self.memory:
            return AgentResult(self.name, [], [], {"matches": []})
        profile = context.agent_state.get("context", {}).get("notes", {}).get("change_profile") or {}
        query = " ".join(str(context.cr.get(k) or "") for k in ("Short description", "Description", "Category", "Sub Category", "Configuration item", "Implementation plan", "Test plan"))
        query += " " + " ".join(str(item) for item in profile.get("archetypes", [])[:4])
        matches = self.memory.search(query, kinds=[MemoryKind.CAB_HISTORY, MemoryKind.SIMILARITY, MemoryKind.POLICY, MemoryKind.EPISODE], limit=10)
        return AgentResult(self.name, [], [], {"matches": [m.__dict__ for m in matches]})


class TestingAgent(BaseAgent):
    name = "testing"

    def run(self, context: AgentContext) -> AgentResult:
        from .requirements import infer_requirements
        from .rules import context_flags, signoff_disposition, evidence_matrix, change_profile
        cr = context.cr
        predictions = infer_requirements(cr)
        uat = next((prediction for prediction in predictions if prediction.name == "UAT"), None)
        flags = context_flags(cr)
        profile = change_profile(cr)
        matrix = evidence_matrix(cr)
        test_plan = str(cr.get("Test plan") or "").strip()
        evidence = str(cr.get("Test Results Evidence") or "").strip()
        journal = " ".join(str(cr.get(field) or "") for field in ("Work notes", "Comments")).lower()
        execution_terms = (
            "tested", "test completed", "testing completed", "validated",
            "validation completed", "sanity check completed", "sanity completed",
            "verified", "verification completed", "passed", "successful",
            "patching completed", "checks completed",
        )
        execution_claimed = bool(matrix["execution_claimed"])
        formal_evidence_positive = bool(matrix["formal_evidence_positive"])
        infrastructure = bool(profile["infrastructure"])
        functional_change = bool(profile["functional"] and not profile["infrastructure"])
        findings: list[Finding] = []
        if not test_plan:
            findings.append(Finding("TEST_PLAN_MISSING", "Testing plan missing", FindingSeverity.WARNING, "No testing plan is populated in the CR.", recommendation="Provide an applicable test approach or documented rationale."))
        execution_status = str(matrix.get("execution_status") or "NOT_RECORDED")
        execution_failed = bool(matrix.get("execution_failed"))
        if execution_failed:
            findings.append(Finding(
                "TEST_EXECUTION_FAILED",
                "Recorded test/validation failure",
                FindingSeverity.BLOCKING if profile["formal_test_evidence_expected"] else FindingSeverity.WARNING,
                "The current CR journal contains language indicating that validation/testing failed or did not pass.",
                recommendation="Resolve the failed validation and record a successful/reconciled result before approval.",
            ))
        elif profile["formal_test_evidence_expected"] and not (formal_evidence_positive or execution_claimed):
            findings.append(Finding(
                "TEST_EXECUTION_UNCONFIRMED", "Test execution is not evidenced", FindingSeverity.WARNING,
                "The change appears functionally relevant, but neither a positive test-evidence disposition nor execution evidence in Work Notes/Comments is recorded.",
                recommendation="Record the executed validation and result before approval.",
            ))
        elif infrastructure and not execution_claimed:
            findings.append(Finding(
                "TECHNICAL_VALIDATION_UNCONFIRMED", "Post-change validation result is not recorded", FindingSeverity.INFO,
                "The infrastructure test approach is defined, but the reviewed CR does not yet contain a positive execution result in Work Notes/Comments.",
                recommendation="Record the post-change sanity-check result when executed.",
            ))
        prior_technical = context.agent_state.get("technical", {}).get("notes", {})
        implementation_present = bool(prior_technical.get("implementation_present", bool(cr.get("Implementation plan"))))
        rollback_aligned = bool(prior_technical.get("rollback_aligned", True))
        coverage_ok = bool(test_plan) and not execution_failed and (not profile["formal_test_evidence_expected"] or formal_evidence_positive or execution_claimed)
        if profile.get("functional") and not implementation_present:
            findings.append(Finding("TEST_TECHNICAL_COVERAGE_GAP", "Testing is defined against an incomplete implementation context", FindingSeverity.WARNING, "The technical specialist did not establish a populated implementation plan for the functional change.", recommendation="Complete the implementation steps before relying on the testing plan as readiness evidence."))
        if profile.get("high_impact") and implementation_present and not rollback_aligned:
            findings.append(Finding("TEST_ROLLBACK_ALIGNMENT_GAP", "Testing context does not confirm rollback alignment", FindingSeverity.WARNING, "The change is higher-impact and the technical specialist could not confirm that rollback is aligned with implementation.", recommendation="Confirm rollback triggers and recovery validation against the implementation sequence."))
        chain = [
            "non-PROD validation mentioned (SIT/UAT/Pre-PROD/lower environment are treated equivalently)" if flags.get("non-prod-validation") else "no non-PROD validation reference stated",
            "test plan is defined" if test_plan else "test plan is missing",
            (
                "formal test-results evidence is recorded"
                if formal_evidence_positive
                else "formal test-results evidence is not recorded"
                if profile["formal_test_evidence_expected"]
                else "formal test-results evidence is not required for this change profile"
            ),
            f"execution status = {execution_status}",

            "UAT required based on change context" if (uat and uat.required) else "UAT not required based on change context",
            "functional coverage appears adequate" if coverage_ok else "functional coverage remains uncertain",
        ]
        findings.append(Finding(
            "TESTING_CONTEXT", "Testing requirement assessed", FindingSeverity.INFO,
            "Testing requirements and evidence were evaluated against the change context.",
            technical_detail=(
                f"UAT required={uat.required if uat else False}; "
                f"{uat.reason if uat else 'no UAT prediction'}; "
                f"non-PROD validation claimed={bool(flags.get('non-prod-validation'))}; "
                f"test-results disposition={signoff_disposition(evidence)}; applicable formal evidence={profile['formal_test_evidence_expected']}"
            ),
        ))
        return AgentResult(
            self.name,
            findings,
            ([{"name": "UAT", "required": uat.required, "reason": uat.reason}] if uat else []),
            {
                "chain": chain,
                "pre_prod_claimed": bool(flags.get("non-prod-validation")),
                "non_prod_validation_claimed": bool(flags.get("non-prod-validation")),
                "execution_claimed": execution_claimed,
                "evidence_present": formal_evidence_positive,
                "functional_coverage_ok": coverage_ok,
                "formal_evidence_present": formal_evidence_positive,
                "non_prod_validation_equivalence": "SIT/UAT/Pre-PROD/lower environment treated as one validation class",
                "change_profile": profile,
                "evidence_matrix": matrix,
                "historical_test_reference_in_plan": matrix["historical_test_reference_in_plan"],
            },
        )


class RiskAgent(BaseAgent):
    name = "risk"

    def run(self, context: AgentContext) -> AgentResult:
        from .rules import context_flags, change_profile
        cr = context.cr
        risk = str(cr.get("Risk") or "").strip()
        impact = str(cr.get("Risk and impact analysis") or "").strip()
        flags = context_flags(cr)
        profile = change_profile(cr)
        elevated_impact = bool(flags.get("elevated-impact"))
        declared_low = risk.lower() in {"low", "none", "minimal", "minimal risk"}
        contradiction = declared_low and elevated_impact
        confidence = 0.5 + (0.2 if risk else 0) + (0.2 if impact else 0) - (0.3 if contradiction else 0)
        confidence = round(max(0.05, min(0.95, confidence)), 2)
        findings: list[Finding] = []
        # Missing Risk is handled centrally by the field policy as advisory metadata.
        # The specialist agent should not emit a second readiness-affecting finding.
        if contradiction:
            findings.append(Finding("RISK_IMPACT_CONTRADICTION", "Declared risk conflicts with impact narrative", FindingSeverity.WARNING, f"Risk is declared {risk!r} but the change context suggests elevated production impact.", technical_detail=impact[:2000], recommendation="Reconcile the declared risk level with the described impact before CAB review."))
        findings.append(Finding("RISK_CONTEXT", "Risk context captured", FindingSeverity.INFO, f"Declared risk: {risk or 'unknown'}.", technical_detail=impact[:4000]))
        impact_level = "elevated" if profile["high_impact"] else "moderate" if impact else "unstated"
        business_notes = context.agent_state.get("business_impact", {}).get("notes", {})
        customer_signal = bool(business_notes.get("customer_visible_signal"))
        chain = [
            f"declared risk = {risk or 'unknown'}",
            f"impact narrative suggests {impact_level} production exposure" if impact else "no impact narrative provided",
            "customer-facing signal confirmed by business-impact agent" if customer_signal else "no customer-facing signal confirmed by business-impact agent",
            "contradiction: declared risk conflicts with impact narrative" if contradiction else "no direct contradiction",
            f"confidence {confidence:.2f}",
        ]
        return AgentResult(self.name, findings, [], {"chain": chain, "risk_confidence": confidence, "contradiction": contradiction, "change_profile": profile})


class EvidenceAgent(BaseAgent):
    name = "evidence"

    def run(self, context: AgentContext) -> AgentResult:
        claim = str(context.cr.get("Test Results Evidence") or "").strip()
        finding = Finding("EVIDENCE_CLAIM", "Evidence claim recorded", FindingSeverity.INFO, "The CR evidence declaration is recorded; attachments require independent verification.", technical_detail=f"Test Results Evidence={claim!r}")
        return AgentResult(self.name, [finding], [], {"requires_document_stage": True})


class CloneAgent(BaseAgent):
    name = "clone"

    def run(self, context: AgentContext) -> AgentResult:
        if not self.memory:
            return AgentResult(self.name, [], [], {"clone_candidates": []})
        from .clone import clone_analysis
        from .retrieval import build_cr_query
        query = build_cr_query(context.cr)
        matches = self.memory.search(query, kinds=[MemoryKind.CAB_HISTORY, MemoryKind.SIMILARITY], limit=12)
        analyses: list[dict[str, Any]] = []
        findings: list[Finding] = []
        for match in matches:
            historical = match.metadata.get("source_record") if isinstance(match.metadata, dict) else None
            if not isinstance(historical, dict):
                continue
            analysis = clone_analysis(context.cr, historical)
            item = {"change_id": analysis.candidate.change_id, "similarity": analysis.candidate.similarity, "historical_decision": match.metadata.get("historical_outcome") or analysis.candidate.historical_decision, "cab_recommendation": analysis.candidate.cab_recommendation, "reusable_fields": list(analysis.reusable_fields), "changed_fields": list(analysis.changed_fields), "revalidation_fields": list(analysis.revalidation_fields), "recommendation": analysis.recommendation, "retrieval_score": match.score}
            analyses.append(item)
        analyses.sort(key=lambda item: (item["similarity"], item.get("retrieval_score") or 0), reverse=True)
        strong = next((item for item in analyses if item["recommendation"] == "CLONE_CANDIDATE"), None)
        if strong:
            findings.append(Finding("CLONE_CANDIDATE_FOUND", "Strong historical clone candidate found", FindingSeverity.INFO, f"A {strong['similarity']:.0%} similar historical Normal CR was found.", technical_detail=f"Historical CR={strong['change_id']}; changed fields={strong['changed_fields']}", recommendation="Reuse only the stable structure and revalidate every changed field/evidence item."))
        if analyses:
            top = analyses[0]
            changed = top.get("changed_fields") or []
            plural = "es" if len(analyses) != 1 else ""
            chain = [f"{len(analyses)} historical match{plural}", f"strongest match was {top['historical_decision']}" if top.get("historical_decision") else "strongest match has no recorded historical outcome", f"current CR differs in: {', '.join(changed[:3])}" if changed else "no material field differences identified vs. strongest match", "historical outcome cannot be directly reused" if top.get("recommendation") != "CLONE_CANDIDATE" else "historical outcome may inform, but does not substitute for, this decision"]
        else:
            chain = ["no historical matches found"]
        return AgentResult(self.name, findings, [], {"clone_candidates": analyses[:5], "chain": chain})


class DecisionAgent(BaseAgent):
    name = "decision"

    def run(self, context: AgentContext) -> AgentResult:
        return AgentResult(self.name, [Finding("DECISION_GATED", "Decision remains policy-gated", FindingSeverity.INFO, "Final readiness is determined after specialist findings, evidence verification and policy gates.")], [], {"generative_brain_is_external": True})

class CrossAgentConsistencyAgent(BaseAgent):
    """Cross-check specialist outputs after domain reasoning has completed."""

    name = "consistency"

    def run(self, context: AgentContext) -> AgentResult:
        state = context.agent_state
        findings: list[Finding] = []
        chains: list[str] = []

        def notes(name: str) -> dict[str, Any]:
            item = state.get(name, {})
            return item.get("notes", {}) if isinstance(item, dict) else {}

        context_notes = notes("context")
        technical_notes = notes("technical")
        testing_notes = notes("testing")
        risk_notes = notes("risk")
        business_notes = notes("business_impact")
        profile = context_notes.get("change_profile") or {}

        if profile.get("functional") and technical_notes.get("implementation_present") is False:
            findings.append(Finding(
                "AGENT_CONSISTENCY_IMPLEMENTATION",
                "Agent disagreement: functional change lacks implementation detail",
                FindingSeverity.WARNING,
                "The context agent classified the CR as functionally relevant, while the technical agent found no populated implementation plan.",
                recommendation="Reconcile the implementation description before CAB.",
            ))
            chains.append("context=FUNCTIONAL; technical=implementation_missing")

        if profile.get("customer_facing") and not business_notes.get("customer_visible_signal"):
            findings.append(Finding(
                "AGENT_CONSISTENCY_CUSTOMER_SIGNAL",
                "Agent disagreement: customer impact signal is unresolved",
                FindingSeverity.WARNING,
                "The context profile indicates customer-facing scope, but the business-impact agent did not confirm a corresponding narrative signal.",
                recommendation="Clarify the actual customer-facing impact and affected service.",
            ))
            chains.append("context=customer-facing; business-impact narrative signal=absent")

        matrix = testing_notes.get("evidence_matrix") or {}
        if matrix.get("execution_status") == "FAILED":
            findings.append(Finding(
                "AGENT_CONSISTENCY_TEST_EXECUTION",
                "Agent consistency check found a failed execution state",
                FindingSeverity.BLOCKING,
                "The testing evidence matrix records FAILED execution; this remains a material blocker regardless of other positive test language.",
                recommendation="Reconcile the execution record and document the final test outcome.",
            ))
            chains.append("testing execution_status=FAILED")

        if risk_notes.get("contradiction") and not profile.get("high_impact"):
            findings.append(Finding(
                "AGENT_CONSISTENCY_RISK_PROFILE",
                "Risk conflict requires profile review",
                FindingSeverity.WARNING,
                "The risk agent found a risk/impact conflict that the shared profile did not classify as high impact.",
                recommendation="Review the impact narrative and risk classification together.",
            ))
            chains.append("risk contradiction=True; profile high_impact=False")

        if not findings:
            chains.append("specialist outputs are mutually consistent on the available signals")

        return AgentResult(
            self.name,
            findings,
            [],
            {"chain": chains, "agent_count_reviewed": len(state), "consistency_clean": not findings},
        )


DEFAULT_AGENT_TYPES = [FieldAgent, ContextAgent, TechnicalAgent, BusinessImpactAgent, MemoryAgent, TestingAgent, RiskAgent, EvidenceAgent, CloneAgent, CrossAgentConsistencyAgent, DecisionAgent]
