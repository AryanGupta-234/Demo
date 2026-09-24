"""Central, explainable policy engine for in-scope Normal change requests.

The rule engine separates four concepts:
1. baseline fields that should normally exist for every Normal CR;
2. context-triggered requirements (UAT, approval, outage, security, data, DB, network);
3. field-quality scoring instead of simple populated/empty checks; and
4. historical field signals, which remain advisory evidence rather than policy truth.

This module deliberately does not invent organization policy. A field becomes a hard gate only when
its applicability is supported by an explicit rule or an existing deterministic gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ApplicabilitySignal:
    requirement: str
    applicable: bool
    confidence: float
    rationale: str
    source: str
    required_fields: tuple[str, ...] = ()
    evidence_fields: tuple[str, ...] = ()
    severity: str = "WARNING"


@dataclass(frozen=True)
class FieldPolicy:
    field: str
    label: str
    domain: str
    baseline: bool = False
    required_when: tuple[str, ...] = ()
    quality_weight: float = 1.0
    missing_severity: str = "WARNING"
    evidence_capable: bool = False


@dataclass(frozen=True)
class FieldQuality:
    field: str
    present: bool
    score: float
    reasons: tuple[str, ...]


MODEL_DESCRIPTIVE_FIELDS: tuple[str, ...] = (
    "Short description", "Description", "Justification", "Implementation plan",
    "Change plan", "Backout plan", "Work notes", "Comments", "Test plan",
)
MODEL_SIGNOFF_FIELDS: tuple[str, ...] = (
    "UAT signoff", "Customer Approval", "TCS QA signoff",
    "Test Results Evidence", "Lower Environment Reference CR/SR",
)
WORKFLOW_DEFAULT_ENVIRONMENT = "PROD"
NON_PROD_VALIDATION_TERMS: tuple[str, ...] = (
    "pre-prod", "pre prod", "preprod", "non-prod", "non prod", "nonprod",
    "uat", "sit", "staging", "lower environment", "test environment",
)
SIGNOFF_POSITIVE_VALUES = frozenset({
    "yes", "y", "approved", "complete", "completed", "passed", "pass",
    "provided", "available", "attached", "verified",
})
SIGNOFF_NEGATIVE_VALUES = frozenset({
    "no", "n", "rejected", "failed", "not approved",
})
SIGNOFF_NOT_APPLICABLE_VALUES = frozenset({
    "", "na", "n/a", "not applicable", "not-applicable",
})


# Canonical fields we have already established from the ServiceNow-normalized schema. Unknown
# fields remain available to the model but are never made mandatory just because they exist.
FIELD_POLICIES: tuple[FieldPolicy, ...] = (
    FieldPolicy("Number", "Change identifier", "identity", baseline=True, missing_severity="BLOCKING"),
    FieldPolicy("Type", "Change type", "identity", baseline=True, missing_severity="BLOCKING"),
    FieldPolicy("Short description", "Change summary", "description", baseline=True),
    FieldPolicy("Description", "Change description", "description", baseline=True),
    FieldPolicy("Justification", "Business / operational reason", "business", baseline=True),
    FieldPolicy("Implementation plan", "Implementation approach", "implementation", baseline=True, missing_severity="BLOCKING"),
    FieldPolicy("Backout plan", "Recovery / rollback path", "recovery", baseline=True, missing_severity="BLOCKING", evidence_capable=True),
    FieldPolicy("Work notes", "Work notes", "journal"),
    FieldPolicy("Comments", "Comments", "journal"),
    FieldPolicy("Change plan", "Change plan", "implementation"),
    FieldPolicy("Test plan", "Testing approach", "testing", baseline=True),
    FieldPolicy("Risk", "Risk classification", "risk", baseline=True, missing_severity="INFO"),
    FieldPolicy("Risk and impact analysis", "Risk and impact analysis", "risk", required_when=("elevated-impact",)),
    FieldPolicy("Configuration item", "Configuration item", "technical", evidence_capable=True),
    FieldPolicy("TCS QA signoff", "TCS QA sign-off", "testing", evidence_capable=True),
    FieldPolicy("Category", "Change category", "classification", required_when=("context-required",)),
    FieldPolicy("Sub Category", "Change sub-category", "classification", required_when=("context-required",), missing_severity="INFO"),
    FieldPolicy("Conflict status", "Conflict status", "governance", required_when=("conflict-check",), missing_severity="INFO"),
    FieldPolicy("Customer Approval", "Customer approval", "approval", required_when=("customer-impact",), evidence_capable=True),
    FieldPolicy("UAT signoff", "UAT sign-off", "testing", required_when=("uat",), evidence_capable=True),
    FieldPolicy("Test Results Evidence", "Test results evidence", "testing", required_when=("testing-evidence",), evidence_capable=True),
    # NOTE: "Planned start"/"Planned end" (not "...date") is what this org's ServiceNow
    # export actually calls these fields once aliased -- see input_loader.normalize_cr_record.
    FieldPolicy("Planned start", "Planned start", "schedule", required_when=("production-change-window",), missing_severity="INFO"),
    FieldPolicy("Planned end", "Planned end", "schedule", required_when=("production-change-window",), missing_severity="INFO"),
    # "Affected Customers" was removed as a hard FieldPolicy: measured against the
    # real 2,013-record historical export, this field is populated in only 28
    # records (1.4%) org-wide -- not just when customer-impact fires, essentially
    # never. The customer-impact context flag itself fires on 58.8% of real CRs,
    # so keeping this as a required field meant a WARNING that was satisfied
    # almost never, on well over half of every report -- noise, not signal. The
    # underlying concern is still carried by the "Customer Approval" field policy
    # and contextual signal below, which the real data shows the org actually uses.
    # "Service impact", "Downtime", "Security review", "Privacy review", "Database
    # validation", "Network approval", "Monitoring plan" were deliberately removed from
    # this list. They do not exist as fields anywhere in this org's real ServiceNow
    # export (verified against 3,548 real historical CR records across 141 distinct
    # field names) -- they were generic ServiceNow-textbook field names, not this
    # org's schema, so every real CR was being flagged MISSING for fields it can
    # never populate. The underlying concerns (impact detail, security/privacy
    # relevance, DB/network risk, monitoring) are still surfaced -- see the
    # advisory-only ApplicabilitySignal entries in contextual_signals() below,
    # which report applicability without demanding a field that isn't there.
)


_BASELINE_FIELDS = {policy.field for policy in FIELD_POLICIES if policy.baseline}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _blob(cr: dict[str, Any]) -> str:
    keys = (
        "Short description", "Description", "Justification", "Risk and impact analysis",
        "Category", "Sub Category", "Configuration item", "Environment", "Service impact",
        "Implementation plan", "Change plan", "Backout plan", "Work notes", "Comments",
        "Test plan", "Lower Environment Reference CR/SR",
    )
    return " ".join(_text(cr.get(key)) for key in keys if _text(cr.get(key)))


def effective_environment(cr: dict[str, Any]) -> str:
    raw = _text(cr.get("Environment"))
    return raw.upper() if raw else WORKFLOW_DEFAULT_ENVIRONMENT


def non_prod_validation_state(cr: dict[str, Any]) -> dict[str, Any]:
    fields = MODEL_DESCRIPTIVE_FIELDS + ("Lower Environment Reference CR/SR",)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", " ".join(str(cr.get(field) or "") for field in fields).lower())
    validation_terms = ("test", "tested", "testing", "validate", "validated", "validation", "sanity", "verified", "verification", "success", "successful")
    matches: list[str] = []
    for sentence in sentences:
        env_terms = [term for term in NON_PROD_VALIDATION_TERMS if re.search(rf"\b{re.escape(term)}\b", sentence)]
        if not env_terms:
            continue
        if any(term in sentence for term in validation_terms):
            matches.extend(env_terms)
    # An explicit lower-environment reference can itself establish the validation class,
    # except when the field is explicitly No/NA.
    ref = _text(cr.get("Lower Environment Reference CR/SR"))
    if ref and ref not in {"no", "na", "n/a", "not applicable"}:
        matches.append("lower environment reference")
    return {
        "claimed": bool(matches),
        "class": "NON_PROD_VALIDATION" if matches else "NOT_MENTIONED",
        "matched_terms": list(dict.fromkeys(matches)),
        "equivalence": "SIT/UAT/Pre-PROD/lower or test environment references are treated as one non-PROD validation class when explicitly described as validation/testing.",
    }


def signoff_disposition(value: Any) -> str:
    text = _text(value)
    if text in SIGNOFF_NOT_APPLICABLE_VALUES:
        return "NOT_RECORDED" if not text else "NOT_APPLICABLE"
    if text in SIGNOFF_POSITIVE_VALUES:
        return "POSITIVE"
    if text in SIGNOFF_NEGATIVE_VALUES:
        return "NEGATIVE"
    return "RECORDED_VALUE"


def model_field_validation(cr: dict[str, Any]) -> dict[str, Any]:
    descriptive = {}
    for field in MODEL_DESCRIPTIVE_FIELDS:
        value = _text(cr.get(field))
        descriptive[field] = {
            "status": "PRESENT" if value else "MISSING_UNSET",
            "length": len(value),
            "placeholder": value in {"n/a", "na", "not applicable", "unknown", "tbd", "to be decided"},
        }
    signoffs = {}
    for field in MODEL_SIGNOFF_FIELDS:
        raw = cr.get(field)
        signoffs[field] = {
            "value": str(raw).strip() if raw not in (None, "", [], {}) else None,
            "disposition": signoff_disposition(raw),
        }
    return {
        "descriptive": descriptive,
        "signoffs": signoffs,
        "effective_environment": effective_environment(cr),
        "environment_derived": not bool(_text(cr.get("Environment"))),
        "non_prod_validation": non_prod_validation_state(cr),
    }


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


@dataclass(frozen=True)
class ChangeProfile:
    archetypes: tuple[str, ...]
    primary_archetype: str
    infrastructure: bool
    functional: bool
    security: bool
    database: bool
    network: bool
    customer_facing: bool
    service_restart: bool
    high_impact: bool
    production: bool
    formal_test_evidence_expected: bool
    non_prod_validation_expected: bool
    approval_expected: bool
    confidence: float


def _has_negated_signal(text: str, phrase: str) -> bool:
    """Return True when a signal is explicitly negated in nearby language."""
    patterns = (
        rf"\bno\s+{re.escape(phrase)}\b",
        rf"\bwithout\s+{re.escape(phrase)}\b",
        rf"\bnot\s+{re.escape(phrase)}\b",
        rf"\bdoes\s+not\s+{re.escape(phrase)}\b",
        rf"\bno\s+customer[- ]facing\s+{re.escape(phrase)}\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _contextual_phrase(text: str, phrases: Iterable[str], *, exclude: Iterable[str] = ()) -> bool:
    blocked = tuple(exclude)
    for phrase in phrases:
        if phrase in text and not any(neg in text for neg in blocked):
            return True
    return False


def _score_signal(text: str, positive: Iterable[str], negative: Iterable[str] = ()) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[str] = []
    for phrase in positive:
        if phrase in text:
            if any(re.search(rf"\b{re.escape(neg)}\b", text[max(0, text.find(phrase)-60):text.find(phrase)+len(phrase)+60]) for neg in negative):
                continue
            score += 1.0
            hits.append(phrase)
    return score, hits


def change_profile(cr: dict[str, Any]) -> dict[str, Any]:
    """Build one explainable semantic profile reused by validation, agents and the model."""
    text = _blob(cr)
    category = _text(cr.get("Category"))
    short_desc = _text(cr.get("Short description"))
    description = _text(cr.get("Description"))
    implementation = _text(cr.get("Implementation plan"))
    test_plan = _text(cr.get("Test plan"))
    risk = _text(cr.get("Risk"))
    production = _text(cr.get("Environment")) in {"", "prod", "production", "production environment", "live"}

    infrastructure_score, infrastructure_hits = _score_signal(
        text,
        ("infrastructure", "infra patch", "os patch", "server patch", "security patch", "server maintenance",
         "server restart", "reboot", "middleware", "storage", "virtual machine", "vm"),
    )
    functional_score, functional_hits = _score_signal(
        text,
        ("enhancement", "defect", "bug fix", "workflow", "functional change", "interface change",
         "application change", "api change", "user interface"),
    )
    customer_score, customer_hits = _score_signal(
        text,
        ("customer-facing", "customer facing", "external user", "customer impact", "customer transaction",
         "payment processing", "branch transaction", "end-user", "end user"),
        negative=("no", "without", "not"),
    )
    security_score, security_hits = _score_signal(
        text,
        ("security", "vulnerability", "zero-day", "patch", "certificate", "credential", "firewall",
         "authentication", "authorization", "encryption"),
    )
    database_score, database_hits = _score_signal(
        text, ("database", "schema", "stored procedure", "migration", "sql")
    )
    network_score, network_hits = _score_signal(
        text, ("firewall", "network", "routing", "load balancer", "proxy", "dns", "connectivity")
    )
    restart_score, restart_hits = _score_signal(
        text, ("restart", "reboot", "downtime", "maintenance window", "unavailable", "outage")
    )

    infrastructure = category in {"infrastructure", "infra"} or infrastructure_score >= 1
    functional = functional_score >= 1 or (
        not infrastructure and any(term in text for term in ("transaction enhancement", "customer-facing", "bug", "defect"))
    )
    customer_facing = customer_score >= 1
    security = security_score >= 1
    database = database_score >= 1
    network = network_score >= 1
    service_restart = restart_score >= 1

    # "critical" alone is not enough. High impact should combine explicit impact language,
    # high/critical declared risk, or a customer/service outage signal.
    explicit_high_risk = risk in {"high", "critical", "1 - high", "2 - high", "very high"}
    high_impact = explicit_high_risk or any(term in text for term in (
        "high impact", "major outage", "production outage", "service unavailable", "customer outage",
        "payment disruption", "transaction outage",
    ))
    archetypes: list[str] = []
    if infrastructure:
        archetypes.append("INFRASTRUCTURE")
    if security:
        archetypes.append("SECURITY")
    if database:
        archetypes.append("DATABASE")
    if network:
        archetypes.append("NETWORK")
    if functional:
        archetypes.append("FUNCTIONAL")
    if customer_facing:
        archetypes.append("CUSTOMER_FACING")
    if service_restart:
        archetypes.append("SERVICE_RESTART")
    if not archetypes:
        archetypes.append("GENERAL")
    priority = (
        "FUNCTIONAL" if functional and not infrastructure else
        "DATABASE" if database and not infrastructure else
        "NETWORK" if network and not infrastructure else
        "SECURITY" if security and infrastructure else
        "INFRASTRUCTURE" if infrastructure else
        "CUSTOMER_FACING" if customer_facing else
        "GENERAL"
    )
    # Formal evidence is primarily a functional/application control. Infra maintenance still
    # needs executable validation, but the execution record can live in Work Notes/Comments.
    formal_test = functional and not infrastructure
    non_prod_expected = functional and not infrastructure and customer_facing
    approval_expected = customer_facing
    active_signals = sum(bool(x) for x in (
        infrastructure, functional, security, database, network, customer_facing, service_restart, high_impact
    ))
    confidence = min(0.99, 0.58 + min(active_signals, 5) * 0.07)
    return {
        "archetypes": archetypes,
        "primary_archetype": priority,
        "infrastructure": infrastructure,
        "functional": functional,
        "security": security,
        "database": database,
        "network": network,
        "customer_facing": customer_facing,
        "service_restart": service_restart,
        "high_impact": high_impact,
        "production": production,
        "formal_test_evidence_expected": formal_test,
        "non_prod_validation_expected": non_prod_expected,
        "approval_expected": approval_expected,
        "confidence": confidence,
        "signal_hits": {
            "infrastructure": infrastructure_hits,
            "functional": functional_hits,
            "customer_facing": customer_hits,
            "security": security_hits,
            "database": database_hits,
            "network": network_hits,
            "service_restart": restart_hits,
        },
        "evidence_expectations": {
            "implementation": True,
            "rollback": True,
            "test_plan": True,
            "formal_test_results": formal_test,
            "post_change_validation": True,
            "non_prod_validation": non_prod_expected,
            "customer_approval": approval_expected,
            "change_window": production,
        },
    }


def evidence_matrix(cr: dict[str, Any]) -> dict[str, Any]:
    """Evaluate evidence quality and applicability without collapsing all gaps into blockers."""
    profile = change_profile(cr)
    implementation = _text(cr.get("Implementation plan"))
    backout = _text(cr.get("Backout plan"))
    test_plan = _text(cr.get("Test plan"))
    evidence = _text(cr.get("Test Results Evidence"))
    journal = " ".join(str(cr.get(k) or "") for k in ("Work notes", "Comments")).strip().lower()
    execution_terms = (
        "tested", "testing completed", "test completed", "validation completed", "validated",
        "verified", "verification completed", "sanity check completed", "passed", "successful",
        "patching completed", "checks completed",
    )
    execution_claimed = bool(journal) and any(term in journal for term in execution_terms)
    historical_ref = bool(re.search(r"\b(?:chg|cr|change)\s*\d{5,}\b", test_plan.lower()))
    evidence_positive = signoff_disposition(evidence) == "POSITIVE"
    dimensions = {
        "implementation": bool(implementation),
        "rollback": bool(backout) and backout not in {"na", "n/a", "none", "not applicable"},
        "test_plan": bool(test_plan),
        "formal_test_results": evidence_positive,
        "post_change_execution": execution_claimed,
    }
    return {
        "profile": profile,
        "dimensions": dimensions,
        "execution_claimed": execution_claimed,
        "formal_evidence_positive": evidence_positive,
        "historical_test_reference_in_plan": historical_ref,
        "test_plan_is_currently_executable": bool(test_plan) and not (
            historical_ref and len(test_plan) < 80
        ),
        "applicable_gaps": [
            key for key, value in dimensions.items()
            if not value and profile["evidence_expectations"].get(key, False)
        ],
        "interpretation": (
            "Infrastructure maintenance: test plan + post-change validation are relevant; formal Test Results Evidence "
            "and non-PROD validation are not universal gates."
            if profile["infrastructure"]
            else "Functional/application change: test execution or formal evidence is expected before readiness is established."
        ),
    }


def contradiction_signals(cr: dict[str, Any]) -> list[dict[str, str]]:
    """Detect direct claim conflicts before the LLM sees the record."""
    profile = change_profile(cr)
    blob = _blob(cr)
    signals: list[dict[str, str]] = []
    impact = _text(cr.get("Risk and impact analysis"))
    risk = _text(cr.get("Risk"))
    test_plan = _text(cr.get("Test plan"))
    work = " ".join(str(cr.get(k) or "") for k in ("Work notes", "Comments")).lower()

    if "no outage" in impact and any(term in blob for term in ("downtime", "outage", "unavailable")):
        signals.append({"code": "IMPACT_CONTRADICTION", "message": "Impact narrative says no outage while another current field mentions outage/downtime."})
    if risk in {"low", "minimal", "minimal risk"} and profile["high_impact"]:
        signals.append({"code": "RISK_IMPACT_CONTRADICTION", "message": "Declared low risk conflicts with explicit high-impact signals."})
    if profile["functional"] and not profile["infrastructure"] and not test_plan:
        signals.append({"code": "FUNCTIONAL_TESTING_GAP", "message": "Functional/application change lacks a test plan."})
    if work and any(term in work for term in ("completed successfully", "tested successfully", "validation passed")) and test_plan and "tbd" in test_plan:
        signals.append({"code": "TEST_PLAN_EXECUTION_CONTRADICTION", "message": "Journal claims successful execution while the structured test plan still contains TBD language."})
    if profile["infrastructure"] and profile["formal_test_evidence_expected"]:
        signals.append({"code": "PROFILE_AMBIGUOUS", "message": "Change contains both infrastructure and functional signals; formal test evidence applicability should be reviewed rather than assumed."})
    return signals


def _has_word(text: str, terms: Iterable[str]) -> bool:
    """Whole-word match, not substring.

    Several short/generic terms below (db, table, port, api, ...) were measured
    against the real 2,013-record historical Normal-CR export as plain substring
    checks and turned out to be dominated by false positives: e.g. bare "port"
    matched "reporting"/"support"/"important" 96% of the time it fired, "table"
    matched "-portable"/"-acceptable"/"-suitable" endings ~34% of the time, "db"
    matched embedded inside CI/hostnames like "PRDDBSRV01" ~38% of the time.
    Word-boundary matching is used only for terms measured to have this problem;
    intentionally substring-style terms (e.g. bare "patch" catching "patching"/
    "patches") are left as plain _has_any substring checks.
    """
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def context_flags(cr: dict[str, Any]) -> dict[str, bool]:
    """Derive one consistent semantic context map from the shared change profile."""
    profile = change_profile(cr)
    sensitive_text = _blob(cr)
    sensitive_data = _has_any(sensitive_text, (
        "pii", "personal data", "customer data", "sensitive data", "card data", "account data", "privacy",
    ))
    return {
        "infrastructure": bool(profile["infrastructure"]),
        "customer-impact": bool(profile["customer_facing"]),
        "functional": bool(profile["functional"]),
        "uat": bool(profile["functional"] and profile["customer_facing"] and not profile["infrastructure"]),
        "non-prod-validation": bool(non_prod_validation_state(cr)["claimed"]),
        "service-restart": bool(profile["service_restart"]),
        "security-change": bool(profile["security"]),
        "sensitive-data": sensitive_data,
        "database-change": bool(profile["database"]),
        "network-change": bool(profile["network"]),
        "high-impact": bool(profile["high_impact"]),
        "elevated-impact": bool(profile["high_impact"] or profile["customer_facing"]),
        "production-technical": bool(profile["production"] and (profile["infrastructure"] or profile["database"] or profile["network"] or profile["functional"])),
        "production-change-window": bool(profile["production"]),
        "impact-detail": bool(profile["production"] and (profile["service_restart"] or profile["customer_facing"] or profile["high_impact"])),
        "testing-evidence": bool(profile["formal_test_evidence_expected"]),
        "context-required": True,
        "customer-approval": bool(profile["approval_expected"]),
        "conflict-check": True,
    }

def _quality(value: Any, *, field: str) -> FieldQuality:
    text = _text(value)
    if not text:
        return FieldQuality(field, False, 0.0, ("missing",))
    if text in {"n/a", "na", "none", "not applicable", "unknown", "tbd", "to be decided"}:
        return FieldQuality(field, True, 0.20, ("placeholder-or-unjustified-value",))

    score = 0.45
    reasons: list[str] = []
    if len(text) >= 30:
        score += 0.15
        reasons.append("sufficient-detail")
    if len(text) >= 100:
        score += 0.10
        reasons.append("strong-detail")
    if any(token in text for token in ("step", "validate", "verify", "backup", "restore", "rollback", "monitor", "restart")):
        score += 0.12
        reasons.append("operational-signal")
    if any(token in text for token in ("because", "impact", "dependency", "if", "when", "after", "before")):
        score += 0.08
        reasons.append("reasoning-signal")
    if any(token in text for token in ("tbd", "later", "same as", "see above", "etc")):
        score -= 0.12
        reasons.append("weak-or-vague-language")
    return FieldQuality(field, True, max(0.0, min(1.0, score)), tuple(reasons) or ("present",))


def field_quality(cr: dict[str, Any]) -> list[FieldQuality]:
    return [_quality(cr.get(policy.field), field=policy.field) for policy in FIELD_POLICIES]


def field_policies() -> tuple[FieldPolicy, ...]:
    return FIELD_POLICIES


def required_field_policies(cr: dict[str, Any]) -> list[tuple[FieldPolicy, bool, tuple[str, ...]]]:
    flags = context_flags(cr)
    result: list[tuple[FieldPolicy, bool, tuple[str, ...]]] = []
    for policy in FIELD_POLICIES:
        reasons: list[str] = []
        required = policy.baseline
        if policy.required_when:
            matched = [flag for flag in policy.required_when if flags.get(flag, False)]
            if matched:
                required = True
                reasons.extend(matched)
        result.append((policy, required, tuple(reasons)))
    return result


def contextual_signals(cr: dict[str, Any]) -> list[ApplicabilitySignal]:
    flags = context_flags(cr)
    return [
        ApplicabilitySignal(
            "UAT", flags["uat"], 0.92 if flags["uat"] else 0.86,
            "Functional/customer-facing behavior generally requires appropriate functional validation; infrastructure maintenance does not automatically trigger UAT.",
            "contextual-rules-v2", required_fields=("Test plan", "UAT signoff"), evidence_fields=("UAT signoff", "Test Results Evidence"), severity="WARNING",
        ),
        ApplicabilitySignal(
            "Customer Approval", flags["customer-approval"], 0.88 if flags["customer-approval"] else 0.82,
            "Customer-visible impact is a signal to verify the applicable approval path; the CR is not assumed approved merely because an approval field exists.",
            "contextual-rules-v2", required_fields=("Customer Approval",), evidence_fields=("Customer Approval",), severity="WARNING",
        ),
        ApplicabilitySignal(
            "Outage / impact detail", flags["impact-detail"], 0.88 if flags["impact-detail"] else 0.80,
            "Production changes with restart, downtime, or material impact should state expected service impact "
            "and timing; this org has no dedicated field for it, so assess from the Description/Justification narrative.",
            "contextual-rules-v2", required_fields=("Description", "Justification"), severity="WARNING",
        ),
        ApplicabilitySignal(
            "Security review", flags["security-change"], 0.90 if flags["security-change"] else 0.80,
            "Security-relevant changes should have their review path documented; this org has no dedicated field "
            "for it, so assess from the Description/Justification/Risk and impact analysis narrative.",
            "contextual-rules-v2", required_fields=("Description", "Risk and impact analysis"), severity="WARNING",
        ),
        ApplicabilitySignal(
            "Privacy / data review", flags["sensitive-data"], 0.90 if flags["sensitive-data"] else 0.80,
            "Sensitive or personal data signals require the applicable privacy/data-control review to be "
            "documented; this org has no dedicated field for it, so assess from the Description/Justification narrative.",
            "contextual-rules-v2", required_fields=("Description", "Justification"), severity="WARNING",
        ),
        ApplicabilitySignal(
            "Database validation", flags["database-change"], 0.91 if flags["database-change"] else 0.80,
            "Database/schema changes need explicit validation and recovery consideration; this org has no dedicated "
            "field for it, so assess from the Implementation plan/Backout plan narrative.",
            "contextual-rules-v2", required_fields=("Implementation plan", "Backout plan"), severity="WARNING",
        ),
        ApplicabilitySignal(
            "Network approval", flags["network-change"], 0.90 if flags["network-change"] else 0.80,
            "Network/firewall/connectivity changes should have their approval path documented; this org has no "
            "dedicated field for it, so assess from the Implementation plan/Backout plan narrative.",
            "contextual-rules-v2", required_fields=("Implementation plan", "Backout plan"), severity="WARNING",
        ),
        ApplicabilitySignal(
            "Monitoring", flags["high-impact"], 0.86 if flags["high-impact"] else 0.78,
            "Higher-impact changes benefit from explicit post-change monitoring/validation; this org has no "
            "dedicated field for it, so assess from the Implementation plan/Test plan narrative.",
            "contextual-rules-v2", required_fields=("Implementation plan", "Test plan"), severity="WARNING",
        ),
        # "Rollback / recovery" and "Conflict check" were removed: both were
        # unconditional (applicable=True, confidence=0.99) duplicates of the
        # "Backout plan" and "Conflict status" FieldPolicy entries above, which
        # already generate their own real findings (BACKOUT_OK/BACKOUT_WEAK,
        # CONFLICT) against the actual field. Keeping both meant every report
        # listed the same requirement twice under two different names.
    ]


def rule_catalog_summary() -> dict[str, Any]:
    """Machine-readable catalog for reporting, debugging, and future rule governance."""
    return {
        "version": "2.0",
        "baseline_fields": sorted(_BASELINE_FIELDS),
        "conditional_fields": {
            policy.field: list(policy.required_when)
            for policy in FIELD_POLICIES
            if policy.required_when
        },
        "policy_counts": {
            "total": len(FIELD_POLICIES),
            "baseline": len(_BASELINE_FIELDS),
            "conditional": sum(bool(p.required_when) for p in FIELD_POLICIES),
            "evidence_capable": sum(p.evidence_capable for p in FIELD_POLICIES),
        },
        "scope": {"primary": "Normal", "excluded": ["Emergency"]},
    }
