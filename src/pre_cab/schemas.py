"""Shared contracts used by every Pre-CAB component and agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Strictness(str, Enum):
    LENIENT = "lenient"
    BALANCED = "balanced"
    STRICT = "strict"


class Decision(str, Enum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    NOT_READY = "NOT_READY"


class FindingSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    BLOCKING = "BLOCKING"


@dataclass(frozen=True)
class Finding:
    code: str
    title: str
    severity: FindingSeverity
    message: str
    technical_detail: str = ""
    evidence_refs: tuple[str, ...] = ()
    recommendation: str = ""


@dataclass(frozen=True)
class Requirement:
    name: str
    required: bool
    reason: str
    source: str = "context"


@dataclass
class ValidationResult:
    decision: Decision
    confidence: float
    score: float
    strictness: Strictness
    findings: list[Finding] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    technical_summary: str = ""
    cab_summary: str = ""
    historical_matches: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == FindingSeverity.BLOCKING]


@dataclass(frozen=True)
class AgentContext:
    cr: dict[str, Any]
    retrieved_memory: list[dict[str, Any]] = ()
    evidence: list[dict[str, Any]] = ()
    strictness: Strictness = Strictness.BALANCED
    prior_findings: tuple[Finding, ...] = ()
