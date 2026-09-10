"""Data-driven field-requirement engine.

Answers one question per field, per CR: *is this field required, and does the
CR satisfy it* -- without hand-coded per-category keyword lists. The answer
comes from a rule table mined from historical CRs (scripts/mine_field_requirements.py),
not from logic baked into this module. Feeding it a new export with categories
this code has never seen requires zero code changes: unseen buckets fall back
to the category level, then to the global level, and are labelled as such so
the gap is visible rather than silently guessed at.

Rule lifecycle (see spec: candidate rule lifecycle):
    CANDIDATE -> VALIDATED -> ACTIVE -> DEPRECATED
The miner only ever produces CANDIDATE tables. Promoting a table to ACTIVE is
a deliberate operational step (e.g. after benchmarking it against held-out
historical outcomes) -- this module will happily load and apply a table at any
lifecycle stage, but callers should not treat CANDIDATE output as policy truth
without that validation step having happened.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .schemas import FindingSeverity

DEFAULT_RULE_TABLE_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "field_requirements.generated.json"


class RequirementLevel(str, Enum):
    REQUIRED = "REQUIRED"
    CONDITIONAL = "CONDITIONAL"
    RECOMMENDED = "RECOMMENDED"
    OPTIONAL = "OPTIONAL"
    NOT_OBSERVED = "NOT_OBSERVED"


# How a mined requirement level maps to a rule-engine severity when the field
# is missing. NOT_OBSERVED means the historical data never showed this field
# populated in this bucket, i.e. it's very likely not applicable at all here.
_MISSING_SEVERITY = {
    RequirementLevel.REQUIRED: FindingSeverity.BLOCKING,
    RequirementLevel.CONDITIONAL: FindingSeverity.WARNING,
    RequirementLevel.RECOMMENDED: FindingSeverity.WARNING,
    RequirementLevel.OPTIONAL: FindingSeverity.INFO,
    RequirementLevel.NOT_OBSERVED: FindingSeverity.INFO,
}

_POSITIVE_VALUES = {"yes", "yes sr", "yes cr"}
_NOT_APPLICABLE_VALUES = {"not applicable", "na", "n/a"}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _blank(value: Any) -> bool:
    text = _text(value)
    return text == "" or text.lower() in {"null", "none"}


@dataclass(frozen=True)
class FieldRequirementFinding:
    field: str
    requirement_level: RequirementLevel
    satisfied: bool
    severity: FindingSeverity
    confidence: float
    rationale: str
    evidence: str
    rule_scope: str  # "bucket:<Category>::<SubCategory>" | "category:<Category>" | "global"
    support: int


@dataclass(frozen=True)
class FieldRequirementReport:
    category: str
    sub_category: str
    resolved_scope: str
    findings: list[FieldRequirementFinding] = field(default_factory=list)

    @property
    def blocking(self) -> list[FieldRequirementFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.BLOCKING]

    @property
    def warnings(self) -> list[FieldRequirementFinding]:
        return [f for f in self.findings if f.severity == FindingSeverity.WARNING]


class FieldRequirementEngine:
    """Applies a mined field-requirement rule table to a CR.

    Resolution order per (Category, Sub Category, field):
        1. exact bucket rule  (most specific, only exists with >= min-support historical evidence)
        2. category-level rule (folds sub-categories together when a bucket had too little evidence)
        3. global rule (whatever the org does across *all* Normal CRs)
    This is what makes the engine evolvable rather than brittle: a brand-new
    Category/Sub Category pair that has never been mined still gets a sensible,
    explainable answer instead of an exception or a silent skip.
    """

    def __init__(self, table: dict[str, Any] | None = None, *, table_path: Path | str | None = None):
        if table is not None:
            self._table = table
        else:
            path = Path(table_path) if table_path else DEFAULT_RULE_TABLE_PATH
            self._table = self._load(path)
        self._path = Path(table_path) if table_path else DEFAULT_RULE_TABLE_PATH

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {
                "rule_table_version": 0,
                "lifecycle_status": "MISSING",
                "buckets": {},
                "categories": {},
                "global": {"support": 0, "fields": {}},
            }
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def reload(self) -> None:
        """Pick up a freshly re-mined table without restarting the process."""
        self._table = self._load(self._path)

    @property
    def version(self) -> int:
        return int(self._table.get("rule_table_version", 0))

    @property
    def lifecycle_status(self) -> str:
        return str(self._table.get("lifecycle_status", "UNKNOWN"))

    def _resolve_fields(self, category: str, sub_category: str) -> tuple[dict[str, Any], str, int]:
        bucket_key = f"{category}::{sub_category}"
        bucket = self._table.get("buckets", {}).get(bucket_key)
        if bucket:
            return bucket["fields"], f"bucket:{bucket_key}", bucket["support"]

        cat = self._table.get("categories", {}).get(category)
        if cat:
            return cat["fields"], f"category:{category}", cat["support"]

        glob = self._table.get("global", {"support": 0, "fields": {}})
        return glob.get("fields", {}), "global", glob.get("support", 0)

    def evaluate(self, cr: dict[str, Any]) -> FieldRequirementReport:
        category = _text(cr.get("Category")) or "Unknown"
        sub_category = _text(cr.get("Sub Category")) or "Unknown"
        rules, scope, _bucket_support = self._resolve_fields(category, sub_category)

        findings: list[FieldRequirementFinding] = []
        for field_name, rule in rules.items():
            try:
                level = RequirementLevel(rule.get("requirement_level", "OPTIONAL"))
            except ValueError:
                level = RequirementLevel.OPTIONAL

            satisfied = self._is_satisfied(field_name, rule, cr.get(field_name))
            severity = FindingSeverity.INFO if satisfied else _MISSING_SEVERITY[level]

            findings.append(
                FieldRequirementFinding(
                    field=field_name,
                    requirement_level=level,
                    satisfied=satisfied,
                    severity=severity,
                    confidence=float(rule.get("confidence", 0.5)),
                    rationale=self._rationale(field_name, level, satisfied, scope),
                    evidence=str(rule.get("evidence", "")),
                    rule_scope=scope,
                    support=int(rule.get("support", 0)),
                )
            )

        return FieldRequirementReport(
            category=category, sub_category=sub_category, resolved_scope=scope, findings=findings
        )

    @staticmethod
    def _is_satisfied(field_name: str, rule: dict[str, Any], value: Any) -> bool:
        if rule.get("kind") == "signoff":
            text = _text(value).lower()
            if _blank(value):
                return False
            # A recorded "Not Applicable" satisfies a CONDITIONAL/OPTIONAL field
            # (it is a legitimate disposition), but for a REQUIRED field the
            # historical data says this bucket overwhelmingly needs a real
            # Yes/No, so NA alone is treated as unresolved.
            if rule.get("requirement_level") == "REQUIRED" and text in _NOT_APPLICABLE_VALUES:
                return False
            return True
        # Descriptive fields: presence is the bar.
        return not _blank(value)

    @staticmethod
    def _rationale(field_name: str, level: RequirementLevel, satisfied: bool, scope: str) -> str:
        if satisfied:
            return f"{field_name} is populated, satisfying its {level.value.lower()} requirement ({scope})."
        if level == RequirementLevel.REQUIRED:
            return (
                f"{field_name} is required for this change's category/sub-category based on "
                f"historical CAB intake patterns ({scope}), but is missing or unresolved."
            )
        if level == RequirementLevel.CONDITIONAL:
            return (
                f"{field_name} is applicable more often than not for this bucket ({scope}) but "
                "was not recorded; verify whether it applies here."
            )
        if level == RequirementLevel.RECOMMENDED:
            return f"{field_name} is commonly present for this bucket ({scope}); its absence is worth noting."
        return f"{field_name} is rarely used for this bucket ({scope}); absence is not penalized."
