"""Mine historical Normal-CR data into a versioned field-requirement rule table.

This is the "learning" half of the evolvable rule engine (see src/pre_cab/
field_requirement_engine.py for the "applying" half). It never hand-codes which
fields matter for which Category/Sub Category — it *measures* it from whatever
historical export it is pointed at, so a new category showing up in next
quarter's data does not require a code change.

Usage:
    python scripts/mine_field_requirements.py private_data/converted.json \
        --out config/field_requirements.generated.json

Output is a CANDIDATE-status rule table (see schemas in field_requirement_engine.py
for the lifecycle: CANDIDATE -> VALIDATED -> ACTIVE -> DEPRECATED). Nothing here
promotes itself to ACTIVE automatically -- that is a deliberate human/benchmark
gate per the "never let the system automatically rewrite production rules
without validation" constraint.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Fields whose *presence* is what matters (free-text plans, descriptions, CI links).
DESCRIPTIVE_FIELDS = [
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
]

# Fields whose *disposition* matters (Yes / No / Not Applicable evidence-signoff
# style fields). These need applicability + compliance separated out, because a
# high "Not Applicable" rate means the field is legitimately conditional, not
# missing -- exactly the "UAT is NOT universally mandatory" nuance the naive
# fill-rate approach gets wrong.
SIGNOFF_FIELDS = [
    "UAT signoff",
    "Customer Approval",
    "TCS QA signoff",
    "Test Results Evidence",
    "Lower Environment Reference CR/SR",
]

POSITIVE_VALUES = {"yes", "yes sr", "yes cr"}
NEGATIVE_VALUES = {"no"}
NOT_APPLICABLE_VALUES = {"not applicable", "na", "n/a"}

MIN_SUPPORT_FOR_BUCKET_RULE = 5
RULE_TABLE_VERSION = 1


def _blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"null", "none"}


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _bucket_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("Category") or "Unknown").strip(), str(record.get("Sub Category") or "Unknown").strip())


def _confidence_from_support(support: int) -> float:
    """More historical records -> more confidence, saturating well before it gets silly."""
    return round(min(0.97, 0.45 + 0.13 * math.log2(support + 1)), 3)


def _descriptive_rule(field: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    support = len(records)
    filled = sum(1 for r in records if not _blank(r.get(field)))
    fill_rate = filled / support if support else 0.0
    if fill_rate >= 0.90:
        level = "REQUIRED"
    elif fill_rate >= 0.50:
        level = "RECOMMENDED"
    elif fill_rate >= 0.05:
        level = "OPTIONAL"
    else:
        level = "NOT_OBSERVED"
    return {
        "kind": "descriptive",
        "requirement_level": level,
        "support": support,
        "fill_rate": round(fill_rate, 3),
        "confidence": _confidence_from_support(support),
        "evidence": f"Populated in {filled}/{support} historical Normal CRs in this bucket.",
    }


def _signoff_rule(field: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    support = len(records)
    values = [_norm(r.get(field)) for r in records if not _blank(r.get(field))]
    yes = sum(1 for v in values if v in POSITIVE_VALUES)
    no = sum(1 for v in values if v in NEGATIVE_VALUES)
    na = sum(1 for v in values if v in NOT_APPLICABLE_VALUES)
    decided = yes + no + na
    decided_rate = decided / support if support else 0.0
    applicable = yes + no
    applicability_rate = (applicable / decided) if decided else 0.0
    compliance_rate = (yes / applicable) if applicable else None

    # A field only earns REQUIRED if the org actually bothers to disposition it
    # (decided_rate) *and* it is applicable more often than not when they do.
    if decided_rate >= 0.30 and applicability_rate >= 0.60:
        level = "REQUIRED"
    elif decided_rate >= 0.15 and applicability_rate >= 0.35:
        level = "CONDITIONAL"
    elif applicability_rate > 0 or decided_rate > 0:
        level = "OPTIONAL"
    else:
        level = "NOT_OBSERVED"

    return {
        "kind": "signoff",
        "requirement_level": level,
        "support": support,
        "decided_rate": round(decided_rate, 3),
        "applicability_rate": round(applicability_rate, 3),
        "compliance_rate": round(compliance_rate, 3) if compliance_rate is not None else None,
        "confidence": _confidence_from_support(decided),
        "evidence": (
            f"Of {support} historical CRs, {decided} recorded a disposition "
            f"({yes} Yes / {no} No / {na} Not Applicable)."
        ),
    }


def mine(records: list[dict[str, Any]]) -> dict[str, Any]:
    normal = [r for r in records if _norm(r.get("Type")) == "normal"]

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normal:
        key = _bucket_key(record)
        buckets[key].append(record)
        by_category[key[0]].append(record)

    def build_rules(records_subset: list[dict[str, Any]]) -> dict[str, Any]:
        rules: dict[str, Any] = {}
        for field in DESCRIPTIVE_FIELDS:
            rules[field] = _descriptive_rule(field, records_subset)
        for field in SIGNOFF_FIELDS:
            rules[field] = _signoff_rule(field, records_subset)
        return rules

    bucket_rules = {}
    for key, recs in buckets.items():
        if len(recs) < MIN_SUPPORT_FOR_BUCKET_RULE:
            continue  # too little evidence for a bucket-specific rule; falls back to category/global
        bucket_rules[f"{key[0]}::{key[1]}"] = {
            "category": key[0],
            "sub_category": key[1],
            "support": len(recs),
            "fields": build_rules(recs),
        }

    category_rules = {}
    for category, recs in by_category.items():
        if len(recs) < MIN_SUPPORT_FOR_BUCKET_RULE:
            continue
        category_rules[category] = {
            "category": category,
            "support": len(recs),
            "fields": build_rules(recs),
        }

    global_rules = {
        "support": len(normal),
        "fields": build_rules(normal),
    }

    dropped_buckets = sorted(
        {f"{k[0]}::{k[1]}" for k, v in buckets.items() if len(v) < MIN_SUPPORT_FOR_BUCKET_RULE}
    )

    return {
        "rule_table_version": RULE_TABLE_VERSION,
        "lifecycle_status": "CANDIDATE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_record_count": len(records),
        "normal_record_count": len(normal),
        "min_support_for_bucket_rule": MIN_SUPPORT_FOR_BUCKET_RULE,
        "low_support_buckets_folded_to_category_or_global": dropped_buckets,
        "buckets": bucket_rules,
        "categories": category_rules,
        "global": global_rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Historical CR export JSON (list of records).")
    parser.add_argument(
        "--out", default="config/field_requirements.generated.json", help="Output rule table path."
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8-sig") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise SystemExit("Expected the input JSON to be a list of CR records.")

    table = mine(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2, ensure_ascii=False)

    print(
        f"Mined {table['normal_record_count']} Normal CRs into "
        f"{len(table['buckets'])} category/sub-category rules "
        f"(+{len(table['categories'])} category-level fallback rules) -> {out_path}"
    )


if __name__ == "__main__":
    main()
