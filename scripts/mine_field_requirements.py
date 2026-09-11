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
import re
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
    "Risk and impact analysis",
    "Environment",
    "Change Class",
]

# Free-text history/reasoning fields. Not scored by fill-rate (they're near-100%
# filled with trivial "CR created" noise) -- mined separately for *content* by
# mine_work_note_signals below.
NOTE_FIELDS = ["Comments and Work notes", "Work notes"]

# Outcome fields used only to label historical CRs for the note-signal miner.
# They are never themselves treated as intake requirements: a CR author doesn't
# "fill in" its own Approval/CAB recommendation, CAB does.
OUTCOME_FIELD_APPROVAL = "Approval"
OUTCOME_FIELD_CAB_RECOMMENDATION = "CAB recommendation"

_NEGATIVE_APPROVAL_VALUES = {"rejected"}

# ServiceNow work-note entries are logged as "DD-MM-YYYY HH:MM:SS - Author Name
# (Work notes)\n<body>". Strip the header so mining sees content, not authors/timestamps.
_NOTE_HEADER_RE = re.compile(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} - ([^()]+?)\s*\([^)]*\)\s*")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")
_STOPWORDS = frozenset(
    "the a an is are was were to of and or for in on at by with this that from "
    "as it its into be been being will would can could should not no yes".split()
)
NOTE_SIGNAL_MIN_NEGATIVE_SUPPORT = 4
NOTE_SIGNAL_MIN_LIFT = 3.0
NOTE_SIGNAL_TOP_N = 30

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


def _note_text(record: dict[str, Any]) -> str:
    for field in NOTE_FIELDS:
        value = record.get(field)
        if not _blank(value):
            return str(value)
    return ""


def _outcome_label(record: dict[str, Any]) -> str | None:
    """Historical label used only to mine work-note language, never to grade a live CR."""
    approval = _norm(record.get(OUTCOME_FIELD_APPROVAL))
    cab_recommendation = _norm(record.get(OUTCOME_FIELD_CAB_RECOMMENDATION))
    if approval in _NEGATIVE_APPROVAL_VALUES or "cancel" in cab_recommendation:
        return "negative"
    if approval == "approved":
        return "positive"
    return None


def _extract_author_tokens(note_text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _NOTE_HEADER_RE.finditer(note_text):
        tokens.update(w.lower() for w in _WORD_RE.findall(match.group(1)) if len(w) > 2)
    return tokens


def _strip_headers(note_text: str) -> str:
    return _NOTE_HEADER_RE.sub(" ", note_text)


def _ngrams(text: str, n: int, author_tokens: set[str]) -> set[str]:
    words = [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS]
    grams = {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    # Drop grams that are entirely author-name tokens (routine approver/assignee
    # names cluster around cancellations for organizational reasons, not content
    # reasons -- keep phrases that mix a name with real content, drop pure names).
    return {g for g in grams if not all(tok in author_tokens for tok in g.split())}


def mine_work_note_signals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Learn which work-note language historically preceded a rejected/cancelled CR.

    This is a frequency-lift heuristic over a genuinely small historical sample
    (a few hundred negative-labeled CRs at most) -- treat every phrase here as a
    CANDIDATE worth a human glance, not a validated policy. It is deliberately
    explainable (raw counts, not a black-box score) so that glance is easy.
    """
    labeled = [(r, _outcome_label(r)) for r in records]
    negative = [r for r, label in labeled if label == "negative"]
    positive = [r for r, label in labeled if label == "positive"]
    if not negative:
        return []

    author_tokens: set[str] = set()
    for record in records:
        author_tokens |= _extract_author_tokens(_note_text(record))

    def phrase_counts(subset: list[dict[str, Any]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for record in subset:
            body = _strip_headers(_note_text(record))
            for n in (1, 2, 3):
                counts.update(_ngrams(body, n, author_tokens))
        return counts

    negative_counts = phrase_counts(negative)
    positive_counts = phrase_counts(positive)

    candidates = []
    for phrase, neg_support in negative_counts.items():
        if neg_support < NOTE_SIGNAL_MIN_NEGATIVE_SUPPORT:
            continue
        pos_support = positive_counts.get(phrase, 0)
        neg_rate = neg_support / len(negative)
        pos_rate = (pos_support + 1) / (len(positive) + 1)  # Laplace-smoothed
        lift = neg_rate / pos_rate
        if lift < NOTE_SIGNAL_MIN_LIFT:
            continue
        candidates.append(
            {
                "phrase": phrase,
                "lift": round(lift, 2),
                "negative_support": neg_support,
                "positive_support": pos_support,
                "confidence": _confidence_from_support(neg_support),
                "evidence": (
                    f"Seen in {neg_support}/{len(negative)} historical CRs later rejected/cancelled, "
                    f"vs {pos_support}/{len(positive)} approved CRs."
                ),
            }
        )
    candidates.sort(key=lambda c: (-c["lift"], -c["negative_support"]))
    return candidates[:NOTE_SIGNAL_TOP_N]


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
        "work_note_signals": mine_work_note_signals(normal),
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
        f"(+{len(table['categories'])} category-level fallback rules) "
        f"and {len(table['work_note_signals'])} work-note risk-language signals -> {out_path}"
    )


if __name__ == "__main__":
    main()
