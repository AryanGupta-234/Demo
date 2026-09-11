from pre_cab.field_requirement_engine import FieldRequirementEngine, RequirementLevel
from pre_cab.schemas import FindingSeverity


def _table():
    return {
        "rule_table_version": 1,
        "lifecycle_status": "CANDIDATE",
        "buckets": {
            "Core::branchchannel": {
                "category": "Core",
                "sub_category": "branchchannel",
                "support": 654,
                "fields": {
                    "Backout plan": {
                        "kind": "descriptive",
                        "requirement_level": "REQUIRED",
                        "support": 654,
                        "fill_rate": 0.997,
                        "confidence": 0.97,
                        "evidence": "Populated in 652/654 historical Normal CRs in this bucket.",
                    },
                    "Customer Approval": {
                        "kind": "signoff",
                        "requirement_level": "REQUIRED",
                        "support": 654,
                        "decided_rate": 0.456,
                        "applicability_rate": 0.829,
                        "compliance_rate": 0.858,
                        "confidence": 0.97,
                        "evidence": "212 Yes / 35 No / 51 Not Applicable.",
                    },
                    "Lower Environment Reference CR/SR": {
                        "kind": "signoff",
                        "requirement_level": "OPTIONAL",
                        "support": 654,
                        "decided_rate": 0.456,
                        "applicability_rate": 0.252,
                        "compliance_rate": 0.147,
                        "confidence": 0.97,
                        "evidence": "11 Yes / 64 No / 223 Not Applicable.",
                    },
                },
            }
        },
        "categories": {
            "Core": {
                "category": "Core",
                "support": 654,
                "fields": {
                    "Backout plan": {
                        "kind": "descriptive",
                        "requirement_level": "RECOMMENDED",
                        "support": 654,
                        "fill_rate": 0.7,
                        "confidence": 0.9,
                        "evidence": "category fallback",
                    }
                },
            }
        },
        "global": {
            "support": 2013,
            "fields": {
                "Backout plan": {
                    "kind": "descriptive",
                    "requirement_level": "RECOMMENDED",
                    "support": 2013,
                    "fill_rate": 0.6,
                    "confidence": 0.9,
                    "evidence": "global fallback",
                }
            },
        },
    }


def test_exact_bucket_match_flags_missing_required_field_as_blocking():
    engine = FieldRequirementEngine(table=_table())
    cr = {"Category": "Core", "Sub Category": "branchchannel", "Backout plan": ""}
    report = engine.evaluate(cr)
    backout = next(f for f in report.findings if f.field == "Backout plan")
    assert backout.requirement_level == RequirementLevel.REQUIRED
    assert not backout.satisfied
    assert backout.severity == FindingSeverity.BLOCKING
    assert report.resolved_scope == "bucket:Core::branchchannel"


def test_satisfied_required_field_is_info_not_blocking():
    engine = FieldRequirementEngine(table=_table())
    cr = {"Category": "Core", "Sub Category": "branchchannel", "Backout plan": "Restore from snapshot."}
    report = engine.evaluate(cr)
    backout = next(f for f in report.findings if f.field == "Backout plan")
    assert backout.satisfied
    assert backout.severity == FindingSeverity.INFO


def test_not_applicable_does_not_satisfy_a_required_signoff_field():
    engine = FieldRequirementEngine(table=_table())
    cr = {
        "Category": "Core",
        "Sub Category": "branchchannel",
        "Customer Approval": "Not Applicable",
    }
    report = engine.evaluate(cr)
    approval = next(f for f in report.findings if f.field == "Customer Approval")
    assert approval.requirement_level == RequirementLevel.REQUIRED
    assert not approval.satisfied


def test_not_applicable_satisfies_an_optional_signoff_field():
    engine = FieldRequirementEngine(table=_table())
    cr = {
        "Category": "Core",
        "Sub Category": "branchchannel",
        "Lower Environment Reference CR/SR": "Not Applicable",
    }
    report = engine.evaluate(cr)
    field = next(f for f in report.findings if f.field == "Lower Environment Reference CR/SR")
    assert field.satisfied


def test_unseen_bucket_falls_back_to_category_then_global():
    engine = FieldRequirementEngine(table=_table())
    cr = {"Category": "Core", "Sub Category": "SomethingBrandNew", "Backout plan": ""}
    report = engine.evaluate(cr)
    assert report.resolved_scope == "category:Core"

    cr2 = {"Category": "TotallyUnknownCategory", "Sub Category": "X", "Backout plan": ""}
    report2 = engine.evaluate(cr2)
    assert report2.resolved_scope == "global"


def test_missing_rule_table_degrades_to_empty_global_without_crashing(tmp_path):
    engine = FieldRequirementEngine(table_path=tmp_path / "does_not_exist.json")
    report = engine.evaluate({"Category": "Any", "Sub Category": "Thing"})
    assert report.findings == []
    assert engine.lifecycle_status == "MISSING"


def _table_with_note_signal():
    table = _table()
    table["work_note_signals"] = [
        {
            "phrase": "wrong category",
            "lift": 39.7,
            "negative_support": 7,
            "positive_support": 0,
            "confidence": 0.9,
            "evidence": "Seen in 7/285 rejected/cancelled CRs, vs 0/1615 approved CRs.",
        }
    ]
    return table


def test_emergency_cr_is_out_of_scope_and_untouched():
    engine = FieldRequirementEngine(table=_table_with_note_signal())
    cr = {
        "Type": "Emergency",
        "Category": "Core",
        "Sub Category": "branchchannel",
        "Backout plan": "",
        "Comments and Work notes": "raised wrong category, please cancel",
    }
    report = engine.evaluate(cr)
    assert not report.in_scope
    assert report.resolved_scope == "out_of_scope:emergency"
    assert report.findings == []
    assert report.note_signals == []


def test_work_note_phrase_match_is_surfaced_with_evidence():
    engine = FieldRequirementEngine(table=_table_with_note_signal())
    cr = {
        "Type": "Normal",
        "Category": "Core",
        "Sub Category": "branchchannel",
        "Backout plan": "Restore from snapshot.",
        "Comments and Work notes": (
            "12-08-2026 12:05:09 - Jane Doe (Work notes)\nRaised under wrong category, please cancel.\n"
        ),
    }
    report = engine.evaluate(cr)
    assert len(report.note_signals) == 1
    assert report.note_signals[0].phrase == "wrong category"
    assert report.note_signals[0].severity == FindingSeverity.WARNING


def test_no_note_text_yields_no_signals():
    engine = FieldRequirementEngine(table=_table_with_note_signal())
    cr = {"Type": "Normal", "Category": "Core", "Sub Category": "branchchannel"}
    report = engine.evaluate(cr)
    assert report.note_signals == []
