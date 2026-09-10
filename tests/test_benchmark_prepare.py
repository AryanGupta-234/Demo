from pre_cab.benchmark_prepare import normalize_outcome, prepare_normal_benchmark
from pre_cab.schemas import Decision


def test_normal_benchmark_hides_outcome_fields():
    records = [{
        "Number": "CHG-001",
        "Type": "Normal",
        "Short description": "Example",
        "CAB recommendation": "Conditional approval",
        "CAB Outcome": "Conditional",
    }]
    examples = prepare_normal_benchmark(records)
    assert len(examples) == 1
    assert examples[0].actual == Decision.CONDITIONAL
    assert "CAB Outcome" not in examples[0].input_record
    assert "CAB recommendation" not in examples[0].input_record


def test_emergency_is_excluded():
    records = [{"Number": "CHG-002", "Type": "Emergency", "CAB Outcome": "Approved"}]
    assert prepare_normal_benchmark(records) == []


def test_ambiguous_outcome_is_not_guessed():
    assert normalize_outcome({"CAB Outcome": "Needs discussion"}) is None
