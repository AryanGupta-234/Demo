from pre_cab.benchmark_leakage import POST_DECISION_FIELDS, strip_post_decision_fields
from pre_cab.benchmark_prepare import prepare_normal_benchmark
from pre_cab.benchmark_split import split_records


def test_post_decision_fields_are_removed():
    row = {
        "Number": "CHG-1",
        "Type": "Normal",
        "Description": "example",
        "CAB recommendation": "Approved",
        "State": "Closed",
        "Close code": "Successful",
        "Implementation plan": "deploy",
    }
    clean = strip_post_decision_fields(row)
    assert "CAB recommendation" not in clean
    assert "State" not in clean
    assert "Implementation plan" in clean
    assert POST_DECISION_FIELDS


def test_prepare_normal_benchmark_strips_answers():
    rows = [
        {"Number": "CHG-1", "Type": "Normal", "Description": "A", "CAB recommendation": "Approved"},
        {"Number": "CHG-2", "Type": "Emergency", "Description": "B", "CAB recommendation": "Approved"},
    ]
    examples = prepare_normal_benchmark(rows)
    assert len(examples) == 1
    assert examples[0].actual is not None
    assert "CAB recommendation" not in examples[0].input_record


def test_split_is_deterministic_and_disjoint():
    rows = []
    for i in range(20):
        outcome = "Approved" if i % 2 else "Cancellation requested"
        rows.append({"Number": f"CHG-{i}", "Type": "Normal", "Description": str(i), "CAB recommendation": outcome})
    a = split_records(rows, seed=11)
    b = split_records(rows, seed=11)
    assert [[r["Number"] for r in part] for part in a] == [[r["Number"] for r in part] for part in b]
    ids = [set(r["Number"] for r in part) for part in a]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
