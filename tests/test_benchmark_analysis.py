from pre_cab.benchmark_analysis import failure_buckets
from pre_cab.benchmark_runner import BenchmarkPrediction
from pre_cab.schemas import Decision, Strictness


def test_failure_buckets_separate_safety_errors():
    rows = [
        BenchmarkPrediction("1", Decision.NOT_READY, Decision.PASS, .8, Strictness.BALANCED, ("MISSING",)),
        BenchmarkPrediction("2", Decision.PASS, Decision.NOT_READY, .7, Strictness.BALANCED, ("RISK",)),
    ]
    result = failure_buckets(rows)
    assert result["false_pass_count"] == 1
    assert result["false_fail_count"] == 1
    assert result["false_pass_finding_codes"] == {"RISK": 1}
