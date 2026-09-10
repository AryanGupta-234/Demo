from pre_cab.requirement_metrics import requirement_metrics


def test_requirement_metrics_reports_one_vs_rest_scores() -> None:
    result = requirement_metrics(
        [
            {"requirement": "UAT", "actual": "required", "predicted": "required"},
            {"requirement": "UAT", "actual": "required", "predicted": "not_required"},
            {"requirement": "UAT", "actual": "not_required", "predicted": "not_required"},
            {"requirement": "UAT", "actual": "not_required", "predicted": "required"},
            {"requirement": "Customer Approval", "actual": "required", "predicted": "required"},
            {"requirement": "UAT", "actual": None, "predicted": "required"},
        ]
    )

    uat = result["by_requirement"]["UAT"]["labels"]
    assert result["total_scored"] == 5
    assert result["accuracy"] == 0.6
    assert uat["required"]["tp"] == 1
    assert uat["required"]["fp"] == 1
    assert uat["required"]["fn"] == 1
    assert uat["required"]["precision"] == 0.5
    assert uat["required"]["recall"] == 0.5
    assert uat["required"]["f1"] == 0.5
