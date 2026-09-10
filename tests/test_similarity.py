from pre_cab.similarity import clone_candidates, compare_for_clone


def test_similar_change_is_a_clone_candidate():
    current = {
        "Number": "CHG-NEW",
        "Type": "Normal",
        "Category": "Debit Card",
        "Sub Category": "ATM Transactions",
        "Configuration item": "ATM-PROD",
        "Short description": "ATM transaction enhancement",
        "Description": "Fix pending fee recovery in ATM transactions.",
        "Implementation plan": "Backup artifact, deploy artifact, restart service.",
        "Backout plan": "Restore backed-up artifact.",
    }
    historical = dict(current)
    historical.update({"Number": "CHG-OLD", "CAB recommendation": "Approved"})

    match = compare_for_clone(current, historical)
    assert match.similarity > 0.9
    assert match.cr_number == "CHG-OLD"
    assert match.outcome == "Approved"


def test_clone_candidates_are_ranked():
    current = {
        "Type": "Normal",
        "Category": "Infrastructure",
        "Sub Category": "Infra Patching",
        "Configuration item": "MGMT-PROD",
        "Short description": "Monthly OS security patching",
        "Description": "Apply operating system security updates.",
        "Implementation plan": "Install patches and reboot.",
        "Backout plan": "Restore previous image from backup.",
    }
    candidates = clone_candidates(
        current,
        [
            {**current, "Number": "CHG-B", "CAB recommendation": "Conditional"},
            {**current, "Number": "CHG-A", "CAB recommendation": "Approved"},
        ],
    )
    assert candidates[0].cr_number in {"CHG-A", "CHG-B"}
    assert candidates[0].similarity >= candidates[-1].similarity
