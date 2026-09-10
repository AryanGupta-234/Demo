from pre_cab.benchmark_manifest import build_manifest, canonical_json_sha256, dataset_fingerprint


def test_canonical_hash_is_order_stable():
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256({"a": 1, "b": 2})


def test_dataset_fingerprint_counts_only_normal_for_benchmark_identity():
    records = [
        {"Number": "CHG001", "Type": "Normal", "Short description": "x"},
        {"Number": "CHG002", "Type": "Emergency", "Short description": "y"},
    ]
    fingerprint = dataset_fingerprint(records)
    assert fingerprint["record_count"] == 2
    assert fingerprint["normal_record_count"] == 1
    assert len(fingerprint["normal_records_sha256"]) == 64


def test_manifest_contains_configuration_without_raw_records():
    records = [{"Number": "CHG001", "Type": "Normal", "Description": "private"}]
    manifest = build_manifest(records, seed=7, strictness="balanced", limit=50, provider="auto", llm_enabled=False)
    assert manifest["seed"] == 7
    assert manifest["dataset"]["normal_record_count"] == 1
    assert "private" not in str(manifest)
