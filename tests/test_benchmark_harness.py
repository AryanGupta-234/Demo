import json

from pre_cab.benchmark_harness import prepare_benchmark_artifacts


def test_prepare_artifacts_keeps_labels_out_of_model_input(tmp_path):
    source = tmp_path / "cr.json"
    source.write_text(json.dumps([
        {"Number": "CHG1", "Type": "Normal", "Short description": "x", "CAB Outcome": "Approved", "State": "Closed"},
        {"Number": "CHG2", "Type": "Emergency", "CAB Outcome": "Approved"},
    ]), encoding="utf-8")
    examples, manifest = prepare_benchmark_artifacts(source, tmp_path / "out")
    assert len(examples) == 1
    model_input = json.loads((tmp_path / "out" / "normal_inputs.json").read_text())
    assert "CAB Outcome" not in model_input[0]
    assert "State" not in model_input[0]
    assert manifest["dataset"]["normal_record_count"] == 1
    assert len(manifest["source_file_sha256"]) == 64
