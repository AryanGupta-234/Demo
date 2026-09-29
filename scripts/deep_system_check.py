"""Deep Pre-CAB system audit for manager demos and regression checks.

This is additive diagnostics: it does not replace the production pipeline.
It checks repository structure, imports, deterministic decision monotonicity,
CR-scoped evidence isolation, bounded parallel processing, and the generative
reasoning contract. Ollama is optional and can be exercised with --ollama.
"""
from __future__ import annotations

import argparse
import compileall
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pre_cab.local_attachments import inventory_workspace
from pre_cab.parallel import process_crs_parallel
from pre_cab.pipeline import _merge_decisions, run_pre_cab
from pre_cab.schemas import Decision, Strictness


REQUIRED_PATHS = (
    "src/pre_cab/pipeline.py",
    "src/pre_cab/evidence.py",
    "src/pre_cab/local_attachments.py",
    "src/pre_cab/final_reasoning.py",
    "src/pre_cab/brain_loop.py",
    "src/pre_cab/narrative.py",
    "src/pre_cab/semantic_memory.py",
    "src/pre_cab/parallel.py",
    "src/pre_cab/servicenow_readonly.py",
    "scripts/demo.py",
    "scripts/diagnose.py",
    "scripts/run_batch.py",
    "tests",
    "config",
)


def _cr(number: str) -> dict[str, Any]:
    return {
        "Number": number,
        "Type": "Normal",
        "Short description": f"Controlled application change for {number}",
        "Description": "A bounded application change with documented implementation and recovery.",
        "Justification": "Operational maintenance.",
        "Implementation plan": "Back up the current artifact, deploy the approved change, run smoke validation.",
        "Backout plan": "Restore the previous artifact and verify service health.",
        "Test plan": "Run smoke tests and record execution results.",
        "Customer Approval": "Yes",
        "UAT signoff": "Not Applicable",
        "Conflict status": "No Conflict",
        "Configuration item": "Demo Application",
        "Risk": "Moderate",
    }


class _FakeModel:
    model_name = "deep-audit-fake-model"

    def generate(self, **_: Any):
        from pre_cab.models import ModelResponse

        payload = {
            "prediction": "PASS",
            "confidence": 0.91,
            "facts": ["Implementation and recovery are documented."],
            "inferences": ["The supplied record is structurally complete for this synthetic case."],
            "uncertainties": [],
            "contradictions": [],
            "technical_reasoning": "The synthetic CR contains implementation, rollback and validation context.",
            "cab_reasoning": "The synthetic change has the expected evidence structure for a controlled CAB review.",
            "cab_questions": [],
            "recommendations": ["Confirm the live evidence before CAB."],
            "self_critique": ["Checked for unsupported claims and contradictions."],
        }
        return ModelResponse(json.dumps(payload), self.model_name, {})


def check_structure(root: Path) -> list[str]:
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    return [f"missing: {path}" for path in missing]


def check_imports() -> list[str]:
    modules = (
        "pre_cab.pipeline",
        "pre_cab.evidence",
        "pre_cab.local_attachments",
        "pre_cab.final_reasoning",
        "pre_cab.brain_loop",
        "pre_cab.narrative",
        "pre_cab.semantic_memory",
        "pre_cab.parallel",
        "pre_cab.servicenow_readonly",
    )
    failures = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def check_algorithms() -> list[str]:
    failures: list[str] = []

    matrix = (
        (Decision.PASS, Decision.PASS, Decision.PASS),
        (Decision.CONDITIONAL, Decision.PASS, Decision.CONDITIONAL),
        (Decision.PASS, Decision.CONDITIONAL, Decision.CONDITIONAL),
        (Decision.PASS, Decision.NOT_READY, Decision.NOT_READY),
        (Decision.NOT_READY, Decision.PASS, Decision.NOT_READY),
    )
    for left, right, expected in matrix:
        actual = _merge_decisions(left, right)
        if actual != expected:
            failures.append(f"decision gate: {left}/{right} -> {actual}, expected {expected}")

    with tempfile.TemporaryDirectory(prefix="pre_cab_audit_") as raw:
        root = Path(raw) / "evidence"
        for number in ("CHG-AUDIT-001", "CHG-AUDIT-002"):
            folder = root / number
            folder.mkdir(parents=True)
            (folder / f"{number}_test.txt").write_text(
                f"{number} test execution result PASS",
                encoding="utf-8",
            )

        results = process_crs_parallel(
            [_cr("CHG-AUDIT-001"), _cr("CHG-AUDIT-002")],
            attachment_root=str(root),
            max_workers=2,
        )
        if [item.cr_number for item in results] != ["CHG-AUDIT-001", "CHG-AUDIT-002"]:
            failures.append("parallel processor did not preserve input order")
        for item in results:
            if item.error or item.result is None:
                failures.append(f"parallel processor failed for {item.cr_number}: {item.error}")
                continue
            refs = [doc.ref for doc in item.result.documents]
            if not refs or any(item.cr_number not in ref for ref in refs):
                failures.append(f"evidence leakage/isolation failure for {item.cr_number}")

        inventory = inventory_workspace(root, "CHG-AUDIT-001")
        if not inventory:
            failures.append("workspace inventory returned no files for a populated CR workspace")

        model_result = run_pre_cab(
            _cr("CHG-AUDIT-GEN"),
            attachment_root=root,
            model=_FakeModel(),
            strictness=Strictness.BALANCED,
        )
        if model_result.final_reasoning is None or model_result.final_reasoning.payload is None:
            failures.append("generative reasoning contract did not produce a valid structured payload")
        if model_result.final_decision not in set(Decision):
            failures.append("final decision is outside the Decision enum")

    return failures


def check_ollama() -> list[str]:
    try:
        from pre_cab.runtime_provider import build_runtime_provider

        model = build_runtime_provider("ollama")
        return [f"Ollama provider available: {getattr(model, 'model_name', type(model).__name__)}"]
    except Exception as exc:
        return [f"Ollama check skipped/unavailable: {type(exc).__name__}: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep structural and algorithmic Pre-CAB audit")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    parser.add_argument("--ollama", action="store_true", help="Also initialize the configured Ollama provider")
    args = parser.parse_args()

    root = args.root.resolve()
    structure = check_structure(root)
    compile_ok = compileall.compile_dir(str(root / "src"), quiet=1) and compileall.compile_dir(str(root / "scripts"), quiet=1)
    import_failures = check_imports()
    algorithm_failures = check_algorithms()
    ollama = check_ollama() if args.ollama else []

    checks = {
        "structure": "PASS" if not structure else "FAIL",
        "compile": "PASS" if compile_ok else "FAIL",
        "imports": "PASS" if not import_failures else "FAIL",
        "deterministic_and_pipeline_algorithms": "PASS" if not algorithm_failures else "FAIL",
        "generative_contract": "PASS" if not any("generative reasoning contract" in item for item in algorithm_failures) else "FAIL",
    }
    failures = structure + import_failures + algorithm_failures
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "ollama": ollama,
        "explanation": {
            "architecture": "Deterministic validation remains authoritative; evidence verification is CR-scoped; neural reasoning adds conservative context; NLG is presentation-only.",
            "evidence_isolation": "Each CR is resolved to its own workspace before extraction and parallel workers preserve CR identity.",
            "generative_ai": "The model must return structured facts/inferences/uncertainties/contradictions and cannot upgrade deterministic or evidence blockers.",
            "manager_summary": "The audit verifies that the system can explain not only the final decision, but also the evidence path, unresolved gaps, and technical basis.",
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("PRE-CAB DEEP SYSTEM AUDIT")
        print("=" * 72)
        for name, status in checks.items():
            print(f"{status:>5}  {name}")
        for line in failures:
            print(f"FAIL  {line}")
        for line in ollama:
            print(f"INFO  {line}")
        print()
        print("Manager explanation:")
        print("  Deterministic gates establish the authoritative readiness boundary.")
        print("  Evidence is discovered, inventoried and verified per CR workspace.")
        print("  Parallel workers process independent CRs without sharing evidence.")
        print("  Generative reasoning explains facts, inferences, uncertainty and contradictions.")
        print("  NLG improves CAB/technical wording without changing the locked decision.")
        print()
        print(f"OVERALL: {'PASS' if not failures else 'FAIL'}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
