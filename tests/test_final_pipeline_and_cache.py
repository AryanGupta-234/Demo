from __future__ import annotations

import json
from pathlib import Path

from pre_cab.audit_store import SQLiteAuditStore
from pre_cab.model_cache import CachingProvider
from pre_cab.models import ModelResponse
from pre_cab.pipeline import run_pre_cab
from pre_cab.schemas import Decision, Strictness
from pre_cab.evidence import EvidenceDocument


class CountingModel:
    model_name = "openai/gpt-oss-120b"

    def __init__(self, prediction: str = "PASS") -> None:
        self.calls = 0
        self.prediction = prediction

    def generate(self, **kwargs):
        self.calls += 1
        payload = {
            "prediction": self.prediction,
            "confidence": 0.87,
            "facts": ["evidence reviewed"],
            "inferences": [],
            "uncertainties": [],
            "contradictions": [],
            "technical_reasoning": "Evidence-aware technical assessment.",
            "cab_reasoning": "Evidence-aware CAB assessment.",
            "cab_questions": [],
            "recommendations": [],
            "self_critique": ["Checked contradictions."],
        }
        return ModelResponse(json.dumps(payload), self.model_name, {})


def _cr() -> dict:
    return {
        "Number": "CHG-DEMO-FINAL",
        "Type": "Normal",
        "Short description": "Customer transaction enhancement",
        "Description": "Update transaction behavior for customers",
        "Justification": "Functional fix",
        "Implementation plan": "Back up current component, deploy update, restart service",
        "Backout plan": "Restore backup and restart service",
        "Test plan": "UAT completed",
        "UAT signoff": "Yes",
        "Customer Approval": "Yes",
        "Conflict status": "No Conflict",
        "Configuration item": "Demo Production",
        "Risk": "Moderate",
    }


def _docs() -> list[EvidenceDocument]:
    return [
        EvidenceDocument(
            "uat",
            "CHG-DEMO-FINAL_UAT.txt",
            "CHG-DEMO-FINAL UAT test case expected result actual result PASS",
        ),
        EvidenceDocument(
            "approval",
            "CHG-DEMO-FINAL_approval.txt",
            "CHG-DEMO-FINAL customer approval: customer approved",
        ),
    ]


def test_public_pipeline_uses_one_final_model_call_after_evidence() -> None:
    model = CountingModel("PASS")
    result = run_pre_cab(_cr(), documents=_docs(), strictness=Strictness.BALANCED, model=model)
    assert model.calls == 1
    assert result.stage2 is not None
    assert result.stage2.verified["testing"] is True
    assert result.final_reasoning is not None
    assert result.final_reasoning.model_prediction == Decision.PASS


def test_final_model_cannot_upgrade_evidence_blocker() -> None:
    model = CountingModel("PASS")
    docs = [EvidenceDocument("dev", "CHG-DEMO-FINAL_test.txt", "CHG-DEMO-FINAL DEV testing completed expected result PASS")]
    result = run_pre_cab(_cr(), documents=docs, strictness=Strictness.BALANCED, model=model)
    assert result.stage2 is not None
    assert result.stage2.decision == Decision.NOT_READY
    assert result.final_decision == Decision.NOT_READY


def test_model_cache_avoids_second_provider_call(tmp_path: Path) -> None:
    model = CountingModel()
    cached = CachingProvider(model, tmp_path / "cache.sqlite3")
    first = cached.generate(system="s", user="u")
    second = cached.generate(system="s", user="u")
    assert model.calls == 1
    assert first.raw["pre_cab_cache"]["hit"] is False
    assert second.raw["pre_cab_cache"]["hit"] is True


def test_audit_store_round_trip(tmp_path: Path) -> None:
    store = SQLiteAuditStore(tmp_path / "audit.sqlite3")
    run_id = store.record(
        cr_number="CHG-DEMO-FINAL",
        strictness="balanced",
        stage1_decision="PASS",
        stage2_decision="PASS",
        final_decision="PASS",
        model="gpt-oss-120b",
        payload={"decision": "PASS"},
    )
    loaded = store.get(run_id)
    assert loaded is not None
    assert loaded.cr_number == "CHG-DEMO-FINAL"
    assert loaded.payload["decision"] == "PASS"
