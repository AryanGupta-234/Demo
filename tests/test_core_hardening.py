from __future__ import annotations

import json

import pytest

from pre_cab.benchmark_metrics import compute_safety_metrics
from pre_cab.budget import BudgetGuard, InferenceBudget
from pre_cab.budgeted_provider import BudgetedProvider
from pre_cab.memory import InMemoryUnifiedMemory, MemoryKind, MemoryRecord
from pre_cab.models import ModelResponse
from pre_cab.sanitization import sanitize_cr_for_reasoning
from pre_cab.schemas import Decision


class StaticProvider:
    model_name = "static"

    def generate(self, **kwargs):
        return ModelResponse("{}", "static", {"usage": {"prompt_tokens": 2, "completion_tokens": 1}})


class TinyEmbedding:
    model_name = "tiny"

    def embed(self, texts):
        result = []
        for text in texts:
            low = text.lower()
            result.append([1.0 if "wallet" in low else 0.0, 1.0 if "patch" in low else 0.0])
        return result


def test_pass_precision_and_recall_use_true_passes() -> None:
    rows = [
        (Decision.PASS, Decision.PASS),
        (Decision.PASS, Decision.CONDITIONAL),
        (Decision.CONDITIONAL, Decision.PASS),
        (Decision.NOT_READY, Decision.NOT_READY),
    ]
    metrics = compute_safety_metrics(rows)
    assert metrics.pass_precision == 0.5
    assert metrics.pass_recall == 0.5
    assert metrics.false_passes == 1


def test_sanitization_redacts_secrets_but_keeps_technical_context() -> None:
    cr = {
        "Type": "Normal",
        "Implementation plan": "deploy wallet.jar to server-01",
        "api_key": "gsk_supersecret123456789",
        "Notes": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
    }
    clean = sanitize_cr_for_reasoning(cr)
    assert clean["Implementation plan"] == "deploy wallet.jar to server-01"
    assert clean["api_key"] == "[REDACTED_SECRET]"
    assert "abcdefghijklmnopqrstuvwxyz" not in clean["Notes"]


def test_budget_wrapper_denies_oversized_call() -> None:
    guard = BudgetGuard(InferenceBudget(tokens_per_minute=10, max_input_tokens_per_call=10, reserved_output_tokens=5))
    provider = BudgetedProvider(StaticProvider(), guard)
    with pytest.raises(RuntimeError):
        provider.generate(system="x" * 40, user="y" * 40)


def test_memory_uses_semantic_signal_when_available() -> None:
    memory = InMemoryUnifiedMemory(embedding_provider=TinyEmbedding())
    memory.remember(MemoryRecord("wallet", MemoryKind.CAB_HISTORY, "application issue"))
    memory.remember(MemoryRecord("patch", MemoryKind.CAB_HISTORY, "infrastructure change"))
    # Vectors are generated from record text, so put semantic term into metadata/text for the test.
    memory = InMemoryUnifiedMemory(embedding_provider=TinyEmbedding())
    memory.remember(MemoryRecord("wallet", MemoryKind.CAB_HISTORY, "wallet application issue"))
    memory.remember(MemoryRecord("patch", MemoryKind.CAB_HISTORY, "server patch change"))
    results = memory.search("wallet failure", limit=1)
    assert results[0].memory_id == "wallet"
