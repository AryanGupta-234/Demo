from pathlib import Path

from pre_cab.embeddings import cosine_similarity
from pre_cab.memory import MemoryKind, MemoryRecord
from pre_cab.semantic_memory import SemanticSQLiteUnifiedMemory


class FakeEmbeddingProvider:
    model_name = "test-local"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            low = text.lower()
            vectors.append([
                1.0 if "rollback" in low else 0.0,
                1.0 if "database" in low else 0.0,
                1.0 if "customer" in low else 0.0,
            ])
        return vectors


def test_persistent_semantic_recall_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    memory = SemanticSQLiteUnifiedMemory(path, embedding_provider=FakeEmbeddingProvider())
    memory.remember(MemoryRecord(
        "episode-1", MemoryKind.EPISODE,
        "Database rollback validation was completed for a customer-facing change.",
        {"confidence": 0.9},
    ))

    reopened = SemanticSQLiteUnifiedMemory(path, embedding_provider=FakeEmbeddingProvider())
    results = reopened.search("customer database recovery", kinds=[MemoryKind.EPISODE])
    assert results
    assert results[0].memory_id == "episode-1"
    assert results[0].score is not None


def test_semantic_vectors_are_persistent(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    memory = SemanticSQLiteUnifiedMemory(path, embedding_provider=FakeEmbeddingProvider())
    memory.remember(MemoryRecord("r1", MemoryKind.FACT, "rollback database", {}))
    memory.remember(MemoryRecord("r2", MemoryKind.FACT, "customer", {}))
    reopened = SemanticSQLiteUnifiedMemory(path, embedding_provider=FakeEmbeddingProvider())
    assert reopened.search("rollback database", kinds=[MemoryKind.FACT], limit=1)[0].memory_id == "r1"
    assert cosine_similarity([1.0, 1.0], [1.0, 1.0]) == 2.0
