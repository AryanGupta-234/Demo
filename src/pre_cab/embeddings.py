"""Optional local embedding provider for semantic retrieval.

The default implementation uses sentence-transformers when installed. The rest of the system only
sees this small protocol, so a hosted embedding service can be substituted later.
"""
from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class LocalSentenceTransformer:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the 'embeddings' extra to enable local semantic retrieval") from exc
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
