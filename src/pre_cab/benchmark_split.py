"""Deterministic, label-stratified Normal-CR benchmark splitting."""
from __future__ import annotations

import random
from typing import Any, Iterable

from .cab_outcomes import normalize_cab_recommendation


def split_records(
    records: Iterable[dict[str, Any]],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split scorable Normal CRs by CAB label; unscorable records are excluded."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must satisfy 0 < train < 1, 0 <= validation < 1, train+validation < 1")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if str(row.get("Type") or "").strip().lower() != "normal":
            continue
        label = normalize_cab_recommendation(row.get("CAB Outcome") or row.get("CAB recommendation"))
        if label is None:
            continue
        buckets.setdefault(label.value, []).append(dict(row))

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for rows in buckets.values():
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * train_fraction)
        n_validation = int(n * validation_fraction)
        train.extend(rows[:n_train])
        validation.extend(rows[n_train:n_train + n_validation])
        test.extend(rows[n_train + n_validation:])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test
