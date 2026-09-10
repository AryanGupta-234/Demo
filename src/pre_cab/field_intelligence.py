from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .cab_outcomes import normalize_cab_recommendation


@dataclass(frozen=True)
class FieldProfile:
    field: str
    rows: int
    populated: int
    population_rate: float
    distinct_values: int
    common_values: tuple[tuple[str, int], ...]
    outcome_by_common_value: tuple[tuple[str, dict[str, int]], ...] = ()


def _present(value: Any) -> bool:
    return value not in (None, "", [], {}) and str(value).strip().lower() not in {"nan", "none"}


def profile_fields(records: Iterable[dict[str, Any]], top_values: int = 5) -> list[FieldProfile]:
    rows_data = list(records)
    fields = sorted({key for row in rows_data for key in row})
    result: list[FieldProfile] = []
    rows = len(rows_data)
    for field in fields:
        values = [str(row.get(field)).strip() for row in rows_data if _present(row.get(field))]
        counts = Counter(values)
        common = counts.most_common(top_values)
        outcome_maps: list[tuple[str, dict[str, int]]] = []
        for value, _count in common:
            outcomes: Counter[str] = Counter()
            for row in rows_data:
                if str(row.get(field)).strip() != value:
                    continue
                label = normalize_cab_recommendation(row.get("CAB Outcome") or row.get("CAB recommendation"))
                outcomes[label.value if label else "UNSCORABLE"] += 1
            outcome_maps.append((value, dict(outcomes)))
        result.append(
            FieldProfile(
                field=field,
                rows=rows,
                populated=len(values),
                population_rate=len(values) / rows if rows else 0.0,
                distinct_values=len(counts),
                common_values=tuple(common),
                outcome_by_common_value=tuple(outcome_maps),
            )
        )
    return result


def contextual_population(
    records: Iterable[dict[str, Any]],
    context_fields: tuple[str, ...] = ("Type", "Category", "Sub Category"),
) -> dict[tuple[str, ...], list[FieldProfile]]:
    groups: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = tuple(str(row.get(field) or "").strip() for field in context_fields)
        groups[key].append(row)
    return {key: profile_fields(rows) for key, rows in groups.items()}
