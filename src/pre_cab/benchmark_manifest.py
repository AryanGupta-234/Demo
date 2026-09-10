"""Reproducibility metadata for leakage-safe benchmark runs."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Iterable


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_fingerprint(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    normal = [row for row in rows if str(row.get("Type") or "").strip().lower() == "normal"]
    ids = [str(row.get("Number") or row.get("Effective number") or "") for row in normal]
    return {
        "record_count": len(rows),
        "normal_record_count": len(normal),
        "normal_ids_sha256": canonical_json_sha256(sorted(ids)),
        "normal_records_sha256": canonical_json_sha256(normal),
    }


def build_manifest(
    records: Iterable[dict[str, Any]],
    *,
    seed: int,
    strictness: str,
    limit: int,
    provider: str,
    llm_enabled: bool,
    mode: str = "benchmark",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "strictness": strictness,
        "limit": limit,
        "provider": provider,
        "llm_enabled": llm_enabled,
        "dataset": dataset_fingerprint(records),
    }
