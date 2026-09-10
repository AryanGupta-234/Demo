"""Small immutable audit-event model for traceable Pre-CAB decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    actor: str
    cr_number: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(self, "timestamp", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
