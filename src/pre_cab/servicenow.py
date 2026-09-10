"""Provider-neutral ServiceNow adapter contracts.

No credentials or network calls live in the core. A production adapter can implement this protocol
using the organization's approved ServiceNow integration method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ServiceNowChange:
    number: str
    fields: dict[str, Any]
    attachment_refs: tuple[str, ...] = ()


class ServiceNowAdapter(Protocol):
    def get_change(self, number: str) -> ServiceNowChange:
        ...

    def list_attachments(self, number: str) -> list[dict[str, Any]]:
        ...

    def add_work_note(self, number: str, note: str) -> None:
        ...

    def update_change(self, number: str, fields: dict[str, Any]) -> None:
        ...
