"""Read-only ServiceNow REST client for controlled ingestion."""
from __future__ import annotations
import base64, json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

@dataclass(frozen=True)
class ServiceNowCredentials:
    base_url: str
    username: str
    password: str

class ReadOnlyServiceNowClient:
    def __init__(self, credentials: ServiceNowCredentials, table: str = "change_request") -> None:
        self.credentials = credentials
        self.table = table

    def _get(self, path: str) -> dict[str, Any]:
        request = Request(self.credentials.base_url.rstrip("/") + "/" + path.lstrip("/"), method="GET")
        token = base64.b64encode(f"{self.credentials.username}:{self.credentials.password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
        request.add_header("Accept", "application/json")
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if isinstance(data, dict) else {"result": data}

    def get_change(self, number: str) -> dict[str, Any]:
        encoded = quote(f"number={number}", safe="")
        data = self._get(f"api/now/table/{self.table}?sysparm_query={encoded}&sysparm_limit=1")
        rows = data.get("result") or []
        if not rows:
            raise KeyError(f"Change request not found: {number}")
        return dict(rows[0])

    def list_attachments(self, sys_id: str) -> list[dict[str, Any]]:
        encoded = quote(f"table_sys_id={sys_id}", safe="")
        data = self._get(f"api/now/attachment?sysparm_query={encoded}")
        return [dict(row) for row in data.get("result") or []]
