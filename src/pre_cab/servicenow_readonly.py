"""Read-only ServiceNow REST client for controlled ingestion.

The client exposes GET-only Change Request and attachment operations. Credentials are never included in
returned metadata or logs. Binary attachments can be downloaded in-memory and converted into
EvidenceDocument records without granting any write capability.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from .evidence import EvidenceDocument


@dataclass(frozen=True)
class ServiceNowCredentials:
    base_url: str
    username: str
    password: str


class ReadOnlyServiceNowClient:
    def __init__(self, credentials: ServiceNowCredentials, table: str = "change_request") -> None:
        self.credentials = credentials
        self.table = table

    def _request(self, path: str, *, accept: str = "application/json") -> Request:
        request = Request(
            self.credentials.base_url.rstrip("/") + "/" + path.lstrip("/"),
            method="GET",
        )
        token = base64.b64encode(
            f"{self.credentials.username}:{self.credentials.password}".encode()
        ).decode()
        request.add_header("Authorization", f"Basic {token}")
        request.add_header("Accept", accept)
        return request

    def _get(self, path: str) -> dict[str, Any]:
        with urlopen(self._request(path), timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data if isinstance(data, dict) else {"result": data}

    def _get_bytes(self, path: str) -> bytes:
        with urlopen(self._request(path, accept="*/*"), timeout=60) as response:
            return response.read()

    def get_change(self, number: str) -> dict[str, Any]:
        encoded = quote(f"number={number}", safe="")
        data = self._get(
            f"api/now/table/{self.table}?sysparm_query={encoded}&sysparm_limit=1&sysparm_display_value=true"
        )
        rows = data.get("result") or []
        if not rows:
            raise KeyError(f"Change request not found: {number}")
        return dict(rows[0])

    def list_attachments(self, sys_id: str) -> list[dict[str, Any]]:
        encoded = quote(f"table_sys_id={sys_id}", safe="")
        data = self._get(f"api/now/attachment?sysparm_query={encoded}")
        return [dict(row) for row in data.get("result") or []]

    def download_attachment(self, attachment_sys_id: str) -> bytes:
        return self._get_bytes(f"api/now/attachment/{quote(attachment_sys_id, safe='')}/file")

    @staticmethod
    def _extract_attachment_text(name: str, content: bytes) -> str:
        suffix = Path(name).suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError("Install the 'docs' extra to parse PDF attachments") from exc
            reader = PdfReader(io.BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if suffix in {".xlsx", ".xlsm"}:
            try:
                from openpyxl import load_workbook
            except ImportError as exc:
                raise RuntimeError("Install the 'docs' extra to parse XLSX attachments") from exc
            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            lines: list[str] = []
            for sheet in workbook.worksheets:
                lines.append(f"[Sheet: {sheet.title}]")
                for row in sheet.iter_rows(values_only=True):
                    values = [str(value) for value in row if value not in (None, "")]
                    if values:
                        lines.append(" | ".join(values))
            return "\n".join(lines)
        return content.decode("utf-8", errors="replace")

    def evidence_documents(self, change: dict[str, Any]) -> list[EvidenceDocument]:
        sys_id = str(change.get("sys_id") or "").strip()
        if not sys_id:
            return []
        documents: list[EvidenceDocument] = []
        for attachment in self.list_attachments(sys_id):
            attachment_id = str(attachment.get("sys_id") or "").strip()
            if not attachment_id:
                continue
            name = str(attachment.get("file_name") or attachment_id)
            try:
                content = self.download_attachment(attachment_id)
                text = self._extract_attachment_text(name, content)
                extraction_error = None
            except (OSError, RuntimeError, ValueError) as exc:
                text = ""
                extraction_error = f"{type(exc).__name__}: {exc}"
            documents.append(
                EvidenceDocument(
                    ref=f"servicenow:{attachment_id}",
                    name=name,
                    text=text,
                    document_type=str(attachment.get("content_type") or Path(name).suffix.lstrip(".") or "unknown"),
                    metadata={
                        "attachment_sys_id": attachment_id,
                        "size_bytes": attachment.get("size_bytes"),
                        "content_type": attachment.get("content_type"),
                        "extraction_error": extraction_error,
                    },
                )
            )
        return documents

    def ingest_change(self, number: str) -> tuple[dict[str, Any], list[EvidenceDocument]]:
        change = self.get_change(number)
        return change, self.evidence_documents(change)
