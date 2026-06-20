"""Salesforce connector: SOQL query results via the REST API.

Each record becomes one :class:`~vincio.core.types.Document` (string fields
rendered as ``Field: value`` lines), following ``nextRecordsUrl`` paging. Runs
on the core ``httpx`` dependency with a bearer access token + instance URL;
accepts an injected client for tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from .base import managed_client, register_connector, row_text

__all__ = ["SalesforceConnector"]


@register_connector("salesforce")
class SalesforceConnector:
    name = "salesforce"

    def __init__(
        self,
        instance_url: str,
        access_token: str,
        soql: str,
        *,
        api_version: str = "v60.0",
        title_field: str = "Name",
        text_fields: list[str] | None = None,
        client: httpx.AsyncClient | None = None,
        max_records: int = 200,
    ) -> None:
        self.instance_url = instance_url.rstrip("/")
        self.access_token = access_token
        self.soql = soql
        self.api_version = api_version
        self.title_field = title_field
        self.text_fields = text_fields
        self.client = client
        self.max_records = max_records

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def _record_document(self, record: dict[str, Any], index: int) -> Document:
        attributes = record.get("attributes", {}) or {}
        sobject = attributes.get("type", "sObject")
        record_id = record.get("Id") or str(index)
        # Drop Salesforce's per-record metadata envelope before rendering.
        fields = {k: v for k, v in record.items() if k != "attributes"}
        title = str(fields.get(self.title_field) or f"{sobject} {record_id}")
        return Document(
            source_uri=f"{self.instance_url}/{record_id}",
            title=title,
            text=row_text(fields, self.text_fields),
            metadata={
                "connector": self.name,
                "sobject": sobject,
                "record_id": record_id,
                "fields": {k: str(v) for k, v in fields.items()},
            },
        )

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                documents: list[Document] = []
                url = f"{self.instance_url}/services/data/{self.api_version}/query"
                params: dict[str, Any] | None = {"q": self.soql}
                while url and len(documents) < self.max_records:
                    response = await client.get(url, params=params, headers=self._headers())
                    response.raise_for_status()
                    payload = response.json()
                    for record in payload.get("records", []):
                        documents.append(self._record_document(record, len(documents)))
                        if len(documents) >= self.max_records:
                            break
                    next_url = payload.get("nextRecordsUrl")
                    if not next_url or payload.get("done", True):
                        break
                    # nextRecordsUrl is a server-absolute path; params are baked in.
                    url = f"{self.instance_url}{next_url}"
                    params = None
                return documents
            except httpx.HTTPError as exc:
                raise LoaderError(f"salesforce connector failed: {exc}") from exc
