"""Google Drive connector: files via the Drive API v3.

Lists files (optionally scoped by a Drive ``q`` query or a folder), exports
Google Docs/Sheets/Slides to text and downloads plain-text files, turning each
into a :class:`~vincio.core.types.Document`. Runs on the core ``httpx``
dependency with a bearer access token; accepts an injected client for tests.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..core.errors import LoaderError
from ..core.types import Document
from ..documents.parsers import strip_html
from .base import managed_client, register_connector

__all__ = ["GoogleDriveConnector"]

# Google-native types must be exported (no direct media download); map each to
# the export MIME type Drive should render.
_EXPORT_TYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
_DOWNLOAD_TYPES = ("text/", "application/json", "application/xml")


@register_connector("gdrive")
class GoogleDriveConnector:
    name = "gdrive"

    def __init__(
        self,
        access_token: str,
        *,
        query: str | None = None,
        folder_id: str | None = None,
        api_base: str = "https://www.googleapis.com/drive/v3",
        client: httpx.AsyncClient | None = None,
        max_files: int = 100,
        page_size: int = 50,
    ) -> None:
        self.access_token = access_token
        self.folder_id = folder_id
        if query is not None:
            self.query = query
        elif folder_id:
            self.query = f"'{folder_id}' in parents and trashed = false"
        else:
            self.query = "trashed = false"
        self.api_base = api_base.rstrip("/")
        self.client = client
        self.max_files = max_files
        self.page_size = page_size

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    async def _file_text(self, client: httpx.AsyncClient, file: dict[str, Any]) -> str | None:
        mime = str(file.get("mimeType", ""))
        file_id = file.get("id")
        if mime in _EXPORT_TYPES:
            response = await client.get(
                f"{self.api_base}/files/{file_id}/export",
                params={"mimeType": _EXPORT_TYPES[mime]},
                headers=self._headers(),
            )
        elif mime.startswith(_DOWNLOAD_TYPES) or mime == "text/html":
            response = await client.get(
                f"{self.api_base}/files/{file_id}",
                params={"alt": "media"},
                headers=self._headers(),
            )
        else:
            return None  # skip binary (images, PDFs handled by the document engine elsewhere)
        response.raise_for_status()
        text = response.text
        return strip_html(text) if mime == "text/html" else text

    async def load(self) -> list[Document]:
        async with managed_client(self.client) as client:
            try:
                documents: list[Document] = []
                page_token: str | None = None
                while len(documents) < self.max_files:
                    params: dict[str, Any] = {
                        "q": self.query,
                        "pageSize": min(self.page_size, self.max_files - len(documents)),
                        "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,webViewLink)",
                    }
                    if page_token:
                        params["pageToken"] = page_token
                    response = await client.get(
                        f"{self.api_base}/files", params=params, headers=self._headers()
                    )
                    response.raise_for_status()
                    payload = response.json()
                    for file in payload.get("files", []):
                        text = await self._file_text(client, file)
                        if text is None:
                            continue
                        extra: dict[str, Any] = {}
                        if file.get("modifiedTime"):
                            extra["created_at"] = file["modifiedTime"]
                        documents.append(
                            Document(
                                source_uri=file.get("webViewLink")
                                or f"https://drive.google.com/file/d/{file.get('id')}",
                                title=file.get("name", file.get("id", "untitled")),
                                text=text,
                                metadata={
                                    "connector": self.name,
                                    "file_id": file.get("id"),
                                    "mime_type": file.get("mimeType"),
                                },
                                **extra,
                            )
                        )
                        if len(documents) >= self.max_files:
                            break
                    page_token = payload.get("nextPageToken")
                    if not page_token:
                        break
                return documents
            except httpx.HTTPError as exc:
                raise LoaderError(f"gdrive connector failed: {exc}") from exc
