"""GCS connector: text blobs from a Google Cloud Storage bucket
(``pip install "vincio[gcs]"``)."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import LoaderError
from ..core.types import Document
from .base import register_connector

__all__ = ["GCSConnector"]

_TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".html", ".log")


@register_connector("gcs")
class GCSConnector:
    name = "gcs"

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        client: Any | None = None,
        extensions: tuple[str, ...] = _TEXT_EXTENSIONS,
        max_objects: int = 200,
        encoding: str = "utf-8",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.client = client
        self.extensions = extensions
        self.max_objects = max_objects
        self.encoding = encoding

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - import guard
            raise LoaderError(
                'GCS connector requires google-cloud-storage: pip install "vincio[gcs]"'
            ) from exc
        return storage.Client()

    def _load_sync(self) -> list[Document]:
        client = self._client()
        try:
            blobs = client.list_blobs(self.bucket, prefix=self.prefix)
            documents: list[Document] = []
            for blob in blobs:
                if len(documents) >= self.max_objects:
                    break
                if not blob.name.lower().endswith(self.extensions):
                    continue
                payload = blob.download_as_bytes()
                text = payload.decode(self.encoding, errors="replace") if isinstance(payload, bytes) else str(payload)
                extra: dict[str, Any] = {}
                if getattr(blob, "updated", None):
                    extra["created_at"] = blob.updated
                documents.append(
                    Document(
                        source_uri=f"gs://{self.bucket}/{blob.name}",
                        title=blob.name.rsplit("/", 1)[-1],
                        text=text,
                        metadata={"connector": self.name, "bucket": self.bucket, "key": blob.name},
                        **extra,
                    )
                )
            return documents
        except LoaderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize client errors
            raise LoaderError(f"gcs connector failed for bucket {self.bucket!r}: {exc}") from exc

    async def load(self) -> list[Document]:
        return await asyncio.to_thread(self._load_sync)
