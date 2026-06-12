"""S3 connector: text objects from an S3 bucket (``pip install "vincio[s3]"``)."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import LoaderError
from ..core.types import Document
from .base import register_connector

__all__ = ["S3Connector"]

_TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".html", ".log")


@register_connector("s3")
class S3Connector:
    name = "s3"

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
            import boto3
        except ImportError as exc:  # pragma: no cover - import guard
            raise LoaderError('S3 connector requires boto3: pip install "vincio[s3]"') from exc
        return boto3.client("s3")

    def _load_sync(self) -> list[Document]:
        client = self._client()
        try:
            keys: list[str] = []
            token: str | None = None
            while len(keys) < self.max_objects:
                kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": self.prefix}
                if token:
                    kwargs["ContinuationToken"] = token
                page = client.list_objects_v2(**kwargs)
                keys.extend(
                    item["Key"]
                    for item in page.get("Contents", [])
                    if item["Key"].lower().endswith(self.extensions)
                )
                if not page.get("IsTruncated"):
                    break
                token = page.get("NextContinuationToken")
            documents: list[Document] = []
            for key in keys[: self.max_objects]:
                obj = client.get_object(Bucket=self.bucket, Key=key)
                body = obj["Body"].read()
                text = body.decode(self.encoding, errors="replace") if isinstance(body, bytes) else str(body)
                extra: dict[str, Any] = {}
                if obj.get("LastModified"):
                    extra["created_at"] = obj["LastModified"]
                documents.append(
                    Document(
                        source_uri=f"s3://{self.bucket}/{key}",
                        title=key.rsplit("/", 1)[-1],
                        text=text,
                        metadata={"connector": self.name, "bucket": self.bucket, "key": key},
                        **extra,
                    )
                )
            return documents
        except LoaderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize client errors
            raise LoaderError(f"s3 connector failed for bucket {self.bucket!r}: {exc}") from exc

    async def load(self) -> list[Document]:
        return await asyncio.to_thread(self._load_sync)
