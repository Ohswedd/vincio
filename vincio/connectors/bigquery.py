"""BigQuery connector: query results into Documents.

Each result row becomes one :class:`~vincio.core.types.Document` (string
columns rendered as ``column: value`` lines). Pass an injected ``client``
(anything exposing ``query(sql).result()`` over dict-convertible rows) for
offline tests; otherwise install ``vincio[bigquery]`` and provide a ``project``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import LoaderError
from ..core.types import Document
from .base import register_connector, row_text, sampled_rows

__all__ = ["BigQueryConnector"]


@register_connector("bigquery")
class BigQueryConnector:
    name = "bigquery"

    def __init__(
        self,
        query: str,
        *,
        project: str | None = None,
        client: Any | None = None,
        id_column: str | None = None,
        title_column: str | None = None,
        text_columns: list[str] | None = None,
        max_rows: int = 1000,
        sample: int | None = None,
        sample_seed: int = 0,
    ) -> None:
        self.query = query
        self.project = project
        self.client = client
        self.id_column = id_column
        self.title_column = title_column
        self.text_columns = text_columns
        self.max_rows = max_rows
        self.sample = sample
        self.sample_seed = sample_seed

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # pragma: no cover - optional dep
            raise LoaderError(
                'bigquery connector needs a client= or: pip install "vincio[bigquery]"'
            ) from exc
        return bigquery.Client(project=self.project)

    def _load_sync(self) -> list[Document]:
        try:
            rows = self._client().query(self.query).result()
            pairs = sampled_rows(rows, max_rows=self.max_rows, sample=self.sample, seed=self.sample_seed)
            documents: list[Document] = []
            for index, raw in pairs:
                row = dict(raw)  # BigQuery Row is mapping-convertible
                row_id = str(row.get(self.id_column, index)) if self.id_column else str(index)
                title = (
                    str(row[self.title_column])
                    if self.title_column and row.get(self.title_column) is not None
                    else f"row {row_id}"
                )
                documents.append(
                    Document(
                        source_uri=f"bigquery://{self.project or 'project'}#{row_id}",
                        title=title,
                        text=row_text(row, self.text_columns),
                        metadata={
                            "connector": self.name,
                            "project": self.project,
                            "row": {k: str(v) for k, v in row.items()},
                            "query": self.query,
                        },
                    )
                )
            return documents
        except LoaderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize driver errors
            raise LoaderError(f"bigquery connector failed: {exc}") from exc

    async def load(self) -> list[Document]:
        return await asyncio.to_thread(self._load_sync)
