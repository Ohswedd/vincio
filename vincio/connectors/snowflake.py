"""Snowflake connector: query results into Documents.

Each result row becomes one :class:`~vincio.core.types.Document` (string
columns rendered as ``column: value`` lines). Pass an injected DB-API
``connection`` (``cursor()`` / ``execute()`` / ``fetchmany()`` / ``description``)
for offline tests; otherwise install ``vincio[snowflake]`` and provide
connection parameters (``account``, ``user``, ...).
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import LoaderError
from ..core.types import Document
from .base import register_connector, row_text

__all__ = ["SnowflakeConnector"]


@register_connector("snowflake")
class SnowflakeConnector:
    name = "snowflake"

    def __init__(
        self,
        query: str,
        *,
        connection: Any | None = None,
        account: str | None = None,
        id_column: str | None = None,
        title_column: str | None = None,
        text_columns: list[str] | None = None,
        max_rows: int = 1000,
        **connect_kwargs: Any,
    ) -> None:
        self.query = query
        self.connection = connection
        self.account = account
        self.id_column = id_column
        self.title_column = title_column
        self.text_columns = text_columns
        self.max_rows = max_rows
        self.connect_kwargs = connect_kwargs

    def _connect(self) -> tuple[Any, bool]:
        if self.connection is not None:
            return self.connection, False
        try:
            import snowflake.connector
        except ImportError as exc:  # pragma: no cover - optional dep
            raise LoaderError(
                'snowflake connector needs a connection= or: pip install "vincio[snowflake]"'
            ) from exc
        return snowflake.connector.connect(account=self.account, **self.connect_kwargs), True

    def _load_sync(self) -> list[Document]:
        connection, owned = self._connect()
        try:
            cursor = connection.cursor()
            cursor.execute(self.query)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchmany(self.max_rows) if hasattr(cursor, "fetchmany") else cursor.fetchall()
            documents: list[Document] = []
            for index, values in enumerate(rows):
                row = dict(zip(columns, values, strict=False))
                row_id = str(row.get(self.id_column, index)) if self.id_column else str(index)
                title = (
                    str(row[self.title_column])
                    if self.title_column and row.get(self.title_column) is not None
                    else f"row {row_id}"
                )
                documents.append(
                    Document(
                        source_uri=f"snowflake://{self.account or 'account'}#{row_id}",
                        title=title,
                        text=row_text(row, self.text_columns),
                        metadata={
                            "connector": self.name,
                            "account": self.account,
                            "row": {k: str(v) for k, v in row.items()},
                            "query": self.query,
                        },
                    )
                )
            return documents
        except LoaderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize driver errors
            raise LoaderError(f"snowflake connector failed: {exc}") from exc
        finally:
            if owned:
                connection.close()

    async def load(self) -> list[Document]:
        if self.connection is not None:
            # Injected DB-API connections may be thread-bound; use the caller's thread.
            return self._load_sync()
        return await asyncio.to_thread(self._load_sync)
