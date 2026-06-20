"""SQL connector: query rows into Documents.

Works with SQLite out of the box (stdlib) via ``url="sqlite:///path.db"``,
or any DB-API 2.0 connection you pass in (psycopg, mysql-connector, ...) —
the connector only needs ``cursor()``/``execute()``/``fetchall()``. When
injecting a sqlite3 connection that other threads may load (e.g. via
``app.add_source``), open it with ``check_same_thread=False``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.errors import LoaderError
from ..core.types import Document
from .base import register_connector, row_text

__all__ = ["SQLConnector"]


@register_connector("sql")
class SQLConnector:
    name = "sql"

    def __init__(
        self,
        query: str,
        *,
        url: str | None = None,
        connection: Any | None = None,
        id_column: str | None = None,
        title_column: str | None = None,
        text_columns: list[str] | None = None,
        max_rows: int = 1000,
    ) -> None:
        if url is None and connection is None:
            raise LoaderError("sql connector needs a url or a DB-API connection")
        self.query = query
        self.url = url
        self.connection = connection
        self.id_column = id_column
        self.title_column = title_column
        self.text_columns = text_columns
        self.max_rows = max_rows

    def _connect(self) -> tuple[Any, bool]:
        if self.connection is not None:
            return self.connection, False
        url = self.url or ""
        if url.startswith("sqlite://"):
            import sqlite3

            path = url.removeprefix("sqlite:///").removeprefix("sqlite://")
            return sqlite3.connect(path), True
        raise LoaderError(
            f"sql connector cannot open {url!r}; pass a DB-API connection= for non-SQLite databases"
        )

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
                text = row_text(row, self.text_columns)
                row_id = str(row.get(self.id_column, index)) if self.id_column else str(index)
                title = str(row[self.title_column]) if self.title_column and row.get(self.title_column) else f"row {row_id}"
                documents.append(
                    Document(
                        source_uri=f"sql://{self.url or 'connection'}#{row_id}",
                        title=title,
                        text=text,
                        metadata={
                            "connector": self.name,
                            "row": {k: str(v) for k, v in row.items()},
                            "query": self.query,
                        },
                    )
                )
            return documents
        except LoaderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize driver errors
            raise LoaderError(f"sql connector failed: {exc}") from exc
        finally:
            if owned:
                connection.close()

    async def load(self) -> list[Document]:
        if self.connection is not None:
            # Injected DB-API connections may be thread-bound (sqlite3
            # check_same_thread): use them on the caller's thread.
            return self._load_sync()
        return await asyncio.to_thread(self._load_sync)
