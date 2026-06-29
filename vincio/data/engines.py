"""Optional, dependency-backed query engines for execution at scale.

The offline default — :class:`~vincio.data.InProcessSqlEngine`, the
standard-library ``sqlite3`` engine — runs every governed query dependency-free
and derives **cell-exact lineage** for the analyst's common shapes. For datasets
larger than is comfortable in process, :class:`DuckDbQueryEngine` runs the *same
verified read-only SQL* on DuckDB, the embedded analytical engine, behind the
``vincio[data]`` extra.

The contract is identical to any :class:`~vincio.data.QueryEngine`: a query
reaches the engine only after the structural read-only screen and schema
grounding have passed, and the engine re-asserts read-only at its boundary as
defense in depth. The accelerator reports **result-level** lineage — the result
still re-derives from the content-hashed source on
:meth:`~vincio.data.QueryResult.verify` by re-execution — while the offline
``sqlite3`` engine remains the path that derives per-cell citations. Coverage is
always stated, never silently downgraded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.errors import DataError, QueryError
from .core import DataType
from .provenance import LineageCoverage
from .query import (
    DataCatalog,
    ExecutedRows,
    QueryEngine,
    _quote_ident,
    _referenced_tables,
    _to_sql_value,
    assert_read_only_sql,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = ["DuckDbQueryEngine"]


def _duckdb_decl(dtype: DataType) -> str:
    if dtype is DataType.INT or dtype is DataType.BOOL:
        return "BIGINT"
    if dtype is DataType.FLOAT:
        return "DOUBLE"
    return "VARCHAR"


def _import_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise DataError(
            "DuckDbQueryEngine needs the 'duckdb' package; install it with "
            'pip install "vincio[data]"'
        ) from exc
    return duckdb  # pragma: no cover - reached only with the extra installed


class DuckDbQueryEngine(QueryEngine):
    """Execute a verified, read-only ``SELECT`` over the catalog on DuckDB.

    A drop-in :class:`~vincio.data.QueryEngine` for execution at scale: the same
    verified SQL the offline engine would run, executed by DuckDB instead. The
    read-only guarantee is re-asserted structurally at the engine boundary
    (defense in depth beneath the plan-time screen), and the result re-derives
    from the content-hashed source on verification. Lineage is reported at the
    **result level**; the offline ``sqlite3`` engine is the path that derives
    per-cell citations.
    """

    def __init__(self, *, database: str = ":memory:") -> None:
        self._database = database

    def _build(self, catalog: DataCatalog) -> Any:  # pragma: no cover - needs duckdb
        duckdb = _import_duckdb()
        conn = duckdb.connect(self._database)
        for name, ds in catalog.tables.items():
            cols = ds.columns
            decls = ", ".join(f"{_quote_ident(c.name)} {_duckdb_decl(c.dtype)}" for c in cols)
            conn.execute(f"CREATE TABLE {_quote_ident(name)} ({decls})")
            if ds.row_count:
                placeholders = ", ".join("?" for _ in cols)
                insert = f"INSERT INTO {_quote_ident(name)} VALUES ({placeholders})"
                conn.executemany(insert, [[_to_sql_value(v) for v in row] for row in ds.rows()])
        return conn

    def dry_run(self, sql: str, catalog: DataCatalog) -> str:  # pragma: no cover - needs duckdb
        assert_read_only_sql(sql)
        _referenced_tables(sql, catalog)
        conn = self._build(catalog)
        try:
            rows = conn.execute(f"EXPLAIN {sql}").fetchall()
            return "; ".join(str(r[-1]) for r in rows)
        except Exception as exc:  # noqa: BLE001 - duckdb raises its own error types
            raise QueryError(f"query failed to compile: {exc}") from exc
        finally:
            conn.close()

    def execute(  # pragma: no cover - needs duckdb
        self, sql: str, catalog: DataCatalog, *, max_rows: int
    ) -> ExecutedRows:
        # Defense in depth: re-assert read-only at the engine boundary (DuckDB has
        # no sqlite-style authorizer for an in-memory build).
        assert_read_only_sql(sql)
        _referenced_tables(sql, catalog)
        conn = self._build(catalog)
        try:
            cur = conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            fetched = cur.fetchmany(max_rows + 1)
            if len(fetched) > max_rows:
                raise QueryError(
                    f"query returned more than max_rows={max_rows} rows; tighten the "
                    "query or raise max_rows"
                )
            rows = [list(r) for r in fetched]
        except QueryError:
            raise
        except Exception as exc:  # noqa: BLE001 - duckdb raises its own error types
            raise QueryError(f"query execution failed: {exc}") from exc
        finally:
            conn.close()
        return ExecutedRows(
            columns=columns, rows=rows, provenance=[], coverage=LineageCoverage.RESULT
        )
