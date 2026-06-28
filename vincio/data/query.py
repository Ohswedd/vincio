"""Governed text-to-query over a registered dataset, with cell-level provenance.

The core analyst capability of the data plane: a question becomes a query that is
**schema-grounded and verified before it runs**, executed **where the data lives**
rather than by materializing rows into the prompt, and whose answer **cites the
exact rows and cells it rests on**, offline-verifiable the way a cited report is.

The pipeline is deterministic and, by default, offline:

1. **Plan.** A question (or an explicit query) is turned into a :class:`QueryPlan`
   — a SQL ``SELECT`` or a dataframe op pipeline — grounded against a
   :class:`DataCatalog` of registered datasets. An unknown table or column is
   refused here, before anything runs.
2. **Verify (read-only).** The query is screened **structurally** for being
   provably read-only: a single statement, a ``SELECT``/``WITH`` head, no write,
   DDL, or stacked statement, and no injection signal in the question. A breach
   raises :class:`~vincio.core.errors.UnsafeQueryError`. The same guarantee is
   available as a :class:`~vincio.verify.ToolContract` (:func:`make_query_contract`)
   so the capability refuses a write structurally when it rides the tool runtime.
3. **Dry-run / cost-bound.** The query is compiled and its plan inspected without
   fetching; a row ceiling (``max_rows``) bounds the result.
4. **Execute.** A pluggable :class:`QueryEngine` runs it. The default
   :class:`InProcessSqlEngine` is the standard-library ``sqlite3`` engine — a real
   SQL engine, dependency-free and offline — opened read-only with an authorizer
   that denies every non-read action (defense in depth beneath the screen). A
   pushdown engine can run the verified SQL against a live source instead.
5. **Cite.** The result is a :class:`~vincio.data.provenance` -bearing
   :class:`QueryResult`: a schema-bearing result :class:`~vincio.data.Dataset`,
   per-row source-cell lineage, and a content-bound hash so ``verify()``
   re-executes the query against the hashed source and confirms the answer — and
   every cited cell — re-derives from the bytes alone.

Everything here is deterministic, dependency-free, and offline.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ..core import tabular
from ..core.errors import QueryError, UnsafeQueryError
from ..core.utils import stable_hash
from .core import ColumnSchema, Dataset, DataType
from .provenance import CellCitation, LineageCoverage, RowProvenance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.injection import InjectionDetector
    from ..verify.programs import ProgramOp
    from .evidence import TableEvidence

__all__ = [
    "QueryDialect",
    "DataCatalog",
    "QueryPlan",
    "QueryResult",
    "QueryEngine",
    "InProcessSqlEngine",
    "HeuristicQueryPlanner",
    "make_query_contract",
    "query_dataset",
    "is_read_only_sql",
    "assert_read_only_sql",
]


class QueryDialect(StrEnum):
    """The two forms a grounded query takes. ``SQL`` is a read-only ``SELECT``
    executed by the engine; ``DATAFRAME`` is an ordered pipeline of whitelisted,
    intrinsically read-only dataframe ops (select / filter / derive / rename)."""

    SQL = "sql"
    DATAFRAME = "dataframe"


# --------------------------------------------------------------------------- #
# Catalog: the grounding source                                               #
# --------------------------------------------------------------------------- #


class DataCatalog:
    """A named set of registered :class:`~vincio.data.Dataset`\\s a query grounds
    against and executes over.

    A query may reference only tables in the catalog and columns those tables
    declare; anything else is refused as ungrounded. The catalog also content-hashes
    its tables, so a :class:`QueryResult` binds the exact data it was computed from
    and a tampered source is caught on :meth:`QueryResult.verify`.
    """

    def __init__(self, datasets: dict[str, Dataset] | None = None) -> None:
        self._tables: dict[str, Dataset] = {}
        for name, ds in (datasets or {}).items():
            self.add(ds, name=name)

    def add(self, dataset: Dataset, *, name: str = "") -> str:
        """Register *dataset* under *name* (defaulting to its own name, then
        ``data``). Returns the resolved table name."""
        table = name or dataset.name or "data"
        if not _IDENT_RE.fullmatch(table):
            raise QueryError(
                f"table name {table!r} is not a simple identifier; pass an explicit "
                "name= of letters, digits, and underscores"
            )
        self._tables[table] = dataset
        return table

    def get(self, name: str) -> Dataset:
        try:
            return self._tables[name]
        except KeyError as exc:
            raise QueryError(
                f"no registered table {name!r}; known tables: {sorted(self._tables)}"
            ) from exc

    @property
    def tables(self) -> dict[str, Dataset]:
        """The registered tables, by name (a copy)."""
        return dict(self._tables)

    @property
    def names(self) -> list[str]:
        return sorted(self._tables)

    def columns(self, table: str) -> list[str]:
        return self.get(table).column_names

    def content_hashes(self) -> dict[str, str]:
        """A content hash per registered table, binding its schema and cells."""
        return {
            name: stable_hash([ds.column_names, ds.dtypes, ds.rows()])
            for name, ds in sorted(self._tables.items())
        }

    @classmethod
    def of(cls, dataset: Dataset, *, name: str = "") -> DataCatalog:
        """A single-table catalog over one dataset."""
        catalog = cls()
        catalog.add(dataset, name=name)
        return catalog


# --------------------------------------------------------------------------- #
# Read-only verification (the structural guard)                               #
# --------------------------------------------------------------------------- #

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

# Keywords that mutate state or reach outside a read. Their presence anywhere as a
# token refuses the query — the read-only guarantee is structural, never a
# best-effort string match.
_WRITE_KEYWORDS = frozenset(
    {
        "insert",
        "update",
        "delete",
        "replace",
        "upsert",
        "merge",
        "drop",
        "create",
        "alter",
        "truncate",
        "rename",
        "attach",
        "detach",
        "reindex",
        "vacuum",
        "pragma",
        "grant",
        "revoke",
        "commit",
        "rollback",
        "savepoint",
        "begin",
        "analyze",
        "load_extension",
    }
)

_LEADING_KEYWORDS = frozenset({"select", "with"})

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'")
# Quoted identifiers: double-quoted (SQL standard), backtick (MySQL), and bracketed
# (SQL Server). Their contents are names, never keywords or statement boundaries.
_QUOTED_IDENT_RE = re.compile(r'"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\]')


def _strip_sql_literals(sql: str) -> str:
    """Blank out comments and string literals so keyword scanning never trips on
    a write keyword that lives inside a quoted value or a comment."""
    without_comments = _COMMENT_RE.sub(" ", sql)
    return _STRING_RE.sub("''", without_comments)


def _strip_for_keyword_scan(sql: str) -> str:
    """Strip comments, string literals, **and quoted identifiers**, replacing each
    quoted identifier with a neutral placeholder.

    Used by the read-only screen so a column or table named with a reserved word
    (``SELECT "update" FROM t``) or containing a semicolon (``"col;drop"``) is not
    misread as a write keyword or a stacked statement — the contents of a quoted
    identifier are a name, never SQL structure."""
    return _QUOTED_IDENT_RE.sub(" _id_ ", _strip_sql_literals(sql))


def _statements(sql: str) -> list[str]:
    """Split *sql* (already literal-stripped) into non-empty top-level statements
    on a ``;`` that is not inside parentheses."""
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == ";" and depth == 0:
            chunk = "".join(current).strip()
            if chunk:
                out.append(chunk)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        out.append(tail)
    return out


def is_read_only_sql(sql: str) -> bool:
    """Whether *sql* is provably a single read-only query.

    True only when, after stripping comments, string literals, and quoted
    identifiers, there is exactly one statement, it begins with ``SELECT`` or
    ``WITH``, and it contains no write / DDL / side-effecting keyword as a token.
    Deterministic; never gated on a model."""
    stripped = _strip_for_keyword_scan(sql)
    statements = _statements(stripped)
    if len(statements) != 1:
        return False
    statement = statements[0]
    tokens = [t.lower() for t in _IDENT_RE.findall(statement)]
    if not tokens or tokens[0] not in _LEADING_KEYWORDS:
        return False
    return not any(tok in _WRITE_KEYWORDS for tok in tokens)


def assert_read_only_sql(sql: str) -> None:
    """Raise :class:`~vincio.core.errors.UnsafeQueryError` unless *sql* is provably
    read-only (see :func:`is_read_only_sql`)."""
    if not is_read_only_sql(sql):
        raise UnsafeQueryError(
            "query refused: not provably read-only (a single read-only SELECT is "
            f"required, write/DDL/stacked statements are not): {sql.strip()[:200]!r}"
        )


# --------------------------------------------------------------------------- #
# Lightweight single-table SQL shape analysis (for cell-exact lineage)        #
# --------------------------------------------------------------------------- #

_CLAUSE_KEYWORDS = ("select", "from", "where", "group by", "having", "order by", "limit")
_AGG_RE = re.compile(r"\b(count|sum|avg|min|max|total|group_concat)\s*\(", re.IGNORECASE)
_SUBQUERY_RE = re.compile(r"\(\s*select\b", re.IGNORECASE)


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split *text* on *sep* at parenthesis depth zero."""
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(ch)
    out.append("".join(current))
    return out


def _clauses(sql: str) -> dict[str, str]:
    """Segment a single, simple ``SELECT`` into its top-level clauses. Returns an
    empty mapping when the shape is not a flat single-statement select (the caller
    then degrades to result-level lineage)."""
    text = " ".join(_strip_sql_literals(sql).split())
    lowered = text.lower()
    if not lowered.startswith("select "):
        return {}
    # Find each clause keyword at the top level, in order.
    marks: list[tuple[int, str]] = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if depth == 0:
            for kw in _CLAUSE_KEYWORDS:
                if lowered.startswith(kw, i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
                    after = i + len(kw)
                    if after >= len(text) or not (text[after].isalnum() or text[after] == "_"):
                        marks.append((i, kw))
                        i = after
                        break
            else:
                i += 1
            continue
        i += 1
    if not marks or marks[0][1] != "select":
        return {}
    out: dict[str, str] = {}
    for idx, (pos, kw) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        out[kw] = text[pos + len(kw) : end].strip()
    return out


def _unquote_ident(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'`":
        return token[1:-1].replace(token[0] * 2, token[0])
    if len(token) >= 2 and token[0] == "[" and token[-1] == "]":
        return token[1:-1]
    return token


def _base_columns(expr: str, columns: Sequence[str]) -> list[str]:
    """Every distinct source column an output expression references, in first-seen
    order. A bare column yields one; an arithmetic blend (``revenue + tax``) yields
    all its operands; a constant or ``COUNT(*)`` yields none."""
    found: list[str] = []
    lowered = {c.lower(): c for c in columns}
    for ident in _IDENT_RE.findall(expr):
        canonical = lowered.get(ident.lower())
        if canonical and canonical not in found:
            found.append(canonical)
    return found


def _base_column(expr: str, columns: Sequence[str]) -> str | None:
    """The single source column an output expression rests on, or ``None`` when it
    references zero or several (used for group-by keys, which must be plain
    columns)."""
    found = _base_columns(expr, columns)
    return found[0] if len(found) == 1 else None


class _Shape(BaseModel):
    """The analyzed lineage shape of a single-table query."""

    table: str
    has_aggregate: bool
    group_by: list[str] = Field(default_factory=list)
    # Per output column, the source columns it rests on (a projection keeps one; a
    # derived expression keeps all its operands; a constant keeps none).
    select_bases: list[list[str]] = Field(default_factory=list)
    distinct: bool = False
    where: str = ""


def _analyze_shape(sql: str, catalog: DataCatalog, tables: list[str]) -> _Shape | None:
    """Classify *sql* as a single-table projection/filter or group-by aggregation,
    returning its :class:`_Shape`, or ``None`` when it is outside the cell-lineage
    grammar (a join, a subquery, a CTE, or multiple tables)."""
    if len(tables) != 1 or _SUBQUERY_RE.search(_strip_sql_literals(sql)):
        return None
    # Cell lineage threads sqlite's implicit rowid; a user column that shadows it
    # (``rowid`` / ``oid`` / ``_rowid_``) would make that reference ambiguous, so
    # degrade to result-level lineage (the result still verifies) rather than cite
    # the wrong source rows.
    if {c.lower() for c in catalog.columns(tables[0])} & {"rowid", "oid", "_rowid_"}:
        return None
    clauses = _clauses(sql)
    if not clauses or "from" not in clauses:
        return None
    from_clause = clauses["from"].strip()
    # one base table only: no comma, no join, no subquery, no parens
    if any(t in from_clause.lower() for t in (",", " join ")) or "(" in from_clause:
        return None
    table = tables[0]
    columns = catalog.columns(table)
    select_text = clauses["select"].strip()
    distinct = select_text.lower().startswith("distinct ")
    if distinct:
        select_text = select_text[len("distinct ") :].strip()
    items = [s.strip() for s in _split_top_level(select_text, ",") if s.strip()]
    select_bases: list[list[str]] = []
    for item in items:
        if item == "*":
            select_bases.extend([c] for c in columns)
            continue
        # drop a trailing alias ("expr AS name")
        expr = re.sub(r"\s+as\s+\w+$", "", item, flags=re.IGNORECASE)
        select_bases.append(_base_columns(expr, columns))
    group_by: list[str] = []
    if "group by" in clauses:
        for tok in _split_top_level(clauses["group by"], ","):
            base = _base_column(tok, columns)
            if base is None:
                return None  # group key not a plain column → not cell-traceable
            group_by.append(base)
    return _Shape(
        table=table,
        has_aggregate=bool(_AGG_RE.search(select_text)) or bool(group_by),
        group_by=group_by,
        select_bases=select_bases,
        distinct=distinct,
        where=clauses.get("where", ""),
    )


# --------------------------------------------------------------------------- #
# Grounding                                                                    #
# --------------------------------------------------------------------------- #

def _referenced_tables(sql: str, catalog: DataCatalog) -> list[str]:
    """The catalog tables a FROM/JOIN clause references, validated to exist."""
    text = _strip_sql_literals(sql)
    known = {n.lower(): n for n in catalog.names}
    found: list[str] = []
    for match in re.finditer(r"\b(from|join)\s+([A-Za-z_]\w*|\"[^\"]+\"|`[^`]+`|\[[^\]]+\])", text, re.IGNORECASE):
        name = _unquote_ident(match.group(2))
        canonical = known.get(name.lower())
        if canonical is None:
            raise QueryError(
                f"query references unknown table {name!r}; registered tables: {catalog.names}"
            )
        if canonical not in found:
            found.append(canonical)
    if not found:
        raise QueryError("query references no registered table")
    return found


# --------------------------------------------------------------------------- #
# The query engine                                                            #
# --------------------------------------------------------------------------- #

# sqlite authorizer: allow only read actions, deny everything else (writes, DDL,
# ATTACH, PRAGMA, transactions). Belt-and-suspenders beneath the read-only screen.
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", 33),
    }
)


def _deny_writes(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _to_sql_value(value: Any) -> Any:
    """Adapt a Python cell value to a sqlite-storable one (bool→int, temporal→ISO)."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


class ExecutedRows(BaseModel):
    """The raw output of a :class:`QueryEngine`: result column names and rows, plus
    the per-result-row source lineage the engine could establish."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    columns: list[str]
    rows: list[list[Any]] = Field(default_factory=list)
    provenance: list[RowProvenance] = Field(default_factory=list)
    coverage: LineageCoverage = LineageCoverage.RESULT
    plan_detail: str = ""


class QueryEngine:
    """The execution boundary for a verified query. Implementations execute a
    read-only query against the catalog's data and return rows plus lineage. The
    default :class:`InProcessSqlEngine` runs in-process on ``sqlite3``; a pushdown
    engine runs the same verified SQL where the data lives."""

    def dry_run(self, sql: str, catalog: DataCatalog) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def execute(self, sql: str, catalog: DataCatalog, *, max_rows: int) -> ExecutedRows:  # pragma: no cover
        raise NotImplementedError


class InProcessSqlEngine(QueryEngine):
    """Execute a verified read-only ``SELECT`` over the catalog with the
    standard-library ``sqlite3`` engine — a real SQL engine, dependency-free and
    offline.

    The connection is opened read-only: ``PRAGMA query_only`` is set and an
    authorizer denies every non-read action, so a write or DDL that somehow passed
    the screen is still structurally refused by the engine. The engine also derives
    **cell-exact lineage** for single-table projection/filter and group-by
    aggregation queries (the analyst's common shapes); other shapes execute and
    re-derive at the result level."""

    def _build(self, catalog: DataCatalog) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        for name, ds in catalog.tables.items():
            cols = ds.columns
            decls = ", ".join(f"{_quote_ident(c.name)} {_sqlite_decl(c.dtype)}" for c in cols)
            conn.execute(f"CREATE TABLE {_quote_ident(name)} ({decls})")
            placeholders = ", ".join("?" for _ in cols)
            insert = f"INSERT INTO {_quote_ident(name)} VALUES ({placeholders})"
            conn.executemany(insert, [[_to_sql_value(v) for v in row] for row in ds.rows()])
        conn.commit()
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_deny_writes)
        return conn

    def dry_run(self, sql: str, catalog: DataCatalog) -> str:
        conn = self._build(catalog)
        try:
            # Compiling and planning the statement validates it parses and binds to
            # the schema, and surfaces the access path, all without fetching a row.
            rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
            return "; ".join(str(r[-1]) for r in rows)
        except sqlite3.Error as exc:
            raise QueryError(f"query failed to compile: {exc}") from exc
        finally:
            conn.close()

    def execute(self, sql: str, catalog: DataCatalog, *, max_rows: int) -> ExecutedRows:
        conn = self._build(catalog)
        try:
            tables = _referenced_tables(sql, catalog)
            shape = _analyze_shape(sql, catalog, tables)
            if shape is not None and not shape.has_aggregate and not shape.distinct:
                return self._execute_projection(conn, sql, catalog, shape, max_rows)
            if shape is not None and shape.has_aggregate and shape.group_by:
                return self._execute_grouped(conn, sql, catalog, shape, max_rows)
            return self._execute_opaque(conn, sql, catalog, tables, max_rows)
        except sqlite3.Error as exc:
            raise QueryError(f"query execution failed: {exc}") from exc
        finally:
            conn.close()

    # -- execution strategies --------------------------------------------------

    def _fetch(self, conn: sqlite3.Connection, sql: str, max_rows: int) -> tuple[list[str], list[list[Any]]]:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchmany(max_rows + 1)]
        if len(rows) > max_rows:
            raise QueryError(
                f"query returned more than max_rows={max_rows} rows; tighten the "
                "query or raise max_rows"
            )
        return columns, rows

    def _execute_opaque(
        self, conn: sqlite3.Connection, sql: str, catalog: DataCatalog, tables: list[str], max_rows: int
    ) -> ExecutedRows:
        columns, rows = self._fetch(conn, sql, max_rows)
        return ExecutedRows(columns=columns, rows=rows, provenance=[], coverage=LineageCoverage.RESULT)

    def _execute_projection(
        self, conn: sqlite3.Connection, sql: str, catalog: DataCatalog, shape: _Shape, max_rows: int
    ) -> ExecutedRows:
        clauses = _clauses(sql)
        rowid_sql = self._rebuild_with_rowid(clauses, shape.table)
        cur = conn.execute(rowid_sql)
        description = [d[0] for d in cur.description]
        fetched = [list(r) for r in cur.fetchmany(max_rows + 1)]
        if len(fetched) > max_rows:
            raise QueryError(f"query returned more than max_rows={max_rows} rows")
        columns = description[1:]  # drop the leading __vrow__
        ds = catalog.get(shape.table)
        rows: list[list[Any]] = []
        provenance: list[RowProvenance] = []
        for result_row, raw in enumerate(fetched):
            src_row = int(raw[0]) - 1  # sqlite rowid is 1-based insertion order
            values = raw[1:]
            rows.append(values)
            cells: list[CellCitation] = []
            for col_index, bases in enumerate(shape.select_bases[: len(values)]):
                out_col = columns[col_index] if col_index < len(columns) else ""
                for base in bases:
                    cells.append(
                        CellCitation(
                            table=shape.table,
                            row=src_row,
                            column=base,
                            value=_source_cell(ds, src_row, base),
                            result_column=out_col,
                        )
                    )
            provenance.append(RowProvenance(result_row=result_row, cells=_dedup_cells(cells), exact=True))
        return ExecutedRows(columns=columns, rows=rows, provenance=provenance, coverage=LineageCoverage.CELL)

    def _execute_grouped(
        self, conn: sqlite3.Connection, sql: str, catalog: DataCatalog, shape: _Shape, max_rows: int
    ) -> ExecutedRows:
        columns, rows = self._fetch(conn, sql, max_rows)
        # Result columns must surface the group keys by name for cell lineage.
        lowered = {c.lower(): i for i, c in enumerate(columns)}
        key_index = [lowered.get(g.lower()) for g in shape.group_by]
        if any(k is None for k in key_index):
            return ExecutedRows(columns=columns, rows=rows, provenance=[], coverage=LineageCoverage.RESULT)
        # A lineage query maps each group key to the contributing source rows.
        group_to_rows = self._group_rowids(conn, shape)
        ds = catalog.get(shape.table)
        # Each output column rests on its own source columns across the contributing
        # rows: a group key on the group column, an aggregate on its operand(s).
        provenance: list[RowProvenance] = []
        for result_row, row in enumerate(rows):
            key = tuple(row[i] for i in key_index)  # type: ignore[index]
            src_rows = group_to_rows.get(key, [])
            cells: list[CellCitation] = []
            for src_row in src_rows:
                for col_index, bases in enumerate(shape.select_bases[: len(columns)]):
                    out_col = columns[col_index]
                    for base in bases:
                        cells.append(
                            CellCitation(
                                table=shape.table,
                                row=src_row,
                                column=base,
                                value=_source_cell(ds, src_row, base),
                                result_column=out_col,
                            )
                        )
            provenance.append(RowProvenance(result_row=result_row, cells=_dedup_cells(cells), exact=True))
        return ExecutedRows(
            columns=columns, rows=rows, provenance=provenance, coverage=LineageCoverage.CELL
        )

    def _group_rowids(self, conn: sqlite3.Connection, shape: _Shape) -> dict[tuple[Any, ...], list[int]]:
        keys_sql = ", ".join(_quote_ident(g) for g in shape.group_by)
        # The witness query maps every (post-WHERE) group key to its source rowids.
        # It deliberately omits HAVING / ORDER BY / LIMIT: attribution is restricted
        # to the groups that actually appear in the result by looking up each result
        # row's group key, so a group HAVING filtered out is simply never looked up —
        # and a HAVING that references a SELECT alias (which this query does not
        # define) cannot break the witness.
        lineage_sql = f"SELECT {keys_sql}, group_concat(rowid) FROM {_quote_ident(shape.table)}"
        if shape.where:
            lineage_sql += f" WHERE {shape.where}"
        lineage_sql += f" GROUP BY {keys_sql}"
        out: dict[tuple[Any, ...], list[int]] = {}
        for raw in conn.execute(lineage_sql).fetchall():
            key = tuple(raw[: len(shape.group_by)])
            rowids = [int(x) - 1 for x in str(raw[-1]).split(",") if x != ""]
            out[key] = rowids
        return out

    @staticmethod
    def _rebuild_with_rowid(clauses: dict[str, str], table: str) -> str:
        select = clauses["select"]
        parts = [f"SELECT {_quote_ident(table)}.rowid AS __vrow__, {select}", f"FROM {clauses['from']}"]
        for kw in ("where", "group by", "having", "order by", "limit"):
            if kw in clauses:
                parts.append(f"{kw.upper()} {clauses[kw]}")
        return " ".join(parts)


def _sqlite_decl(dtype: DataType) -> str:
    if dtype is DataType.INT or dtype is DataType.BOOL:
        return "INTEGER"
    if dtype is DataType.FLOAT:
        return "REAL"
    return "TEXT"


def _source_cell(ds: Dataset, row: int, column: str) -> Any:
    if 0 <= row < ds.row_count and column in ds.column_names:
        return ds.column(column)[row]
    return None


# --------------------------------------------------------------------------- #
# QueryPlan & QueryResult                                                      #
# --------------------------------------------------------------------------- #


class QueryPlan(BaseModel):
    """A schema-grounded, read-only-verified query that has **not yet run**.

    The artifact of the verify-before-execute step: the SQL (or dataframe ops), the
    registered tables it grounds to, the proof it is read-only, and a dry-run plan /
    row ceiling. Build it with :meth:`for_sql`; execute it with :meth:`run` (or the
    :func:`query_dataset` one-shot)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str = ""
    dialect: QueryDialect = QueryDialect.SQL
    sql: str = ""
    ops: list[Any] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    read_only: bool = True
    max_rows: int = 10_000
    plan_detail: str = ""

    @classmethod
    def for_sql(
        cls,
        sql: str,
        catalog: DataCatalog,
        *,
        question: str = "",
        max_rows: int = 10_000,
        engine: QueryEngine | None = None,
    ) -> QueryPlan:
        """Ground and verify *sql* against *catalog* without running it. Refuses an
        unsafe (non-read-only) or ungrounded query here, before execution."""
        assert_read_only_sql(sql)
        tables = _referenced_tables(sql, catalog)
        # Column grounding is enforced authoritatively by the dry-run compile
        # below: an unknown column refuses the plan before it ever executes.
        engine = engine or InProcessSqlEngine()
        plan_detail = engine.dry_run(sql, catalog)
        return cls(
            question=question,
            dialect=QueryDialect.SQL,
            sql=sql.strip(),
            tables=tables,
            read_only=True,
            max_rows=max_rows,
            plan_detail=plan_detail,
        )

    def run(self, catalog: DataCatalog, *, engine: QueryEngine | None = None) -> QueryResult:
        """Execute the verified plan and return a cited :class:`QueryResult`."""
        if self.dialect is QueryDialect.DATAFRAME:
            return _run_dataframe(self, catalog)
        engine = engine or InProcessSqlEngine()
        executed = engine.execute(self.sql, catalog, max_rows=self.max_rows)
        return QueryResult.build(self, catalog, executed)


class QueryResult(BaseModel):
    """A query's result, schema-bearing and **cell-level cited**.

    Carries the result as a :class:`~vincio.data.Dataset`, the per-result-row
    source-cell :class:`~vincio.data.provenance.RowProvenance`, the content hashes
    of the source tables, and a result hash binding them. :meth:`verify`
    re-executes the plan against a catalog, re-derives the result and every cited
    cell from the bytes, and confirms nothing was tampered — the analytics analogue
    of a cited report's offline verification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: QueryPlan
    dataset: Dataset
    provenance: list[RowProvenance] = Field(default_factory=list)
    coverage: LineageCoverage = LineageCoverage.RESULT
    source_hashes: dict[str, str] = Field(default_factory=dict)
    result_hash: str = ""

    # -- construction ----------------------------------------------------------

    @classmethod
    def build(cls, plan: QueryPlan, catalog: DataCatalog, executed: ExecutedRows) -> QueryResult:
        dataset = _result_dataset(plan, executed)
        source_hashes = {t: catalog.content_hashes()[t] for t in plan.tables}
        result = cls(
            plan=plan,
            dataset=dataset,
            provenance=executed.provenance,
            coverage=executed.coverage,
            source_hashes=source_hashes,
        )
        result.result_hash = result._compute_hash()
        return result

    def _compute_hash(self) -> str:
        return stable_hash(
            [
                _normalize_sql(self.plan.sql) if self.plan.dialect is QueryDialect.SQL else self.plan.ops,
                str(self.plan.dialect),
                sorted(self.source_hashes.items()),
                self.dataset.column_names,
                self.dataset.rows(),
            ]
        )

    # -- access ----------------------------------------------------------------

    @property
    def rows(self) -> list[list[Any]]:
        """The result rows."""
        return self.dataset.rows()

    @property
    def columns(self) -> list[str]:
        """The result column names."""
        return self.dataset.column_names

    @property
    def row_count(self) -> int:
        return self.dataset.row_count

    def value(self, row: int = 0, column: str | int = 0) -> Any:
        """One result cell, by row index and column name or index."""
        col = column if isinstance(column, str) else self.dataset.column_names[column]
        return self.dataset.column(col)[row]

    def citations(self, row: int, column: str | int | None = None) -> list[CellCitation]:
        """The source cells a result cell (or whole result row) rests on."""
        if row >= len(self.provenance):
            return []
        prov = self.provenance[row]
        if column is None:
            return list(prov.cells)
        col = column if isinstance(column, str) else self.dataset.column_names[column]
        return prov.citations_for(col)

    def cite_refs(self, row: int, column: str | int | None = None) -> list[str]:
        """The distinct stable cell locators (``table#r<row>!<col>``) a result cell
        (or whole result row) rests on."""
        seen: set[str] = set()
        out: list[str] = []
        for c in self.citations(row, column):
            if c.ref not in seen:
                seen.add(c.ref)
                out.append(c.ref)
        return out

    # -- verification ----------------------------------------------------------

    def verify(self, catalog: DataCatalog, *, engine: QueryEngine | None = None) -> bool:
        """Re-run the plan against *catalog* and confirm the result, the source
        hashes, and every cited cell re-derive from the bytes. Returns ``False`` on
        any divergence (a tampered result, a tampered source, or a flipped cell)."""
        for table, expected in self.source_hashes.items():
            if catalog.content_hashes().get(table) != expected:
                return False
        if self._compute_hash() != self.result_hash:
            return False
        try:
            if self.plan.dialect is QueryDialect.DATAFRAME:
                replay = _run_dataframe(self.plan, catalog)
            else:
                executed = (engine or InProcessSqlEngine()).execute(
                    self.plan.sql, catalog, max_rows=self.plan.max_rows
                )
                replay = QueryResult.build(self.plan, catalog, executed)
        except QueryError:
            return False
        if replay.dataset.rows() != self.dataset.rows() or replay.columns != self.columns:
            return False
        # every cited source cell still holds the value it was bound to
        for prov in self.provenance:
            for cell in prov.cells:
                if catalog.tables.get(cell.table) is None:
                    return False
                if _source_cell(catalog.get(cell.table), cell.row, cell.column) != cell.value:
                    return False
        return True

    # -- projection ------------------------------------------------------------

    def to_evidence(self, *, source_id: str = "", caption: str = "", **kwargs: Any) -> TableEvidence:
        """Project the result table into cited ``modality="table"`` evidence the
        context compiler scores, budgets, orders, and cites."""
        ev = self.dataset.to_evidence(
            source_id=source_id or self.dataset.name or "query_result", caption=caption, **kwargs
        )
        ev.metadata = {
            **ev.metadata,
            "query": self.plan.sql or "dataframe",
            "result_hash": self.result_hash,
            "lineage_coverage": str(self.coverage),
            "source_tables": self.plan.tables,
        }
        return ev


# --------------------------------------------------------------------------- #
# Dataframe-op dialect (intrinsically read-only, always cell-exact)           #
# --------------------------------------------------------------------------- #


def _run_dataframe(plan: QueryPlan, catalog: DataCatalog) -> QueryResult:
    """Execute a dataframe-op plan over a single table with exact per-cell lineage.

    Reuses the whitelisted, ``eval``-free :class:`~vincio.verify.ProgramOp`
    transforms (select / filter / derive / rename), which are read-only by
    construction, and threads each output row's source index so every output cell
    cites the exact source cell it rests on."""
    if len(plan.tables) != 1:
        raise QueryError("the dataframe dialect runs over exactly one registered table")
    table = plan.tables[0]
    ds = catalog.get(table)
    rows: list[tuple[int, dict[str, Any]]] = list(enumerate(ds.records()))
    # Each output column tracks the source columns it rests on (a projection keeps
    # one; a derive references however many its expression names).
    col_sources: dict[str, list[str]] = {c: [c] for c in ds.column_names}
    for op in plan.ops:
        rows, col_sources = _apply_op(op, rows, col_sources, ds.column_names)
    columns = list(rows[0][1].keys()) if rows else list(col_sources)
    out_rows = [[rec.get(c) for c in columns] for _, rec in rows]
    provenance: list[RowProvenance] = []
    for result_row, (src_index, _) in enumerate(rows):
        cells: list[CellCitation] = []
        for c in columns:
            for base in col_sources.get(c, []):
                cells.append(
                    CellCitation(
                        table=table,
                        row=src_index,
                        column=base,
                        value=_source_cell(ds, src_index, base),
                        result_column=c,
                    )
                )
        provenance.append(RowProvenance(result_row=result_row, cells=_dedup_cells(cells), exact=True))
    executed = ExecutedRows(
        columns=columns, rows=out_rows, provenance=provenance, coverage=LineageCoverage.CELL
    )
    return QueryResult.build(plan, catalog, executed)


def _apply_op(
    op: ProgramOp,
    rows: list[tuple[int, dict[str, Any]]],
    col_sources: dict[str, list[str]],
    source_columns: list[str],
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, list[str]]]:
    transformed: list[tuple[int, dict[str, Any]]] = []
    for idx, rec in rows:
        out = op.apply([rec])  # filter may drop the row; other ops return exactly one
        if out:
            transformed.append((idx, out[0]))
    if op.op == "select":
        new_sources = {k: col_sources.get(k, []) for k in op.fields if k in col_sources}
    elif op.op == "rename":
        new_sources = {op.mapping.get(k, k): v for k, v in col_sources.items()}
    elif op.op == "derive":
        idents: list[str] = []
        for ident in _IDENT_RE.findall(op.expr):
            if ident in source_columns and ident not in idents:
                idents.append(ident)
        new_sources = dict(col_sources)
        new_sources[op.field] = idents
    else:  # filter — columns unchanged
        new_sources = dict(col_sources)
    return transformed, new_sources


def _dedup_cells(cells: list[CellCitation]) -> list[CellCitation]:
    """De-duplicate cell citations by (table, row, column, result_column),
    preserving order — the same source cell may legitimately support more than one
    output column."""
    seen: set[tuple[str, int, str, str]] = set()
    out: list[CellCitation] = []
    for cell in cells:
        key = (cell.table, cell.row, cell.column, cell.result_column)
        if key not in seen:
            seen.add(key)
            out.append(cell)
    return out


# --------------------------------------------------------------------------- #
# Result dataset assembly                                                      #
# --------------------------------------------------------------------------- #


def _result_dataset(plan: QueryPlan, executed: ExecutedRows) -> Dataset:
    columns = executed.columns
    cells = [[row[j] if j < len(row) else None for row in executed.rows] for j in range(len(columns))]
    schema_cols = [
        ColumnSchema(
            name=name,
            dtype=DataType(tabular.infer_dtype(cells[j])),
            nullable=any(v is None for v in cells[j]),
        )
        for j, name in enumerate(columns)
    ]
    name = (plan.tables[0] + "_query") if plan.tables else "query_result"
    return Dataset(name=name, columns=schema_cols, cells=cells)


def _normalize_sql(sql: str) -> str:
    """Whitespace-normalize SQL for stable content binding (semantics preserved)."""
    return " ".join(_strip_sql_literals(sql).split()).lower()


# --------------------------------------------------------------------------- #
# Tool contract (the read-only guarantee as an enforced boundary)             #
# --------------------------------------------------------------------------- #


def make_query_contract(*, max_rows: int = 10_000) -> Any:
    """A :class:`~vincio.verify.ToolContract` that refuses a non-read-only query and
    bounds the result row count — so a ``query_data`` tool **structurally** refuses
    a write or DDL when it rides the permissioned tool runtime."""
    from ..verify.programs import ToolContract

    contract = ToolContract()
    contract.requires_that(
        "query is provably read-only (a single SELECT, no write/DDL/stacked statement)",
        lambda args: is_read_only_sql(str(args.get("sql", args.get("query", "")))),
    )
    contract.ensures_that(
        f"result is bounded to {max_rows} rows",
        lambda args, result: _result_row_count(result) <= max_rows,
    )
    return contract


def _result_row_count(result: Any) -> int:
    if isinstance(result, QueryResult):
        return result.row_count
    if isinstance(result, Dataset):
        return result.row_count
    if isinstance(result, (list, tuple)):
        return len(result)
    return 0


# --------------------------------------------------------------------------- #
# Heuristic offline planner (deterministic NL → SQL for common shapes)        #
# --------------------------------------------------------------------------- #


class HeuristicQueryPlanner:
    """A small, deterministic natural-language-to-SQL planner for the canonical
    analyst questions, used offline where no model is configured.

    It grounds against the catalog's schema and handles counts, single-column
    aggregates, and group-by aggregates (``"total revenue by region"``). It is
    intentionally bounded: a question it cannot ground confidently returns
    ``None``, so the caller falls back to an explicit query or the model planner —
    it never guesses an ungrounded query."""

    _AGGREGATES = (
        ("average", "AVG"),
        ("avg", "AVG"),
        ("mean", "AVG"),
        ("total", "SUM"),
        ("sum", "SUM"),
        ("maximum", "MAX"),
        ("max", "MAX"),
        ("largest", "MAX"),
        ("highest", "MAX"),
        ("minimum", "MIN"),
        ("min", "MIN"),
        ("smallest", "MIN"),
        ("lowest", "MIN"),
    )

    def plan(self, question: str, catalog: DataCatalog, *, table: str | None = None) -> str | None:
        """Ground *question* to a read-only ``SELECT`` over *table* (or the single
        registered table), or ``None`` when it cannot be grounded confidently."""
        name = table or (catalog.names[0] if len(catalog.names) == 1 else None)
        if name is None:
            return None
        ds = catalog.get(name)
        columns = ds.column_names
        words = re.findall(r"[a-z0-9_]+", question.lower())
        wordset = set(words)
        numeric = [c.name for c in ds.columns if c.dtype in (DataType.INT, DataType.FLOAT)]
        mentioned = [c for c in columns if c.lower() in wordset]
        group_cols = [c for c in mentioned if c not in numeric]
        agg = next((fn for kw, fn in self._AGGREGATES if kw in wordset), None)
        is_count = "count" in wordset or "how many" in question.lower() or "number of" in question.lower()
        quoted = _quote_ident(name)

        if agg:
            measure = next((c for c in mentioned if c in numeric), None) or (numeric[0] if numeric else None)
            if measure is None:
                return None
            if group_cols:
                g = group_cols[0]
                return (
                    f"SELECT {_quote_ident(g)}, {agg}({_quote_ident(measure)}) AS {agg.lower()}_{measure} "
                    f"FROM {quoted} GROUP BY {_quote_ident(g)} ORDER BY {_quote_ident(g)}"
                )
            return f"SELECT {agg}({_quote_ident(measure)}) AS {agg.lower()}_{measure} FROM {quoted}"

        if is_count:
            if group_cols:
                g = group_cols[0]
                return (
                    f"SELECT {_quote_ident(g)}, COUNT(*) AS count FROM {quoted} "
                    f"GROUP BY {_quote_ident(g)} ORDER BY {_quote_ident(g)}"
                )
            return f"SELECT COUNT(*) AS count FROM {quoted}"

        return None


# --------------------------------------------------------------------------- #
# One-shot pipeline                                                           #
# --------------------------------------------------------------------------- #


def query_dataset(
    request: str,
    data: Dataset | DataCatalog | dict[str, Dataset],
    *,
    dialect: QueryDialect | str = QueryDialect.SQL,
    question: str = "",
    ops: Iterable[Any] | None = None,
    table: str | None = None,
    max_rows: int = 10_000,
    engine: QueryEngine | None = None,
    injection_detector: InjectionDetector | None = None,
    screen_question: bool = True,
) -> QueryResult:
    """Plan → verify → execute → cite, in one call.

    *request* is either the SQL to run, or (when ``question`` is set or *request*
    is plainly a question) a natural-language question the offline
    :class:`HeuristicQueryPlanner` grounds to SQL. *data* is a single
    :class:`~vincio.data.Dataset`, a mapping of name→dataset, or a
    :class:`DataCatalog`. The natural-language question is screened for injection
    (the same detector the text rails use), the query is verified read-only and
    grounded, and the result is returned cell-level cited and offline-verifiable.
    """
    catalog = _as_catalog(data, table=table)
    dialect = QueryDialect(dialect)

    if dialect is QueryDialect.DATAFRAME:
        if ops is None:
            raise QueryError("the dataframe dialect requires ops=[...] (ProgramOp pipeline)")
        name = table or (catalog.names[0] if len(catalog.names) == 1 else None)
        if name is None:
            raise QueryError("specify table= when the catalog has more than one table")
        if screen_question and question:
            _screen_question(question, injection_detector)
        plan = QueryPlan(
            question=question, dialect=QueryDialect.DATAFRAME, ops=list(ops), tables=[name], max_rows=max_rows
        )
        return plan.run(catalog)

    # SQL dialect: an explicit SQL statement (any leading statement keyword) is
    # taken verbatim and verified read-only (a write/DDL is refused as unsafe); a
    # natural-language request is screened and grounded by the offline planner.
    nl = ""
    if question:
        nl = question
    elif not _is_sql_statement(request):
        nl = request

    if nl:
        if screen_question:
            _screen_question(nl, injection_detector)
        planned = HeuristicQueryPlanner().plan(nl, catalog, table=table)
        if planned is None:
            raise QueryError(
                "could not ground the question to a query offline; pass explicit SQL "
                "or configure a model planner via app.query_data"
            )
        sql = planned
    else:
        sql = request

    plan = QueryPlan.for_sql(sql, catalog, question=nl, max_rows=max_rows, engine=engine)
    return plan.run(catalog, engine=engine)


def _as_catalog(data: Dataset | DataCatalog | dict[str, Dataset], *, table: str | None) -> DataCatalog:
    if isinstance(data, DataCatalog):
        return data
    if isinstance(data, Dataset):
        return DataCatalog.of(data, name=table or data.name or "data")
    if isinstance(data, dict):
        return DataCatalog(data)
    raise QueryError(f"cannot build a catalog from {type(data).__name__}")


# Leading keywords that mark a string as an explicit SQL statement (rather than a
# natural-language question). A non-read-only statement here is taken verbatim and
# refused by the read-only guard, not silently routed to the NL planner.
_SQL_STATEMENT_KEYWORDS = _LEADING_KEYWORDS | _WRITE_KEYWORDS | {"explain", "values", "table"}


def _is_sql_statement(text: str) -> bool:
    tokens = _IDENT_RE.findall(text.strip())
    return bool(tokens) and tokens[0].lower() in _SQL_STATEMENT_KEYWORDS


def _screen_question(question: str, detector: InjectionDetector | None) -> None:
    if detector is None:
        from ..security.injection import InjectionDetector as _Det

        detector = _Det()
    if detector.detect(question).detected:
        raise UnsafeQueryError(
            "question refused: a prompt-injection signal was detected in the "
            "natural-language question before it became a query"
        )
