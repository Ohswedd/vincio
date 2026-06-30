"""The semantic layer: measures, dimensions, and derived columns defined once.

An analyst's "revenue" is ``price × qty`` summed — but a question that asks for it
by name should never have to *re-derive* it, and two questions that ask for it two
ways must never compute it two ways. This module defines the analytical vocabulary
**once** so a natural-language question maps to a **governed metric** rather than a
raw column, and that metric is compiled to SQL one way everywhere — the
single-source discipline the governance refactors already use, brought to the data
plane.

* :class:`DerivedColumn` — a row-level calculation declared once
  (``revenue = price × qty``) that measures and dimensions reference by name.
* :class:`Dimension` — a groupable attribute (a column, or an expression like a
  month bucket) a metric is broken down by.
* :class:`Measure` — an **aggregated, governed metric**: a ``SUM`` / ``AVG`` /
  ``COUNT`` (… ) over a column or a derived column, an optional row filter, or a
  **ratio** of two other measures (``avg_order_value = revenue ÷ orders``).
* :class:`SemanticLayer` — the named set of those definitions over one registered
  table. It **compiles** a :class:`MetricQuery` to a single read-only ``SELECT``
  and **runs** it through the *existing* governed query plane, so the answer is
  cell-level cited and offline-verifiable like any other query — never a parallel
  stack.
* :class:`MetricResult` — the governed answer. :meth:`MetricResult.verify` proves
  three things from the bytes alone: the layer's definitions are unchanged, the SQL
  that ran is the layer's **canonical** compilation of the metric (an ad-hoc number
  cannot pass as the governed one), and the result re-derives from the hashed
  source.
* :class:`MetricLineage` — a metric's **column-level provenance**: the base columns
  and source table it rests on, resolving the derived-column graph and any ratio
  references — the link the governance lineage and right-to-erasure machinery
  follow into the dataset plane.

Everything here is deterministic, dependency-free, and offline. The compiled query
is always re-screened read-only and dry-run-grounded by
:meth:`~vincio.data.QueryPlan.for_sql`, so the semantic layer can never smuggle a
write past the read-only guard.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.errors import SemanticLayerError
from ..core.utils import stable_hash
from .core import Dataset, DataType
from .provenance import LineageCoverage
from .query import (
    DataCatalog,
    QueryEngine,
    QueryPlan,
    QueryResult,
    _screen_question,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.injection import InjectionDetector

__all__ = [
    "Aggregation",
    "DerivedColumn",
    "Dimension",
    "Measure",
    "MetricQuery",
    "MetricResult",
    "MetricLineage",
    "SemanticLayer",
    "query_metric",
]


_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_SIMPLE_IDENT_RE = re.compile(r"[A-Za-z_]\w*\Z")
# A single-quoted SQL string literal (an embedded quote doubled) — its contents are
# data, never an identifier, so derived-column substitution must skip it.
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")

# Identifiers that appear in a resolved metric expression but name SQL machinery,
# not source columns — excluded when deriving a metric's column-level lineage
# without a catalog to ground against.
_SQL_NONCOLUMN_TOKENS = frozenset(
    {
        "sum", "avg", "min", "max", "count", "total", "distinct", "as", "case",
        "when", "then", "else", "end", "and", "or", "not", "null", "is", "in",
        "like", "between", "cast", "real", "integer", "text", "nullif", "coalesce",
        "abs", "round", "strftime", "date", "datetime", "substr", "length", "true",
        "false",
    }
)


class Aggregation(StrEnum):
    """The closed set of aggregations a :class:`Measure` may declare. Each maps to
    a standard SQL aggregate the offline ``sqlite3`` engine computes; a ratio
    measure declares no aggregation and divides two other measures instead."""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"


_SQL_FN = {
    Aggregation.SUM: "SUM",
    Aggregation.AVG: "AVG",
    Aggregation.MIN: "MIN",
    Aggregation.MAX: "MAX",
    Aggregation.COUNT: "COUNT",
    Aggregation.COUNT_DISTINCT: "COUNT",
}


def _q(name: str) -> str:
    """Double-quote a SQL identifier (doubling any embedded quote)."""
    return '"' + name.replace('"', '""') + '"'


def _require_identifier(name: str, *, what: str) -> str:
    if not name or not _SIMPLE_IDENT_RE.match(name):
        raise SemanticLayerError(
            f"{what} name {name!r} is not a simple identifier (letters, digits, "
            "and underscores; not starting with a digit)"
        )
    return name


def _reject_statement_break(expression: str, *, what: str) -> str:
    """Refuse an expression that could break out of its clause. A semicolon (a
    stacked statement) or an unbalanced parenthesis is rejected early with a clear
    error; every compiled query is *additionally* re-screened read-only, so this is
    a friendlier first line, not the only one."""
    if ";" in expression:
        raise SemanticLayerError(
            f"{what} expression may not contain ';' (a stacked statement): {expression!r}"
        )
    if expression.count("(") != expression.count(")"):
        raise SemanticLayerError(
            f"{what} expression has unbalanced parentheses: {expression!r}"
        )
    return expression


class DerivedColumn(BaseModel):
    """A row-level calculation declared once and referenced by name.

    ``revenue = price * qty`` is a derived column: its ``expression`` is a SQL
    fragment over the table's base columns (and other derived columns, which
    compose), substituted in wherever a measure or dimension names ``revenue``. The
    optional ``dtype`` / ``unit`` are documentation for the metric catalog; the
    expression is the source of truth."""

    name: str
    expression: str
    dtype: DataType = DataType.FLOAT
    unit: str | None = None
    description: str = ""

    @model_validator(mode="after")
    def _validate(self) -> DerivedColumn:
        _require_identifier(self.name, what="derived column")
        if not self.expression.strip():
            raise SemanticLayerError(f"derived column {self.name!r} has an empty expression")
        _reject_statement_break(self.expression, what="derived column")
        return self


class Dimension(BaseModel):
    """A groupable attribute a metric is broken down by.

    A plain dimension is a column (``region``); an expression dimension buckets one
    (``order_month = strftime('%Y-%m', order_date)``). ``synonyms`` let a
    natural-language question ground to the dimension by a friendly word. A plain
    dimension carries cell-exact lineage through the query plane; an expression
    dimension reports result-level lineage (always stated, never silently
    downgraded)."""

    name: str
    expression: str = ""
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> Dimension:
        _require_identifier(self.name, what="dimension")
        if self.expression:
            _reject_statement_break(self.expression, what="dimension")
        return self

    @property
    def is_plain(self) -> bool:
        """Whether the dimension is a bare column (no derived expression)."""
        return not self.expression


class Measure(BaseModel):
    """An aggregated, governed metric — the unit a question maps to.

    A measure is either an **aggregation** (``agg`` over ``expression``, e.g.
    ``SUM(revenue)``; ``COUNT`` defaults its expression to ``*``) optionally
    narrowed by ``filters`` (row predicates folded into a ``CASE``), or a **ratio**
    (``numerator`` ÷ ``denominator``, naming two other measures) for a governed
    rate like average order value. Exactly one of the two forms is declared.
    ``synonyms`` ground a natural-language question to the metric by a friendly
    word."""

    name: str
    agg: Aggregation | None = None
    expression: str = ""
    numerator: str = ""
    denominator: str = ""
    filters: list[str] = Field(default_factory=list)
    unit: str | None = None
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> Measure:
        _require_identifier(self.name, what="measure")
        if self.is_ratio:
            if self.agg is not None or self.expression:
                raise SemanticLayerError(
                    f"ratio measure {self.name!r} declares both a ratio "
                    "(numerator/denominator) and an aggregation; declare one form"
                )
            _require_identifier(self.numerator, what="ratio numerator")
            _require_identifier(self.denominator, what="ratio denominator")
        else:
            if self.numerator or self.denominator:
                raise SemanticLayerError(
                    f"measure {self.name!r} declares only one of numerator/denominator; "
                    "a ratio needs both"
                )
            if self.agg is None:
                raise SemanticLayerError(
                    f"measure {self.name!r} declares no aggregation and is not a ratio; "
                    "set agg= or numerator=/denominator="
                )
            if self.agg is not Aggregation.COUNT and not self.expression.strip():
                raise SemanticLayerError(
                    f"measure {self.name!r} ({self.agg}) needs an expression to aggregate"
                )
            if self.expression:
                _reject_statement_break(self.expression, what="measure")
        for predicate in self.filters:
            if not predicate.strip():
                raise SemanticLayerError(f"measure {self.name!r} has an empty filter")
            _reject_statement_break(predicate, what="measure filter")
        return self

    @property
    def is_ratio(self) -> bool:
        """Whether the measure is a ratio of two other measures."""
        return bool(self.numerator and self.denominator)


class MetricQuery(BaseModel):
    """A request for one or more governed metrics, optionally broken down.

    ``metrics`` and ``dimensions`` name measures and dimensions the layer defines;
    ``filters`` are read-only predicate fragments ANDed into the ``WHERE`` clause.
    The layer compiles this to exactly one read-only ``SELECT`` — the single source
    of truth for how the metric is computed."""

    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    order_by: str = ""
    descending: bool = False
    limit: int | None = None

    @model_validator(mode="after")
    def _validate(self) -> MetricQuery:
        if not self.metrics:
            raise SemanticLayerError("a metric query must request at least one metric")
        for predicate in self.filters:
            _reject_statement_break(predicate, what="query filter")
        return self


class MetricLineage(BaseModel):
    """A metric's column-level provenance — where the number comes from.

    Resolves the derived-column graph and any ratio references down to the **base
    columns** of the source table the metric ultimately rests on, plus the derived
    columns and underlying measures traversed and the governed aggregate SQL. The
    governance lineage index fills ``source`` with the registered source the table
    was ingested under, so a metric's provenance reaches the same right-to-erasure
    machinery a document's does."""

    metric: str
    table: str
    source: str = ""
    base_columns: list[str] = Field(default_factory=list)
    derived_via: list[str] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=list)
    expression_sql: str = ""


class SemanticLayer(BaseModel):
    """Measures, dimensions, and derived columns defined once over one table.

    Build it declaratively or with the chaining :meth:`add_derived` /
    :meth:`add_dimension` / :meth:`add_measure` helpers, then :meth:`query` a metric
    by name or natural-language question. The metric compiles to a single read-only
    ``SELECT`` (the same one every time, the single-source discipline) executed by
    the *existing* governed query plane, so the answer is cell-level cited and
    offline-verifiable. :meth:`column_lineage` reports a metric's base columns and
    :meth:`digest` content-binds the definitions so a :class:`MetricResult` can
    prove it was computed the governed way."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    table: str
    name: str = ""
    description: str = ""
    derived: list[DerivedColumn] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> SemanticLayer:
        self._revalidate()
        return self

    def _revalidate(self) -> None:
        """Re-check the layer's invariants (unique names across one namespace, an
        acyclic derived-column graph, and well-formed ratio references) — run on
        construction and after every builder mutation."""
        _require_identifier(self.table, what="table")
        names: dict[str, str] = {}
        for kind, items in (
            ("derived column", self.derived),
            ("dimension", self.dimensions),
            ("measure", self.measures),
        ):
            for item in items:
                clash = names.get(item.name)
                if clash is not None:
                    raise SemanticLayerError(
                        f"name {item.name!r} is declared twice ({clash} and {kind}); "
                        "derived columns, dimensions, and measures share one namespace"
                    )
                names[item.name] = kind
        self._check_derived_acyclic()
        self._check_ratio_refs()

    # -- builders --------------------------------------------------------------

    def add_derived(
        self,
        name: str,
        expression: str,
        *,
        dtype: DataType | str = DataType.FLOAT,
        unit: str | None = None,
        description: str = "",
    ) -> SemanticLayer:
        """Declare a row-level derived column (``revenue = price * qty``) and return
        the layer for chaining."""
        self.derived.append(
            DerivedColumn(
                name=name,
                expression=expression,
                dtype=DataType(dtype),
                unit=unit,
                description=description,
            )
        )
        self._revalidate()
        return self

    def add_dimension(
        self,
        name: str,
        *,
        expression: str = "",
        description: str = "",
        synonyms: Iterable[str] = (),
    ) -> SemanticLayer:
        """Declare a groupable dimension and return the layer for chaining."""
        self.dimensions.append(
            Dimension(
                name=name, expression=expression, description=description, synonyms=list(synonyms)
            )
        )
        self._revalidate()
        return self

    def add_measure(
        self,
        name: str,
        agg: Aggregation | str | None = None,
        expression: str = "",
        *,
        numerator: str = "",
        denominator: str = "",
        filters: Iterable[str] = (),
        unit: str | None = None,
        description: str = "",
        synonyms: Iterable[str] = (),
    ) -> SemanticLayer:
        """Declare an aggregated metric (or a ratio of two metrics) and return the
        layer for chaining."""
        self.measures.append(
            Measure(
                name=name,
                agg=Aggregation(agg) if agg is not None else None,
                expression=expression,
                numerator=numerator,
                denominator=denominator,
                filters=list(filters),
                unit=unit,
                description=description,
                synonyms=list(synonyms),
            )
        )
        self._revalidate()
        return self

    # -- lookups ---------------------------------------------------------------

    @property
    def metric_names(self) -> list[str]:
        """The names of every governed metric (measure) the layer defines."""
        return [m.name for m in self.measures]

    @property
    def dimension_names(self) -> list[str]:
        """The names of every dimension the layer defines."""
        return [d.name for d in self.dimensions]

    def measure(self, name: str) -> Measure:
        """The named measure, or raise :class:`~vincio.core.errors.SemanticLayerError`."""
        for m in self.measures:
            if m.name == name:
                return m
        raise SemanticLayerError(
            f"no measure named {name!r}; defined metrics: {self.metric_names}"
        )

    def dimension(self, name: str) -> Dimension:
        """The named dimension, or raise :class:`~vincio.core.errors.SemanticLayerError`."""
        for d in self.dimensions:
            if d.name == name:
                return d
        raise SemanticLayerError(
            f"no dimension named {name!r}; defined dimensions: {self.dimension_names}"
        )

    def _derived(self, name: str) -> DerivedColumn | None:
        for d in self.derived:
            if d.name == name:
                return d
        return None

    # -- definition checks -----------------------------------------------------

    def _check_derived_acyclic(self) -> None:
        derived = {d.name: d for d in self.derived}

        def walk(name: str, stack: tuple[str, ...]) -> None:
            if name in stack:
                cycle = " → ".join((*stack[stack.index(name) :], name))
                raise SemanticLayerError(f"derived columns form a cycle: {cycle}")
            col = derived.get(name)
            if col is None:
                return
            for ident in _IDENT_RE.findall(col.expression):
                if ident in derived:
                    walk(ident, (*stack, name))

        for d in self.derived:
            walk(d.name, ())

    def _check_ratio_refs(self) -> None:
        by_name = {m.name: m for m in self.measures}

        def walk(name: str, stack: tuple[str, ...]) -> None:
            if name in stack:
                cycle = " → ".join((*stack[stack.index(name) :], name))
                raise SemanticLayerError(f"ratio measures form a cycle: {cycle}")
            m = by_name.get(name)
            if m is None:
                raise SemanticLayerError(
                    f"ratio measure {stack[-1]!r} references unknown measure {name!r}"
                )
            if m.is_ratio:
                walk(m.numerator, (*stack, name))
                walk(m.denominator, (*stack, name))

        for m in self.measures:
            if m.is_ratio:
                walk(m.numerator, (m.name,))
                walk(m.denominator, (m.name,))

    # -- expression resolution -------------------------------------------------

    def _resolve_expr(self, expression: str, *, _stack: tuple[str, ...] = ()) -> str:
        """Substitute every derived-column reference in *expression* with its own
        (recursively resolved) parenthesized SQL, leaving base-column references as
        bare identifiers the engine resolves. String literals are copied verbatim, so
        a literal whose text equals a derived-column name is never rewritten."""

        def repl(match: re.Match[str]) -> str:
            ident = match.group(0)
            col = self._derived(ident)
            if col is None or ident in _stack:
                return ident
            return "(" + self._resolve_expr(col.expression, _stack=(*_stack, ident)) + ")"

        out: list[str] = []
        pos = 0
        for literal in _STRING_LITERAL_RE.finditer(expression):
            out.append(_IDENT_RE.sub(repl, expression[pos : literal.start()]))
            out.append(literal.group(0))  # literal text is data, copied verbatim
            pos = literal.end()
        out.append(_IDENT_RE.sub(repl, expression[pos:]))
        return "".join(out)

    def _measure_sql(self, measure: Measure) -> str:
        """The governed SQL expression for one measure (no alias)."""
        if measure.is_ratio:
            num = self._measure_sql(self.measure(measure.numerator))
            den = self._measure_sql(self.measure(measure.denominator))
            # NULLIF guards a zero denominator (the ratio is NULL, never a crash).
            return f"(CAST({num} AS REAL) / NULLIF({den}, 0))"
        assert measure.agg is not None  # noqa: S101 - a non-ratio measure always has an agg (model validator)
        fn = _SQL_FN[measure.agg]
        if measure.agg is Aggregation.COUNT and not measure.expression.strip():
            inner = "*"
        else:
            inner = self._resolve_expr(measure.expression)
        if measure.filters:
            predicate = " AND ".join(f"({self._resolve_expr(p)})" for p in measure.filters)
            kept = "1" if inner == "*" else inner
            inner = f"CASE WHEN {predicate} THEN {kept} ELSE NULL END"
        if measure.agg is Aggregation.COUNT_DISTINCT:
            return f"COUNT(DISTINCT {inner})"
        return f"{fn}({inner})"

    def _dimension_sql(self, dimension: Dimension) -> str:
        """The governed SQL expression for one dimension (no alias)."""
        if dimension.is_plain:
            return _q(dimension.name)
        return self._resolve_expr(dimension.expression)

    # -- compilation -----------------------------------------------------------

    def compile(self, query: MetricQuery) -> str:
        """Compile a :class:`MetricQuery` to the single read-only ``SELECT`` that
        computes it — the one canonical SQL for the metric, every time."""
        measures = [self.measure(name) for name in query.metrics]
        dimensions = [self.dimension(name) for name in query.dimensions]
        select_parts: list[str] = []
        for dim in dimensions:
            select_parts.append(f"{self._dimension_sql(dim)} AS {_q(dim.name)}")
        for m in measures:
            select_parts.append(f"{self._measure_sql(m)} AS {_q(m.name)}")
        sql = f"SELECT {', '.join(select_parts)} FROM {_q(self.table)}"
        if query.filters:
            where = " AND ".join(f"({self._resolve_expr(p)})" for p in query.filters)
            sql += f" WHERE {where}"
        if dimensions:
            group = ", ".join(self._dimension_sql(d) for d in dimensions)
            sql += f" GROUP BY {group}"
        order = self._order_clause(query, measures, dimensions)
        if order:
            sql += f" ORDER BY {order}"
        if query.limit is not None:
            if query.limit < 0:
                raise SemanticLayerError("limit must be non-negative")
            sql += f" LIMIT {int(query.limit)}"
        return sql

    def _order_clause(
        self, query: MetricQuery, measures: list[Measure], dimensions: list[Dimension]
    ) -> str:
        selected = {m.name for m in measures} | {d.name for d in dimensions}
        if query.order_by:
            if query.order_by not in selected:
                raise SemanticLayerError(
                    f"order_by {query.order_by!r} is not one of the selected metrics or "
                    f"dimensions {sorted(selected)}"
                )
            key = _q(query.order_by)
        elif dimensions:
            key = _q(dimensions[0].name)
        else:
            return ""
        return f"{key} DESC" if query.descending else key

    # -- grounding & execution -------------------------------------------------

    def resolve(self, question: str) -> MetricQuery | None:
        """Ground a natural-language *question* to a governed :class:`MetricQuery`,
        or ``None`` when no defined metric is mentioned.

        Deterministic and bounded — the governed analogue of the offline
        :class:`~vincio.data.HeuristicQueryPlanner`: it never guesses an ungrounded
        metric. A metric (or dimension) is mentioned when its name, its
        space-separated form, or one of its synonyms appears in the question."""
        text = " " + " ".join(re.findall(r"[a-z0-9_]+", question.lower())) + " "

        def mentioned(name: str, synonyms: Sequence[str]) -> bool:
            for token in (name, name.replace("_", " "), *synonyms):
                token = token.lower().strip()
                if token and f" {token} " in text:
                    return True
            return False

        metrics = [m.name for m in self.measures if mentioned(m.name, m.synonyms)]
        if not metrics:
            return None
        dimensions = [d.name for d in self.dimensions if mentioned(d.name, d.synonyms)]
        return MetricQuery(metrics=metrics, dimensions=dimensions)

    def build_query(
        self,
        request: str | MetricQuery | Sequence[str],
        *,
        by: Sequence[str] | None = None,
        where: Sequence[str] | None = None,
        order_by: str = "",
        descending: bool = False,
        limit: int | None = None,
        injection_detector: InjectionDetector | None = None,
        screen: bool = True,
    ) -> MetricQuery:
        """Resolve *request* (a metric name, a list of metric names, a
        :class:`MetricQuery`, or a natural-language question) into a concrete
        :class:`MetricQuery`. A natural-language question is injection-screened
        before it grounds, the same way the query plane screens a question."""
        if isinstance(request, MetricQuery):
            return request
        if isinstance(request, str) and request in {m.name for m in self.measures}:
            metrics: list[str] = [request]
        elif not isinstance(request, str):
            metrics = list(request)
        else:
            if screen:
                _screen_question(request, injection_detector)
            grounded = self.resolve(request)
            if grounded is None:
                raise SemanticLayerError(
                    f"could not ground {request!r} to a defined metric; defined "
                    f"metrics: {self.metric_names}"
                )
            # An explicit by=/order_by= refines the grounded breakdown.
            if by is None and grounded.dimensions:
                by = grounded.dimensions
            metrics = grounded.metrics
        return MetricQuery(
            metrics=metrics,
            dimensions=list(by or []),
            filters=list(where or []),
            order_by=order_by,
            descending=descending,
            limit=limit,
        )

    def run(
        self,
        query: MetricQuery,
        data: Dataset | DataCatalog | dict[str, Dataset],
        *,
        engine: QueryEngine | None = None,
        max_rows: int = 10_000,
    ) -> MetricResult:
        """Compile *query* to the governed SQL and execute it through the existing
        read-only-verified query plane, returning a cited, verifiable
        :class:`MetricResult`."""
        catalog = _as_catalog(data, table=self.table)
        sql = self.compile(query)
        plan = QueryPlan.for_sql(sql, catalog, max_rows=max_rows, engine=engine)
        result = plan.run(catalog, engine=engine)
        return MetricResult(spec=query, result=result, layer_hash=self.digest())

    def query(
        self,
        request: str | MetricQuery | Sequence[str],
        data: Dataset | DataCatalog | dict[str, Dataset],
        *,
        by: Sequence[str] | None = None,
        where: Sequence[str] | None = None,
        order_by: str = "",
        descending: bool = False,
        limit: int | None = None,
        engine: QueryEngine | None = None,
        max_rows: int = 10_000,
        injection_detector: InjectionDetector | None = None,
        screen: bool = True,
    ) -> MetricResult:
        """Resolve *request*, compile the governed SQL, and run it — plan → verify →
        execute → cite, in one call, over *data*."""
        query = self.build_query(
            request,
            by=by,
            where=where,
            order_by=order_by,
            descending=descending,
            limit=limit,
            injection_detector=injection_detector,
            screen=screen,
        )
        return self.run(query, data, engine=engine, max_rows=max_rows)

    # -- lineage & identity ----------------------------------------------------

    def column_lineage(
        self, metric: str, *, catalog: DataCatalog | None = None
    ) -> MetricLineage:
        """The base columns and source table a metric rests on — its column-level
        provenance, resolving derived columns and ratio references.

        When *catalog* is given, the base columns are grounded to the table's real
        columns; otherwise SQL function names are excluded heuristically. The
        governance lineage index fills :attr:`MetricLineage.source`."""
        measure = self.measure(metric)
        derived_via: list[str] = []
        measures: list[str] = []
        self._gather(measure, derived_via, measures)
        expression_sql = self._measure_sql(measure)
        known = (
            {c.lower() for c in catalog.columns(self.table)} if catalog is not None else None
        )
        base: list[str] = []
        for ident in _IDENT_RE.findall(self._resolve_expr_for_lineage(measure)):
            lower = ident.lower()
            if lower in _SQL_NONCOLUMN_TOKENS or ident in {d.name for d in self.derived}:
                continue
            if known is not None and lower not in known:
                continue
            if ident not in base:
                base.append(ident)
        return MetricLineage(
            metric=metric,
            table=self.table,
            base_columns=base,
            derived_via=derived_via,
            measures=measures,
            expression_sql=expression_sql,
        )

    def _resolve_expr_for_lineage(self, measure: Measure) -> str:
        if measure.is_ratio:
            return " ".join(
                self._resolve_expr_for_lineage(self.measure(name))
                for name in (measure.numerator, measure.denominator)
            )
        parts = [self._resolve_expr(measure.expression)] if measure.expression else []
        parts.extend(self._resolve_expr(p) for p in measure.filters)
        return " ".join(parts)

    def _gather(self, measure: Measure, derived_via: list[str], measures: list[str]) -> None:
        if measure.name not in measures:
            measures.append(measure.name)
        if measure.is_ratio:
            self._gather(self.measure(measure.numerator), derived_via, measures)
            self._gather(self.measure(measure.denominator), derived_via, measures)
            return
        for expr in (measure.expression, *measure.filters):
            for ident in _IDENT_RE.findall(expr):
                col = self._derived(ident)
                if col is not None:
                    if ident not in derived_via:
                        derived_via.append(ident)
                    for nested in _IDENT_RE.findall(col.expression):
                        nested_col = self._derived(nested)
                        if nested_col is not None and nested not in derived_via:
                            derived_via.append(nested)

    def base_columns(self) -> list[str]:
        """Every base column any metric or dimension in the layer references — the
        column-level footprint the lineage index records for erasure."""
        seen: list[str] = []
        for measure in self.measures:
            for col in self.column_lineage(measure.name).base_columns:
                if col not in seen:
                    seen.append(col)
        for dim in self.dimensions:
            if dim.is_plain:
                if dim.name not in seen:
                    seen.append(dim.name)
            else:
                for ident in _IDENT_RE.findall(self._resolve_expr(dim.expression)):
                    if (
                        ident.lower() not in _SQL_NONCOLUMN_TOKENS
                        and ident not in {d.name for d in self.derived}
                        and ident not in seen
                    ):
                        seen.append(ident)
        return seen

    def validate_against(self, data: Dataset | DataCatalog | dict[str, Dataset]) -> None:
        """Ground every metric and dimension against real data: compile and dry-run
        each, raising :class:`~vincio.core.errors.SemanticLayerError` on the first
        that references an unknown column or fails to compile."""
        catalog = _as_catalog(data, table=self.table)
        if self.table not in catalog.tables:
            raise SemanticLayerError(
                f"layer table {self.table!r} is not registered; known tables: "
                f"{catalog.names}"
            )
        from ..core.errors import QueryError

        for measure in self.measures:
            try:
                QueryPlan.for_sql(
                    self.compile(MetricQuery(metrics=[measure.name])), catalog, max_rows=1
                )
            except QueryError as exc:
                raise SemanticLayerError(
                    f"measure {measure.name!r} does not ground to {self.table!r}: {exc}"
                ) from exc
        anchor = self.measures[0].name if self.measures else None
        for dim in self.dimensions:
            if anchor is None:
                break
            try:
                QueryPlan.for_sql(
                    self.compile(MetricQuery(metrics=[anchor], dimensions=[dim.name])),
                    catalog,
                    max_rows=1,
                )
            except QueryError as exc:
                raise SemanticLayerError(
                    f"dimension {dim.name!r} does not ground to {self.table!r}: {exc}"
                ) from exc

    def digest(self) -> str:
        """A stable content hash binding the layer's definitions, so a
        :class:`MetricResult` can prove it was computed by *these* definitions."""
        return stable_hash(
            [
                self.table,
                [d.model_dump(mode="json") for d in self.derived],
                [d.model_dump(mode="json") for d in self.dimensions],
                [m.model_dump(mode="json") for m in self.measures],
            ]
        )


class MetricResult(BaseModel):
    """A governed metric's answer — cited and provably computed the one way.

    Wraps the underlying cell-cited :class:`~vincio.data.QueryResult` with the
    :class:`MetricQuery` it answered and the layer's content hash.
    :meth:`verify` proves, from the bytes alone, that the layer's definitions are
    unchanged, that the SQL that ran is the layer's **canonical** compilation of the
    metric (an ad-hoc query cannot pass as the governed one), and that the result
    re-derives from the hashed source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: MetricQuery
    result: QueryResult
    layer_hash: str = ""

    @property
    def metrics(self) -> list[str]:
        """The governed metrics this result computed."""
        return list(self.spec.metrics)

    @property
    def dimensions(self) -> list[str]:
        """The dimensions the metrics were broken down by."""
        return list(self.spec.dimensions)

    @property
    def sql(self) -> str:
        """The governed SQL that computed the metric."""
        return self.result.plan.sql

    @property
    def rows(self) -> list[list[Any]]:
        """The result rows."""
        return self.result.rows

    @property
    def columns(self) -> list[str]:
        """The result column names (the dimensions then the metrics)."""
        return self.result.columns

    @property
    def row_count(self) -> int:
        """The number of result rows."""
        return self.result.row_count

    @property
    def coverage(self) -> LineageCoverage:
        """The lineage coverage of the underlying query result."""
        return self.result.coverage

    def value(self, row: int = 0, column: str | int | None = None) -> Any:
        """One result cell. ``column`` defaults to the **first metric** (so a grouped
        result returns the measure, not the leading dimension); pass a column name or
        index to address another."""
        if column is None:
            column = self.spec.metrics[0]
        return self.result.value(row, column)

    def cite_refs(self, row: int = 0, column: str | int | None = None) -> list[str]:
        """The exact source-cell locators a result cell (or row) rests on."""
        return self.result.cite_refs(row, column)

    def verify(
        self,
        layer: SemanticLayer,
        data: Dataset | DataCatalog | dict[str, Dataset],
        *,
        engine: QueryEngine | None = None,
    ) -> bool:
        """Prove the answer is the governed metric, re-derived from the bytes.

        Returns ``False`` unless the layer's definitions still hash to
        :attr:`layer_hash`, the SQL that ran equals the layer's canonical
        compilation of :attr:`spec`, and the underlying query result re-executes and
        re-derives every cited cell from the hashed source."""
        if layer.digest() != self.layer_hash:
            return False
        try:
            expected = layer.compile(self.spec)
        except SemanticLayerError:
            return False
        if _canon_sql(expected) != _canon_sql(self.result.plan.sql):
            return False
        catalog = _as_catalog(data, table=layer.table)
        return self.result.verify(catalog, engine=engine)


def _canon_sql(sql: str) -> str:
    """Whitespace-canonical SQL for comparing two compilations of the same metric."""
    return " ".join(sql.split())


def _as_catalog(
    data: Dataset | DataCatalog | dict[str, Dataset], *, table: str
) -> DataCatalog:
    if isinstance(data, DataCatalog):
        return data
    if isinstance(data, Dataset):
        return DataCatalog.of(data, name=table or data.name or "data")
    if isinstance(data, dict):
        return DataCatalog(data)
    raise SemanticLayerError(f"cannot build a catalog from {type(data).__name__}")


def query_metric(
    request: str | MetricQuery | Sequence[str],
    data: Dataset | DataCatalog | dict[str, Dataset],
    *,
    layer: SemanticLayer,
    by: Sequence[str] | None = None,
    where: Sequence[str] | None = None,
    order_by: str = "",
    descending: bool = False,
    limit: int | None = None,
    engine: QueryEngine | None = None,
    max_rows: int = 10_000,
    injection_detector: InjectionDetector | None = None,
    screen: bool = True,
) -> MetricResult:
    """Resolve a governed metric over *data* with *layer* and run it — the one-shot
    free function behind :meth:`SemanticLayer.query`.

    *request* is a metric name, a list of metric names, a :class:`MetricQuery`, or a
    natural-language question the layer grounds to a governed metric (the question
    is injection-screened first). The compiled query is verified read-only and
    executed by the offline query plane; the answer is cell-level cited and
    offline-verifiable via :meth:`MetricResult.verify`."""
    return layer.query(
        request,
        data,
        by=by,
        where=where,
        order_by=order_by,
        descending=descending,
        limit=limit,
        engine=engine,
        max_rows=max_rows,
        injection_detector=injection_detector,
        screen=screen,
    )
