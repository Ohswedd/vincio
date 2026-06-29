"""The data-analysis agent: bounded, multi-step EDA with a cited narrative.

A real analytical question over a table is rarely answered by a single query. An
analyst *explores*: they size the table up, summarize the measures, break a
measure down by a dimension, notice where it concentrates, and drill into the
part that dominates — then write up what they found, pointing at the figures that
back each statement. :func:`analyze_dataset` (offline) and :class:`AnalysisAgent`
(``app.analyze_data``) run exactly that loop, composed from the data plane's
existing organs so every finding is grounded and cited *by construction*:

* **plan** — a deterministic plan is grounded against the table's schema: an
  overview, the question itself (grounded by the same offline planner
  :func:`~vincio.data.query_dataset` uses), the measures' extremes and totals, and
  a measure-by-dimension breakdown;
* **query** — each step runs through the **governed query plane**
  (:class:`~vincio.data.QueryPlan`): schema-grounded, **read-only-verified**,
  cost-bounded, and executed where the data lives — never materialized into a
  prompt;
* **inspect** — each result is inspected deterministically (an empty or dominated
  result is noticed) and turned into a finding that **cites the exact source
  cells** it rests on (``sales#r3!revenue``);
* **refine** — while depth budget remains, a dominant group is drilled into for
  another, narrower finding;
* **synthesize** — the findings become a **cited analytical narrative**, the
  analytics analogue of a cited report, that re-derives from the bytes:
  :meth:`AnalysisResult.verify` re-executes every step's query and confirms the
  narrative and every cited cell still hold.

The whole loop is bounded by an explicit :class:`AnalysisBudget` and, through
:class:`AnalysisAgent`, audited on the app's chain. The verifier is not a model:
it is the query plane's offline re-execution and cell re-derivation, so a finding
is only carried when its underlying query verifies. Everything here is
deterministic, dependency-free, and offline; a :class:`~vincio.data.QueryEngine`
(e.g. the DuckDB accelerator behind ``vincio[data]``) can be supplied to push the
queries down to where the data lives at scale.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import AnalysisError, QueryError
from ..core.utils import stable_hash
from .core import Dataset, DataType
from .provenance import LineageCoverage
from .query import (
    DataCatalog,
    HeuristicQueryPlanner,
    QueryEngine,
    QueryPlan,
    QueryResult,
    _as_catalog,
    _screen_question,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.app import ContextApp
    from ..core.types import EvidenceItem
    from ..security.injection import InjectionDetector
    from .evidence import TableEvidence

__all__ = [
    "AnalysisStepKind",
    "AnalysisBudget",
    "AnalysisStep",
    "AnalysisResult",
    "AnalysisAgent",
    "analyze_dataset",
]


class AnalysisStepKind(StrEnum):
    """The role a step plays in the analysis.

    ``OVERVIEW`` sizes the table; ``QUESTION`` is the user's objective grounded to
    a query; ``EXTREME`` finds where a measure peaks or bottoms; ``TOTAL``
    aggregates a measure; ``BREAKDOWN`` groups a measure by a dimension; ``DRILL``
    is a refinement that narrows into the group that dominated a breakdown.
    """

    OVERVIEW = "overview"
    QUESTION = "question"
    EXTREME = "extreme"
    TOTAL = "total"
    BREAKDOWN = "breakdown"
    DRILL = "drill"


class AnalysisBudget(BaseModel):
    """Explicit bounds for one analysis run.

    ``max_steps`` caps the total number of queries the agent runs (the plan is
    truncated to fit); ``max_refinements`` caps how many dominant groups are
    drilled into; ``max_rows`` bounds each query's result; ``top_k`` is how many
    rows of a breakdown a finding names; ``max_breakdowns`` caps how many
    measure-by-dimension breakdowns are planned.
    """

    max_steps: int = 8
    max_refinements: int = 2
    max_rows: int = 10_000
    top_k: int = 3
    max_breakdowns: int = 2


class AnalysisStep(BaseModel):
    """One executed step of the analysis: its question, the query that answered it,
    and the cited finding it produced.

    ``result`` is the cell-level-cited :class:`~vincio.data.QueryResult` the step
    ran (``None`` only for a step that could not execute). ``finding`` is the
    human-readable claim, and ``cite_refs`` are the exact source-cell locators
    (``sales#r3!revenue``) it rests on. ``primary`` marks the step that answers
    the user's objective directly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: AnalysisStepKind
    question: str
    query: str = ""
    finding: str = ""
    cite_refs: list[str] = Field(default_factory=list)
    coverage: LineageCoverage = LineageCoverage.RESULT
    primary: bool = False
    refinement: bool = False
    result: QueryResult | None = None

    @property
    def cited(self) -> str:
        """The finding with its cell citations appended, e.g. ``... [sales#r0!revenue]``."""
        if not self.cite_refs:
            return self.finding
        return f"{self.finding} [{', '.join(self.cite_refs)}]"


class AnalysisResult(BaseModel):
    """A bounded, multi-step analysis rendered as a **cited analytical narrative**.

    Carries the objective, the executed :class:`AnalysisStep`\\s (each a verified,
    cell-cited query), the assembled narrative, and a content hash binding them.
    :meth:`verify` re-executes every step against a catalog and confirms the
    narrative and every cited cell re-derive from the bytes — the analytics
    analogue of a cited report's offline verification. The coverage is the weakest
    across the steps and is always stated, never silently downgraded.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    objective: str
    table: str = ""
    narrative: str = ""
    steps: list[AnalysisStep] = Field(default_factory=list)
    coverage: LineageCoverage = LineageCoverage.RESULT
    result_hash: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)

    # -- construction ----------------------------------------------------------

    @classmethod
    def _assemble(cls, objective: str, table: str, steps: list[AnalysisStep]) -> AnalysisResult:
        narrative = _build_narrative(objective, table, steps)
        executed = [s for s in steps if s.result is not None]
        coverage = (
            LineageCoverage.CELL
            if executed and all(s.coverage is LineageCoverage.CELL for s in executed)
            else LineageCoverage.RESULT
        )
        cited = sum(1 for s in executed if s.cite_refs)
        metrics = {
            "steps": float(len(steps)),
            "queries": float(len(executed)),
            "findings": float(sum(1 for s in steps if s.finding)),
            "cited_findings": float(cited),
            "citation_coverage": round(cited / len(executed), 4) if executed else 0.0,
        }
        result = cls(
            objective=objective,
            table=table,
            narrative=narrative,
            steps=steps,
            coverage=coverage,
            metrics=metrics,
        )
        result.result_hash = result._compute_hash()
        return result

    def _compute_hash(self) -> str:
        return stable_hash(
            [
                self.objective,
                self.table,
                self.narrative,
                [(s.kind.value, s.query, s.result.result_hash if s.result else "") for s in self.steps],
            ]
        )

    # -- access ----------------------------------------------------------------

    def primary_step(self) -> AnalysisStep | None:
        """The step that answers the user's objective directly, if one grounded."""
        return next((s for s in self.steps if s.primary), None)

    def answer(self) -> Any:
        """The headline answer to the objective: the primary step's scalar value
        when it is a single cell, else its result rows, else ``None``."""
        step = self.primary_step()
        if step is None or step.result is None:
            return None
        qr = step.result
        if qr.row_count == 1 and len(qr.columns) == 1:
            return qr.value(0, 0)
        return qr.rows

    def findings(self) -> list[str]:
        """The cited findings, in order — each a claim with its cell citations."""
        return [s.cited for s in self.steps if s.finding]

    def cite_refs(self) -> list[str]:
        """Every distinct source-cell locator the narrative rests on, in order."""
        seen: set[str] = set()
        out: list[str] = []
        for step in self.steps:
            for ref in step.cite_refs:
                if ref not in seen:
                    seen.add(ref)
                    out.append(ref)
        return out

    # -- verification ----------------------------------------------------------

    def verify(self, catalog: DataCatalog, *, engine: QueryEngine | None = None) -> bool:
        """Re-execute every step against *catalog* and confirm the narrative, each
        step's query, and every cited cell re-derive from the bytes. Returns
        ``False`` on any divergence (a tampered narrative, source, or cell)."""
        if self._compute_hash() != self.result_hash:
            return False
        for step in self.steps:
            if step.result is not None and not step.result.verify(catalog, engine=engine):
                return False
        return True

    # -- projection ------------------------------------------------------------

    def render(self, fmt: str = "markdown") -> str:
        """Render the narrative as plain text (``markdown``) or with each finding's
        underlying query shown (``report``)."""
        if fmt == "report":
            lines = [f"# Analysis: {self.objective}", ""]
            for step in self.steps:
                if not step.finding:
                    continue
                lines.append(f"- {step.cited}")
                if step.query:
                    lines.append(f"  > `{step.query}`")
            return "\n".join(lines)
        return self.narrative

    def to_evidence(self, *, source_id: str = "", caption: str = "", **kwargs: Any) -> TableEvidence:
        """Project the analysis into cited table evidence the context compiler
        scores, budgets, orders, and cites — one row per finding."""
        records = [
            {"finding": s.finding, "sources": ", ".join(s.cite_refs), "query": s.query}
            for s in self.steps
            if s.finding
        ]
        dataset = Dataset.from_records(records or [{"finding": self.narrative, "sources": "", "query": ""}],
                                       name=f"{self.table or 'analysis'}_findings")
        ev = dataset.to_evidence(
            source_id=source_id or f"{self.table or 'analysis'}_analysis",
            caption=caption or f"Analysis of {self.table or 'dataset'}: {self.objective}",
            **kwargs,
        )
        ev.metadata = {
            **ev.metadata,
            "objective": self.objective,
            "result_hash": self.result_hash,
            "lineage_coverage": str(self.coverage),
            "steps": len(self.steps),
        }
        return ev

    def to_evidence_item(self, **kwargs: Any) -> EvidenceItem:
        """Project straight to a ``modality='table'`` evidence item."""
        return self.to_evidence(**kwargs).to_evidence_item()

    def summary(self) -> str:
        """A one-line human summary of the analysis."""
        return (
            f"{self.table or 'dataset'}: {len(self.steps)} steps, "
            f"{int(self.metrics.get('queries', 0))} queries, "
            f"{int(self.metrics.get('cited_findings', 0))} cited findings ({self.coverage})"
        )


# --------------------------------------------------------------------------- #
# Deterministic offline planning + execution                                  #
# --------------------------------------------------------------------------- #


def _q(name: str) -> str:
    """Double-quote a SQL identifier (doubling any embedded quote)."""
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: Any) -> str:
    """Render a Python value as a SQL literal (a string single-quoted, an embedded
    quote doubled; a bool as 0/1; ``None`` as NULL)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        # NaN / ±inf have no SQL literal form; render as NULL (an ``= NULL`` predicate
        # is never true, so a group keyed on a non-finite value simply yields no drill).
        return "NULL" if not math.isfinite(value) else repr(value)
    if isinstance(value, int):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _measure_columns(ds: Dataset) -> list[str]:
    return [c.name for c in ds.columns if c.dtype in (DataType.INT, DataType.FLOAT)]


def _dimension_columns(ds: Dataset, *, max_cardinality: int) -> list[str]:
    """Low-cardinality, non-measure columns suitable for a group-by, most
    discriminating first (fewest distinct groups), ties broken by column order."""
    measures = set(_measure_columns(ds))
    candidates: list[tuple[int, int, str]] = []
    for order, col in enumerate(ds.columns):
        if col.name in measures:
            continue
        distinct = len({v for v in ds.column(col.name) if v is not None})
        if 1 < distinct <= max_cardinality:
            candidates.append((distinct, order, col.name))
    candidates.sort()
    return [name for _, _, name in candidates]


def _run_sql(
    sql: str,
    catalog: DataCatalog,
    *,
    max_rows: int,
    engine: QueryEngine | None,
) -> QueryResult | None:
    """Verify and run a generated analytical query, returning ``None`` if it
    cannot be grounded or executed (so one bad step never sinks the analysis)."""
    try:
        plan = QueryPlan.for_sql(sql, catalog, max_rows=max_rows, engine=engine)
        return plan.run(catalog, engine=engine)
    except QueryError:
        return None


def _summarize_rows(qr: QueryResult, *, limit: int) -> tuple[str, list[str]]:
    """A compact ``key=value`` description of a result's top rows and the distinct
    source cells they rest on."""
    cols = qr.columns
    pieces: list[str] = []
    refs: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(qr.rows[:limit]):
        if len(cols) >= 2:
            pieces.append(f"{row[0]}={_fmt(row[-1])}")
        else:
            pieces.append(_fmt(row[0]))
        for ref in qr.cite_refs(i):
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return ", ".join(pieces), refs


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _plan_initial(
    catalog: DataCatalog,
    table: str,
    objective: str,
    budget: AnalysisBudget,
    *,
    engine: QueryEngine | None,
    extra_questions: list[str] | None,
) -> list[AnalysisStep]:
    """Build and execute the deterministic, schema-grounded plan."""
    ds = catalog.get(table)
    steps: list[AnalysisStep] = []
    measures = _measure_columns(ds)
    dimensions = _dimension_columns(ds, max_cardinality=max(budget.top_k * 4, 20))

    def remaining() -> int:
        return budget.max_steps - len(steps)

    # 1. Overview — size the table up.
    overview = _run_sql(f"SELECT COUNT(*) AS row_count FROM {_q(table)}", catalog,
                        max_rows=budget.max_rows, engine=engine)
    n_rows = int(overview.value(0, 0)) if overview and overview.row_count else ds.row_count
    steps.append(
        AnalysisStep(
            kind=AnalysisStepKind.OVERVIEW,
            question=f"How large is {table}?",
            query=overview.plan.sql if overview else "",
            finding=(
                f"The {table} dataset has {n_rows:,} rows across {ds.width} columns "
                f"({', '.join(ds.column_names)})."
            ),
            coverage=overview.coverage if overview else LineageCoverage.RESULT,
            result=overview,
        )
    )

    # 2. The objective itself, grounded by the offline planner (cell-cited when it
    #    is a single-table projection/filter or group-by aggregation).
    planner = HeuristicQueryPlanner()
    grounded = planner.plan(objective, catalog, table=table) if objective else None
    if grounded is not None and remaining() > 0:
        qr = _run_sql(grounded, catalog, max_rows=budget.max_rows, engine=engine)
        if qr is not None:
            finding, refs = _describe_question(objective, qr, budget.top_k)
            steps.append(
                AnalysisStep(
                    kind=AnalysisStepKind.QUESTION,
                    question=objective,
                    query=qr.plan.sql,
                    finding=finding,
                    cite_refs=refs,
                    coverage=qr.coverage,
                    primary=True,
                    result=qr,
                )
            )

    # 3. Measure extremes + totals (cell-cited extreme; result-level total).
    label = dimensions[0] if dimensions else (ds.column_names[0] if ds.column_names else "")
    for measure in measures:
        if remaining() <= 1:
            break
        steps.extend(_measure_steps(catalog, table, measure, label, n_rows, budget, engine))

    # 4. Measure-by-dimension breakdowns, with a drill into the dominant group.
    breakdowns = 0
    refinements = 0
    for dimension in dimensions:
        if breakdowns >= budget.max_breakdowns or remaining() <= 0 or not measures:
            break
        measure = measures[0]
        step = _breakdown_step(catalog, table, measure, dimension, budget, engine)
        if step is None:
            continue
        steps.append(step)
        breakdowns += 1
        # 5. Refine: drill into the group that dominates the breakdown, capped by the
        #    total refinement budget across all breakdowns.
        if (
            refinements < budget.max_refinements
            and step.result is not None
            and step.result.row_count > 1
        ):
            drilled = _drill_steps(catalog, table, measure, dimension, dimensions, step, budget, engine,
                                   remaining())
            if drilled:
                refinements += 1
                steps.extend(drilled)

    # 6. Any model-proposed follow-up questions (grounded + verified like the rest).
    for question in extra_questions or []:
        if remaining() <= 0:
            break
        grounded = planner.plan(question, catalog, table=table)
        if grounded is None:
            continue
        qr = _run_sql(grounded, catalog, max_rows=budget.max_rows, engine=engine)
        if qr is None:
            continue
        finding, refs = _describe_question(question, qr, budget.top_k)
        steps.append(
            AnalysisStep(
                kind=AnalysisStepKind.QUESTION,
                question=question,
                query=qr.plan.sql,
                finding=finding,
                cite_refs=refs,
                coverage=qr.coverage,
                result=qr,
            )
        )

    return steps[: budget.max_steps]


def _measure_steps(
    catalog: DataCatalog,
    table: str,
    measure: str,
    label: str,
    n_rows: int,
    budget: AnalysisBudget,
    engine: QueryEngine | None,
) -> list[AnalysisStep]:
    steps: list[AnalysisStep] = []
    # Extreme — a cell-cited projection of the top row by the measure. When a
    # distinct label column exists, project it alongside the measure so the finding
    # can name where the peak sits; otherwise project the measure alone.
    has_label = bool(label) and label != measure
    if has_label:
        extreme_sql = (
            f"SELECT {_q(label)}, {_q(measure)} FROM {_q(table)} "
            f"ORDER BY {_q(measure)} DESC, {_q(label)} LIMIT 1"
        )
    else:
        extreme_sql = (
            f"SELECT {_q(measure)} FROM {_q(table)} ORDER BY {_q(measure)} DESC LIMIT 1"
        )
    qr = _run_sql(extreme_sql, catalog, max_rows=budget.max_rows, engine=engine)
    if qr is not None and qr.row_count:
        if has_label:
            finding = (
                f"The highest {measure} is {_fmt(qr.value(0, 1))} "
                f"(at {label}={_fmt(qr.value(0, 0))})."
            )
        else:
            finding = f"The highest {measure} is {_fmt(qr.value(0, 0))}."
        steps.append(
            AnalysisStep(
                kind=AnalysisStepKind.EXTREME,
                question=f"Where does {measure} peak?",
                query=qr.plan.sql,
                finding=finding,
                cite_refs=qr.cite_refs(0),
                coverage=qr.coverage,
                result=qr,
            )
        )
    # Total — a result-level aggregate over the whole measure.
    total = _run_sql(f"SELECT SUM({_q(measure)}) AS total FROM {_q(table)}", catalog,
                     max_rows=budget.max_rows, engine=engine)
    if total is not None and total.row_count and total.value(0, 0) is not None:
        steps.append(
            AnalysisStep(
                kind=AnalysisStepKind.TOTAL,
                question=f"What is the total {measure}?",
                query=total.plan.sql,
                finding=f"The total {measure} across all {n_rows:,} rows is {_fmt(total.value(0, 0))}.",
                coverage=total.coverage,
                result=total,
            )
        )
    return steps


def _breakdown_step(
    catalog: DataCatalog,
    table: str,
    measure: str,
    dimension: str,
    budget: AnalysisBudget,
    engine: QueryEngine | None,
) -> AnalysisStep | None:
    sql = (
        f"SELECT {_q(dimension)}, SUM({_q(measure)}) AS sum_{_safe(measure)} "
        f"FROM {_q(table)} GROUP BY {_q(dimension)} "
        f"ORDER BY sum_{_safe(measure)} DESC, {_q(dimension)}"
    )
    qr = _run_sql(sql, catalog, max_rows=budget.max_rows, engine=engine)
    if qr is None or not qr.row_count:
        return None
    summary, refs = _summarize_rows(qr, limit=budget.top_k)
    top_key = qr.value(0, 0)
    top_val = qr.value(0, 1)
    finding = (
        f"By {dimension}, {measure} concentrates in {_fmt(top_key)} ({_fmt(top_val)}); "
        f"the breakdown is {summary}."
    )
    return AnalysisStep(
        kind=AnalysisStepKind.BREAKDOWN,
        question=f"How does {measure} break down by {dimension}?",
        query=qr.plan.sql,
        finding=finding,
        cite_refs=refs,
        coverage=qr.coverage,
        result=qr,
    )


def _drill_steps(
    catalog: DataCatalog,
    table: str,
    measure: str,
    dimension: str,
    dimensions: list[str],
    breakdown: AnalysisStep,
    budget: AnalysisBudget,
    engine: QueryEngine | None,
    remaining: int,
) -> list[AnalysisStep]:
    if remaining <= 0 or budget.max_refinements <= 0 or breakdown.result is None:
        return []
    secondary = next((d for d in dimensions if d != dimension), None)
    if secondary is None:
        return []
    dominant = breakdown.result.value(0, 0)
    sql = (
        f"SELECT {_q(secondary)}, SUM({_q(measure)}) AS sum_{_safe(measure)} "
        f"FROM {_q(table)} WHERE {_q(dimension)} = {_sql_literal(dominant)} "
        f"GROUP BY {_q(secondary)} ORDER BY sum_{_safe(measure)} DESC, {_q(secondary)}"
    )
    qr = _run_sql(sql, catalog, max_rows=budget.max_rows, engine=engine)
    if qr is None or not qr.row_count:
        return []
    summary, refs = _summarize_rows(qr, limit=budget.top_k)
    finding = (
        f"Within {dimension}={_fmt(dominant)} (the largest group), {measure} breaks down "
        f"by {secondary} as {summary}."
    )
    return [
        AnalysisStep(
            kind=AnalysisStepKind.DRILL,
            question=f"Within {dimension}={_fmt(dominant)}, how does {measure} split by {secondary}?",
            query=qr.plan.sql,
            finding=finding,
            cite_refs=refs,
            coverage=qr.coverage,
            refinement=True,
            result=qr,
        )
    ]


def _safe(name: str) -> str:
    """A bare alias token derived from a column name (for ``SUM(x) AS sum_x``)."""
    token = "".join(ch if ch.isalnum() else "_" for ch in name)
    return token or "measure"


def _describe_question(objective: str, qr: QueryResult, top_k: int) -> tuple[str, list[str]]:
    """A finding describing the objective-grounded result."""
    if qr.row_count == 0:
        return (f"No rows answer “{objective}”.", [])
    if qr.row_count == 1 and len(qr.columns) == 1:
        refs = qr.cite_refs(0)
        return (f"In answer to “{objective}”: {qr.columns[0]} is {_fmt(qr.value(0, 0))}.", refs)
    summary, refs = _summarize_rows(qr, limit=top_k)
    return (f"For “{objective}”: {summary}.", refs)


def _build_narrative(objective: str, table: str, steps: list[AnalysisStep]) -> str:
    """Assemble the cited narrative from the executed steps, deterministically."""
    head = f"Analysis of {table or 'the dataset'}" + (f" — {objective}" if objective else "") + ":"
    body = [f"- {step.cited}" for step in steps if step.finding]
    if not body:
        return f"{head}\n- No groundable findings for this dataset."
    return head + "\n" + "\n".join(body)


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #


def _resolve_table(catalog: DataCatalog, table: str | None) -> str:
    if table is not None:
        if table not in catalog.tables:
            raise AnalysisError(
                f"no registered table {table!r}; known tables: {catalog.names}"
            )
        return table
    if len(catalog.names) == 1:
        return catalog.names[0]
    if not catalog.names:
        raise AnalysisError("no dataset to analyze; register a dataset or pass data=")
    raise AnalysisError(
        f"the catalog has {len(catalog.names)} tables {catalog.names}; pass table= to choose one"
    )


def analyze_dataset(
    objective: str,
    data: Dataset | DataCatalog | dict[str, Dataset],
    *,
    table: str | None = None,
    budget: AnalysisBudget | None = None,
    engine: QueryEngine | None = None,
    injection_detector: InjectionDetector | None = None,
    screen: bool = True,
    extra_questions: list[str] | None = None,
) -> AnalysisResult:
    """Run a bounded, multi-step analysis over a dataset and return a cited
    analytical narrative — the offline, deterministic core of the data-analysis
    agent.

    *objective* is the analytical question; *data* is a single
    :class:`~vincio.data.Dataset`, a mapping of name→dataset, or a
    :class:`~vincio.data.DataCatalog`. The objective is screened for injection
    (the same detector the text rails use), a deterministic plan is grounded
    against the schema, every step runs through the
    governed, read-only-verified query plane, and the findings become a narrative
    that **cites the exact source cells** and re-derives from the bytes via
    :meth:`AnalysisResult.verify`. Pass an *engine* (e.g. the DuckDB accelerator)
    to push the queries down to where the data lives.
    """
    catalog = _as_catalog(data, table=table)
    table = _resolve_table(catalog, table)
    budget = budget or AnalysisBudget()
    if screen and objective:
        _screen_question(objective, injection_detector)
    steps = _plan_initial(
        catalog, table, objective, budget, engine=engine, extra_questions=extra_questions
    )
    return AnalysisResult._assemble(objective, table, steps)


class AnalysisAgent:
    """Plan → query → inspect → refine → synthesize, cited and budget-bounded.

    Wraps the deterministic :func:`analyze_dataset` core with an app: it resolves
    the catalog (the app's registered datasets or a one-shot ``dataset=``), screens
    the objective with the app's injection detector, optionally asks the configured
    model for additional analytical questions (each still grounded and verified by
    the query plane — the model never produces a query that bypasses the screen),
    and audits the run on the app's chain. Offline (or whenever the model returns
    nothing groundable) the agent is byte-for-byte the deterministic core, so it
    stays reproducible and air-gapped-safe.
    """

    def __init__(
        self,
        app: ContextApp,
        *,
        budget: AnalysisBudget | None = None,
        engine: QueryEngine | None = None,
        propose_followups: bool = True,
        max_followups: int = 3,
    ) -> None:
        self.app = app
        self.budget = budget or AnalysisBudget()
        self.engine = engine
        self.propose_followups = propose_followups
        self.max_followups = max_followups

    async def arun(
        self,
        objective: str,
        *,
        table: str | None = None,
        dataset: Any | None = None,
    ) -> AnalysisResult:
        catalog = self._catalog(dataset, table)
        resolved = _resolve_table(catalog, table)
        # Screen the objective once (the generated analytical queries are safe by
        # construction); a write/DDL/injection signal refuses the run.
        if objective:
            _screen_question(objective, self._detector())
        extra = await self._propose(objective, catalog, resolved) if self.propose_followups else []
        result = analyze_dataset(
            objective,
            catalog,
            table=resolved,
            budget=self.budget,
            engine=self.engine,
            screen=False,  # already screened above
            extra_questions=extra,
        )
        self._audit(objective, resolved, result)
        return result

    def run(
        self,
        objective: str,
        *,
        table: str | None = None,
        dataset: Any | None = None,
    ) -> AnalysisResult:
        """Synchronous wrapper around :meth:`arun`."""
        from ..providers.base import run_sync

        result: AnalysisResult = run_sync(  # type: ignore[no-untyped-call]
            self.arun(objective, table=table, dataset=dataset)
        )
        return result

    # -- helpers --------------------------------------------------------------

    def _catalog(self, dataset: Any | None, table: str | None) -> DataCatalog:
        if dataset is not None:
            ds = self.app._coerce_dataset(dataset, name=table or "")
            return DataCatalog.of(ds, name=table or ds.name or "data")
        catalog: DataCatalog = self.app.data_catalog()
        return catalog

    def _detector(self) -> InjectionDetector | None:
        return getattr(self.app, "injection_detector", None)

    async def _propose(self, objective: str, catalog: DataCatalog, table: str) -> list[str]:
        """Ask the model for up to ``max_followups`` extra analytical questions.

        Returns only questions the offline planner can ground (so a junk or empty
        model response degrades to no extra steps). Any provider failure → ``[]``.
        """
        from ..core.types import Message, ModelRequest
        from ..providers.base import run_sync

        ds = catalog.get(table)
        try:
            provider = self.app._base_provider()
            request = ModelRequest(
                model=self.app.model or "",
                messages=[
                    Message(
                        role="system",
                        content=(
                            "You are a data analyst. Propose up to "
                            f"{self.max_followups} short follow-up questions that can be "
                            "answered by aggregating one column grouped by another. One "
                            "question per line, no numbering."
                        ),
                    ),
                    Message(
                        role="user",
                        content=(
                            f"Table {table} columns: {', '.join(ds.column_names)}.\n"
                            f"Objective: {objective}"
                        ),
                    ),
                ],
                temperature=0.0,
                max_output_tokens=256,
            )
            response = run_sync(provider.generate(request))  # type: ignore[no-untyped-call]
            text = response.text or ""
        except Exception:  # noqa: BLE001 - any provider failure → no extra steps
            return []
        planner = HeuristicQueryPlanner()
        out: list[str] = []
        for line in text.splitlines():
            question = line.strip(" -*0123456789.").strip()
            if question and planner.plan(question, catalog, table=table) is not None:
                out.append(question)
            if len(out) >= self.max_followups:
                break
        return out

    def _audit(self, objective: str, table: str, result: AnalysisResult) -> None:
        self.app.audit.record(
            "data_analysis",
            decision="allow",
            resource=table,
            details={
                "objective": objective[:200],
                "steps": len(result.steps),
                "queries": int(result.metrics.get("queries", 0)),
                "cited_findings": int(result.metrics.get("cited_findings", 0)),
                "lineage_coverage": str(result.coverage),
                "result_hash": result.result_hash,
            },
        )
        self.app.events.emit(
            "data_analysis.completed",
            {"objective": objective[:200], "table": table, "steps": len(result.steps)},
        )
