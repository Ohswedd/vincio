"""Tabular evidence, profiling, sampling, and data-quality rails.

The data plane treats a dataset as *first-class, schema-bearing, columnar
evidence* — never flattened to prose or dumped as ``json.dumps``. The model sees
a compact, token-oriented encoding in which the schema, types, units, and
null-handling are declared **once** and the cells follow as delimited rows, plus
a deterministic profile and a representative sample when the rows themselves
would not fit.

* :class:`Dataset` — a typed :class:`DataSchema` (per-column name, type, unit,
  nullability) over column-major cells. Build one from rows, records, columns, a
  legacy ``TableData``, or a compact encoding.
* :class:`DataEncoder` — renders a dataset header-once in a compact, lossless,
  round-trippable form and reports its columnar-accurate token cost; also
  encodes arbitrary JSON-like values, the token-efficient replacement for
  ``json.dumps``.
* :class:`TableEvidence` — projects a dataset into the context evidence the
  compiler scores, budgets, orders, and cites.
* :func:`profile_dataset` / :class:`DatasetProfile` — a deterministic,
  bounded-memory column profile (cardinality, histograms, percentiles, null
  rate, exemplars) that is itself fixed-size evidence the compiler can score.
* :func:`sample_dataset` (and :func:`reservoir_sample` / :func:`stratified_sample`)
  — a representative sample that stands in for the whole, replacing a biased
  first-N cutoff.
* :func:`fit_to_window` / :class:`WindowFit` — profile + representative sample
  fitted into a fixed token budget, so a table far larger than the window is
  represented faithfully inside it.
* :class:`DataQualityRails` — screen a dataset for schema violations, constraint
  breaks, and anomalies on the same deterministic rail path PII and injection
  detection ride.
* :func:`query_dataset` / :class:`QueryResult` — turn a question (or explicit
  SQL / dataframe ops) over a registered :class:`DataCatalog` into a
  schema-grounded, read-only-verified, cost-bounded query, executed offline by
  the standard-library ``sqlite3`` engine (:class:`InProcessSqlEngine`), whose
  answer **cites the exact source cells** it rests on
  (:class:`~vincio.data.provenance.CellCitation`) and re-derives from the bytes
  via :meth:`QueryResult.verify`. :class:`DuckDbQueryEngine` runs the same
  verified SQL on DuckDB (behind the ``vincio[data]`` extra) for execution at
  scale.
* :func:`analyze_dataset` / :class:`AnalysisAgent` / :class:`AnalysisResult` —
  a bounded, multi-step analysis agent that plans, queries, inspects, and refines
  over a dataset through the governed query plane, producing a **cited analytical
  narrative** that re-derives from the bytes via :meth:`AnalysisResult.verify`.
* :func:`generate_chart` / :class:`Chart` / :class:`ChartSpec` — turn a cited
  :class:`QueryResult` into a spec-driven chart that is **content-bound** (a C2PA
  data-driven credential bound to its bytes) and **data-bound** (a back-reference
  to the exact source cells it was built from), offline-verifiable via
  :meth:`Chart.verify`. The default :class:`VegaLiteRenderer` is dependency-free;
  :class:`MatplotlibRenderer` (behind the ``vincio[charts]`` extra) rasterizes the
  same spec to a PNG.
* :class:`RowStream` — a lazy, re-iterable, schema-bearing handle over a row
  source larger than memory (records, a generator factory, or a CSV / JSON-Lines
  file read line by line). Profile, fit, sample, :func:`stream_aggregate`, or
  :func:`encode_stream` it in a single bounded pass whose footprint is invariant
  to the row count, or run an analytical transform over it at scale through the
  :class:`~vincio.providers.BatchRunner` with :func:`stream_map`.
* :class:`SemanticLayer` — define :class:`Measure`\\s, :class:`Dimension`\\s, and
  :class:`DerivedColumn`\\s **once** so a natural-language question maps to a
  **governed metric** (:func:`query_metric` / :class:`MetricResult`) compiled one
  way everywhere through the read-only-verified query plane, and a metric's
  column-level provenance (:class:`MetricLineage`) reaches the lineage and
  right-to-erasure machinery.
* :class:`StreamWindow` — the profiling, query, governed-metric, and quality
  primitives re-expressed over an **unbounded event stream** instead of a bounded
  :class:`Dataset`, computed one window at a time (``tumbling`` / ``sliding`` /
  ``session``) so the working set stays invariant to the event volume. Each closed
  window emits a result (:class:`WindowedProfile` / :class:`WindowedQueryResult` /
  :class:`WindowedMetricResult` / :class:`WindowedQualityReport`) that cites the
  exact source **events** it rests on (:class:`EventCitation`, ``stream@<offset>``)
  and ``verify()``s offline against its bounded :class:`CapturedWindow`. The
  app-governed :class:`StreamingAnalytics` (``app.stream_analytics``) audits each
  window and drives a **live** realtime session as readily as a replayed log.
* :class:`DataEngagement` — the capstone facade (``app.data_engagement``) that
  threads the whole plane (register → profile → sample → fit → screen → query →
  analyze → chart → governed metric → cite) behind one governed, audited
  call-path and seals it into a hash-chained, signed :class:`DataNarrative` that
  :meth:`DataNarrative.verify`\\s offline and is **data-bound** — every captured
  finding re-executes against the content-hashed source. The analytics analogue
  of :class:`~vincio.settlement.CrossOrgEngagement`.
* :class:`FederatedDataEngagement` — the cross-org facade
  (``app.federated_data_engagement``) that runs a :class:`FederatedQuery` across
  several organizations' data planes over the existing cross-org fabric:
  negotiated as a :class:`~vincio.negotiation.Contract`, choreographed as a
  :class:`~vincio.choreography.Saga` whose steps run each org's governed query
  plane **locally** and return only the aggregated, cell-cited
  :class:`MetricResult` — never the raw rows — reconciled into one signed,
  offline-verifiable :class:`FederatedNarrative` whose every :class:`FederatedFinding`
  re-derives from each org's content-hashed source. Residency egress refusal, the
  consent ledger, and the differential-privacy accountant apply at the boundary
  exactly as for a local query.

Everything here is deterministic, dependency-free, and offline. ``Dataset`` and
the schema types are exported from this subpackage (the top-level ``Dataset``
name belongs to :mod:`vincio.evals`); :class:`DataEncoder`,
:class:`TableEvidence`, :class:`DatasetProfile`, :class:`DataQualityRails`, and
:class:`DataQualityReport` are also re-exported at the package top level.

    from vincio.data import Dataset, profile_dataset, sample_dataset, DataQualityRails

    ds = Dataset.from_records(
        [{"region": "NA", "revenue": 1200.5}, {"region": "EU", "revenue": 980.0}],
        name="sales",
    )
    print(ds.encode())                 # sales{#2,region:str,revenue:float}\nNA,1200.5\nEU,980.0
    profile = profile_dataset(ds)      # deterministic column profile (fixed-size evidence)
    report = DataQualityRails.from_dataset(ds).check(ds)   # screen for schema/quality breaches
"""

from __future__ import annotations

from ..core.errors import (
    AnalysisError,
    ChartError,
    DataError,
    DataQualityError,
    QueryError,
    SemanticLayerError,
    StreamError,
    UnsafeQueryError,
)
from .analysis import (
    AnalysisAgent,
    AnalysisBudget,
    AnalysisResult,
    AnalysisStep,
    AnalysisStepKind,
    analyze_dataset,
)
from .charts import (
    Chart,
    ChartChannel,
    ChartEncoding,
    ChartRenderer,
    ChartSpec,
    ChartType,
    MatplotlibRenderer,
    VegaLiteRenderer,
    generate_chart,
)
from .core import ColumnSchema, DataSchema, Dataset, DataType
from .encoders import DataEncoder
from .engagement import (
    DataEngagement,
    DataEngagementSignature,
    DataEngagementVerification,
    DataNarrative,
    DataStage,
)
from .engines import DuckDbQueryEngine
from .evidence import TableEvidence
from .federated import (
    FederatedContribution,
    FederatedDataEngagement,
    FederatedFinding,
    FederatedMember,
    FederatedNarrative,
    FederatedQuery,
    FederatedSignature,
    FederatedStage,
    FederatedVerification,
)
from .profile import (
    ColumnProfile,
    DatasetProfile,
    HistogramBin,
    profile_dataset,
    profile_stream,
)
from .provenance import CellCitation, LineageCoverage, RowProvenance
from .quality import (
    ColumnConstraint,
    DataQualityRails,
    DataQualityReport,
    DataQualityViolation,
)
from .query import (
    DataCatalog,
    HeuristicQueryPlanner,
    InProcessSqlEngine,
    QueryDialect,
    QueryEngine,
    QueryPlan,
    QueryResult,
    assert_read_only_sql,
    is_read_only_sql,
    make_query_contract,
    query_dataset,
)
from .sampling import (
    SampleMethod,
    reservoir_sample,
    sample_dataset,
    stratified_sample,
    systematic_sample,
)
from .semantic import (
    Aggregation,
    DerivedColumn,
    Dimension,
    Measure,
    MetricLineage,
    MetricQuery,
    MetricResult,
    SemanticLayer,
    query_metric,
)
from .streaming import (
    BulkMapResult,
    RowStream,
    StreamAggregation,
    encode_stream,
    stream_aggregate,
    stream_map,
)
from .streaming_analytics import (
    CapturedWindow,
    EventCitation,
    StreamingAnalytics,
    StreamWindow,
    WindowedAggregation,
    WindowedMetricResult,
    WindowedProfile,
    WindowedQualityReport,
    WindowedQueryResult,
    WindowKind,
)
from .window import WindowFit, fit_stream, fit_to_window

__all__ = [
    "Dataset",
    "DataSchema",
    "ColumnSchema",
    "DataType",
    "DataEncoder",
    "TableEvidence",
    "DataError",
    "DataQualityError",
    "StreamError",
    "QueryError",
    "UnsafeQueryError",
    "AnalysisError",
    "ChartError",
    "SemanticLayerError",
    # profiling
    "HistogramBin",
    "ColumnProfile",
    "DatasetProfile",
    "profile_dataset",
    "profile_stream",
    # sampling
    "SampleMethod",
    "reservoir_sample",
    "stratified_sample",
    "systematic_sample",
    "sample_dataset",
    # fit-in-window
    "WindowFit",
    "fit_to_window",
    "fit_stream",
    # data-quality rails
    "ColumnConstraint",
    "DataQualityViolation",
    "DataQualityReport",
    "DataQualityRails",
    # governed text-to-query & cell-level provenance
    "DataCatalog",
    "QueryDialect",
    "QueryPlan",
    "QueryResult",
    "QueryEngine",
    "InProcessSqlEngine",
    "HeuristicQueryPlanner",
    "CellCitation",
    "RowProvenance",
    "LineageCoverage",
    "query_dataset",
    "make_query_contract",
    "is_read_only_sql",
    "assert_read_only_sql",
    "DuckDbQueryEngine",
    # data-analysis agent & multi-step EDA
    "AnalysisStepKind",
    "AnalysisBudget",
    "AnalysisStep",
    "AnalysisResult",
    "AnalysisAgent",
    "analyze_dataset",
    # charts & cited analytical artifacts
    "ChartType",
    "ChartChannel",
    "ChartEncoding",
    "ChartSpec",
    "Chart",
    "ChartRenderer",
    "VegaLiteRenderer",
    "MatplotlibRenderer",
    "generate_chart",
    # streaming & out-of-core bulk processing
    "RowStream",
    "StreamAggregation",
    "stream_aggregate",
    "encode_stream",
    "BulkMapResult",
    "stream_map",
    # real-time & streaming analytics (windowed primitives)
    "WindowKind",
    "StreamWindow",
    "CapturedWindow",
    "EventCitation",
    "WindowedProfile",
    "WindowedQueryResult",
    "WindowedMetricResult",
    "WindowedQualityReport",
    "WindowedAggregation",
    "StreamingAnalytics",
    # semantic layer & governed metrics
    "Aggregation",
    "DerivedColumn",
    "Dimension",
    "Measure",
    "MetricQuery",
    "MetricResult",
    "MetricLineage",
    "SemanticLayer",
    "query_metric",
    # data & analytics capstone — the engagement lifecycle facade
    "DataStage",
    "DataEngagementSignature",
    "DataEngagementVerification",
    "DataNarrative",
    "DataEngagement",
    # cross-org / federated analytics
    "FederatedQuery",
    "FederatedMember",
    "FederatedContribution",
    "FederatedFinding",
    "FederatedStage",
    "FederatedSignature",
    "FederatedVerification",
    "FederatedNarrative",
    "FederatedDataEngagement",
]
