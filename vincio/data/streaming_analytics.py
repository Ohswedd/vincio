"""Real-time analytics over an unbounded event stream.

The data plane's batch primitives — :func:`~vincio.data.profile_dataset`,
governed text-to-query (:meth:`~vincio.core.ContextApp.query_data`), the governed
metric (:meth:`~vincio.core.ContextApp.query_metric`), and the data-quality rails
(:class:`~vincio.data.DataQualityRails`) — all assume a *bounded*
:class:`~vincio.data.Dataset`. A live event feed has no end. This module
re-expresses those primitives over an **unbounded stream**, computed one
**window** at a time so the working set stays invariant to how many events have
flowed, and every per-window answer cites the exact events it rests on and
re-derives offline against the captured window.

* :class:`StreamWindow` — the windowing policy (``tumbling`` / ``sliding`` /
  ``session``). It carries the streaming analogues of the batch primitives:
  :meth:`~StreamWindow.profile`, :meth:`~StreamWindow.query`,
  :meth:`~StreamWindow.query_metric`, :meth:`~StreamWindow.screen`, and a
  bounded-memory :meth:`~StreamWindow.aggregate`, each iterating an unbounded
  :class:`~vincio.data.RowStream` and emitting one result per closed window. The
  resident set holds only the *open* windows (one per key for a tumbling window),
  never the whole stream, so a billion-event feed is processed inside the same
  footprint as a thousand-event one.
* :class:`CapturedWindow` — a single closed window's bounded snapshot: the events
  it held (as a schema-bearing :class:`~vincio.data.Dataset`), their stable
  stream offsets, the window bounds, and a content hash binding them. It is the
  trusted anchor a windowed result re-derives against, fully offline.
* :class:`EventCitation` — a reference to one source *event* cell
  (``stream@<offset>!<column>``), the streaming analogue of
  :class:`~vincio.data.provenance.CellCitation`'s ``table#r<row>!<column>``: a
  windowed answer cites the exact events (not rows) it rests on.
* :class:`WindowedProfile` / :class:`WindowedQueryResult` /
  :class:`WindowedMetricResult` / :class:`WindowedQualityReport` /
  :class:`WindowedAggregation` — the per-window results. Each carries its
  :class:`CapturedWindow` and ``verify()``s offline against it.

Everything here is deterministic, dependency-free, and offline. It rides the
streaming primitives the out-of-core rung already shipped
(:class:`~vincio.data.RowStream` / :func:`~vincio.data.stream_aggregate`) and the
governed, read-only-verified query plane — so a live dashboard tile or an
alerting rule is just a governed, cited, budget-bounded query over a window,
never a hosted stream processor.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.errors import StreamError
from ..core.utils import stable_hash
from .core import ColumnSchema, Dataset, DataType
from .profile import DatasetProfile, profile_dataset
from .provenance import CellCitation, LineageCoverage
from .quality import DataQualityRails, DataQualityReport
from .query import DataCatalog, QueryEngine, QueryResult, query_dataset
from .semantic import MetricResult, SemanticLayer
from .streaming import RowStream, StreamAggregation, _coerce_stream, stream_aggregate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..security.injection import InjectionDetector
    from .evidence import TableEvidence

__all__ = [
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
    "DEFAULT_STREAM_TABLE",
    "DEFAULT_MAX_OPEN_WINDOWS",
]

DEFAULT_STREAM_TABLE = "events"
# A hard ceiling on the number of windows held open at once. With in-order events
# a tumbling window keeps one open window per key and a sliding window keeps
# ``size / slide``; a stream so out of order that it would exceed this is refused
# rather than growing the working set without bound.
DEFAULT_MAX_OPEN_WINDOWS = 4_096


class WindowKind(StrEnum):
    """The three windowing strategies over an event stream.

    ``TUMBLING`` — fixed-size, non-overlapping windows that partition time
    (``slide == size``). ``SLIDING`` — fixed-size windows that advance by a
    ``slide`` smaller than ``size``, so they overlap and an event belongs to more
    than one. ``SESSION`` — dynamic windows that grow while events keep arriving
    within a ``gap`` and close once the feed (for a key) goes quiet longer than
    the gap."""

    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"


# --------------------------------------------------------------------------- #
# Event-level provenance                                                       #
# --------------------------------------------------------------------------- #


class EventCitation(BaseModel):
    """A reference to one source *event* cell a windowed answer rests on.

    ``ref`` renders the stable, parseable locator ``stream@<offset>!<column>``
    (the offset is the event's 0-based position in the stream — its stable
    identity, the streaming analogue of a row index). ``value`` binds the cell's
    value at compute time, so a later check can confirm the captured event still
    holds it. ``result_column`` is the output column this source cell contributes
    to, so a derived value cites exactly its operands."""

    stream: str
    offset: int
    column: str
    value: Any = None
    result_column: str = ""

    @property
    def ref(self) -> str:
        """The stable event-cell locator, e.g. ``orders@418!amount``."""
        return f"{self.stream}@{self.offset}!{self.column}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.ref


# --------------------------------------------------------------------------- #
# A closed window's bounded, content-bound snapshot                            #
# --------------------------------------------------------------------------- #


class CapturedWindow(BaseModel):
    """The bounded snapshot of one closed window — the trusted anchor a windowed
    result re-derives against, offline.

    Holds the events the window contained as a schema-bearing
    :class:`~vincio.data.Dataset` (``dataset``, one row per event in arrival
    order), their stable stream ``offsets`` (``offsets[i]`` is the global offset
    of dataset row ``i``), the window's ``[start, end)`` time bounds, the
    partition ``key`` it belongs to, and a ``content_hash`` binding all of it. Its
    size is bounded by the window, never the stream, so capturing it is the
    bounded-memory step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stream: str
    kind: WindowKind
    start: float
    end: float
    key: list[Any] = Field(default_factory=list)
    offsets: list[int] = Field(default_factory=list)
    dataset: Dataset
    closed_by: str = "watermark"
    content_hash: str = ""

    @classmethod
    def build(
        cls,
        *,
        stream: str,
        kind: WindowKind,
        start: float,
        end: float,
        key: Sequence[Any],
        offsets: Sequence[int],
        dataset: Dataset,
        closed_by: str,
    ) -> CapturedWindow:
        window = cls(
            stream=stream,
            kind=kind,
            start=start,
            end=end,
            key=list(key),
            offsets=list(offsets),
            dataset=dataset,
            closed_by=closed_by,
        )
        window.content_hash = window._compute_hash()
        return window

    def _compute_hash(self) -> str:
        return stable_hash(
            [
                self.stream,
                str(self.kind),
                round(self.start, 9),
                round(self.end, 9),
                self.key,
                self.offsets,
                self.dataset.column_names,
                self.dataset.dtypes,
                self.dataset.rows(),
            ]
        )

    def integrity(self) -> bool:
        """Whether the snapshot is internally consistent — the recomputed content
        hash matches, and the offset list lines up with the captured rows."""
        return (
            len(self.offsets) == self.dataset.row_count
            and self._compute_hash() == self.content_hash
        )

    @property
    def event_count(self) -> int:
        """The number of events the window captured."""
        return self.dataset.row_count

    def catalog(self, *, table: str | None = None) -> DataCatalog:
        """A single-table :class:`~vincio.data.DataCatalog` over the captured
        events, registered under *table* (defaulting to the stream name) — the
        grounding source a windowed query re-executes against."""
        return DataCatalog.of(self.dataset, name=table or self.stream)

    def to_stream(self) -> RowStream:
        """The captured events as a :class:`~vincio.data.RowStream` (so the
        bounded-pass operators apply uniformly)."""
        return RowStream.from_dataset(self.dataset)

    def event_offset(self, row: int) -> int:
        """The global stream offset of captured dataset row *row*."""
        return self.offsets[row]

    def label(self) -> str:
        """A short, human-readable window label, e.g. ``[0.0, 60.0) NA``."""
        keyed = f" {','.join(str(k) for k in self.key)}" if self.key else ""
        return f"[{self.start:g}, {self.end:g}){keyed}"


# --------------------------------------------------------------------------- #
# Per-window results                                                           #
# --------------------------------------------------------------------------- #


def _remap_cells(
    cells: Iterable[CellCitation], window: CapturedWindow
) -> list[EventCitation]:
    """Re-key window-local cell citations (``table#r<i>!col``) to stream-offset
    event citations (``stream@<offset>!col``), de-duplicated in order."""
    seen: set[tuple[int, str, str]] = set()
    out: list[EventCitation] = []
    for cell in cells:
        if not 0 <= cell.row < len(window.offsets):
            continue
        offset = window.offsets[cell.row]
        key = (offset, cell.column, cell.result_column)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            EventCitation(
                stream=window.stream,
                offset=offset,
                column=cell.column,
                value=cell.value,
                result_column=cell.result_column,
            )
        )
    return out


class WindowedProfile(BaseModel):
    """A deterministic, bounded-memory column profile of one closed window.

    Wraps a :class:`~vincio.data.DatasetProfile` of the window's events with the
    :class:`CapturedWindow` it was computed over. :meth:`verify` re-profiles the
    captured window and confirms the profile re-derives from it, offline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window: CapturedWindow
    profile: DatasetProfile

    @property
    def event_offsets(self) -> list[int]:
        """Every event offset the profile rests on (the whole window)."""
        return list(self.window.offsets)

    def cite_events(self) -> list[str]:
        """The stable event locators (``stream@<offset>``) the profile rests on."""
        return [f"{self.window.stream}@{o}" for o in self.window.offsets]

    def verify(self) -> bool:
        """Re-profile the captured window and confirm the profile re-derives from
        it. Returns ``False`` on any divergence or a broken snapshot."""
        if not self.window.integrity():
            return False
        return profile_dataset(self.window.dataset) == self.profile

    def summary(self) -> str:
        """A one-line human summary of the windowed profile."""
        return f"profile {self.window.label()}: {self.window.event_count:,} events"


class WindowedQueryResult(BaseModel):
    """A governed query's result over one closed window, **event-level cited**.

    Wraps the cell-cited :class:`~vincio.data.QueryResult` computed over the
    captured window with the :class:`CapturedWindow` itself. :meth:`cite_events`
    reports the exact source *events* (not rows) a result cell rests on, and
    :meth:`verify` re-executes the query against the captured window and confirms
    the answer — and every cited event — re-derives from it, offline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window: CapturedWindow
    result: QueryResult

    @property
    def rows(self) -> list[list[Any]]:
        """The result rows."""
        return self.result.rows

    @property
    def columns(self) -> list[str]:
        """The result column names."""
        return self.result.columns

    @property
    def row_count(self) -> int:
        """The number of result rows."""
        return self.result.row_count

    @property
    def coverage(self) -> LineageCoverage:
        """The lineage coverage of the underlying query result."""
        return self.result.coverage

    def value(self, row: int = 0, column: str | int = 0) -> Any:
        """One result cell, by row index and column name or index."""
        return self.result.value(row, column)

    def event_citations(self, row: int, column: str | int | None = None) -> list[EventCitation]:
        """The source events a result cell (or whole result row) rests on."""
        return _remap_cells(self.result.citations(row, column), self.window)

    def cite_events(self, row: int, column: str | int | None = None) -> list[str]:
        """The distinct stable event locators (``stream@<offset>!<col>``) a result
        cell (or whole result row) rests on."""
        return [c.ref for c in self.event_citations(row, column)]

    def verify(self, *, engine: QueryEngine | None = None) -> bool:
        """Re-execute the query against the captured window and confirm the result,
        every cited cell, and the captured events re-derive from the bytes. Also
        confirms every cited event offset belongs to the window. Returns ``False``
        on any divergence."""
        if not self.window.integrity():
            return False
        table = self.result.plan.tables[0] if self.result.plan.tables else self.window.stream
        if not self.result.verify(self.window.catalog(table=table), engine=engine):
            return False
        valid = set(self.window.offsets)
        for prov in self.result.provenance:
            for cell in prov.cells:
                if not 0 <= cell.row < len(self.window.offsets):
                    return False
                if self.window.offsets[cell.row] not in valid:
                    return False
        return True

    def to_evidence(self, **kwargs: Any) -> TableEvidence:
        """Project the result into cited ``modality="table"`` evidence the context
        compiler scores, budgets, orders, and cites; the window's bounds and the
        stream travel in the evidence metadata."""
        ev = self.result.to_evidence(**kwargs)
        ev.metadata = {
            **ev.metadata,
            "stream": self.window.stream,
            "window_kind": str(self.window.kind),
            "window_start": self.window.start,
            "window_end": self.window.end,
            "window_events": self.window.event_count,
        }
        return ev

    def summary(self) -> str:
        """A one-line human summary of the windowed query result."""
        return (
            f"query {self.window.label()}: {self.row_count} row(s) "
            f"over {self.window.event_count:,} events"
        )


class WindowedMetricResult(BaseModel):
    """A governed metric's answer over one closed window, **event-level cited**.

    Wraps the :class:`~vincio.data.MetricResult` computed over the captured window
    with the :class:`SemanticLayer` it was resolved through (so :meth:`verify` is
    self-contained). :meth:`verify` proves, offline, that the SQL was the layer's
    canonical compilation of the metric and that the answer — and every cited
    event — re-derives from the captured window."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window: CapturedWindow
    result: MetricResult
    layer: SemanticLayer

    @property
    def metrics(self) -> list[str]:
        """The governed metrics this result computed."""
        return self.result.metrics

    @property
    def dimensions(self) -> list[str]:
        """The dimensions the metrics were broken down by."""
        return self.result.dimensions

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
        """One result cell (``column`` defaults to the first metric)."""
        return self.result.value(row, column)

    def event_citations(self, row: int = 0, column: str | int | None = None) -> list[EventCitation]:
        """The source events a metric cell (or row) rests on."""
        return _remap_cells(self.result.result.citations(row, column), self.window)

    def cite_events(self, row: int = 0, column: str | int | None = None) -> list[str]:
        """The distinct stable event locators a metric cell (or row) rests on."""
        return [c.ref for c in self.event_citations(row, column)]

    def verify(self, *, engine: QueryEngine | None = None) -> bool:
        """Prove the answer is the governed metric, re-derived from the captured
        window. Returns ``False`` unless the layer's definitions are unchanged, the
        SQL equals the layer's canonical compilation, and the result re-derives."""
        if not self.window.integrity():
            return False
        table = self.result.result.plan.tables[0] if self.result.result.plan.tables else self.layer.table
        return self.result.verify(self.layer, self.window.catalog(table=table), engine=engine)

    def summary(self) -> str:
        """A one-line human summary of the windowed metric result."""
        return (
            f"metric {self.window.label()}: {', '.join(self.metrics)} "
            f"over {self.window.event_count:,} events"
        )


class WindowedQualityReport(BaseModel):
    """The data-quality screen of one closed window.

    Wraps the :class:`~vincio.data.DataQualityReport` of the window's events with
    the :class:`CapturedWindow` and the :class:`~vincio.data.DataQualityRails`
    that produced it (so :meth:`verify` re-screens self-contained). The screen
    covers the exact events the window captured; :meth:`offending_events` maps each
    violation to the offsets of the events that breached it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window: CapturedWindow
    report: DataQualityReport
    rails: DataQualityRails
    offenders: dict[str, list[int]] = Field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        """Whether the window passed (no blocking violation fired)."""
        return self.report.allowed

    @property
    def event_offsets(self) -> list[int]:
        """Every event offset the screen ran over (the whole window)."""
        return list(self.window.offsets)

    def offending_events(self, column: str, rule: str) -> list[int]:
        """The stream offsets of the events that breached ``column``'s ``rule``."""
        return list(self.offenders.get(f"{column}:{rule}", []))

    def verify(self) -> bool:
        """Re-screen the captured window with the same rails and confirm the report
        re-derives from it. Returns ``False`` on any divergence or a broken
        snapshot."""
        if not self.window.integrity():
            return False
        return self.rails.check(self.window.dataset) == self.report

    def summary(self) -> str:
        """A one-line human summary of the windowed screen."""
        verdict = "passed" if self.allowed else "blocked"
        return (
            f"quality {self.window.label()}: {verdict}, "
            f"{len(self.report.violations)} finding(s) over {self.window.event_count:,} events"
        )


class WindowedAggregation(BaseModel):
    """A bounded-memory group-by over one closed window.

    Wraps the :class:`~vincio.data.StreamAggregation` computed over the captured
    window — the fastest windowed reduction, riding
    :func:`~vincio.data.stream_aggregate` directly. Aggregate lineage is
    window-level (each group row rests on the window's events); use
    :meth:`StreamWindow.query` for a cited, cell-level group-by. :meth:`verify`
    re-aggregates the captured window and confirms the result re-derives."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window: CapturedWindow
    aggregation: StreamAggregation

    @property
    def result(self) -> Dataset:
        """The aggregated result table — one row per group."""
        return self.aggregation.result

    @property
    def event_offsets(self) -> list[int]:
        """Every event offset the aggregation rests on (the whole window)."""
        return list(self.window.offsets)

    def verify(self) -> bool:
        """Re-aggregate the captured window and confirm the result re-derives.
        Returns ``False`` on any divergence or a broken snapshot."""
        if not self.window.integrity():
            return False
        replay = stream_aggregate(
            self.window.to_stream(),
            group_by=self.aggregation.group_by,
            measures=_measures_from_columns(
                self.aggregation.group_by, self.aggregation.measure_columns
            ),
            max_groups=self.aggregation.max_groups,
        )
        return replay.result.rows() == self.result.rows() and replay.result.column_names == self.result.column_names

    def summary(self) -> str:
        """A one-line human summary of the windowed aggregation."""
        return (
            f"aggregate {self.window.label()}: {self.aggregation.group_count:,} group(s) "
            f"over {self.window.event_count:,} events"
        )


def _measures_from_columns(group_by: Sequence[str], measure_columns: Sequence[str]) -> dict[str, list[str]]:
    """Reconstruct the ``{column: [agg, ...]}`` measure spec from a
    :class:`StreamAggregation`'s ``<col>_<agg>`` output column names (for replay
    verification)."""
    measures: dict[str, list[str]] = {}
    for column in measure_columns:
        col, _, agg = column.rpartition("_")
        if col and agg:
            measures.setdefault(col, []).append(agg)
    return measures


# --------------------------------------------------------------------------- #
# The stateful, bounded window assigner                                        #
# --------------------------------------------------------------------------- #


class _OpenWindow:
    """One in-flight (not yet closed) window's accumulating event buffer."""

    __slots__ = ("start", "end", "key", "offsets", "rows", "last_time")

    def __init__(self, start: float, end: float, key: tuple[Any, ...]) -> None:
        self.start = start
        self.end = end
        self.key = key
        self.offsets: list[int] = []
        self.rows: list[list[Any]] = []
        self.last_time = start

    def add(self, offset: int, row: Sequence[Any], time: float) -> None:
        self.offsets.append(offset)
        self.rows.append(list(row))
        self.last_time = max(self.last_time, time)


class _WindowAssigner:
    """Assigns stream events to windows by event time and emits each window once
    it closes, holding only the open windows resident.

    Tumbling and sliding windows are aligned to a fixed grid and closed when the
    watermark (the latest event time seen) passes their end plus the allowed
    lateness; an event older than that — belonging to an already-closed window —
    is dropped and counted, never silently folded into a stale window. Session
    windows grow while events keep arriving within the gap (per key) and close
    once the watermark moves past the last event plus the gap."""

    def __init__(self, window: StreamWindow, columns: Sequence[ColumnSchema], *, table: str) -> None:
        self.window = window
        self.columns = list(columns)
        self.table = table
        self.schema = list(columns)
        names = [c.name for c in columns]
        self._time_index: int | None = None
        if window.time_column is not None:
            if window.time_column not in names:
                raise StreamError(
                    f"no column named {window.time_column!r} to use as the event-time "
                    f"column; stream columns are {names}"
                )
            self._time_index = names.index(window.time_column)
        self._key_indices: list[int] = []
        for key in window.key_by:
            if key not in names:
                raise StreamError(f"no column named {key!r} to partition windows by")
            self._key_indices.append(names.index(key))
        self._open: dict[tuple[Any, ...], _OpenWindow] = {}
        self._closed_sessions: list[CapturedWindow] = []
        self._watermark = -math.inf
        self.events_seen = 0
        self.late_events = 0

    # -- event time / key extraction ------------------------------------------

    def _time_of(self, offset: int, row: Sequence[Any]) -> float:
        if self._time_index is None:
            return float(offset)  # processing time: the arrival offset is the clock
        value = row[self._time_index] if self._time_index < len(row) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise StreamError(
                f"event-time column {self.window.time_column!r} must be a number "
                f"(epoch seconds or a logical counter), got {value!r}"
            )
        return float(value)

    def _key_of(self, row: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(row[i] if i < len(row) else None for i in self._key_indices)

    # -- the driving step ------------------------------------------------------

    def push(self, row: Sequence[Any]) -> list[CapturedWindow]:
        """Admit one event; return the windows this event closed (possibly none)."""
        offset = self.events_seen
        self.events_seen += 1
        time = self._time_of(offset, row)
        key = self._key_of(row)
        self._watermark = max(self._watermark, time)
        if self.window.kind is WindowKind.SESSION:
            self._add_session(offset, row, time, key)
        else:
            self._add_grid(offset, row, time, key)
        return self._harvest()

    def drain(self) -> list[CapturedWindow]:
        """Close every still-open window at end of stream, in deterministic
        order."""
        closed = sorted(self._open.values(), key=lambda w: (w.end, w.start, _sort_key(w.key)))
        self._open.clear()
        return [self._capture(w, closed_by="drain") for w in closed]

    # -- assignment ------------------------------------------------------------

    def _add_grid(self, offset: int, row: Sequence[Any], time: float, key: tuple[Any, ...]) -> None:
        for start in self._covering_starts(time):
            handle = (key, start)
            window = self._open.get(handle)
            if window is None:
                end = start + self.window.size
                if end + self.window.lateness <= self._watermark:
                    # The window is already past its close horizon — this is a late
                    # event for a window that has been emitted; drop and count it.
                    self.late_events += 1
                    continue
                window = _OpenWindow(start, end, key)
                self._open[handle] = window
                self._guard_open_count()
            window.add(offset, row, time)

    def _covering_starts(self, time: float) -> list[float]:
        size = self.window.size
        slide = self.window.slide
        origin = self.window.origin
        if time < origin:
            self.late_events += 1
            return []
        # Highest aligned start at or before ``time``; then step back by ``slide``
        # while the window still covers ``time`` (start + size > time).
        highest = origin + math.floor((time - origin) / slide) * slide
        starts: list[float] = []
        start = highest
        while start + size > time and start >= origin:
            if start <= time:
                starts.append(round(start, 9))
            start -= slide
        return starts

    def _add_session(self, offset: int, row: Sequence[Any], time: float, key: tuple[Any, ...]) -> None:
        gap = self.window.gap
        window = self._open.get((key,))
        if window is not None and time > window.last_time + gap:
            # The quiet gap closed the prior session before this event opens a new
            # one; emit it now (ahead of any watermark-closed windows this push).
            self._open.pop((key,))
            self._closed_sessions.append(self._capture(window, closed_by="gap"))
            window = None
        if window is None:
            window = _OpenWindow(time, time, key)
            self._open[(key,)] = window
            self._guard_open_count()
        window.add(offset, row, time)
        window.end = window.last_time  # the session spans [first event, last event]

    def _harvest(self) -> list[CapturedWindow]:
        closed, self._closed_sessions = self._closed_sessions, []
        ready: list[tuple[tuple[Any, ...], _OpenWindow]] = []
        for handle, window in self._open.items():
            horizon = (window.last_time + self.window.gap) if self.window.kind is WindowKind.SESSION else window.end
            if horizon + self.window.lateness <= self._watermark:
                ready.append((handle, window))
        for handle, _ in ready:
            self._open.pop(handle, None)
        ready.sort(key=lambda hw: (hw[1].end, hw[1].start, _sort_key(hw[1].key)))
        emitted = [self._capture(w, closed_by="watermark") for _, w in ready]
        return closed + emitted

    # -- capture ---------------------------------------------------------------

    def _capture(self, window: _OpenWindow, *, closed_by: str) -> CapturedWindow:
        dataset = Dataset.from_rows(
            window.rows, list(self.columns), name=self.table, source=self.table
        )
        return CapturedWindow.build(
            stream=self.table,
            kind=self.window.kind,
            start=window.start,
            end=window.end,
            key=list(window.key),
            offsets=window.offsets,
            dataset=dataset,
            closed_by=closed_by,
        )

    def _guard_open_count(self) -> None:
        if len(self._open) > self.window.max_open_windows:
            raise StreamError(
                f"the number of simultaneously-open windows exceeded the bound of "
                f"{self.window.max_open_windows:,}; the stream is more out of order "
                "than the allowed lateness, or the key cardinality is unbounded — "
                "raise max_open_windows, coarsen key_by, or widen the window"
            )


def _sort_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(str(k) for k in key)


# --------------------------------------------------------------------------- #
# The windowing policy + the streaming analogues of the batch primitives       #
# --------------------------------------------------------------------------- #


class StreamWindow(BaseModel):
    """A windowing policy over an unbounded event stream, carrying the streaming
    analogues of the data plane's batch primitives.

    Build one with :meth:`tumbling`, :meth:`sliding`, or :meth:`session`, then
    drive an unbounded :class:`~vincio.data.RowStream` through one of the
    operators — :meth:`profile`, :meth:`query`, :meth:`query_metric`,
    :meth:`screen`, or :meth:`aggregate`. Each returns a lazy iterator that emits
    one result per closed window, holding only the open windows resident.

    Events are windowed by **event time** (the ``time_column``, a number — epoch
    seconds or a logical counter) or, when none is given, by **processing time**
    (the arrival offset). ``key_by`` partitions the stream into independent
    per-key windows. Windows close on a watermark (the latest event time seen)
    past their end plus ``lateness``; a late event for an already-closed window is
    dropped and counted, never silently folded in."""

    model_config = ConfigDict(frozen=True)

    kind: WindowKind
    size: float = 0.0
    slide: float = 0.0
    gap: float = 0.0
    origin: float = 0.0
    time_column: str | None = None
    key_by: tuple[str, ...] = ()
    lateness: float = 0.0
    table: str = DEFAULT_STREAM_TABLE
    max_rows: int = 10_000
    max_open_windows: int = DEFAULT_MAX_OPEN_WINDOWS

    @model_validator(mode="after")
    def _validate(self) -> StreamWindow:
        if self.kind in (WindowKind.TUMBLING, WindowKind.SLIDING):
            if self.size <= 0:
                raise StreamError(f"a {self.kind} window needs size > 0, got {self.size}")
            if self.slide <= 0:
                raise StreamError(f"a {self.kind} window needs slide > 0, got {self.slide}")
            if self.slide > self.size:
                raise StreamError(
                    f"slide ({self.slide}) cannot exceed size ({self.size}); for "
                    "non-overlapping windows use StreamWindow.tumbling"
                )
        elif self.kind is WindowKind.SESSION:
            if self.gap <= 0:
                raise StreamError(f"a session window needs gap > 0, got {self.gap}")
        if self.lateness < 0:
            raise StreamError(f"lateness cannot be negative, got {self.lateness}")
        return self

    # -- construction ----------------------------------------------------------

    @classmethod
    def tumbling(
        cls,
        size: float,
        *,
        time_column: str | None = None,
        key_by: Sequence[str] = (),
        origin: float = 0.0,
        lateness: float = 0.0,
        table: str = DEFAULT_STREAM_TABLE,
        max_rows: int = 10_000,
        max_open_windows: int = DEFAULT_MAX_OPEN_WINDOWS,
    ) -> StreamWindow:
        """A fixed-size, non-overlapping window of width *size* (in the event-time
        unit, or in events for processing time)."""
        return cls(
            kind=WindowKind.TUMBLING,
            size=size,
            slide=size,
            origin=origin,
            time_column=time_column,
            key_by=tuple(key_by),
            lateness=lateness,
            table=table,
            max_rows=max_rows,
            max_open_windows=max_open_windows,
        )

    @classmethod
    def sliding(
        cls,
        size: float,
        slide: float,
        *,
        time_column: str | None = None,
        key_by: Sequence[str] = (),
        origin: float = 0.0,
        lateness: float = 0.0,
        table: str = DEFAULT_STREAM_TABLE,
        max_rows: int = 10_000,
        max_open_windows: int = DEFAULT_MAX_OPEN_WINDOWS,
    ) -> StreamWindow:
        """A fixed-size window of width *size* that advances by *slide* (< *size*),
        so windows overlap and an event belongs to ``ceil(size / slide)`` of
        them."""
        return cls(
            kind=WindowKind.SLIDING,
            size=size,
            slide=slide,
            origin=origin,
            time_column=time_column,
            key_by=tuple(key_by),
            lateness=lateness,
            table=table,
            max_rows=max_rows,
            max_open_windows=max_open_windows,
        )

    @classmethod
    def session(
        cls,
        gap: float,
        *,
        time_column: str | None = None,
        key_by: Sequence[str] = (),
        lateness: float = 0.0,
        table: str = DEFAULT_STREAM_TABLE,
        max_rows: int = 10_000,
        max_open_windows: int = DEFAULT_MAX_OPEN_WINDOWS,
    ) -> StreamWindow:
        """A dynamic window that grows while events keep arriving within *gap* and
        closes once the feed (for a key) goes quiet longer than the gap. Session
        windows are almost always partitioned (``key_by``)."""
        return cls(
            kind=WindowKind.SESSION,
            gap=gap,
            time_column=time_column,
            key_by=tuple(key_by),
            lateness=lateness,
            table=table,
            max_rows=max_rows,
            max_open_windows=max_open_windows,
        )

    # -- window assignment -----------------------------------------------------

    def assign(self, stream: RowStream | Dataset, *, table: str | None = None) -> Iterator[CapturedWindow]:
        """Iterate an unbounded stream and yield each window's bounded
        :class:`CapturedWindow` as it closes, holding only the open windows
        resident. The escape hatch under every operator."""
        rows = _coerce_stream(stream)
        assigner = _WindowAssigner(self, rows.columns, table=table or self.table or rows.name or DEFAULT_STREAM_TABLE)
        for row in rows.rows():
            yield from assigner.push(row)
        yield from assigner.drain()

    async def assign_async(
        self,
        source: AsyncIterator[Sequence[Any]] | AsyncIterator[Mapping[str, Any]],
        columns: Sequence[ColumnSchema] | Sequence[str],
        *,
        table: str | None = None,
        extract: Callable[[Any], Sequence[Any] | Mapping[str, Any] | None] | None = None,
    ) -> AsyncIterator[CapturedWindow]:
        """Iterate a live **async** event source (e.g. a realtime session's
        events) and yield each window's :class:`CapturedWindow` as it closes — the
        live counterpart of :meth:`assign`.

        ``columns`` declares the event schema the rows align to. Each item is a row
        (positional) or a mapping; pass ``extract`` to pull a row/mapping out of a
        richer event object (returning ``None`` skips it)."""
        schema = _coerce_columns(columns)
        names = [c.name for c in schema]
        assigner = _WindowAssigner(self, schema, table=table or self.table or DEFAULT_STREAM_TABLE)
        async for item in source:
            row = extract(item) if extract is not None else item
            if row is None:
                continue
            if isinstance(row, Mapping):
                row = [row.get(n) for n in names]
            for captured in assigner.push(row):
                yield captured
        for captured in assigner.drain():
            yield captured

    # -- per-window appliers ---------------------------------------------------

    def profile_window(self, window: CapturedWindow, **kwargs: Any) -> WindowedProfile:
        """Profile one already-captured window."""
        return WindowedProfile(window=window, profile=profile_dataset(window.dataset, **kwargs))

    def query_window(
        self,
        window: CapturedWindow,
        request: str,
        *,
        dialect: str = "sql",
        ops: list[Any] | None = None,
        question: str = "",
        engine: QueryEngine | None = None,
        injection_detector: InjectionDetector | None = None,
        screen_question: bool = True,
    ) -> WindowedQueryResult:
        """Run a governed, read-only-verified query over one already-captured
        window."""
        result = query_dataset(
            request,
            window.catalog(),
            dialect=dialect,
            ops=ops,
            question=question,
            table=window.stream,
            max_rows=self.max_rows,
            engine=engine,
            injection_detector=injection_detector,
            screen_question=screen_question,
        )
        return WindowedQueryResult(window=window, result=result)

    def metric_window(
        self,
        window: CapturedWindow,
        request: Any,
        *,
        layer: SemanticLayer,
        by: Sequence[str] | None = None,
        where: Sequence[str] | None = None,
        order_by: str = "",
        descending: bool = False,
        limit: int | None = None,
        engine: QueryEngine | None = None,
        injection_detector: InjectionDetector | None = None,
        screen: bool = True,
    ) -> WindowedMetricResult:
        """Compute a governed metric over one already-captured window."""
        from .semantic import query_metric

        result = query_metric(
            request,
            window.catalog(table=layer.table),
            layer=layer,
            by=by,
            where=where,
            order_by=order_by,
            descending=descending,
            limit=limit,
            engine=engine,
            max_rows=self.max_rows,
            injection_detector=injection_detector,
            screen=screen,
        )
        return WindowedMetricResult(window=window, result=result, layer=layer)

    def screen_window(self, window: CapturedWindow, rails: DataQualityRails) -> WindowedQualityReport:
        """Screen one already-captured window with the data-quality rails."""
        report = rails.check(window.dataset)
        offenders = _offending_offsets(window, rails)
        return WindowedQualityReport(window=window, report=report, rails=rails, offenders=offenders)

    def aggregate_window(
        self,
        window: CapturedWindow,
        *,
        group_by: str | Sequence[str],
        measures: Mapping[str, str | Sequence[str]] | None = None,
        max_groups: int = 1_000_000,
    ) -> WindowedAggregation:
        """Reduce one already-captured window with a bounded-memory group-by,
        riding :func:`~vincio.data.stream_aggregate`."""
        aggregation = stream_aggregate(
            window.to_stream(), group_by=group_by, measures=measures, max_groups=max_groups
        )
        return WindowedAggregation(window=window, aggregation=aggregation)

    # -- streaming operators (one result per closed window) --------------------

    def profile(self, stream: RowStream | Dataset, *, table: str | None = None, **kwargs: Any) -> Iterator[WindowedProfile]:
        """Profile each window of the stream — a deterministic, bounded-memory
        column profile per closed window."""
        for window in self.assign(stream, table=table):
            yield self.profile_window(window, **kwargs)

    def query(
        self,
        stream: RowStream | Dataset,
        request: str,
        *,
        table: str | None = None,
        **kwargs: Any,
    ) -> Iterator[WindowedQueryResult]:
        """Run a governed, read-only-verified, **event-cited** query over each
        window of the stream."""
        for window in self.assign(stream, table=table):
            yield self.query_window(window, request, **kwargs)

    def query_metric(
        self,
        stream: RowStream | Dataset,
        request: Any,
        *,
        layer: SemanticLayer,
        **kwargs: Any,
    ) -> Iterator[WindowedMetricResult]:
        """Compute a governed metric over each window of the stream."""
        for window in self.assign(stream, table=layer.table):
            yield self.metric_window(window, request, layer=layer, **kwargs)

    def screen(self, stream: RowStream | Dataset, rails: DataQualityRails, *, table: str | None = None) -> Iterator[WindowedQualityReport]:
        """Screen each window of the stream against the data-quality rails."""
        for window in self.assign(stream, table=table):
            yield self.screen_window(window, rails)

    def aggregate(
        self,
        stream: RowStream | Dataset,
        *,
        group_by: str | Sequence[str],
        measures: Mapping[str, str | Sequence[str]] | None = None,
        max_groups: int = 1_000_000,
        table: str | None = None,
    ) -> Iterator[WindowedAggregation]:
        """Reduce each window of the stream with a bounded-memory group-by, riding
        :func:`~vincio.data.stream_aggregate`."""
        for window in self.assign(stream, table=table):
            yield self.aggregate_window(window, group_by=group_by, measures=measures, max_groups=max_groups)


# --------------------------------------------------------------------------- #
# The app-governed driver — audited windows + the live realtime path            #
# --------------------------------------------------------------------------- #


class StreamingAnalytics:
    """Governed real-time analytics bound to a :class:`~vincio.core.ContextApp`.

    The app-aware front for :class:`StreamWindow`: every emitted window lands on
    the app's hash-chained audit log (``stream_window``), the app's injection
    detector screens any natural-language question, and the same window operators —
    :meth:`profile`, :meth:`query`, :meth:`query_metric`, :meth:`screen`,
    :meth:`aggregate` — are available over a **replayed** stream (sync) or a
    **live** async source such as a realtime session (:meth:`drive`). Build one
    with :meth:`~vincio.core.ContextApp.stream_analytics`.

    Each operator yields one governed, cited, offline-verifiable result per closed
    window — so a live dashboard tile or an alerting rule is just a query over a
    window, never a hosted stream processor."""

    def __init__(
        self,
        app: Any,
        window: StreamWindow,
        *,
        table: str = DEFAULT_STREAM_TABLE,
        layer: SemanticLayer | None = None,
    ) -> None:
        self.app = app
        self.window = window
        self.table = table
        self.layer = layer
        self._injection: Any = None

    # -- audit -----------------------------------------------------------------

    def _audit(self, operation: str, window: CapturedWindow, *, details: dict[str, Any] | None = None) -> None:
        self.app.audit.record(
            "stream_window",
            resource=window.stream,
            details={
                "operation": operation,
                "kind": str(window.kind),
                "key": [str(k) for k in window.key],
                "start": window.start,
                "end": window.end,
                "events": window.event_count,
                "content_hash": window.content_hash,
                **(details or {}),
            },
        )

    def _detector(self) -> Any:
        # One detector per driver — every window's natural-language question is
        # screened on the same deterministic rail (created once, not per window).
        if self._injection is None:
            from ..security.injection import InjectionDetector

            self._injection = InjectionDetector()
        return self._injection

    # -- replayed (sync) operators ---------------------------------------------

    def profile(self, stream: RowStream | Dataset, **kwargs: Any) -> Iterator[WindowedProfile]:
        """Profile each window of a replayed stream, auditing every result."""
        for window in self.window.assign(stream, table=self.table):
            result = self.window.profile_window(window, **kwargs)
            self._audit("profile", window)
            yield result

    def query(self, stream: RowStream | Dataset, request: str, **kwargs: Any) -> Iterator[WindowedQueryResult]:
        """Run a governed, event-cited query over each window of a replayed
        stream, auditing every result."""
        kwargs.setdefault("injection_detector", self._detector())
        for window in self.window.assign(stream, table=self.table):
            result = self.window.query_window(window, request, **kwargs)
            self._audit(
                "query",
                window,
                details={
                    "row_count": result.row_count,
                    "lineage_coverage": str(result.coverage),
                    "result_hash": result.result.result_hash,
                },
            )
            yield result

    def query_metric(
        self,
        stream: RowStream | Dataset,
        request: Any,
        *,
        layer: SemanticLayer | None = None,
        **kwargs: Any,
    ) -> Iterator[WindowedMetricResult]:
        """Compute a governed metric over each window of a replayed stream,
        auditing every result."""
        resolved = layer or self.layer
        if resolved is None:
            raise StreamError("query_metric needs a SemanticLayer (pass layer= or set it on the driver)")
        kwargs.setdefault("injection_detector", self._detector())
        for window in self.window.assign(stream, table=resolved.table):
            result = self.window.metric_window(window, request, layer=resolved, **kwargs)
            self._audit(
                "metric",
                window,
                details={
                    "metrics": result.metrics,
                    "dimensions": result.dimensions,
                    "row_count": result.row_count,
                    "result_hash": result.result.result.result_hash,
                },
            )
            yield result

    def screen(self, stream: RowStream | Dataset, rails: DataQualityRails, **kwargs: Any) -> Iterator[WindowedQualityReport]:
        """Screen each window of a replayed stream, auditing every verdict."""
        for window in self.window.assign(stream, table=self.table):
            result = self.window.screen_window(window, rails)
            self._audit(
                "screen",
                window,
                details={"allowed": result.allowed, "violations": len(result.report.violations)},
            )
            yield result

    def aggregate(
        self,
        stream: RowStream | Dataset,
        *,
        group_by: str | Sequence[str],
        measures: Mapping[str, str | Sequence[str]] | None = None,
        max_groups: int = 1_000_000,
    ) -> Iterator[WindowedAggregation]:
        """Reduce each window of a replayed stream with a bounded-memory group-by,
        auditing every result."""
        for window in self.window.assign(stream, table=self.table):
            result = self.window.aggregate_window(
                window, group_by=group_by, measures=measures, max_groups=max_groups
            )
            self._audit("aggregate", window, details={"groups": result.aggregation.group_count})
            yield result

    # -- live (async) driver ---------------------------------------------------

    async def drive(
        self,
        source: AsyncIterator[Any],
        columns: Sequence[ColumnSchema] | Sequence[str],
        *,
        apply: str = "query",
        request: Any = None,
        rails: DataQualityRails | None = None,
        layer: SemanticLayer | None = None,
        group_by: str | Sequence[str] | None = None,
        measures: Mapping[str, str | Sequence[str]] | None = None,
        extract: Callable[[Any], Sequence[Any] | Mapping[str, Any] | None] | None = None,
        on_window: Callable[[Any], Any] | None = None,
        max_windows: int | None = None,
    ) -> list[Any]:
        """Drive the chosen operator over a **live** async event source — a
        realtime session's events, a queue, any async iterator — and apply it to
        each window as it closes.

        ``apply`` selects the operator (``"query"`` / ``"profile"`` / ``"screen"``
        / ``"metric"`` / ``"aggregate"``); ``columns`` declares the event schema;
        ``extract`` pulls a row/mapping out of a richer event object (e.g.
        ``lambda e: e.data.get("row")``). Each window's governed result is audited,
        passed to ``on_window`` (awaited if it is a coroutine), and collected.
        ``max_windows`` bounds a live run. Returns the collected results."""
        resolved_layer = layer or self.layer
        table = resolved_layer.table if (apply == "metric" and resolved_layer) else self.table
        results: list[Any] = []
        async for window in self.window.assign_async(source, columns, table=table, extract=extract):
            result = self._apply(
                apply,
                window,
                request=request,
                rails=rails,
                layer=resolved_layer,
                group_by=group_by,
                measures=measures,
            )
            results.append(result)
            if on_window is not None:
                outcome = on_window(result)
                if hasattr(outcome, "__await__"):
                    await outcome
            if max_windows is not None and len(results) >= max_windows:
                break
        return results

    def _apply(
        self,
        apply: str,
        window: CapturedWindow,
        *,
        request: Any,
        rails: DataQualityRails | None,
        layer: SemanticLayer | None,
        group_by: str | Sequence[str] | None,
        measures: Mapping[str, str | Sequence[str]] | None,
    ) -> Any:
        result: Any
        if apply == "profile":
            result = self.window.profile_window(window)
            self._audit("profile", window)
        elif apply == "query":
            if request is None:
                raise StreamError("apply='query' needs a request= (SQL or a question)")
            result = self.window.query_window(window, request, injection_detector=self._detector())
            self._audit(
                "query",
                window,
                details={"row_count": result.row_count, "result_hash": result.result.result_hash},
            )
        elif apply == "metric":
            if layer is None or request is None:
                raise StreamError("apply='metric' needs a layer= and a request=")
            result = self.window.metric_window(window, request, layer=layer, injection_detector=self._detector())
            self._audit("metric", window, details={"metrics": result.metrics, "row_count": result.row_count})
        elif apply == "screen":
            if rails is None:
                raise StreamError("apply='screen' needs rails=")
            result = self.window.screen_window(window, rails)
            self._audit("screen", window, details={"allowed": result.allowed})
        elif apply == "aggregate":
            if group_by is None:
                raise StreamError("apply='aggregate' needs group_by=")
            result = self.window.aggregate_window(window, group_by=group_by, measures=measures)
            self._audit("aggregate", window, details={"groups": result.aggregation.group_count})
        else:
            raise StreamError(
                f"unknown operator {apply!r}; choose query / profile / screen / metric / aggregate"
            )
        return result


def _coerce_columns(columns: Sequence[ColumnSchema] | Sequence[str]) -> list[ColumnSchema]:
    items = list(columns)
    if items and isinstance(items[0], ColumnSchema):
        return [c for c in items if isinstance(c, ColumnSchema)]
    return [ColumnSchema(name=str(c), dtype=DataType.STR, nullable=True) for c in items]


def _offending_offsets(window: CapturedWindow, rails: DataQualityRails) -> dict[str, list[int]]:
    """Map each value-level violation to the stream offsets of the events that
    breached it.

    The window is bounded, so it is re-screened with a clone of the rails that
    captures *every* offender (not just the report's truncated examples), then the
    full set of breaching values is matched back to the captured column — so the
    pinpointing is complete, not example-capped. A breach without offending values
    (a null-rate or count-only finding) carries no per-event offsets."""
    detailed = DataQualityRails(
        list(rails.constraints),
        detect_anomalies=rails.detect_anomalies,
        anomaly_threshold=rails.anomaly_threshold,
        anomaly_action=rails.anomaly_action,
        max_examples=max(window.event_count, 1),
        pii_detector=rails._pii,
        secret_scanner=rails._secrets,
        injection_detector=rails._injection,
    )
    report = detailed.check(window.dataset)
    offenders: dict[str, list[int]] = {}
    for violation in report.violations:
        if not violation.examples or violation.column not in window.dataset.column_names:
            continue
        bad = set(violation.examples)
        column = window.dataset.column(violation.column)
        hits = [window.offsets[i] for i, value in enumerate(column) if value in bad]
        if hits:
            offenders[f"{violation.column}:{violation.rule}"] = hits
    return offenders
