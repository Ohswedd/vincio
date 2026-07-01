# Real-time and streaming analytics

The data plane's batch primitives — [profiling](dataset-profiling.md), governed
[text-to-query](governed-text-to-query.md), the
[governed metric](semantic-layer-and-governed-metrics.md), and the
[data-quality rails](dataset-profiling.md) — all assume a *bounded*
[`Dataset`](tabular-evidence.md). A live event feed has no end. This rung
re-expresses those primitives over an **unbounded event stream**, computed one
**window** at a time so the working set stays invariant to the event volume, with
every per-window answer citing the exact events it rests on and re-deriving
offline against the captured window.

Everything here is deterministic, dependency-free, and offline. It rides the
[streaming primitives](streaming-and-out-of-core.md) the out-of-core rung already
shipped (`RowStream` / `stream_aggregate`) and the governed, read-only-verified
query plane — never a hosted stream processor.

## StreamWindow — the windowing policy

A [`StreamWindow`](../../vincio/data/streaming_analytics.py) carries the streaming
analogues of the batch primitives. Build one with `tumbling`, `sliding`, or
`session`, then drive an unbounded [`RowStream`](streaming-and-out-of-core.md)
through one of the operators — `profile`, `query`, `query_metric`, `screen`, or a
bounded-memory `aggregate`. Each returns a lazy iterator that emits one result per
**closed window**, holding only the open windows resident.

```python
from vincio.data import StreamWindow, ColumnSchema, DataType, RowStream

schema = [ColumnSchema(name="ts", dtype=DataType.INT),
          ColumnSchema(name="region", dtype=DataType.STR),
          ColumnSchema(name="amount", dtype=DataType.FLOAT)]
stream = RowStream.from_rows(event_log, schema, name="orders")

win = StreamWindow.tumbling(size=60, time_column="ts", table="orders")
for wq in win.query(stream, "SELECT region, sum(amount) AS total FROM orders GROUP BY region"):
    print(wq.window.label(), wq.value(0, "total"))
    wq.cite_events(0, "total")     # the exact source events the figure rests on
    assert wq.verify()             # re-derives offline against the captured window
```

- **Tumbling** — fixed-size, non-overlapping windows that partition time.
- **Sliding** — fixed-size windows that advance by a `slide` smaller than `size`,
  so they overlap and an event belongs to more than one.
- **Session** — dynamic windows that grow while events keep arriving within a
  `gap` and close once the feed (for a key) goes quiet longer than the gap.

Events are windowed by **event time** (the `time_column`, a number — epoch
seconds or a logical counter) or, when none is given, by **processing time** (the
arrival offset). `key_by` partitions the stream into independent per-key windows.
A window closes on a **watermark** (the latest event time seen) past its end plus
an allowed `lateness`; a late event for an already-closed window is **dropped and
counted**, never silently folded into a stale window.

## Event-level provenance

Each closed window is captured into a bounded
[`CapturedWindow`](../../vincio/data/streaming_analytics.py): the events it held
(as a schema-bearing [`Dataset`](tabular-evidence.md)), their stable stream
**offsets**, the window bounds, and a content hash binding them. A windowed answer
cites the exact source **events** it rests on through an
[`EventCitation`](../../vincio/data/streaming_analytics.py) — `stream@<offset>!<column>`,
the streaming analogue of a [cell citation](governed-text-to-query.md)'s
`table#r<row>!<column>`.

`verify()` re-executes the operation against the captured window and confirms the
answer — and every cited event — re-derives from the bytes, fully offline, against
that bounded snapshot. A tampered captured event is caught.

```python
wq.window.event_count        # the bounded number of events the window held
wq.cite_events(0, "total")   # ['orders@0!amount', 'orders@2!amount']
wq.verify()                  # True; re-derives against the captured window
```

The windowed `profile`, `query_metric`, `screen`, and `aggregate` carry the same
discipline: `WindowedProfile`, `WindowedMetricResult`, `WindowedQualityReport`,
and `WindowedAggregation` each hold their captured window and `verify()` against
it. The quality screen additionally maps each violation to the **offsets of the
offending events** (`offending_events(column, rule)`).

## Bounded memory

The resident working set holds only the *open* windows — one per key for a
tumbling window — never the whole stream. A 200,000-event feed is processed inside
the same footprint as a 2,000-event one, as long as the window size is fixed. A
stream more out of order than the allowed lateness (or with unbounded key
cardinality) is **refused** by `max_open_windows` rather than growing without
bound.

## The governed driver and live feeds

`app.stream_analytics(window)` returns a
[`StreamingAnalytics`](../../vincio/data/streaming_analytics.py) — the app-aware
front that audits every emitted window on the hash-chained audit log
(`stream_window`), screens any natural-language question on the same deterministic
injection rail, and drives a **live** async source (a queue, a websocket, a
[realtime session](../../vincio/realtime/session.py)'s events) exactly as it
replays a log.

```python
analytics = app.stream_analytics(win, table="orders")

# Replayed log (sync):
for wq in analytics.query(stream, "SELECT region, sum(amount) AS total FROM orders GROUP BY region"):
    dashboard.update(wq)

# Live feed (async) with an alerting callback:
await analytics.drive(live_source, schema, apply="query", request=QUERY,
                      on_window=alert, max_windows=None)
```

So a live dashboard tile or an alerting rule is just a governed, cited,
budget-bounded query over a window — never a hosted stream processor.

## Guarantees

DataPlaneBench holds the rung under CI-gated budgets and published SLOs: a
**windowed-correctness** check (the windowed group-by over a replayed log equals
the brute-force ground truth of bucketing every event), a **bounded-memory**
invariant (the per-window pipeline's peak resident set for a 100×-longer stream
stays within a small factor of the shorter one), and an **incremental-provenance**
check (every window verifies offline against its capture, every cited offset is
in-window, and a tampered captured event is caught).

## See also

- [Streaming and out-of-core bulk processing](streaming-and-out-of-core.md) — the `RowStream` / `stream_aggregate` primitives this rung rides.
- [Governed text-to-query and cell-level provenance](governed-text-to-query.md) — the read-only-verified query plane each window runs through.
- [Semantic layer and governed metrics](semantic-layer-and-governed-metrics.md) — the governed metric a window computes.
- [Dataset profiling, sampling, and quality rails](dataset-profiling.md) — the bounded-pass profile and the rails applied per window.
- [Cross-org federated analytics](federated-data-engagement.md) — the same governed-metric primitives run across organizations, only aggregated results crossing the trust boundary.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Analyze data](../guides/analyze-data.md)
- [Example: 13_data_and_analytics.py](../../examples/13_data_and_analytics.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
