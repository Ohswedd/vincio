# Streaming and out-of-core bulk processing

A dataset does not have to fit in memory to be first-class evidence. The
[profiling and fit-in-window](dataset-profiling.md) rung already represents a
huge table inside the window as a profile plus a representative sample; this rung
processes a dataset *far larger than memory* in **bounded passes** — at high
throughput, inside a footprint that does not grow with the row count.

Everything here is deterministic (seeded reservoirs over a fixed input order) and
dependency-free; the CSV / JSON-Lines readers use only the standard library.

## RowStream — a lazy, re-iterable, schema-bearing handle

A [`RowStream`](../../vincio/data/streaming.py) is the out-of-core analogue of a
[`Dataset`](tabular-evidence.md): it never materializes its rows. It holds a
*factory* that produces a fresh row iterator on demand, so the same stream can be
profiled, fitted, sampled, aggregated, and encoded — each a single bounded pass.

```python
from vincio.data import RowStream

# From a file read line by line (CSV or JSON-Lines), never loaded whole:
stream = RowStream.open("events.csv")            # types inferred from a bounded peek
stream = RowStream.from_jsonl("events.ndjson")

# From in-memory data or a re-iterable generator factory:
stream = RowStream.from_records(records, name="events")
stream = RowStream.from_rows(lambda: generate(), schema=["id", "region", "amount"])

for chunk in stream.chunks(50_000):              # bounded Dataset chunks
    ...                                          # at most 50k rows resident at a time

profile = stream.profile()                       # one bounded pass; footprint tracks columns, not rows
fit = stream.fit(max_tokens=2_000)               # profile + budget-sized representative sample
sample = stream.sample(1_000, seed=7)            # uniform reservoir sample
```

A bare generator object is single-use; pass a sequence, a zero-argument callable,
or a file so the stream can be read in more than one pass. The CSV reader infers
each column's type from a bounded peek and coerces losslessly — a value that does
not round-trip exactly (a leading-zero id, a thousands-separated number) stays
text. Pass an explicit `schema=` to fix the types without peeking.

## Streaming aggregation — a group-by in a fixed footprint

`stream_aggregate` (and `RowStream.aggregate` / `app.aggregate_stream`) groups a
stream by one or more columns and reduces measures over each group in a single
bounded pass. The working set holds **one accumulator per distinct group**, never
the rows, so a table far larger than memory aggregates inside a fixed footprint.

```python
from vincio.data import stream_aggregate

agg = stream_aggregate(
    stream,
    group_by="region",
    measures={"amount": ["sum", "mean", "min", "max"]},
)
agg.result            # a small Dataset: one row per group (group keys, measures, count)
agg.group_count       # distinct groups — the size of the working set
agg.rows_processed    # rows folded in — can be far larger than memory
```

Each group's row `count` is always emitted. A group cardinality beyond
`max_groups` (default one million) is **refused** with a `StreamError` rather than
growing without bound — group by a coarser key or raise the bound. The result is
an ordinary [`Dataset`](tabular-evidence.md): encode it, profile it, or carry it
as cited evidence (`agg.to_evidence_item()`).

## Streaming compact encoding and compression

`encode_stream` (and `RowStream.encode`) renders a stream to the compact,
lossless [encoding](tabular-evidence.md) **header-once, row-by-row**, so a dataset
larger than memory is compressed in one bounded pass. With `compress=True` the
output is gzip-compressed (the encoding is highly compressible — a column's values
repeat); pass a binary `sink=` to stream the bytes straight to a file.

```python
from vincio.data import encode_stream

data = encode_stream(stream)                     # compact bytes, lossless round-trip
gz = encode_stream(stream, compress=True)        # gzip-compressed
with open("events.tbl", "wb") as handle:
    encode_stream(stream, sink=handle)           # streamed to disk, bounded memory
```

The streaming header omits the row count (a stream does not know its length up
front); the [decoder](../../vincio/core/tabular.py) reads rows to end-of-input, so
the round-trip stays exact.

## The streaming candidate pre-filter

The [context compiler](context-packets.md) collected every candidate and then
scored them all — fine for a normal corpus, but a 10,000+ evidence pool would pay
the full multi-signal scoring, the O(n²) dedup and conflict passes, and an
embedding per candidate before anything was dropped. The **streaming candidate
pre-filter** closes that path.

When `ContextCompilerOptions.max_candidates` (or the config field
`performance.max_context_candidates`) is set and the evidence pool exceeds it, a
single streaming pass keeps only the most promising candidates *before* the
expensive stages run:

- a bounded **min-heap** of at most `max_candidates` survivors ranked by a cheap
  lexical relevance proxy against the query (top-K, never the whole sorted pool);
- a capped **content-fingerprint set** that drops exact duplicates (a
  reservoir-style overflow stops the set growing on a huge stream).

So the expensive stages — and the resident vector footprint — never see more than
the cap, however large the raw corpus grows. Required and non-evidence candidates
(memory, tool results) always pass through, and every drop is recorded in the
excluded report (`prefiltered_low_relevance` / `prefiltered_duplicate`) so the
pruning is auditable. The pre-filter is off by default (the cap is `None`), so the
behavior of an unbounded compile is unchanged.

```python
from vincio.context.compiler import ContextCompiler, ContextCompilerOptions

compiler = ContextCompiler(ContextCompilerOptions(max_candidates=200))
compiled = await compiler.compile(objective=obj, user_input=ui, evidence=huge_pool, budget=budget)
compiler.prefilter_drops          # how many candidates were dropped before scoring
```

## Analytical pipelines at scale on the BatchRunner

`stream_map` (and `app.map_stream`) runs an analytical transform over a stream
*at scale* by chunking it into the existing
[`BatchRunner`](../../vincio/providers/batch.py): each bounded chunk becomes one
provider request (typically a prompt over the chunk's compact encoding), the set
is dispatched to a provider Batch API (half-cost, bounded concurrency), and the
responses are reconciled by chunk index — a missing or failed chunk surfaces as a
failed result rather than being dropped.

```python
def build(chunk, index):
    return ModelRequest(model=model, messages=[Message(role="user", content=chunk.encode())])

result = await app.map_stream(stream, build, chunk_rows=512)
result.succeeded      # one reconciled response per chunk
result.cost_usd       # total, at the discounted batch rate
```

## Guarantees

DataPlaneBench holds the rung under CI-gated budgets and published SLOs: a
**throughput** floor (rows/s through the streaming group-by and tokens/s through
the streaming encoder), a **memory-stays-bounded** invariant (the aggregation's
peak resident set for a 100×-larger stream stays within a small factor of the
smaller one), and a **pre-filter bounds the pool** check (a 10,000-candidate
corpus compiled under a cap keeps only the cap's worth while every relevant item
survives). `benchmarks/competitive.py` compares the streaming group-by's peak
memory head-to-head with loading every row first.

## See also

- [Tabular evidence and the compact encoder](tabular-evidence.md) — the encoding the stream renders.
- [Dataset profiling, sampling, and quality rails](dataset-profiling.md) — the bounded-pass profile and sample a stream reuses.
- [Context packets](context-packets.md) — the compiler the pre-filter bounds.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Analyze data](../guides/analyze-data.md)
- [Guide: Performance & streaming](../guides/performance.md)
- [Example: 13_data_and_analytics.py](../../examples/13_data_and_analytics.py)
- [Concept: Context packets & long-horizon governance](context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
