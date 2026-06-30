"""Real-time & streaming analytics over an unbounded event stream.

The data plane's batch primitives — profiling, governed text-to-query, the
governed metric, and the data-quality rails — all assume a bounded `Dataset`. A
live event feed has no end. This program walks the windowed data plane that
re-expresses those primitives over an **unbounded stream**, computed one window
at a time so the working set stays invariant to the event volume, fully offline
on a replayed event log:

  * `StreamWindow` — the windowing policy (`tumbling` / `sliding` / `session`),
    carrying windowed `profile`, `query`, `query_metric`, `screen`, and a
    bounded-memory `aggregate`, each emitting one result per closed window;
  * event-level provenance — each windowed answer cites the exact source *events*
    (`stream@<offset>`, not rows) it rests on and `verify()`s offline against its
    bounded captured window;
  * bounded memory — only the open windows are resident, so a 200,000-event feed
    is processed inside the same footprint as a 2,000-event one;
  * `app.stream_analytics` — the governed driver that audits every emitted window
    and drives a **live** async feed (a queue, a websocket, a realtime session's
    events) exactly as it replays a log — so a live dashboard tile or an alerting
    rule is just a governed, cited, budget-bounded query over a window.

Everything below is opt-in and additive; none of it touches a database, a network,
or a hosted stream processor.
"""

from __future__ import annotations

import asyncio
import tracemalloc

from _shared import example_provider

from vincio import ContextApp
from vincio.data import (
    Aggregation,
    ColumnConstraint,
    ColumnSchema,
    DataQualityRails,
    DataType,
    RowStream,
    SemanticLayer,
    StreamWindow,
)

SCHEMA = [
    ColumnSchema(name="ts", dtype=DataType.INT, unit="s"),
    ColumnSchema(name="region", dtype=DataType.STR),
    ColumnSchema(name="amount", dtype=DataType.FLOAT, unit="USD"),
]
REGIONS = ["NA", "EU", "APAC", "LATAM"]
WINDOW_QUERY = "SELECT region, sum(amount) AS total FROM orders GROUP BY region ORDER BY region"


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def order_events(n: int):
    """A re-iterable factory standing in for an unbounded order feed: one event
    per second, region round-robin, amount varying."""

    def factory():
        for i in range(n):
            yield [i, REGIONS[i % 4], float((i % 100) + 1)]

    return factory


# ---------------------------------------------------------------------------
# 1. Tumbling windows over a replayed event log, with event-level provenance.
# ---------------------------------------------------------------------------
def section_windowed_query() -> None:
    banner("1. StreamWindow.query — tumbling windows, event-cited & verifiable")

    stream = RowStream.from_rows(order_events(12), SCHEMA, name="orders")
    win = StreamWindow.tumbling(size=4, time_column="ts", table="orders")
    for wq in win.query(stream, WINDOW_QUERY):
        na_row = next((i for i, r in enumerate(wq.rows) if r[0] == "NA"), 0)
        print(f"   window {wq.window.label()}: {wq.rows}")
        print(f"     NA total rests on events {wq.cite_events(na_row, 'total')}")
        print(f"     verifies offline against the captured window: {wq.verify()}")


# ---------------------------------------------------------------------------
# 2. Sliding & session windows — the same stream, different shapes.
# ---------------------------------------------------------------------------
def section_window_shapes() -> None:
    banner("2. sliding & session windows")

    stream = RowStream.from_rows(order_events(8), SCHEMA, name="orders")
    sliding = StreamWindow.sliding(size=4, slide=2, time_column="ts", table="orders")
    print("   sliding (size 4, slide 2) — windows overlap:")
    for cw in sliding.assign(stream):
        print(f"     {cw.label()} holds events {cw.offsets}")

    # A session window grows while events keep arriving within the gap (per key).
    sessions = [[0, "NA", 1.0], [1, "NA", 2.0], [9, "NA", 3.0], [10, "NA", 4.0]]
    se = StreamWindow.session(gap=2, time_column="ts", key_by=["region"], table="orders")
    print("   session (gap 2) — a quiet gap closes a session:")
    for cw in se.assign(RowStream.from_rows(sessions, SCHEMA, name="orders")):
        print(f"     {cw.label()} events {cw.offsets} (closed by {cw.closed_by})")


# ---------------------------------------------------------------------------
# 3. Windowed profile, governed metric, and per-window quality rails.
# ---------------------------------------------------------------------------
def section_windowed_primitives() -> None:
    banner("3. windowed profile, governed metric & quality rails")

    stream = RowStream.from_rows(order_events(8), SCHEMA, name="orders")
    win = StreamWindow.tumbling(size=4, time_column="ts", table="orders")

    wp = next(iter(win.profile(stream)))
    amount = wp.profile.column("amount")
    print(f"   profile {wp.window.label()}: amount in [{amount.min}, {amount.max}], mean {amount.mean}")

    layer = SemanticLayer(table="orders").add_dimension("region").add_measure(
        "total_revenue", Aggregation.SUM, "amount", unit="USD"
    )
    wm = next(iter(win.query_metric(stream, "total_revenue", layer=layer, by=["region"])))
    print(f"   governed metric {wm.window.label()}: {wm.rows} (verifies: {wm.verify()})")

    rails = DataQualityRails([ColumnConstraint(column="amount", max_value=3.0, action="warn")])
    for wr in win.screen(stream, rails):
        offenders = wr.offending_events("amount", "out_of_range")
        if offenders:
            print(f"   quality {wr.window.label()}: {len(offenders)} event(s) over the cap at {offenders}")


# ---------------------------------------------------------------------------
# 4. Bounded memory — the footprint is invariant to the event volume.
# ---------------------------------------------------------------------------
def section_bounded_memory() -> None:
    banner("4. bounded memory — invariant to how many events have flowed")

    win = StreamWindow.tumbling(size=1_000, table="orders")  # processing-time

    def peak_for(n: int) -> int:
        tracemalloc.start()
        for _ in win.query(RowStream.from_rows(order_events(n), SCHEMA, name="orders"), WINDOW_QUERY):
            pass  # consume lazily, retain nothing
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small = peak_for(2_000)
    large = peak_for(200_000)
    print(f"   peak resident set: {small:,} bytes for 2,000 events")
    print(f"                      {large:,} bytes for 200,000 events (100x longer)")
    print(f"   stays within a small factor — only the open window is held: {large <= small * 4 + 1_048_576}")


# ---------------------------------------------------------------------------
# 5. The governed driver — audited windows over a replayed log.
# ---------------------------------------------------------------------------
def section_governed_driver() -> None:
    banner("5. app.stream_analytics — every window audited")

    provider, model = example_provider()
    app = ContextApp(name="realtime", provider=provider, model=model)
    win = StreamWindow.tumbling(size=4, time_column="ts", table="orders")
    analytics = app.stream_analytics(win, table="orders")

    stream = RowStream.from_rows(order_events(12), SCHEMA, name="orders")
    windows = list(analytics.query(stream, WINDOW_QUERY))
    audited = [e for e in app.audit.entries if e.action == "stream_window"]
    print(f"   emitted {len(windows)} windows; audit chain holds: {app.audit.verify_chain()}")
    print(f"   {len(audited)} window(s) on the hash-chained audit log (action 'stream_window')")


# ---------------------------------------------------------------------------
# 6. A live async feed — alerting on a window as it closes.
# ---------------------------------------------------------------------------
async def section_live_feed() -> None:
    banner("6. drive a live async feed — an alerting rule over a window")

    provider, model = example_provider()
    app = ContextApp(name="realtime", provider=provider, model=model)
    win = StreamWindow.tumbling(size=4, time_column="ts", table="orders")
    analytics = app.stream_analytics(win, table="orders")

    async def live_orders():
        """Stand-in for a live feed — a queue, a websocket, or a realtime
        session's events(). Any async iterator works; pass extract= to pull a
        row out of a richer event object. Here NA spikes in the third window."""
        for i in range(16):
            await asyncio.sleep(0)  # yield control, as a real feed would
            region = REGIONS[i % 4]
            amount = 500.0 if (region == "NA" and 8 <= i < 12) else float((i % 50) + 1)
            yield [i, region, amount]

    alerts: list[str] = []

    def alert(result) -> None:
        # Alert when any region's windowed revenue crosses a threshold.
        for row in result.rows:
            if row[1] and row[1] > 200.0:
                alerts.append(f"{result.window.label()} {row[0]}=${row[1]:.0f}")

    results = await analytics.drive(
        live_orders(),
        SCHEMA,
        apply="query",
        request=WINDOW_QUERY,
        on_window=alert,
        max_windows=4,
    )
    print(f"   processed {len(results)} live windows (stopped at max_windows=4)")
    print(f"   all verify offline: {all(r.verify() for r in results)}")
    print(f"   alerts fired: {alerts or 'none'}")


async def main() -> None:
    section_windowed_query()
    section_window_shapes()
    section_windowed_primitives()
    section_bounded_memory()
    section_governed_driver()
    await section_live_feed()
    print("\nDone — an unbounded event stream was analyzed window by window, cited, "
          "verified, and audited, inside a footprint invariant to its volume.")


if __name__ == "__main__":
    asyncio.run(main())
