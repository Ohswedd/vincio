"""Dataset profiling, sampling, fit-in-window, and data-quality rails.

A table far larger than the context window cannot enter a prompt, and truncating
it to the first thousand rows throws away everything the rest would have said.
This program walks the data plane's representation rung, fully offline on the
deterministic mock:

  * `profile_dataset` — a deterministic, bounded-memory column profile
    (cardinality, percentiles, histograms, null rate, exemplars) that is itself
    fixed-size evidence;
  * `sample_dataset` — reservoir and stratified sampling that stand a
    representative sample in for the whole, replacing a biased first-N cutoff;
  * `fit_dataset` / `fit_to_window` — profile + representative sample fitted into
    a fixed token budget, so a ten-million-row table is represented faithfully
    inside the window and the representation does not grow with the rows;
  * `DataQualityRails` / `app.screen_data` — screening for schema violations,
    constraint breaks, anomalies, and PII on the same deterministic rail path
    PII and injection detection ride for text.

Everything below is opt-in and additive; none of it touches a database or a network.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio import ContextApp
from vincio.data import (
    ColumnConstraint,
    ColumnSchema,
    DataQualityRails,
    Dataset,
    DataType,
    fit_stream,
    profile_dataset,
    sample_dataset,
)


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


SALES = [
    {"region": ["NA", "EU", "APAC"][i % 3], "revenue": 100.0 + (i % 50), "units": (i % 7) or None}
    for i in range(300)
]


# ---------------------------------------------------------------------------
# 1. A deterministic, bounded-memory column profile.
# ---------------------------------------------------------------------------
def section_profile() -> None:
    banner("1. profile_dataset — a fixed-size column profile")

    ds = Dataset.from_records(SALES, name="sales")
    profile = profile_dataset(ds)
    print(f"   {profile.summary()}")
    rev = profile.column("revenue")
    print(f"   revenue : min={rev.min} max={rev.max} mean={rev.mean} p95={rev.percentiles['p95']}")
    region = profile.column("region")
    print(f"   region  : distinct={region.distinct} top={region.top_values[:3]}")
    print(f"   units   : null_rate={profile.column('units').null_rate}")
    # The profile is itself compact, schema-once evidence the compiler can score.
    print(f"   profile token cost: {profile.token_cost()} (fixed regardless of row count)")


# ---------------------------------------------------------------------------
# 2. Representative sampling — reservoir and stratified.
# ---------------------------------------------------------------------------
def section_sampling() -> None:
    banner("2. sample_dataset — a representative sample stands in for the whole")

    ds = Dataset.from_records(SALES, name="sales")
    # Reservoir: a uniform sample in a single bounded pass (not the first N).
    uniform = sample_dataset(ds, 12, method="reservoir", seed=1)
    print(f"   reservoir sample: {uniform.row_count} rows · {uniform.metadata['sample']}")

    # Stratified: preserves the distribution of a key column.
    stratified = sample_dataset(ds, 12, method="stratified", by="region", seed=1)
    counts: dict[str, int] = {}
    for r in stratified.column("region"):
        counts[r] = counts.get(r, 0) + 1
    print(f"   stratified by region: {counts} (proportional, no category washed out)")


# ---------------------------------------------------------------------------
# 3. Screening a table on the deterministic rail path.
# ---------------------------------------------------------------------------
def section_quality() -> None:
    banner("3. DataQualityRails — schema, constraints, anomalies, PII")

    # A dataset with one of each defect class seeded in.
    dirty = Dataset.from_records(
        [
            {"id": 1, "region": "NA", "amount": 50.0, "note": "ok"},
            {"id": 2, "region": "EU", "amount": 9_000_000.0, "note": "fine"},  # anomaly + range
            {"id": 3, "region": "ZZ", "amount": -5.0, "note": "reach me at a@b.com"},  # allowed/range/PII
        ],
        name="orders",
    )
    rails = DataQualityRails(
        [
            ColumnConstraint(column="region", allowed_values=["NA", "EU", "APAC"]),
            ColumnConstraint(column="amount", min_value=0, max_value=10_000),
            ColumnConstraint(column="note", detectors=["pii"]),
        ],
        detect_anomalies=True,
    )
    report = rails.check(dirty)
    print(f"   allowed: {report.allowed}")
    for v in report.violations:
        print(f"     - {v.column}:{v.rule} ({v.action}) — {v.message}")


# ---------------------------------------------------------------------------
# 4. Fitting a table far larger than the window into a fixed token budget.
# ---------------------------------------------------------------------------
def section_fit() -> None:
    banner("4. fit_to_window — a huge table under a fixed token budget")

    schema = [
        ColumnSchema(name="id", dtype=DataType.INT),
        ColumnSchema(name="region", dtype=DataType.STR),
        ColumnSchema(name="amount", dtype=DataType.FLOAT),
    ]
    regions = ["NA", "EU", "APAC", "LATAM"]

    def gen(n: int):
        for i in range(n):
            yield [i, regions[i % 4], float(i % 1000)]

    # The same fixed budget represents tables that differ 10× in height — the
    # profile is fixed-size and the sample is budget-bound (single bounded pass).
    for n in (50_000, 500_000):
        fit = fit_stream(gen(n), schema, max_tokens=2000, seed=7, name="txns")
        print(
            f"   {n:>8,} rows -> {fit.token_cost} tokens "
            f"(profile {fit.profile_tokens} + {fit.sample_size}-row sample) within={fit.within_budget}"
        )
    print("   the representation does not grow with the rows — that is the fit-in-window guarantee")


# ---------------------------------------------------------------------------
# 5. The app surface: screen with audit, fit as evidence, answer over it.
# ---------------------------------------------------------------------------
async def section_app() -> None:
    banner("5. app.screen_data (audited) + app.fit_dataset as evidence")

    provider, model = example_provider(default_responder=lambda _r: "NA, EU, and APAC are represented.")
    app = ContextApp(name="data", provider=provider, model=model)

    # Screening lands on the shared, hash-chained audit log like any rail decision.
    report = app.screen_data(SALES, detect_anomalies=True)
    decision = next(e for e in app.audit.entries if e.action == "data_quality")
    print(f"   screen_data allowed={report.allowed} · audit decision={decision.decision}")

    # Fit the table and carry the profile + sample as cited table evidence.
    fit = app.fit_dataset(SALES, max_tokens=1200, seed=1)
    app.pending_evidence.extend(fit.to_evidence_items(source_id="sales"))
    result = await app.arun("Which regions appear in the sales data?")
    print(f"   fit: {fit.summary()}")
    print(f"   answer: {result.output}")


async def main() -> None:
    section_profile()
    section_sampling()
    section_quality()
    section_fit()
    await section_app()
    print("\nDone — a table far larger than the window reached the model faithfully, offline.")


if __name__ == "__main__":
    asyncio.run(main())
