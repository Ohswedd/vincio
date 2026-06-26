"""Representative sampling — reservoir, stratified, systematic.

Covers single-pass uniform sampling over an iterator, distribution-preserving
stratified allocation, deterministic seeding, schema preservation, sample
metadata, and the connector reservoir-sampling path that replaces the first-N
cutoff.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections import Counter

import pytest

from vincio.core.errors import DataError
from vincio.data import (
    Dataset,
    SampleMethod,
    reservoir_sample,
    sample_dataset,
    stratified_sample,
    systematic_sample,
)


def _sales(n: int = 100) -> Dataset:
    return Dataset.from_records(
        [{"region": ["NA", "EU", "APAC"][i % 3], "revenue": 100.0 + i} for i in range(n)],
        name="sales",
    )


# --------------------------------------------------------------------------- #
# reservoir_sample
# --------------------------------------------------------------------------- #


def test_reservoir_returns_k_items():
    assert len(reservoir_sample(range(1000), 50, seed=1)) == 50


def test_reservoir_preserves_order_by_default():
    sample = reservoir_sample(range(1000), 50, seed=1)
    assert sample == sorted(sample)


def test_reservoir_is_deterministic():
    assert reservoir_sample(range(1000), 20, seed=7) == reservoir_sample(range(1000), 20, seed=7)


def test_reservoir_different_seed_differs():
    assert reservoir_sample(range(1000), 20, seed=1) != reservoir_sample(range(1000), 20, seed=2)


def test_reservoir_smaller_than_k_returns_all():
    assert reservoir_sample(range(5), 50, seed=1) == [0, 1, 2, 3, 4]


def test_reservoir_zero_and_negative():
    assert reservoir_sample(range(10), 0) == []
    with pytest.raises(DataError):
        reservoir_sample(range(10), -1)


def test_reservoir_is_unbiased_over_position():
    # Across many seeds, early and late indices are sampled at similar rates.
    n, k, trials = 100, 10, 400
    counts = Counter()
    for seed in range(trials):
        counts.update(reservoir_sample(range(n), k, seed=seed))
    first_half = sum(counts[i] for i in range(n // 2))
    second_half = sum(counts[i] for i in range(n // 2, n))
    # No strong positional bias (a head() cutoff would put everything in first_half).
    assert 0.8 < first_half / second_half < 1.25


def test_reservoir_over_iterator_single_pass():
    # An exhausted generator proves only one pass is taken.
    gen = (i for i in range(500))
    sample = reservoir_sample(gen, 10, seed=3)
    assert len(sample) == 10
    assert list(gen) == []  # fully consumed


# --------------------------------------------------------------------------- #
# stratified_sample
# --------------------------------------------------------------------------- #


def test_stratified_preserves_proportions():
    ds = Dataset.from_records(
        [{"g": "rare"}] * 10 + [{"g": "common"}] * 90,
        name="skewed",
    )
    sample = stratified_sample(ds, 20, by="g", seed=1)
    counts = Counter(sample.column("g"))
    # 10% rare in the source → ~10% (2 of 20) in the sample, not washed out.
    assert counts["rare"] == 2
    assert counts["common"] == 18


def test_stratified_requires_known_column():
    with pytest.raises(DataError):
        stratified_sample(_sales(30), 5, by="nope")


def test_stratified_via_sample_dataset_needs_by():
    with pytest.raises(DataError):
        sample_dataset(_sales(30), 5, method="stratified")


def test_stratified_is_deterministic():
    a = stratified_sample(_sales(99), 30, by="region", seed=2).rows()
    b = stratified_sample(_sales(99), 30, by="region", seed=2).rows()
    assert a == b


# --------------------------------------------------------------------------- #
# systematic & dispatch
# --------------------------------------------------------------------------- #


def test_systematic_is_evenly_spaced_and_deterministic():
    sample = systematic_sample(_sales(100), 10)
    revenues = sample.column("revenue")
    assert len(revenues) == 10
    # Evenly spaced positions: 0, 10, 20, ...
    assert revenues == [100.0 + 10 * i for i in range(10)]


def test_sample_dataset_preserves_schema_and_records_metadata():
    sample = sample_dataset(_sales(100), 10, method=SampleMethod.RESERVOIR, seed=1)
    assert sample.column_names == ["region", "revenue"]
    assert sample.dtypes == _sales(100).dtypes
    meta = sample.metadata["sample"]
    assert meta["method"] == "reservoir"
    assert meta["size"] == 10
    assert meta["of"] == 100


def test_sample_dataset_accepts_records():
    sample = sample_dataset([{"a": 1}, {"a": 2}, {"a": 3}], 2, seed=1)
    assert sample.row_count == 2


def test_head_method_is_first_n():
    sample = sample_dataset(_sales(100), 5, method="head")
    assert sample.column("revenue") == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_reservoir_preserve_order_false_shuffles():
    ordered = reservoir_sample(range(1000), 30, seed=1, preserve_order=True)
    shuffled = reservoir_sample(range(1000), 30, seed=1, preserve_order=False)
    assert sorted(shuffled) == ordered  # same items, different order
    assert shuffled != ordered


def test_sample_dataset_rejects_unsupported_input():
    with pytest.raises(DataError):
        sample_dataset(42, 5)  # type: ignore[arg-type]


def test_systematic_zero_and_full():
    ds = _sales(20)
    assert systematic_sample(ds, 0).row_count == 0
    assert systematic_sample(ds, 50).row_count == 20  # k >= n returns all
    with pytest.raises(DataError):
        systematic_sample(ds, -1)


def test_stratified_k_at_least_n_returns_all():
    ds = _sales(9)
    assert stratified_sample(ds, 50, by="region").row_count == 9


def test_reservoir_zero_via_dispatch():
    assert sample_dataset(_sales(10), 0, method="reservoir").row_count == 0


# --------------------------------------------------------------------------- #
# Connector reservoir sampling (replaces the first-N cutoff)
# --------------------------------------------------------------------------- #


def _sqlite_with_rows(n: int) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("create table t(id integer primary key, name text)")
    con.executemany("insert into t values (?, ?)", [(i, f"row{i}") for i in range(n)])
    con.commit()
    return con


def test_sql_connector_default_is_head_cutoff():
    from vincio.connectors.sql import SQLConnector

    con = _sqlite_with_rows(100)
    docs = asyncio.run(SQLConnector("select * from t", connection=con, id_column="id", max_rows=10).load())
    ids = sorted(int(d.metadata["row"]["id"]) for d in docs)
    assert ids == list(range(10))  # the legacy first-N behavior, unchanged


def test_sql_connector_reservoir_spreads_across_whole_table():
    from vincio.connectors.sql import SQLConnector

    con = _sqlite_with_rows(100)
    docs = asyncio.run(
        SQLConnector("select * from t", connection=con, id_column="id", sample=10, sample_seed=3).load()
    )
    ids = sorted(int(d.metadata["row"]["id"]) for d in docs)
    assert len(ids) == 10
    assert max(ids) > 50  # representative of the whole, not just the head


def test_sql_connector_reservoir_is_deterministic():
    from vincio.connectors.sql import SQLConnector

    def ids(seed: int) -> list[int]:
        con = _sqlite_with_rows(100)
        docs = asyncio.run(
            SQLConnector("select * from t", connection=con, id_column="id", sample=8, sample_seed=seed).load()
        )
        return sorted(int(d.metadata["row"]["id"]) for d in docs)

    assert ids(5) == ids(5)
