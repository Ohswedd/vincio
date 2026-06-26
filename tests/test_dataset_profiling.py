"""Dataset profiling — deterministic, bounded-memory column summaries.

Covers exact statistics over every value, reservoir-estimated percentiles and
histograms for large columns, cardinality with a lower-bound cap, the profile's
projection to fixed-size table evidence, determinism, and the streaming path.
"""

from __future__ import annotations

import pytest

from vincio.core.errors import DataError
from vincio.data import (
    ColumnProfile,
    ColumnSchema,
    Dataset,
    DatasetProfile,
    DataType,
    HistogramBin,
    profile_dataset,
    profile_stream,
)

# --------------------------------------------------------------------------- #
# Exact statistics
# --------------------------------------------------------------------------- #


def _sales(n: int = 50) -> Dataset:
    records = [
        {
            "region": ["NA", "EU", "APAC"][i % 3],
            "revenue": 100.0 + i,
            "units": (i % 7) or None,  # introduces nulls
        }
        for i in range(n)
    ]
    return Dataset.from_records(records, name="sales")


def test_exact_numeric_stats():
    ds = Dataset.from_columns({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    profile = profile_dataset(ds)
    assert isinstance(profile, DatasetProfile)
    col = profile.column("x")
    assert isinstance(col, ColumnProfile)
    assert col.min == 1.0
    assert col.max == 5.0
    assert col.mean == 3.0
    assert col.stddev == pytest.approx(1.4142136, abs=1e-6)
    assert col.count == 5
    assert col.null_count == 0


def test_null_rate_is_exact():
    ds = Dataset.from_columns({"x": [1, None, 3, None, 5]})
    col = profile_dataset(ds).column("x")
    assert col.null_count == 2
    assert col.null_rate == pytest.approx(0.4)
    assert col.count == 3  # non-null count


def test_cardinality_exact_when_small():
    ds = Dataset.from_columns({"c": ["a", "b", "a", "c", "b", "a"]})
    col = profile_dataset(ds).column("c")
    assert col.distinct == 3
    assert col.distinct_is_lower_bound is False


def test_cardinality_is_lower_bound_past_cap():
    ds = Dataset.from_columns({"c": list(range(1000))})
    col = profile_dataset(ds, distinct_cap=100).column("c")
    assert col.distinct == 100
    assert col.distinct_is_lower_bound is True


def test_top_values_ranked_by_frequency():
    ds = Dataset.from_columns({"c": ["a"] * 5 + ["b"] * 3 + ["c"] * 1})
    col = profile_dataset(ds).column("c")
    assert col.top_values[0] == ("a", 5)
    assert col.top_values[1] == ("b", 3)


def test_exemplars_are_distinct_and_capped():
    ds = Dataset.from_columns({"c": ["a", "a", "b", "c", "d"]})
    col = profile_dataset(ds, max_exemplars=2).column("c")
    assert col.exemplars == ["a", "b"]


# --------------------------------------------------------------------------- #
# Percentiles & histograms
# --------------------------------------------------------------------------- #


def test_percentiles_exact_for_small_columns():
    ds = Dataset.from_columns({"x": [float(i) for i in range(1, 101)]})
    col = profile_dataset(ds, reservoir_size=4096).column("x")
    assert col.estimated is False
    assert col.percentiles["p50"] == pytest.approx(50.5, abs=0.5)
    assert col.percentiles["p99"] == pytest.approx(99.0, abs=1.0)


def test_large_column_is_flagged_estimated():
    ds = Dataset.from_columns({"x": [float(i) for i in range(5000)]})
    profile = profile_dataset(ds, reservoir_size=256)
    assert profile.column("x").estimated is True
    assert profile.estimated is True
    # Exact figures stay exact even when percentiles are estimated.
    assert profile.column("x").min == 0.0
    assert profile.column("x").max == 4999.0


def test_numeric_histogram_population_scaled():
    ds = Dataset.from_columns({"x": [float(i) for i in range(100)]})
    col = profile_dataset(ds, bins=10).column("x")
    assert len(col.histogram) == 10
    assert sum(b.count for b in col.histogram) == pytest.approx(100, abs=2)
    assert isinstance(col.histogram[0], HistogramBin)
    assert col.histogram[0].lo == 0.0


def test_categorical_histogram_uses_labels():
    ds = Dataset.from_columns({"c": ["a"] * 3 + ["b"] * 2})
    col = profile_dataset(ds).column("c")
    labels = {b.label: b.count for b in col.histogram}
    assert labels == {"a": 3, "b": 2}


# --------------------------------------------------------------------------- #
# Determinism & evidence projection
# --------------------------------------------------------------------------- #


def test_profile_is_deterministic():
    ds = _sales(5000)
    a = profile_dataset(ds, seed=1).encode()
    b = profile_dataset(ds, seed=1).encode()
    assert a == b


def test_profile_seed_changes_estimates_only_within_tolerance():
    ds = Dataset.from_columns({"x": [float(i) for i in range(10000)]})
    p1 = profile_dataset(ds, reservoir_size=256, seed=1).column("x")
    p2 = profile_dataset(ds, reservoir_size=256, seed=2).column("x")
    # Exact figures identical regardless of seed; estimates close.
    assert p1.min == p2.min and p1.max == p2.max and p1.mean == p2.mean
    assert p1.percentiles["p50"] == pytest.approx(p2.percentiles["p50"], rel=0.1)


def test_profile_projects_to_table_evidence():
    profile = profile_dataset(_sales())  # _sales() is already named "sales"
    item = profile.to_evidence_item()
    assert item.modality == "table"
    assert item.table is not None
    assert "encoding" in item.table
    assert item.token_cost > 0
    # One row of stats per profiled column.
    assert len(item.table["rows"]) == 3


def test_profile_is_bounded_regardless_of_rows():
    # The profile's size tracks the number of columns and the width of the stat
    # values, never the row count — a 50× taller table profiles within a few
    # tokens of a short one (the difference is only larger magnitudes/counts).
    small = profile_dataset(_sales(100)).token_cost()
    large = profile_dataset(_sales(5000)).token_cost()
    assert abs(small - large) <= 30


def test_profile_column_lookup_errors_for_unknown():
    profile = profile_dataset(_sales())
    with pytest.raises(DataError):
        profile.column("nope")


def test_profile_marks_sampled_input():
    ds = _sales()
    ds.metadata["sample"] = {"method": "reservoir", "size": 50, "of": 1000}
    assert profile_dataset(ds).sampled is True


# --------------------------------------------------------------------------- #
# Streaming path
# --------------------------------------------------------------------------- #


def test_profile_stream_matches_in_memory():
    ds = _sales(300)
    schema = ds.columns
    streamed = profile_stream(ds.rows(), schema, name="sales", seed=0)
    in_memory = profile_dataset(ds, seed=0)
    assert streamed.row_count == in_memory.row_count
    assert streamed.column("revenue").min == in_memory.column("revenue").min
    assert streamed.column("revenue").max == in_memory.column("revenue").max
    assert streamed.column("region").distinct == in_memory.column("region").distinct


def test_profile_stream_bounded_memory_large():
    schema = [
        ColumnSchema(name="id", dtype=DataType.INT),
        ColumnSchema(name="amount", dtype=DataType.FLOAT),
    ]

    def gen(n: int):
        for i in range(n):
            yield [i, float(i % 1000)]

    profile = profile_stream(gen(200_000), schema, name="big", reservoir_size=512)
    assert profile.row_count == 200_000
    assert profile.column("amount").min == 0.0
    assert profile.column("amount").max == 999.0
    assert profile.column("amount").estimated is True
    # A 200k-row profile is the same fixed size as a tiny one.
    assert profile.token_cost() < 500
