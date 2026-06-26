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

from ..core.errors import DataError, DataQualityError
from .core import ColumnSchema, DataSchema, Dataset, DataType
from .encoders import DataEncoder
from .evidence import TableEvidence
from .profile import (
    ColumnProfile,
    DatasetProfile,
    HistogramBin,
    profile_dataset,
    profile_stream,
)
from .quality import (
    ColumnConstraint,
    DataQualityRails,
    DataQualityReport,
    DataQualityViolation,
)
from .sampling import (
    SampleMethod,
    reservoir_sample,
    sample_dataset,
    stratified_sample,
    systematic_sample,
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
]
