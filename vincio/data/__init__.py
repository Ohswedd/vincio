"""Tabular evidence and the compact data encoder.

The data plane treats a dataset as *first-class, schema-bearing, columnar
evidence* — never flattened to prose or dumped as ``json.dumps``. The model sees
a compact, token-oriented encoding in which the schema, types, units, and
null-handling are declared **once** and the cells follow as delimited rows.

* :class:`Dataset` — a typed :class:`DataSchema` (per-column name, type, unit,
  nullability) over column-major cells. Build one from rows, records, columns, a
  legacy ``TableData``, or a compact encoding.
* :class:`DataEncoder` — renders a dataset header-once in a compact, lossless,
  round-trippable form and reports its columnar-accurate token cost; also
  encodes arbitrary JSON-like values, the token-efficient replacement for
  ``json.dumps``.
* :class:`TableEvidence` — projects a dataset into the context evidence the
  compiler scores, budgets, orders, and cites, so a table reaches the prompt
  structured and token-cheap rather than as a pipe-joined string.

Everything here is deterministic, dependency-free, and offline. ``Dataset`` and
the schema types are exported from this subpackage (the top-level ``Dataset``
name belongs to :mod:`vincio.evals`); :class:`DataEncoder` and
:class:`TableEvidence` are also re-exported at the package top level.

    from vincio.data import Dataset, DataEncoder, TableEvidence

    ds = Dataset.from_records(
        [{"region": "NA", "revenue": 1200.5}, {"region": "EU", "revenue": 980.0}],
        name="sales",
    )
    print(ds.encode())                 # sales{#2,region:str,revenue:float}\nNA,1200.5\nEU,980.0
    evidence = ds.to_evidence(source_id="sales")   # first-class table evidence
"""

from __future__ import annotations

from ..core.errors import DataError
from .core import ColumnSchema, DataSchema, Dataset, DataType
from .encoders import DataEncoder
from .evidence import TableEvidence

__all__ = [
    "Dataset",
    "DataSchema",
    "ColumnSchema",
    "DataType",
    "DataEncoder",
    "TableEvidence",
    "DataError",
]
