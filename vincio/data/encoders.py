"""The deterministic, token-oriented data encoder.

:class:`DataEncoder` is the user-facing surface over the
:mod:`vincio.core.tabular` kernel: it renders a :class:`~vincio.data.Dataset`
(or a legacy ``TableData``, a list of record mappings, or any JSON-like value)
into a compact, lossless, schema-declared-once string, and decodes that string
back into a typed :class:`~vincio.data.Dataset`. It also reports the columnar-
accurate token cost of an encoding — the replacement for the per-cell heuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core import tabular
from .core import Dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..documents.parsers import TableData

__all__ = ["DataEncoder"]


class DataEncoder:
    """Render tabular data header-once in a compact, token-oriented form.

    The encoder is configured once and reused. ``encode`` accepts a
    :class:`~vincio.data.Dataset`, a legacy ``TableData``, or a list of record
    mappings; ``encode_value`` accepts any JSON-like value (the token-efficient
    replacement for ``json.dumps``); ``decode`` reconstructs a typed dataset; and
    ``token_cost`` reports the exact token footprint of the encoding.
    """

    def __init__(
        self,
        *,
        delimiter: str = ",",
        include_name: bool = True,
        include_count: bool = True,
        include_types: bool = True,
        include_units: bool = True,
        exemplars: int = 0,
        max_rows: int | None = None,
    ) -> None:
        self.options = tabular.EncodeOptions(
            delimiter=delimiter,
            include_name=include_name,
            include_count=include_count,
            include_types=include_types,
            include_units=include_units,
            exemplars=exemplars,
            max_rows=max_rows,
        )

    def _as_dataset(self, data: Dataset | TableData | list[dict[str, Any]]) -> Dataset:
        if isinstance(data, Dataset):
            return data
        if isinstance(data, list):
            return Dataset.from_records(data)
        # Duck-typed legacy TableData (avoid importing documents at module load).
        if hasattr(data, "columns") and hasattr(data, "rows"):
            return Dataset.from_table_data(data)
        raise TypeError(f"cannot encode {type(data).__name__} as a table")

    def encode(self, data: Dataset | TableData | list[dict[str, Any]]) -> str:
        """Encode tabular data into the compact, lossless form."""
        return self._as_dataset(data).encode(options=self.options)

    def encode_value(self, obj: Any) -> str:
        """Encode an arbitrary JSON-like value compactly (the token-efficient
        replacement for ``json.dumps(indent=2)``)."""
        return tabular.encode_value(obj, options=self.options)

    def decode(self, text: str) -> Dataset:
        """Reconstruct a typed :class:`~vincio.data.Dataset` from an encoding."""
        return Dataset.from_encoding(text)

    def token_cost(self, data: Dataset | TableData | list[dict[str, Any]], *, model: str | None = None) -> int:
        """The exact token cost of encoding *data* — columnar-accurate, counting
        the tokens the model actually receives."""
        return self._as_dataset(data).token_cost(model=model, options=self.options)
