"""Spec-driven charts with C2PA provenance and per-figure data binding.

A chart is the data plane's generated *artifact*. :func:`generate_chart` turns a
cell-cited :class:`~vincio.data.QueryResult` into a spec-driven figure that carries
the two guarantees an analytical deliverable needs:

* **Content-bound** — a C2PA-style :class:`~vincio.governance.transparency.ProvenanceManifest`
  bound to the rendered bytes by SHA-256, exactly the credential a generated image
  or audio clip carries. A chart is *data-driven* media (the IPTC ``dataDrivenMedia``
  source type, ``is_synthetic=False``): a faithful, deterministic rendering of real
  values, not model-synthesized content — and the credential says so honestly.
* **Data-bound** — a back-reference to the **exact source rows and cells** the
  figure was built from (the result's
  :class:`~vincio.data.provenance.RowProvenance`), and a
  :meth:`Chart.verify` that re-executes the source query against a catalog,
  confirms the plotted figure is a faithful projection of that verified result,
  and confirms the manifest still binds the bytes — the analytics analogue of a
  cited report's offline verification.

The default :class:`VegaLiteRenderer` emits a portable Vega-Lite v5 JSON spec and
is **dependency-free, deterministic, and offline** — the spec embeds the plotted
values, so a consumer renders it with any Vega-Lite runtime. The optional
:class:`MatplotlibRenderer` (behind the ``vincio[charts]`` extra) rasterizes the
same spec to a PNG that carries its credential embedded in the file. Coverage is
always stated, never silently downgraded.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import ChartError
from ..core.media import media_sha256
from ..core.utils import stable_hash
from ..governance.transparency import (
    ContentSigner,
    ProvenanceManifest,
    embed_provenance,
    mark_data_driven_content,
    verify_manifest,
    write_sidecar_manifest,
)
from .core import Dataset, DataType
from .evidence import TableEvidence
from .provenance import LineageCoverage, RowProvenance
from .query import DataCatalog, QueryEngine, QueryResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.types import EvidenceItem

__all__ = [
    "ChartType",
    "ChartChannel",
    "ChartEncoding",
    "ChartSpec",
    "Chart",
    "ChartRenderer",
    "VegaLiteRenderer",
    "MatplotlibRenderer",
    "generate_chart",
]

_VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
_VEGA_LITE_MEDIA_TYPE = "application/vnd.vega-lite+json"


class ChartType(StrEnum):
    """The closed mark vocabulary a chart declares — the deterministic subset of
    Vega-Lite marks that also rasterizes cleanly through matplotlib.

    ``BAR`` a measure across a dimension, ``LINE`` a measure across an ordered (often
    temporal) dimension, ``POINT`` a scatter of two measures, ``AREA`` a filled
    line, ``ARC`` a part-of-whole (pie)."""

    BAR = "bar"
    LINE = "line"
    POINT = "point"
    AREA = "area"
    ARC = "arc"


def _vega_type(dtype: DataType) -> str:
    """Map a column's data type to a Vega-Lite measurement type."""
    if dtype is DataType.INT or dtype is DataType.FLOAT:
        return "quantitative"
    if dtype is DataType.DATE or dtype is DataType.DATETIME or dtype is DataType.TIME:
        return "temporal"
    return "nominal"


def _is_measure(dtype: DataType) -> bool:
    return dtype is DataType.INT or dtype is DataType.FLOAT


class ChartChannel(BaseModel):
    """One encoding channel: the result column it binds, its Vega-Lite measurement
    type, a display title, and an optional unit carried from the source schema."""

    field: str
    type: str = "nominal"
    title: str = ""
    unit: str | None = None

    def axis_title(self) -> str:
        """The channel's display title (its field name when none was set), with the
        unit appended when the schema declared one (``revenue (USD)``)."""
        base = self.title or self.field
        return f"{base} ({self.unit})" if self.unit else base


class ChartEncoding(BaseModel):
    """How result columns map to a chart's positional/grouping channels: the
    horizontal ``x`` (a dimension, or a measure for a scatter), the vertical ``y``
    (the measure), and an optional ``color`` series (a second dimension)."""

    x: ChartChannel
    y: ChartChannel
    color: ChartChannel | None = None

    @property
    def fields(self) -> list[str]:
        """The distinct result columns this encoding plots, in channel order."""
        out = [self.x.field, self.y.field]
        if self.color is not None and self.color.field not in out:
            out.append(self.color.field)
        return out


class ChartSpec(BaseModel):
    """A spec-driven chart definition: title, mark, channel encoding, the plotted
    columns, and the **values** it depicts (a projection of the source result onto
    the encoded columns). :meth:`to_vega_lite` renders it as a portable, embedded-data
    Vega-Lite v5 spec a consumer can render with any Vega-Lite runtime."""

    title: str = ""
    mark: ChartType = ChartType.BAR
    encoding: ChartEncoding
    columns: list[str] = Field(default_factory=list)
    values: list[dict[str, Any]] = Field(default_factory=list)

    def to_vega_lite(self) -> dict[str, Any]:
        """A complete, deterministic Vega-Lite v5 specification with the data
        embedded inline. An ``arc`` mark uses the ``theta``/``color`` channels a
        part-of-whole chart needs; every other mark uses ``x``/``y`` (+ optional
        ``color``)."""
        enc = self.encoding
        spec: dict[str, Any] = {
            "$schema": _VEGA_LITE_SCHEMA,
            "data": {"values": self.values},
            "mark": {"type": self.mark.value, "tooltip": True},
        }
        if self.title:
            spec["title"] = self.title
        if self.mark is ChartType.ARC:
            spec["encoding"] = {
                "theta": {"field": enc.y.field, "type": enc.y.type, "title": enc.y.axis_title()},
                "color": {"field": enc.x.field, "type": enc.x.type, "title": enc.x.axis_title()},
            }
            return spec
        encoding: dict[str, Any] = {
            "x": {"field": enc.x.field, "type": enc.x.type, "title": enc.x.axis_title()},
            "y": {"field": enc.y.field, "type": enc.y.type, "title": enc.y.axis_title()},
        }
        if enc.color is not None:
            encoding["color"] = {
                "field": enc.color.field,
                "type": enc.color.type,
                "title": enc.color.axis_title(),
            }
        spec["encoding"] = encoding
        return spec

    def to_json(self) -> str:
        """The Vega-Lite spec as deterministic, sorted-key JSON."""
        import json

        return json.dumps(self.to_vega_lite(), sort_keys=True, separators=(",", ":"), default=str)


# --------------------------------------------------------------------------- #
# Renderers                                                                    #
# --------------------------------------------------------------------------- #


@runtime_checkable
class ChartRenderer(Protocol):
    """Renders a :class:`ChartSpec` to bytes of a declared media type."""

    name: str
    media_type: str

    def render(self, spec: ChartSpec) -> bytes: ...


class VegaLiteRenderer:
    """The default, dependency-free renderer: a portable Vega-Lite v5 JSON spec.

    Deterministic and offline — the spec embeds the plotted values, so the bytes a
    consumer receives fully describe the figure, and the C2PA credential binds
    them. This is the offline-first path; reach for :class:`MatplotlibRenderer` only
    when you need a rasterized image."""

    name = "vega-lite"
    media_type = _VEGA_LITE_MEDIA_TYPE

    def render(self, spec: ChartSpec) -> bytes:
        return spec.to_json().encode("utf-8")


class MatplotlibRenderer:
    """Rasterize a chart to a PNG with matplotlib (behind the ``vincio[charts]``
    extra). The same spec the Vega-Lite renderer emits, drawn to an image whose
    embedded C2PA credential travels in the PNG itself — the way a generated image
    carries its manifest. The dependency-free :class:`VegaLiteRenderer` is the
    offline default; this is opt-in."""

    name = "matplotlib"
    media_type = "image/png"

    def __init__(self, *, width: int = 640, height: int = 400, dpi: int = 100) -> None:
        self.width = width
        self.height = height
        self.dpi = dpi

    def render(self, spec: ChartSpec) -> bytes:  # pragma: no cover - needs the extra
        plt = _import_matplotlib()
        import io

        enc = spec.encoding
        xs = [row.get(enc.x.field) for row in spec.values]
        ys = [row.get(enc.y.field) for row in spec.values]
        fig, ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        try:
            if spec.mark is ChartType.ARC:
                ax.pie(ys, labels=[str(x) for x in xs], autopct="%1.1f%%")
            elif spec.mark is ChartType.LINE:
                ax.plot(range(len(xs)), ys, marker="o")
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")
            elif spec.mark is ChartType.POINT:
                ax.scatter(xs, ys)
            elif spec.mark is ChartType.AREA:
                ax.fill_between(range(len(xs)), ys)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")
            else:
                ax.bar([str(x) for x in xs], ys)
            if spec.mark is not ChartType.ARC:
                ax.set_xlabel(enc.x.axis_title())
                ax.set_ylabel(enc.y.axis_title())
            if spec.title:
                ax.set_title(spec.title)
            fig.tight_layout()
            buf = io.BytesIO()
            # Suppress matplotlib's timestamp/software PNG metadata so the bytes are
            # a pure function of the data (a reproducible credential).
            fig.savefig(buf, format="png", metadata={"Software": None})
            return buf.getvalue()
        finally:
            plt.close(fig)


def _import_matplotlib() -> Any:  # pragma: no cover - exercised only with the extra
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ChartError(
            "MatplotlibRenderer needs the 'matplotlib' package; install it with "
            'pip install "vincio[charts]" (the default Vega-Lite renderer needs no extra)'
        ) from exc
    return plt


# --------------------------------------------------------------------------- #
# The chart artifact                                                           #
# --------------------------------------------------------------------------- #


class Chart(BaseModel):
    """A rendered chart, **content-bound and data-bound**.

    Carries the :class:`ChartSpec`, the rendered ``data`` bytes and their media
    type, the C2PA :class:`~vincio.governance.transparency.ProvenanceManifest`
    bound to those bytes, and the lineage back to the rows it was built from — the
    source result's :class:`~vincio.data.provenance.RowProvenance`, the content
    hashes of the source tables, and the source result hash. :meth:`verify`
    re-executes the source query against a catalog, confirms the figure is a
    faithful projection of the verified result, and confirms the credential still
    binds the bytes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ChartSpec
    data: bytes
    media_type: str = _VEGA_LITE_MEDIA_TYPE
    renderer: str = "vega-lite"
    manifest: ProvenanceManifest | None = None
    # Lineage back to the source rows. ``source`` is the cited query result the
    # figure was built from (``None`` for a chart of a bare dataset); ``provenance``
    # is the per-row source-cell lineage; ``coverage`` states how precise it is.
    source: QueryResult | None = None
    provenance: list[RowProvenance] = Field(default_factory=list)
    coverage: LineageCoverage = LineageCoverage.RESULT
    source_hashes: dict[str, str] = Field(default_factory=dict)
    result_hash: str = ""
    chart_hash: str = ""

    # -- construction ----------------------------------------------------------

    @classmethod
    def _build(
        cls,
        spec: ChartSpec,
        data: bytes,
        *,
        media_type: str,
        renderer: str,
        source: QueryResult | None,
        provenance: list[RowProvenance],
        coverage: LineageCoverage,
        source_hashes: dict[str, str],
        result_hash: str,
        manifest: ProvenanceManifest | None,
    ) -> Chart:
        chart = cls(
            spec=spec,
            data=data,
            media_type=media_type,
            renderer=renderer,
            manifest=manifest,
            source=source,
            provenance=provenance,
            coverage=coverage,
            source_hashes=source_hashes,
            result_hash=result_hash,
        )
        chart.chart_hash = chart._compute_hash()
        return chart

    def _compute_hash(self) -> str:
        return stable_hash(
            [
                self.spec.title,
                self.spec.mark.value,
                self.spec.encoding.model_dump(),
                self.spec.columns,
                self.spec.values,
                self.media_type,
                sorted(self.source_hashes.items()),
                self.result_hash,
                media_sha256(self.data),
            ]
        )

    # -- access ----------------------------------------------------------------

    @property
    def point_count(self) -> int:
        """The number of plotted data points (rows the figure depicts)."""
        return len(self.spec.values)

    def cite_refs(self) -> list[str]:
        """Every distinct source-cell locator (``table#r<row>!<col>``) the figure
        rests on, in order — the back-reference to the exact rows it was built from."""
        seen: set[str] = set()
        out: list[str] = []
        for prov in self.provenance:
            for ref in prov.refs:
                if ref not in seen:
                    seen.add(ref)
                    out.append(ref)
        return out

    def to_vega_lite(self) -> dict[str, Any]:
        """The chart's Vega-Lite v5 specification (shorthand for ``spec.to_vega_lite()``)."""
        return self.spec.to_vega_lite()

    # -- verification ----------------------------------------------------------

    def content_bound(self) -> bool:
        """Whether the C2PA credential still binds the rendered bytes (the manifest's
        SHA-256 matches ``data``). ``False`` when there is no manifest."""
        return self.manifest is not None and verify_manifest(self.manifest, self.data)

    def verify(self, catalog: DataCatalog, *, engine: QueryEngine | None = None) -> bool:
        """Confirm the figure is **content-bound and data-bound** against *catalog*.

        Returns ``False`` on any divergence: an edited spec or bytes (the chart
        hash no longer recomputes), a stripped or mismatched credential (the manifest
        no longer binds the bytes), a tampered source (the source query no longer
        re-executes to the same result), or a figure whose plotted values are not a
        faithful projection of that verified result."""
        if self._compute_hash() != self.chart_hash:
            return False
        if self.manifest is not None and not verify_manifest(self.manifest, self.data):
            return False
        if self.source is not None:
            if not self.source.verify(catalog, engine=engine):
                return False
            if _project(self.source.dataset, self.spec.encoding) != self.spec.values:
                return False
        return True

    # -- projection ------------------------------------------------------------

    def save(self, path: str | Path, *, sidecar: bool = True) -> str:
        """Write the rendered bytes to *path*; when ``sidecar`` is set and the bytes
        do not embed the credential, also write a ``<name>.c2pa.json`` manifest
        sidecar beside them. Returns the asset path."""
        target = Path(path)
        target.write_bytes(self.data)
        if sidecar and self.manifest is not None and self.media_type != "image/png":
            write_sidecar_manifest(target, self.manifest)
        return str(target)

    def to_evidence(self, *, source_id: str = "", caption: str = "", **kwargs: Any) -> TableEvidence:
        """Project the figure into cited ``modality="table"`` evidence the context
        compiler scores, budgets, orders, and cites — the plotted values as a small
        table, with the chart spec, lineage, and credential carried in metadata."""
        records = self.spec.values or [dict.fromkeys(self.spec.columns)]
        dataset = Dataset.from_records(records, name=f"{self.spec.title or 'chart'}_data")
        ev = dataset.to_evidence(
            source_id=source_id or f"{self.spec.title or 'chart'}_figure",
            caption=caption or self.spec.title or "Chart",
            **kwargs,
        )
        ev.metadata = {
            **ev.metadata,
            "chart_type": self.spec.mark.value,
            "renderer": self.renderer,
            "media_type": self.media_type,
            "result_hash": self.result_hash,
            "chart_hash": self.chart_hash,
            "lineage_coverage": str(self.coverage),
            "cite_refs": self.cite_refs(),
            "source_tables": sorted(self.source_hashes),
            "content_sha256": self.manifest.content_sha256 if self.manifest else None,
        }
        return ev

    def to_evidence_item(self, **kwargs: Any) -> EvidenceItem:
        """Project straight to a ``modality='table'`` evidence item carrying the
        figure's data, spec, lineage, and credential."""
        return self.to_evidence(**kwargs).to_evidence_item()


# --------------------------------------------------------------------------- #
# Encoding inference & generation                                              #
# --------------------------------------------------------------------------- #


def _channel(dataset: Dataset, field: str) -> ChartChannel:
    col = next((c for c in dataset.columns if c.name == field), None)
    if col is None:
        raise ChartError(
            f"chart encoding names column {field!r}, which the result does not carry "
            f"(has {dataset.column_names})"
        )
    return ChartChannel(field=field, type=_vega_type(col.dtype), title=field, unit=col.unit)


def _infer_encoding(
    dataset: Dataset,
    *,
    mark: ChartType,
    x: str | None,
    y: str | None,
    color: str | None,
) -> ChartEncoding:
    """Resolve the x / y / color channels, inferring any the caller did not pin from
    the result's schema: a dimension on x, a measure on y, a second dimension on
    color. Raises :class:`ChartError` when the result cannot support the chart."""
    if not dataset.column_names:
        raise ChartError("cannot build a chart from a result with no columns")
    measures = [c.name for c in dataset.columns if _is_measure(c.dtype)]
    dimensions = [c.name for c in dataset.columns if not _is_measure(c.dtype)]

    if y is None:
        # A scatter plots a measure against a measure; every other mark plots a
        # measure (y) across a dimension (x).
        if mark is ChartType.POINT and len(measures) >= 2:
            y = measures[1]
        elif measures:
            y = measures[0]
        else:
            raise ChartError(
                "a chart needs a numeric column to plot on its value axis; the result "
                f"has no measure ({dataset.column_names})"
            )
    if x is None:
        if mark is ChartType.POINT and measures:
            x = next((m for m in measures if m != y), measures[0])
        else:
            x = dimensions[0] if dimensions else next(c for c in dataset.column_names if c != y)
    x_ch = _channel(dataset, x)
    y_ch = _channel(dataset, y)
    color_ch: ChartChannel | None = None
    if color is not None:
        color_ch = _channel(dataset, color)
    elif mark is not ChartType.ARC:
        # A second dimension becomes the series, when one is free.
        extra = next((d for d in dimensions if d not in (x, y)), None)
        if extra is not None:
            color_ch = _channel(dataset, extra)
    return ChartEncoding(x=x_ch, y=y_ch, color=color_ch)


def _project(dataset: Dataset, encoding: ChartEncoding) -> list[dict[str, Any]]:
    """The dataset projected onto the encoded columns, as plain records — exactly
    the values a chart depicts (and re-derives against on :meth:`Chart.verify`)."""
    fields = encoding.fields
    cols = {f: dataset.column(f) for f in fields}
    return [{f: cols[f][i] for f in fields} for i in range(dataset.row_count)]


def _default_mark(dataset: Dataset, x: str | None) -> ChartType:
    """A bar chart by default; a line when the x axis is temporal. Resolves the x
    column the same way :func:`_infer_encoding` does (the first dimension) so the
    temporal upgrade fires even when the caller did not pin ``x``."""
    target = x
    if target is None:
        dims = [c.name for c in dataset.columns if not _is_measure(c.dtype)]
        target = dims[0] if dims else None
    if target is not None:
        col = next((c for c in dataset.columns if c.name == target), None)
        if col is not None and _vega_type(col.dtype) == "temporal":
            return ChartType.LINE
    return ChartType.BAR


def _resolve_source(
    result: QueryResult | Dataset | Any,
) -> tuple[Dataset, QueryResult | None, list[RowProvenance], LineageCoverage, dict[str, str], str]:
    """Normalize a chart's input into (dataset, source-result, provenance, coverage,
    source-hashes, result-hash). Accepts a cited :class:`~vincio.data.QueryResult`
    (the first-class, re-derivable path), an :class:`~vincio.data.AnalysisResult`
    (its primary step's result), or a bare :class:`~vincio.data.Dataset` (content-
    bound only — there is no query to re-execute, stated as ``RESULT`` coverage)."""
    if isinstance(result, QueryResult):
        return (
            result.dataset,
            result,
            result.provenance,
            result.coverage,
            result.source_hashes,
            result.result_hash,
        )
    if isinstance(result, Dataset):
        return result, None, [], LineageCoverage.RESULT, {}, ""
    # Duck-type an AnalysisResult without importing it (avoids a heavier dependency
    # cycle); chart its primary step's cited result.
    primary = getattr(result, "primary_step", None)
    if callable(primary):
        step = primary() or next(
            (s for s in getattr(result, "steps", []) if getattr(s, "result", None) is not None),
            None,
        )
        qr = getattr(step, "result", None)
        if isinstance(qr, QueryResult):
            return _resolve_source(qr)
        raise ChartError("the analysis has no grounded result to chart")
    raise ChartError(
        "generate_chart expects a QueryResult, an AnalysisResult, or a Dataset; "
        f"got {type(result).__name__}"
    )


def generate_chart(
    result: QueryResult | Dataset | Any,
    *,
    type: ChartType | str = ChartType.BAR,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str = "",
    renderer: ChartRenderer | None = None,
    signer: ContentSigner | None = None,
    infer_type: bool = True,
) -> Chart:
    """Turn a cited query result into a **content-bound, data-bound** chart.

    Infers the encoding from the result's schema when ``x`` / ``y`` / ``color`` are
    not pinned (a dimension on x, a measure on y, a second dimension as the series),
    renders the spec to bytes through ``renderer`` (the dependency-free
    :class:`VegaLiteRenderer` by default), stamps the bytes with a C2PA
    *data-driven* credential, and carries the result's per-cell lineage as the
    figure's back-reference to the rows it was built from::

        result = app.query_data("revenue by region", table="sales")
        chart = generate_chart(result, title="Revenue by region")
        chart.cite_refs()            # the exact source cells the figure rests on
        chart.verify(catalog)        # re-derives from the bytes + binds the credential

    Pass ``renderer=MatplotlibRenderer()`` (with the ``vincio[charts]`` extra) for a
    rasterized PNG, ``type=`` to choose the mark (a temporal x axis defaults to a
    line when ``infer_type`` is left on), and ``signer=`` to attach a cryptographic
    signature to the credential. Returns a :class:`Chart`."""
    mark = type if isinstance(type, ChartType) else ChartType(type)
    dataset, source, provenance, coverage, source_hashes, result_hash = _resolve_source(result)
    if dataset.row_count == 0:
        raise ChartError("cannot build a chart from an empty result")
    if infer_type and mark is ChartType.BAR:
        # A bar is the default; upgrade it to a line when the x axis is temporal.
        # An explicit non-bar mark, or ``infer_type=False``, is always honored.
        mark = _default_mark(dataset, x)
    encoding = _infer_encoding(dataset, mark=mark, x=x, y=y, color=color)
    values = _project(dataset, encoding)
    spec = ChartSpec(
        title=title,
        mark=mark,
        encoding=encoding,
        columns=encoding.fields,
        values=values,
    )
    renderer = renderer or VegaLiteRenderer()
    data = renderer.render(spec)
    manifest = mark_data_driven_content(
        data, media_type=renderer.media_type, provider="vincio", signer=signer
    )
    stamped = embed_provenance(data, manifest)
    if stamped is not data:
        # A renderer whose container can embed the credential (PNG) changed the
        # bytes; re-bind the manifest to what a consumer will actually receive.
        manifest = mark_data_driven_content(
            stamped, media_type=renderer.media_type, provider="vincio", signer=signer
        )
        data = stamped
    return Chart._build(
        spec,
        data,
        media_type=renderer.media_type,
        renderer=renderer.name,
        source=source,
        provenance=provenance,
        coverage=coverage,
        source_hashes=source_hashes,
        result_hash=result_hash,
        manifest=manifest,
    )
