"""Data-plane verbs (tabular evidence, streaming, query, analytics, metrics) — a private mixin of
:class:`~vincio.core.app.ContextApp`.

Extracted verbatim from ``vincio/core/app.py`` (v7.5 structure line): method
source, decorators, comments, and docstrings are unchanged. ``ContextApp``
composes this class, so every method here remains an ``app.*`` verb; the
``self: ContextApp`` annotations keep attribute access type-checked against
the composed app. The standing hygiene lints (:mod:`vincio._error_contract`,
:mod:`vincio._observable_failure`, :mod:`vincio._assert_robustness`)
deliberately keep ``vincio/core/_app_*.py`` in scope despite the private
filename, so the verb surface stays guarded after the split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import ContextApp


class _DataVerbs:
    """Data-plane verbs (tabular evidence, streaming, query, analytics, metrics). Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        _data_catalog_obj: Any


    # -- tabular evidence & the compact data encoder --------------------

    def table_evidence(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        *,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
        source_id: str = "",
        caption: str = "",
        encoder: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build first-class tabular evidence — a typed, columnar dataset rendered
        header-once — from rows, records, a :class:`~vincio.data.Dataset`, or a
        legacy ``TableData``.

        A dataset is *schema-bearing, columnar evidence*, never a row-flattened
        document: it carries a typed schema (per-column name, type, unit,
        nullability) and reaches the model as the compact, token-oriented encoding
        of :class:`~vincio.data.DataEncoder` (the schema declared once, the cells
        as delimited rows), with a columnar-accurate token cost. Add the result to
        ``app.pending_evidence`` for the next run, or hand it to the context
        compiler's ``evidence`` list directly::

            ev = app.table_evidence(
                [{"region": "NA", "revenue": 1200.5}, {"region": "EU", "revenue": 980.0}],
                name="sales",
            )
            app.pending_evidence.append(ev.to_evidence_item())
            result = await app.arun("Revenue by region?")

        ``data`` may be a list of record mappings, a list of rows (with ``columns``
        or ``schema``), a :class:`~vincio.data.Dataset`, or a ``TableData``.
        Returns a :class:`~vincio.data.TableEvidence`.
        """
        from ..core.errors import DataError
        from ..data import Dataset, TableEvidence

        if isinstance(data, TableEvidence):
            return data
        if isinstance(data, Dataset):
            dataset = data
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            dataset = Dataset.from_records(data, schema=schema, name=name)
        elif isinstance(data, list):
            spec = schema if schema is not None else columns
            if spec is None:
                raise DataError("from rows, pass `columns=` or `schema=` to name the columns")
            dataset = Dataset.from_rows(data, spec, name=name)
        elif hasattr(data, "columns") and hasattr(data, "rows"):
            dataset = Dataset.from_table_data(data, name=name)
        else:
            raise DataError(f"cannot build table evidence from {type(data).__name__}")
        return dataset.to_evidence(
            source_id=source_id or name or "dataset", caption=caption, encoder=encoder, **kwargs
        )

    def _coerce_dataset(  # type: ignore[misc]
        self: ContextApp, data: Any, *, schema: Any | None = None, columns: list[str] | None = None, name: str = ""
    ) -> Any:
        """Coerce records, rows, a ``TableData``, ``TableEvidence``, or a
        ``Dataset`` into a :class:`~vincio.data.Dataset` for the data-plane
        methods (profiling, sampling, screening, fitting)."""
        from ..core.errors import DataError
        from ..data import Dataset, TableEvidence

        if isinstance(data, TableEvidence):
            return data.dataset
        if isinstance(data, Dataset):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return Dataset.from_records(data, schema=schema, name=name)
        if isinstance(data, list):
            spec = schema if schema is not None else columns
            if spec is None:
                raise DataError("from rows, pass `columns=` or `schema=` to name the columns")
            return Dataset.from_rows(data, spec, name=name)
        if hasattr(data, "columns") and hasattr(data, "rows"):
            return Dataset.from_table_data(data, name=name)
        raise DataError(f"cannot build a dataset from {type(data).__name__}")

    def profile_dataset(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        *,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> Any:
        """Compute a deterministic, bounded-memory column profile of a dataset —
        per column its type, null rate, cardinality, extrema, mean/stddev,
        percentiles, a distribution histogram, and exemplars.

        The profile is fixed-size (its footprint depends on the number of columns,
        not the number of rows), so it stands in for a table that will never fit
        and is itself first-class evidence the context compiler scores and cites::

            profile = app.profile_dataset(rows, columns=["region", "revenue"])
            app.pending_evidence.append(profile.to_evidence_item())

        ``data`` may be a list of record mappings, a list of rows (with
        ``columns`` / ``schema``), a :class:`~vincio.data.Dataset`, a
        ``TableData``, or :class:`~vincio.data.TableEvidence`. Returns a
        :class:`~vincio.data.DatasetProfile`.
        """
        from ..data import profile_dataset as _profile

        dataset = self._coerce_dataset(data, schema=schema, columns=columns, name=name)
        return _profile(dataset, **kwargs)

    def sample_dataset(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        n: int,
        *,
        method: Any = "reservoir",
        by: Any = None,
        seed: int = 0,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
    ) -> Any:
        """Draw a representative sample of up to ``n`` rows that stands in for the
        whole dataset, replacing a biased first-N cutoff.

        ``method`` is ``reservoir`` (uniform, single-pass), ``stratified``
        (proportional across the ``by`` column, preserving its distribution),
        ``systematic`` (evenly spaced), or ``head``. Returns a schema-preserving
        :class:`~vincio.data.Dataset` that records how it was drawn in
        ``metadata['sample']`` and can be encoded, profiled, or carried as
        evidence exactly like any other dataset.
        """
        from ..data import sample_dataset as _sample

        dataset = self._coerce_dataset(data, schema=schema, columns=columns, name=name)
        return _sample(dataset, n, method=method, by=by, seed=seed)

    def fit_dataset(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        *,
        max_tokens: int,
        method: Any = "reservoir",
        by: Any = None,
        seed: int = 0,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fit a dataset far larger than the window into a fixed token budget: a
        full-fidelity column profile plus a representative sample sized to whatever
        budget the profile leaves.

        The representation stays within ``max_tokens`` whether the table has ten
        thousand rows or ten million — the profile is fixed-size and the sample is
        budget-bound. Returns a :class:`~vincio.data.WindowFit` whose
        ``to_evidence_items()`` yields the profile and the sample as cited table
        evidence::

            fit = app.fit_dataset(rows, columns=["region", "revenue"], max_tokens=2000)
            app.pending_evidence.extend(fit.to_evidence_items())
        """
        from ..data import fit_to_window

        dataset = self._coerce_dataset(data, schema=schema, columns=columns, name=name)
        return fit_to_window(
            dataset, max_tokens=max_tokens, method=method, by=by, seed=seed, model=model, **kwargs
        )

    def screen_data(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        *,
        rails: Any | None = None,
        constraints: list[Any] | None = None,
        detect_anomalies: bool = False,
        enforce_schema: bool = True,
        raise_on_block: bool = False,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
    ) -> Any:
        """Screen a tabular input for schema violations, constraint breaks, and
        anomalies on the same deterministic rail path PII and injection detection
        ride. The decision lands on the shared audit chain (``data_quality``).

        Pass an explicit :class:`~vincio.data.DataQualityRails` as ``rails``, a
        list of :class:`~vincio.data.ColumnConstraint`s as ``constraints``, or
        neither — in which case (``enforce_schema``) the dataset's own declared
        schema is enforced. With ``raise_on_block`` a blocking finding raises
        :class:`~vincio.core.errors.DataQualityError`. Returns a
        :class:`~vincio.data.DataQualityReport`.
        """
        from ..data import DataQualityRails

        dataset = self._coerce_dataset(data, schema=schema, columns=columns, name=name)
        if rails is None:
            if constraints is not None:
                rails = DataQualityRails(constraints, detect_anomalies=detect_anomalies)
            elif enforce_schema:
                rails = DataQualityRails.from_dataset(dataset, detect_anomalies=detect_anomalies)
            else:
                rails = DataQualityRails(detect_anomalies=detect_anomalies)
        report = rails.check(dataset)
        self.audit.record(
            "data_quality",
            decision="allow" if report.allowed else "deny",
            resource=dataset.name or "dataset",
            details={
                "row_count": report.row_count,
                "column_count": report.column_count,
                "violations": len(report.violations),
                "blocking": [f"{v.column}:{v.rule}" for v in report.blocking],
                "warnings": [f"{v.column}:{v.rule}" for v in report.warnings],
            },
        )
        if raise_on_block:
            report.raise_for_status()
        return report

    # -- streaming & out-of-core bulk processing ------------------------

    def stream_dataset(  # type: ignore[misc]
        self: ContextApp,
        source: Any,
        *,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
        format: str | None = None,
    ) -> Any:
        """Open a dataset larger than memory as a lazy, schema-bearing
        :class:`~vincio.data.RowStream` — the out-of-core handle the streaming
        operators consume in bounded passes.

        ``source`` may be a file path (CSV / JSON-Lines, chosen by ``format`` or
        the extension), a list of record mappings, a list of rows (with
        ``columns`` / ``schema``), a :class:`~vincio.data.Dataset`, or a
        zero-argument callable returning a fresh row iterator. Profile, fit,
        sample, :meth:`~vincio.data.RowStream.aggregate`, or
        :meth:`~vincio.data.RowStream.encode` the result without ever
        materializing the whole table::

            stream = app.stream_dataset("events.csv")
            app.pending_evidence.extend(stream.fit(max_tokens=2000).to_evidence_items())
        """
        from pathlib import Path

        from ..core.errors import DataError
        from ..data import Dataset, RowStream, TableEvidence

        if isinstance(source, RowStream):
            return source
        if isinstance(source, TableEvidence):
            return RowStream.from_dataset(source.dataset)
        if isinstance(source, Dataset):
            return RowStream.from_dataset(source)
        if isinstance(source, (str, Path)) and (format is not None or "\n" not in str(source)):
            return RowStream.open(source, format=format, schema=schema, name=name)
        if isinstance(source, list) and source and isinstance(source[0], dict):
            return RowStream.from_records(source, schema=schema, name=name)
        if isinstance(source, list):
            spec = schema if schema is not None else columns
            if spec is None:
                raise DataError("from rows, pass `columns=` or `schema=` to name the columns")
            return RowStream.from_rows(source, spec, name=name)
        if callable(source):
            spec = schema if schema is not None else columns
            if spec is None:
                raise DataError("from a row factory, pass `columns=` or `schema=` to name the columns")
            return RowStream.from_rows(source, spec, name=name)
        raise DataError(f"cannot stream {type(source).__name__}")

    def aggregate_stream(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        *,
        group_by: Any,
        measures: Any | None = None,
        max_groups: int = 1_000_000,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
    ) -> Any:
        """Group a dataset larger than memory by one or more columns and reduce
        measures over each group in a single bounded-memory pass.

        ``measures`` maps a column to the aggregation(s) to compute over it
        (``"sum"`` / ``"mean"`` / ``"min"`` / ``"max"``; each group's row
        ``count`` is always emitted). The working set tracks the number of
        *groups*, not rows, so a table far larger than memory aggregates inside a
        fixed footprint; a group cardinality beyond ``max_groups`` is refused.
        ``data`` may be a :class:`~vincio.data.RowStream`, a file path, records,
        rows (with ``columns`` / ``schema``), or a
        :class:`~vincio.data.Dataset`. Returns a
        :class:`~vincio.data.StreamAggregation`.
        """
        from ..data import stream_aggregate

        stream = self.stream_dataset(data, schema=schema, columns=columns, name=name)
        return stream_aggregate(
            stream, group_by=group_by, measures=measures, max_groups=max_groups
        )

    async def map_stream(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        build_request: Any,
        *,
        runner: Any | None = None,
        backend: Any | None = None,
        chunk_rows: int = 4_096,
        timeout_s: float | None = None,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
    ) -> Any:
        """Run an analytical transform over a dataset larger than memory *at
        scale* by chunking it into the provider Batch API.

        Each bounded chunk becomes one model request via ``build_request(chunk,
        index)`` (typically a prompt over the chunk's compact encoding), the set
        is dispatched through the existing
        :class:`~vincio.providers.BatchRunner` (half-cost, bounded concurrency),
        and the responses are reconciled by chunk index. Pass a ``runner`` /
        ``backend``, or omit both to use the app's own provider. Returns a
        :class:`~vincio.data.BulkMapResult`.
        """
        from ..data import stream_map

        stream = self.stream_dataset(data, schema=schema, columns=columns, name=name)
        if runner is None and backend is None:
            backend = self.resolve_provider()
        return await stream_map(
            stream,
            build_request,
            runner=runner,
            backend=backend,
            chunk_rows=chunk_rows,
            timeout_s=timeout_s,
        )

    # -- governed text-to-query & cell-level provenance -----------------

    def data_catalog(self: ContextApp) -> Any:  # type: ignore[misc]
        """The app's lazily-created :class:`~vincio.data.DataCatalog` — the grounding
        source for :meth:`query_data` and the catalog a
        :meth:`~vincio.data.QueryResult.verify` re-executes against."""
        catalog = getattr(self, "_data_catalog_obj", None)
        if catalog is None:
            from ..data import DataCatalog

            catalog = DataCatalog()
            self._data_catalog_obj = catalog
        return catalog

    def register_dataset(  # type: ignore[misc]
        self: ContextApp,
        data: Any,
        *,
        name: str = "",
        schema: Any | None = None,
        columns: list[str] | None = None,
        source: str | None = None,
    ) -> str:
        """Register a dataset in the app's data catalog so :meth:`query_data` can
        ground and execute a query against it by name.

        ``data`` may be records, rows (with ``columns`` / ``schema``), a
        :class:`~vincio.data.Dataset`, a ``TableData``, or
        :class:`~vincio.data.TableEvidence`. Returns the resolved table name.

        The dataset is recorded in the **lineage index** under ``source`` (defaulting
        to the dataset's own ``source``, then its table name), with its columns — so
        a governed metric's column-level provenance traces back to it and a
        :meth:`erase_source` sweep removes it alongside the source's documents,
        memories, and artifacts. The registration is audited (``data_register``)::

            app.register_dataset(rows, columns=["region", "revenue"], name="sales")
            result = app.query_data("total revenue by region", table="sales")
        """
        dataset = self._coerce_dataset(data, schema=schema, columns=columns, name=name)
        table = self.data_catalog().add(dataset, name=name)
        resolved_source = source or dataset.source or table
        self.lineage.record_dataset(resolved_source, table, dataset.column_names)
        self.audit.record(
            "data_register",
            resource=table,
            details={
                "row_count": dataset.row_count,
                "column_count": dataset.width,
                "source": resolved_source,
            },
        )
        return table

    def query_data(  # type: ignore[misc]
        self: ContextApp,
        request: str,
        *,
        dataset: Any | None = None,
        table: str | None = None,
        dialect: Any = "sql",
        ops: list[Any] | None = None,
        question: str = "",
        max_rows: int = 10_000,
        engine: Any | None = None,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
        raise_on_refusal: bool = True,
    ) -> Any:
        """Turn a natural-language question (or explicit SQL / dataframe ops) over a
        registered dataset into a query that is **schema-grounded and verified
        before it runs**, executed where the data lives rather than materialized
        into the prompt, and whose answer **cites the exact rows and cells** it
        rests on — the analytics analogue of a cited report, offline-verifiable.

        The query is held **read-only by default**: it is screened structurally (a
        write, DDL, stacked statement, or an injection signal in the question is
        refused, raising :class:`~vincio.core.errors.UnsafeQueryError`) and executed
        by the offline ``sqlite3`` engine under a deny-writes authorizer — the same
        guarantee :func:`~vincio.data.build_query_contract` carries when the
        capability rides the permissioned tool runtime. Every decision lands on the
        audit chain (``data_query``)::

            app.register_dataset(rows, columns=["region", "revenue"], name="sales")
            result = app.query_data("total revenue by region", table="sales")
            result.value(0, "sum_revenue")          # the answer
            result.cite_refs(0, "sum_revenue")      # the exact source cells it rests on
            result.verify(app.data_catalog())       # re-derives from the bytes

        Pass ``dataset=`` for a one-shot over an unregistered table, or
        ``dialect="dataframe"`` with ``ops=`` for the deterministic dataframe-op
        path. Returns a :class:`~vincio.data.QueryResult` (or ``None`` when a
        refusal is caught with ``raise_on_refusal=False``).
        """
        from ..core.errors import QueryError, UnsafeQueryError
        from ..data import DataCatalog, query_dataset

        if dataset is not None:
            ds = self._coerce_dataset(dataset, schema=schema, columns=columns, name=name)
            catalog = DataCatalog.of(ds, name=name or ds.name or "data")
        else:
            catalog = self.data_catalog()
            if not catalog.names:
                raise QueryError(
                    "no dataset registered; pass dataset= or call "
                    "app.register_dataset(...) first"
                )
        try:
            result = query_dataset(
                request,
                catalog,
                dialect=dialect,
                question=question,
                ops=ops,
                table=table,
                max_rows=max_rows,
                engine=engine,
            )
        except UnsafeQueryError as exc:
            self.audit.record(
                "data_query",
                decision="deny",
                resource=table or (catalog.names[0] if catalog.names else "dataset"),
                details={"refused": "unsafe", "reason": str(exc)[:200]},
            )
            if raise_on_refusal:
                raise
            return None
        self.audit.record(
            "data_query",
            decision="allow",
            resource=",".join(result.plan.tables) or "dataset",
            details={
                "dialect": str(result.plan.dialect),
                "row_count": result.row_count,
                "lineage_coverage": str(result.coverage),
                "result_hash": result.result_hash,
            },
        )
        return result

    def analyze_data(  # type: ignore[misc]
        self: ContextApp,
        objective: str,
        *,
        dataset: Any | None = None,
        table: str | None = None,
        budget: Any | None = None,
        max_steps: int | None = None,
        engine: Any | None = None,
        propose_followups: bool = True,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
        raise_on_refusal: bool = True,
    ) -> Any:
        """Run a bounded, multi-step analysis over a registered dataset and return a
        **cited analytical narrative** — the data plane's analyst agent.

        The agent plans (an overview, the objective grounded to a query, the
        measures' extremes and totals, a measure-by-dimension breakdown), queries
        each step through the governed, **read-only-verified** query plane, inspects
        the result, and refines by drilling into the group that dominates — bounded
        by an :class:`~vincio.data.AnalysisBudget`. Every finding **cites the exact
        source cells** it rests on, the narrative re-derives from the bytes via
        :meth:`~vincio.data.AnalysisResult.verify`, and the whole run lands on the
        audit chain (``data_analysis``)::

            app.register_dataset(rows, columns=["region", "revenue"], name="sales")
            analysis = app.analyze_data("how does revenue break down by region?", table="sales")
            print(analysis.narrative)               # the cited narrative
            analysis.verify(app.data_catalog())     # re-derives every finding from the bytes

        The objective is screened by the same injection detector the text rails use
        (a refusal raises :class:`~vincio.core.errors.UnsafeQueryError`); pass
        ``dataset=`` for a one-shot over an unregistered table, ``budget=`` or
        ``max_steps=`` to bound the run, and ``engine=`` (e.g.
        :class:`~vincio.data.DuckDbQueryEngine`) to push the queries down at scale.
        Returns an :class:`~vincio.data.AnalysisResult` (or ``None`` when a refusal
        is caught with ``raise_on_refusal=False``).
        """
        from ..core.errors import AnalysisError, UnsafeQueryError
        from ..data import AnalysisAgent, AnalysisBudget

        if budget is None and max_steps is not None:
            budget = AnalysisBudget(max_steps=max_steps)
        ds = None
        if dataset is not None:
            ds = self._coerce_dataset(dataset, schema=schema, columns=columns, name=name)
        elif not self.data_catalog().names:
            raise AnalysisError(
                "no dataset registered; pass dataset= or call app.register_dataset(...) first"
            )
        agent = AnalysisAgent(
            self, budget=budget, engine=engine, propose_followups=propose_followups
        )
        try:
            return agent.run(objective, table=table, dataset=ds)
        except UnsafeQueryError as exc:
            self.audit.record(
                "data_analysis",
                decision="deny",
                resource=table or (ds.name if ds is not None else "dataset"),
                details={"refused": "unsafe", "reason": str(exc)[:200]},
            )
            if raise_on_refusal:
                raise
            return None

    def generate_chart(  # type: ignore[misc]
        self: ContextApp,
        result: Any,
        *,
        type: Any = "bar",
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        title: str = "",
        renderer: Any | None = None,
        signer: Any | None = None,
        infer_type: bool = True,
        table: str | None = None,
        max_rows: int = 10_000,
        engine: Any | None = None,
    ) -> Any:
        """Turn a cited query result into a **content-bound, data-bound** chart — the
        data plane's generated analytical artifact.

        ``result`` may be a :class:`~vincio.data.QueryResult` (or
        :class:`~vincio.data.AnalysisResult` / :class:`~vincio.data.Dataset`), or a
        natural-language question / SQL string that is first run through the governed,
        read-only-verified query plane (:meth:`query_data`, with ``table=``). The
        figure carries a C2PA *data-driven* credential bound to its rendered bytes and
        a back-reference to the **exact source cells** it was built from, and the run
        lands on the audit chain (``chart_generate``)::

            result = app.query_data("revenue by region", table="sales")
            chart = app.generate_chart(result, title="Revenue by region")
            chart.cite_refs()             # the exact source cells the figure rests on
            chart.verify(app.data_catalog())   # re-derives + binds the credential

        The default renderer is the dependency-free
        :class:`~vincio.data.VegaLiteRenderer`; pass
        ``renderer=MatplotlibRenderer()`` (with the ``vincio[charts]`` extra) for a
        rasterized PNG. Returns a :class:`~vincio.data.Chart`."""
        from ..data import generate_chart as _generate_chart

        if isinstance(result, str):
            result = self.query_data(result, table=table, max_rows=max_rows, engine=engine)
        chart = _generate_chart(
            result,
            type=type,
            x=x,
            y=y,
            color=color,
            title=title,
            renderer=renderer,
            signer=signer,
            infer_type=infer_type,
        )
        self.audit.record(
            "chart_generate",
            resource=title or chart.spec.mark.value,
            details={
                "chart_type": chart.spec.mark.value,
                "renderer": chart.renderer,
                "media_type": chart.media_type,
                "points": chart.point_count,
                "lineage_coverage": str(chart.coverage),
                "result_hash": chart.result_hash,
                "chart_hash": chart.chart_hash,
                # frozen audit-detail key — external consumers bind to it.
                "content_sha256": chart.manifest.content_hash if chart.manifest else None,
            },
        )
        return chart

    # -- semantic layer & governed metrics ------------------------------

    def semantic_layer(  # type: ignore[misc]
        self: ContextApp,
        table: str,
        *,
        measures: list[Any] | None = None,
        dimensions: list[Any] | None = None,
        derived: list[Any] | None = None,
        name: str = "",
        description: str = "",
        register: bool = True,
        validate: bool = True,
    ) -> Any:
        """Define a :class:`~vincio.data.SemanticLayer` over a registered table —
        measures, dimensions, and derived columns declared **once** so a question
        maps to a **governed metric** rather than a raw column.

        ``measures`` / ``dimensions`` / ``derived`` are :class:`~vincio.data.Measure`
        / :class:`~vincio.data.Dimension` / :class:`~vincio.data.DerivedColumn`
        instances (or mappings with the same fields). When ``register`` (the
        default) the layer is kept on the app and resolved by :meth:`query_metric`
        and :meth:`metric_lineage`; when ``validate`` and the table is registered,
        every metric and dimension is dry-run-grounded against it. The definition is
        audited (``semantic_layer_define``)::

            app.register_dataset(rows, columns=["region", "price", "qty"], name="sales")
            layer = app.semantic_layer(
                "sales",
                derived=[DerivedColumn(name="revenue", expression="price * qty")],
                measures=[Measure(name="total_revenue", agg="sum", expression="revenue")],
                dimensions=[Dimension(name="region")],
            )
            result = app.query_metric("total_revenue", by=["region"])

        Returns the :class:`~vincio.data.SemanticLayer`.
        """
        from ..data import DerivedColumn, Dimension, Measure, SemanticLayer

        def _coerce(items: list[Any] | None, cls: type[Any]) -> list[Any]:
            out: list[Any] = []
            for item in items or []:
                out.append(item if isinstance(item, cls) else cls(**item))
            return out

        layer = SemanticLayer(
            table=table,
            name=name,
            description=description,
            derived=_coerce(derived, DerivedColumn),
            dimensions=_coerce(dimensions, Dimension),
            measures=_coerce(measures, Measure),
        )
        if validate and table in self.data_catalog():
            layer.validate_against(self.data_catalog())
        if register:
            self._semantic_layers[table] = layer
        self.audit.record(
            "semantic_layer_define",
            resource=table,
            details={
                "name": name or table,
                "measures": layer.metric_names,
                "dimensions": layer.dimension_names,
                "derived": [d.name for d in layer.derived],
                "registered": register,
            },
        )
        return layer

    def _resolve_layer(self: ContextApp, layer: Any | None, table: str | None) -> Any:  # type: ignore[misc]
        from ..core.errors import SemanticLayerError

        if layer is not None:
            return layer
        if table is not None:
            if table not in self._semantic_layers:
                raise SemanticLayerError(
                    f"no semantic layer registered for table {table!r}; call "
                    "app.semantic_layer(...) first or pass layer="
                )
            return self._semantic_layers[table]
        if len(self._semantic_layers) == 1:
            return next(iter(self._semantic_layers.values()))
        if not self._semantic_layers:
            raise SemanticLayerError(
                "no semantic layer registered; call app.semantic_layer(...) first "
                "or pass layer="
            )
        raise SemanticLayerError(
            "more than one semantic layer registered; pass table= or layer= to "
            f"choose ({sorted(self._semantic_layers)})"
        )

    def query_metric(  # type: ignore[misc]
        self: ContextApp,
        request: Any,
        *,
        layer: Any | None = None,
        table: str | None = None,
        by: list[str] | None = None,
        where: list[str] | None = None,
        order_by: str = "",
        descending: bool = False,
        limit: int | None = None,
        dataset: Any | None = None,
        max_rows: int = 10_000,
        engine: Any | None = None,
        schema: Any | None = None,
        columns: list[str] | None = None,
        name: str = "",
        raise_on_refusal: bool = True,
    ) -> Any:
        """Compute a **governed metric** — a measure resolved through a
        :class:`~vincio.data.SemanticLayer` and computed **one way everywhere**.

        ``request`` is a metric name, a list of metric names, a
        :class:`~vincio.data.MetricQuery`, or a natural-language question the layer
        grounds to a governed metric (the question is injection-screened first). The
        metric compiles to a single read-only ``SELECT`` and runs through the same
        governed, read-only-verified query plane :meth:`query_data` uses, so the
        answer **cites the exact source cells** and re-derives from the bytes — and
        :meth:`~vincio.data.MetricResult.verify` additionally proves the SQL was the
        layer's canonical compilation, so an ad-hoc number cannot pass as the
        governed one. The run is audited (``metric_query``)::

            result = app.query_metric("total_revenue", by=["region"])
            result.value(0)                         # the governed number
            result.cite_refs(0)                     # the exact source cells
            result.verify(layer, app.data_catalog())  # governed + re-derives

        Resolve the layer explicitly (``layer=``), by ``table=``, or implicitly when
        exactly one is registered. Pass ``dataset=`` to compute over an unregistered
        table. Returns a :class:`~vincio.data.MetricResult` (or ``None`` when a
        refusal is caught with ``raise_on_refusal=False``).
        """
        from ..core.errors import SemanticLayerError, UnsafeQueryError
        from ..data import DataCatalog, query_metric

        resolved = self._resolve_layer(layer, table)
        if dataset is not None:
            ds = self._coerce_dataset(dataset, schema=schema, columns=columns, name=name)
            # Ground the one-shot dataset under the layer's table so the compiled
            # metric SQL (which references it by name) resolves.
            data: Any = DataCatalog.of(ds, name=name or resolved.table)
        else:
            data = self.data_catalog()
        try:
            result = query_metric(
                request,
                data,
                layer=resolved,
                by=by,
                where=where,
                order_by=order_by,
                descending=descending,
                limit=limit,
                engine=engine,
                max_rows=max_rows,
            )
        except (UnsafeQueryError, SemanticLayerError) as exc:
            self.audit.record(
                "metric_query",
                decision="deny",
                resource=resolved.table,
                details={
                    "refused": "unsafe" if isinstance(exc, UnsafeQueryError) else "ungrounded",
                    "reason": str(exc)[:200],
                },
            )
            if raise_on_refusal:
                raise
            return None
        self.audit.record(
            "metric_query",
            decision="allow",
            resource=resolved.table,
            details={
                "metrics": result.metrics,
                "dimensions": result.dimensions,
                "row_count": result.row_count,
                "lineage_coverage": str(result.coverage),
                "result_hash": result.result.result_hash,
                "layer_hash": result.layer_hash,
            },
        )
        return result

    def metric_lineage(  # type: ignore[misc]
        self: ContextApp,
        metric: str,
        *,
        layer: Any | None = None,
        table: str | None = None,
    ) -> Any:
        """The **column-level provenance** of a governed metric — the base columns
        and source it rests on, resolving the derived-column graph and any ratio
        references.

        Fills :attr:`~vincio.data.MetricLineage.source` from the lineage index (the
        source the dataset was registered under), so a metric's provenance reaches
        the same machinery a document's lineage and a subject's erasure do. Audited
        (``metric_lineage``)::

            lin = app.metric_lineage("total_revenue")
            lin.base_columns                 # ['price', 'qty']
            lin.source                       # the source the dataset was ingested under
        """
        resolved = self._resolve_layer(layer, table)
        lineage = resolved.column_lineage(metric, catalog=self.data_catalog())
        lineage.source = self.lineage.source_of_table(resolved.table) or resolved.table
        self.audit.record(
            "metric_lineage",
            resource=resolved.table,
            details={
                "metric": metric,
                "base_columns": lineage.base_columns,
                "derived_via": lineage.derived_via,
                "source": lineage.source,
            },
        )
        return lineage

    # -- data & analytics capstone --------------------------------------

    def data_engagement(  # type: ignore[misc]
        self: ContextApp,
        *,
        dataset: str = "",
        question: str = "",
        analyst: str | None = None,
    ) -> Any:
        """Thread the whole data & analytics plane behind one governed call-path.

        Returns a :class:`~vincio.data.DataEngagement` — the capstone facade that
        composes the entire pipeline (register → profile → sample → fit → screen →
        query → analyze → chart → governed metric → cite) into one governed, audited,
        hash-linked narrative. Each lifecycle method delegates to the *same* entry
        point on this app a caller would use directly, so the primitives stay
        unchanged and usable on their own; the facade only captures and **narrates**
        them.

        :meth:`~vincio.data.DataEngagement.seal` mints the content-bound, signed
        :class:`~vincio.data.DataNarrative`, and
        :meth:`~vincio.data.DataEngagement.verify` proves the whole chain — every
        captured artifact's digest, and (given the catalog) every analytical answer's
        re-derivation from the source it cites — verifies offline, so a tamper
        introduced anywhere is caught::

            eng = app.data_engagement(question="how does revenue break down by region?")
            eng.register(rows, columns=["region", "price", "qty"], name="sales")
            eng.profile()
            eng.query("total revenue by region")
            eng.analyze("how does revenue break down by region?")
            eng.chart(eng.result, title="Revenue by region")
            eng.cite(title="Revenue analysis")
            narrative = eng.seal()
            eng.verify(app.contract_signer).valid          # chain + digests + data-bound
            narrative.verify(app.contract_signer).valid     # offline from the bytes alone
        """
        from ..data.engagement import DataEngagement

        return DataEngagement(self, dataset=dataset, question=question, analyst=analyst)

    def federated_data_engagement(  # type: ignore[misc]
        self: ContextApp,
        *,
        query: Any | None = None,
        coordinator: str | None = None,
        layer: Any | None = None,
    ) -> Any:
        """Run a governed analytics query **across organizations** without pooling
        the raw rows — the cross-org / federated twin of :meth:`data_engagement`.

        Returns a :class:`~vincio.data.FederatedDataEngagement`: add each
        participating org with
        :meth:`~vincio.data.FederatedDataEngagement.add_member`, then thread the
        lifecycle — negotiate the :class:`~vincio.data.FederatedQuery` into a signed
        :class:`~vincio.negotiation.Contract`, choreograph a contract-governed
        :class:`~vincio.choreography.Saga` so each org runs the governed metric
        **locally** and returns only its aggregated, cell-cited
        :class:`~vincio.data.MetricResult`, and reconcile the aggregates into one
        signed, offline-verifiable :class:`~vincio.data.FederatedNarrative`. The raw
        rows never cross the trust boundary; residency egress refusal, the consent
        ledger's analytics purpose, the differential-privacy accountant, and the
        ``min_members`` k-anonymity floor all apply at the boundary exactly as they
        would to a local query::

            from vincio.data import FederatedQuery

            q = FederatedQuery.of("total_revenue", table="sales", by=["region"])
            fed = app.federated_data_engagement(query=q)
            fed.add_member("acme", acme_app, region="us-east-1")
            fed.add_member("globex", globex_app, region="eu-west-1")
            findings = fed.run()                 # negotiate → dispatch → reconcile
            narrative = fed.seal()
            fed.verify(app.contract_signer).valid   # chain + digests + data-bound
        """
        from ..data.federated import FederatedDataEngagement

        return FederatedDataEngagement(self, query=query, coordinator=coordinator, layer=layer)

    # -- real-time & streaming analytics --------------------------------

    def stream_analytics(  # type: ignore[misc]
        self: ContextApp,
        window: Any,
        *,
        table: str = "events",
        layer: Any | None = None,
    ) -> Any:
        """Open a governed real-time analytics driver over an **unbounded event
        stream** — the profiling, query, governed-metric, and quality primitives
        re-expressed window by window.

        Pass a :class:`~vincio.data.StreamWindow` (``tumbling`` / ``sliding`` /
        ``session``) and get a :class:`~vincio.data.StreamingAnalytics`: drive a
        replayed :class:`~vincio.data.RowStream` (or a live realtime session)
        through :meth:`~vincio.data.StreamingAnalytics.profile`,
        :meth:`~vincio.data.StreamingAnalytics.query`,
        :meth:`~vincio.data.StreamingAnalytics.query_metric`,
        :meth:`~vincio.data.StreamingAnalytics.screen`, or
        :meth:`~vincio.data.StreamingAnalytics.aggregate`, each emitting one
        result per closed window. The working set holds only the open windows, so
        the footprint is invariant to how many events have flowed; every result
        **cites the exact events** it rests on and ``verify()``s offline against
        its bounded captured window; and each emitted window lands on the audit
        chain (``stream_window``)::

            from vincio.data import StreamWindow, ColumnSchema, DataType, RowStream

            schema = [ColumnSchema(name="ts", dtype=DataType.INT),
                      ColumnSchema(name="region", dtype=DataType.STR),
                      ColumnSchema(name="amount", dtype=DataType.FLOAT)]
            stream = RowStream.from_rows(event_log, schema, name="orders")
            win = StreamWindow.tumbling(size=60, time_column="ts", table="orders")
            analytics = app.stream_analytics(win, table="orders")
            for wq in analytics.query(stream, "SELECT region, sum(amount) AS total "
                                              "FROM orders GROUP BY region"):
                print(wq.window.label(), wq.value(0, "total"))
                wq.cite_events(0, "total")   # the exact events the figure rests on
                assert wq.verify()           # re-derives from the captured window

        Returns a :class:`~vincio.data.StreamingAnalytics`."""
        from ..data.streaming_analytics import StreamingAnalytics

        return StreamingAnalytics(self, window, table=table, layer=layer)
