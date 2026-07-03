"""Source/retrieval and memory verbs — a private mixin of
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

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..documents.loaders import load_directory, load_document
from ..memory.engine import MemoryEngine
from ..memory.policies import MemoryWritePolicy
from ..memory.stores import SQLiteMemoryStore
from ..providers.base import run_sync
from ..retrieval.chunking import chunk_document
from ..retrieval.engine import RetrievalEngine
from ..retrieval.graph_retrieval import EntityGraph
from ..retrieval.indexes import (
    BM25Index,
    VectorIndex,
)
from ..retrieval.late_interaction import LateInteractionIndex
from ..retrieval.rerankers import build_reranker
from ..retrieval.sparse import SparseIndex
from ._app_support import _SourceConfig
from .errors import (
    ConfigError,
    InputError,
)
from .types import (
    EvidenceItem,
    MemoryItem,
)

if TYPE_CHECKING:
    from .app import ContextApp


class _KnowledgeVerbs:
    """Source/retrieval and memory verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        _bm25: BM25Index | None
        _late_interaction: LateInteractionIndex | None
        _sparse: SparseIndex | None
        _vector: VectorIndex | None
        entity_graph: EntityGraph | None
        memory: MemoryEngine | None
        memory_enabled: bool
        retrieval: RetrievalEngine | None


    # -- sources / retrieval ----------------------------------------------------------------

    def _ensure_retrieval(self: ContextApp, retrieval: str) -> None:  # type: ignore[misc]
        if self._bm25 is None:
            self._bm25 = BM25Index()
        if self._vector is None:
            self._vector = VectorIndex(self.embedder)
        if retrieval in ("sparse", "hybrid_full") and self._sparse is None:
            self._sparse = SparseIndex()
        if retrieval in ("late_interaction", "hybrid_full") and self._late_interaction is None:
            self._late_interaction = LateInteractionIndex()
        indexes: list[Any]
        if retrieval == "bm25":
            indexes = [self._bm25]
        elif retrieval in ("dense", "vector"):
            indexes = [self._vector]
        elif retrieval == "sparse":
            indexes = [self._sparse]
        elif retrieval == "late_interaction":
            indexes = [self._late_interaction]
        elif retrieval == "hybrid_full":
            # Lexical + dense + learned-sparse + late-interaction, one RRF.
            indexes = [self._bm25, self._vector, self._sparse, self._late_interaction]
        else:  # hybrid / hybrid_graph
            indexes = [self._bm25, self._vector]
        reranker = build_reranker(self.config.retrieval.reranker)
        self.retrieval = RetrievalEngine(
            indexes,
            reranker=reranker,
            candidate_multiplier=self.config.retrieval.candidate_multiplier,
            query_strategies=self.config.retrieval.query_strategies,
            adaptive_top_k=self.config.retrieval.adaptive_top_k,
            adaptive_top_k_ceiling=self.config.retrieval.adaptive_top_k_ceiling,
        )
        if retrieval in ("graph", "hybrid_graph") and self.entity_graph is None:
            self.entity_graph = EntityGraph()

    def add_source(  # type: ignore[misc]
        self: ContextApp,
        name: str,
        *,
        path: str | None = None,
        documents: list[Any] | None = None,
        connector: Any | None = None,
        loader: str | None = None,
        chunking: str | None = None,
        retrieval: str = "hybrid",
        anchor: bool = False,
        brief_tokens: int = 400,
    ) -> ContextApp:
        """Register a knowledge source: load, chunk, and index documents.

        Sources can come from a local ``path``, in-memory ``documents``, or
        any :class:`~vincio.connectors.Connector` (web, GitHub, SQL, S3,
        GCS, Notion, Confluence, Slack, or custom) via ``connector=``.

        Set ``anchor=True`` to make this a **task frame**: a PRD, spec, brand
        identity, or coding-standards corpus that is 100% needed for the global
        context of a multi-call task but not in full on every call. The docs are
        indexed for on-demand detail *and* distilled once into a compact,
        content-hash-cached brief (bounded by ``brief_tokens``, constraint-first)
        injected as **pinned** evidence into every run — so the frame is always
        present at a flat ~few-hundred-token cost instead of re-pasting the whole
        corpus. Inspect it with :meth:`task_brief`.
        """
        chunking = chunking or self.config.retrieval.chunking
        source = _SourceConfig(
            name=name, path=path, loader=loader, chunking=chunking, retrieval=retrieval,
            anchor=anchor, brief_tokens=brief_tokens,
        )
        self._ensure_retrieval(retrieval)
        docs = list(documents or [])
        if connector is not None:
            docs.extend(run_sync(connector.load()))
        if path is not None:
            target = Path(path)
            if target.is_dir():
                docs.extend(load_directory(target))
            elif target.is_file():
                docs.append(load_document(target))
            else:
                raise ConfigError(f"source path not found: {path}")
        all_chunks = []
        for document in docs:
            document.metadata.setdefault("source", name)
            chunks = chunk_document(
                document,
                strategy=chunking,
                size=self.config.retrieval.chunk_size_tokens,
                overlap=self.config.retrieval.chunk_overlap_tokens,
                cache=self.chunk_cache,
            )
            all_chunks.extend(chunks)
            self.store.save(
                "documents",
                {
                    "id": document.id,
                    "title": document.title,
                    "source": name,
                    "uri": document.source_uri,
                },
            )
        if all_chunks:
            run_sync(self._index_chunks(all_chunks))
            if self.entity_graph is not None:
                self.entity_graph.add_chunks(all_chunks)
        # Lineage: record source → documents → chunks so erasure-by-source
        # and provenance tracing have a precomputed chain.
        self.lineage.record_ingest(name, documents=docs, chunks=all_chunks)
        source.document_count = len(docs)
        source.chunk_count = len(all_chunks)
        self.sources[name] = source
        # Task frame: register the anchor's documents so its always-on brief is
        # (re)built and cached; the same docs also stay indexed for on-demand detail.
        if anchor and docs:
            self.anchors.add(name, docs, brief_tokens=brief_tokens)
        return self

    def task_brief(self: ContextApp) -> str | None:  # type: ignore[misc]
        """The current task-frame brief — the compact, constraint-first digest of
        every ``anchor=True`` source, injected as pinned evidence on every run —
        or ``None`` when no anchors are registered."""
        brief = self.anchors.brief()
        return brief.text if brief is not None else None

    async def _index_chunks(self: ContextApp, chunks: list[Any]) -> None:  # type: ignore[misc]
        if self._bm25 is not None:
            await self._bm25.add(chunks)
        if self._vector is not None:
            await self._vector.add(chunks)
        if self._sparse is not None:
            await self._sparse.add(chunks)
        if self._late_interaction is not None:
            await self._late_interaction.add(chunks)

    async def ingest_files(self: ContextApp, paths: list[str]) -> list[EvidenceItem]:  # type: ignore[misc]
        """Ad-hoc file ingestion for run(files=[...]): load, chunk, index."""
        evidence: list[EvidenceItem] = []
        for path in paths:
            if path in self._ingested_files:
                evidence.extend(self._ingested_files[path])
                continue
            document = load_document(path)
            chunks = chunk_document(
                document,
                strategy=self.config.retrieval.chunking,
                size=self.config.retrieval.chunk_size_tokens,
                overlap=self.config.retrieval.chunk_overlap_tokens,
                cache=self.chunk_cache,
            )
            if self.retrieval is None:
                self._ensure_retrieval("hybrid")
            await self._index_chunks(chunks)
            if self.entity_graph is not None:
                self.entity_graph.add_chunks(chunks)
            self.lineage.record_ingest(path, documents=[document], chunks=chunks)
            items = [
                EvidenceItem(
                    id=chunk.citation_ref,
                    source_id=chunk.document_id,
                    text=chunk.text,
                    page=chunk.page,
                    section_path=chunk.section_path,
                    token_cost=chunk.token_count,
                    relevance=0.5,
                    provenance=0.9,
                )
                for chunk in chunks[:50]
            ]
            self._ingested_files[path] = items
            evidence.extend(items)
        return evidence

    def retrieve_facts(  # type: ignore[misc]
        self: ContextApp,
        query: str,
        *,
        facts: Sequence[str] | Sequence[Any] | Any,
        task: str | None = None,
        where: Any | None = None,
        top_k: int = 12,
        coverage_threshold: float = 0.15,
        per_fact_top_k: int = 3,
    ) -> Any:
        """Retrieve by the facts a task *needs*, reporting per-fact coverage and gaps.

        Reasoning retrieval, instead of one top-k by query similarity: a
        :class:`~vincio.retrieval.FactSchema` declares the facts the task requires
        (a refund decision needs the plan, the payment status, the refund policy),
        and the engine retrieves the task query, then runs a targeted retrieval for
        each fact still uncovered. The result is the merged evidence with a
        :class:`~vincio.retrieval.FactCoverage` per fact and a ``complete`` flag that
        is ``False`` while any *required* fact is missing — the signal an agent uses
        to gather more evidence rather than answer on a gap (the
        insufficient-evidence behaviour).

        Index documents first with :meth:`add_source`. *facts* is a list of fact
        names, a list of :class:`~vincio.retrieval.FactRequirement`, or a ready
        :class:`~vincio.retrieval.FactSchema`. Returns a
        :class:`~vincio.retrieval.FactRetrieval`::

            app.add_source("kb", documents=docs)
            result = app.retrieve_facts(
                "should this refund be approved?",
                facts=["refund_policy", "payment_status", "plan_tier"],
            )
            if not result.complete:
                ask_for(result.missing_facts)
        """
        from ..retrieval.reasoning_retrieval import (
            FactRequirement,
            FactSchema,
            ReasoningRetriever,
        )

        if self.retrieval is None:
            raise InputError(
                "retrieve_facts needs an indexed source; call add_source(...) first"
            )
        if isinstance(facts, FactSchema):
            schema = facts
        else:
            # A lone string is one fact name, not a sequence of single-character
            # facts (a str is iterable) — coerce it so the footgun can't fire.
            items: list[Any] = [facts] if isinstance(facts, str) else list(facts)
            schema_task = task or query
            if items and isinstance(items[0], FactRequirement):
                schema = FactSchema(task=schema_task, facts=list(items))
            else:
                schema = FactSchema.from_names(schema_task, [str(name) for name in items])
        retriever = ReasoningRetriever(
            self.retrieval,
            coverage_threshold=coverage_threshold,
            per_fact_top_k=per_fact_top_k,
        )
        return run_sync(retriever.retrieve_facts(query, schema, where=where, top_k=top_k))

    # -- memory ---------------------------------------------------------------------------------

    def add_memory(  # type: ignore[misc]
        self: ContextApp,
        *,
        scope: str = "user",
        strategy: str = "semantic",
        store: Any | None = None,
        embedder: Any | None = None,
    ) -> ContextApp:
        """Enable the scoped memory engine (hybrid vector+graph recall by default)."""
        if store is None:
            metadata_url = self.config.storage.metadata
            if metadata_url.startswith("sqlite"):
                from ..storage.base import parse_storage_url

                _scheme, location = parse_storage_url(metadata_url)
                store = SQLiteMemoryStore(Path(location).with_name("memory.db"))
            else:
                from ..memory.stores import InMemoryMemoryStore

                store = InMemoryMemoryStore()
        if embedder is None and self.config.memory.hybrid_recall:
            embedder = self.embedder
        self.memory = MemoryEngine(
            store,
            write_policy=MemoryWritePolicy(min_confidence=self.config.memory.min_confidence),
            decay_lambda=self.config.memory.decay_lambda,
            min_confidence=self.config.memory.min_confidence,
            graph_enabled=strategy in ("semantic_graph", "graph"),
            embedder=embedder,
            vector_weight=self.config.memory.vector_weight,
            retention_weight=self.config.memory.retention_weight,
            ttl_days=self.config.memory.ttl_days,
            audit=self.audit,
            consent_ledger=self.consent_ledger,
            privacy_accountant=self.privacy_accountant,
        )
        self.memory_enabled = self.config.memory.enabled
        return self

    def remember(self: ContextApp, content: str, **kwargs: Any) -> MemoryItem:  # type: ignore[misc]
        """Ergonomic memory write; creates the memory engine on first use."""
        if self.memory is None:
            self.add_memory()
        return self.memory.remember(content, **kwargs)  # type: ignore[union-attr]

    def recall(self: ContextApp, query: str, **kwargs: Any) -> list[MemoryItem]:  # type: ignore[misc]
        """Ergonomic memory recall over user/agent/session scopes."""
        if self.memory is None:
            self.add_memory()
        return self.memory.recall(query, **kwargs)  # type: ignore[union-attr]

    def consolidate_memory(  # type: ignore[misc]
        self: ContextApp,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        min_age_days: float = 7.0,
        summarizer: Any | None = None,
    ) -> Any:
        """Run episodic→semantic memory consolidation as a maintenance pass.

        The periodic background tier transition: a session's episodic memories
        (observations, evidence, and tool write-backs) are summarized into a few
        durable semantic memories promoted to the user (or agent) scope,
        near-duplicates are merged, and the source episodes are archived with
        provenance — never silently dropped. With a differential-privacy
        accountant attached (:meth:`add_privacy_accountant`), a consolidation that
        would exceed a subject's budget is refused and recorded rather than run.

        Pass ``session_id`` to consolidate one session now (returns its
        :class:`~vincio.memory.consolidation.ConsolidationReport`), promoting to
        ``user_id`` or ``agent_id`` when given; omit it to sweep every session
        whose episodes have all aged past ``min_age_days`` — the
        scheduled-maintenance form, returning the list of reports for the sessions
        consolidated (each promoted to its own recorded user, so ``agent_id`` does
        not apply to the sweep). Schedule it from your own job runner (a cron, a
        Temporal timer); Vincio stays a library and runs no background loop of its
        own::

            app.consolidate_memory(session_id="sess-42", user_id="u1")  # one now
            app.consolidate_memory(min_age_days=7.0)                     # nightly sweep
        """
        if self.memory is None:
            raise InputError(
                "consolidate_memory needs the memory engine; call add_memory() first"
            )
        if session_id is not None:
            return run_sync(
                self.memory.consolidate(
                    session_id, user_id=user_id, agent_id=agent_id, summarizer=summarizer
                )
            )
        return run_sync(
            self.memory.promote_aged_episodes(
                min_age_days=min_age_days, user_id=user_id, summarizer=summarizer
            )
        )

    def enable_memory_os(  # type: ignore[misc]
        self: ContextApp,
        *,
        scope: str = "agent",
        owner_id: str = "agent",
        max_core_tokens: int = 2000,
        permission: str = "memory:write",
        register_tools: bool = True,
    ):
        """Expose self-editing memory (MemGPT/Letta-class) as permissioned tools.

        Returns a :class:`~vincio.memory.agent_os.MemoryOS` over this app's
        audited memory engine and (by default) registers ``memory_append`` /
        ``memory_replace`` / ``memory_archive`` as write tools and
        ``memory_search`` as a read tool — so an agent can edit its own memory
        on the same RBAC + audit + budget path as any other tool, with a
        context-pressure pager between core and archival memory::

            os = app.enable_memory_os(owner_id="agent-1")
            agent = app.agent(tools=["memory_append", "memory_search"])
        """
        from ..memory.agent_os import MemoryOS

        if self.memory is None:
            self.add_memory()
        assert self.memory is not None  # noqa: S101 - add_memory() above guarantees it
        os = MemoryOS(self.memory, scope=scope, owner_id=owner_id, max_core_tokens=max_core_tokens)
        if register_tools:
            append, replace, search, archive = os.tools()
            self.add_tool(
                append, name="memory_append", permissions=[permission], side_effects="write"
            )
            self.add_tool(
                replace, name="memory_replace", permissions=[permission], side_effects="write"
            )
            self.add_tool(
                archive, name="memory_archive", permissions=[permission], side_effects="write"
            )
            self.add_tool(search, name="memory_search", side_effects="read")
        return os
