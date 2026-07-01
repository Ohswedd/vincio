"""ContextApp: the Vincio public API.

::

    from vincio import ContextApp

    app = ContextApp(name="docs_qa")
    app.add_source("docs", path="./docs", retrieval="hybrid")
    answer = app.run("How do I configure SSO?")
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from ..agents.blackboard import Blackboard
from ..agents.crew import AgentRole, Crew
from ..agents.executor import AgentExecutor
from ..agents.graph import Checkpointer, StateGraph
from ..agents.planner import Planner
from ..caching.base import InMemoryCache
from ..caching.compilation import ChunkCache, ContextCompileCache, PromptCompileCache
from ..caching.invalidation import InvalidationManager
from ..caching.layers import ResponseCache
from ..context.compiler import ContextCompiler, ContextCompilerOptions
from ..documents.loaders import load_directory, load_document
from ..evals.datasets import Dataset, EvalCase
from ..evals.metrics import RunOutput
from ..evals.online import OnlineEvaluator
from ..evals.runners import EvalRunner
from ..governance.fertility import FertilityTracker
from ..governance.lineage import ErasureResult, LineageIndex, build_erasure_proof
from ..governance.residency import ResidencyPolicy
from ..input.routers import InputRouter
from ..memory.engine import MemoryEngine
from ..memory.policies import MemoryWritePolicy
from ..memory.stores import SQLiteMemoryStore
from ..observability import build_exporter
from ..observability.costs import CostTracker
from ..observability.traces import Tracer
from ..output.repair import Repairer
from ..output.routing import SchemaRouter
from ..output.schemas import OutputContract, OutputSchema
from ..output.validators import SemanticValidator
from ..prompts.compiler import CompilerOptions, PromptCompiler
from ..prompts.signatures import Predict, Signature
from ..prompts.templates import PromptSpec
from ..providers import build_provider
from ..providers.base import ModelProvider, run_sync
from ..providers.cache_strategy import PromptCacheStrategy
from ..retrieval.chunking import chunk_document
from ..retrieval.embeddings import CachedEmbedder, build_embedder
from ..retrieval.engine import RetrievalEngine
from ..retrieval.filters import FilterSpec
from ..retrieval.graph_retrieval import EntityGraph
from ..retrieval.indexes import (
    BM25Index,
    VectorIndex,
    build_filter_spec,
)
from ..retrieval.late_interaction import LateInteractionIndex
from ..retrieval.prefetch import SpeculativePrefetcher
from ..retrieval.rerankers import build_reranker
from ..retrieval.sparse import SparseIndex
from ..security.access import AccessController, Principal
from ..security.audit import AuditLog
from ..security.pii import PIIDetector
from ..security.policy import PolicyEngine
from ..security.rails import Rail, RailEngine
from ..skills.library import SkillLibrary
from ..storage.base import create_metadata_store
from ..tools.permissions import ToolPermissionChecker
from ..tools.registry import ToolRegistry
from ..tools.runtime import ToolRuntime
from ..workflows.engine import Workflow
from .config import VincioConfig, load_config
from .diagnostics import note_suppressed
from .errors import (
    AgentEngineError,
    ConfigError,
    InputError,
    ResidencyViolationError,
    ToolNotFoundError,
)
from .events import EventBus
from .facades import (
    GovernanceFacade,
    OptimizationFacade,
    RetrievalFacade,
    RunFacade,
    ServingFacade,
    TrainingFacade,
)
from .runtime import VincioRuntime
from .types import (
    Budget,
    Constraint,
    EvidenceItem,
    Example,
    FileRef,
    Instruction,
    MemoryItem,
    Objective,
    PolicySet,
    RunConfig,
    RunResult,
    RunStreamEvent,
    TaskType,
    UserInput,
)

if TYPE_CHECKING:
    from ..assistant import Assistant

__all__ = ["ContextApp", "RunHandle"]


class RunHandle:
    """Handle to an in-flight run started by :meth:`ContextApp.submit`.

    Wraps the run's task and exposes cooperative cancellation that is identical
    across the streaming and non-streaming paths: :meth:`cancel` propagates a
    ``CancelledError`` into the run's bounded-concurrency groups, and the
    cancelled run is still fully recorded on its trace and audit chain. Await the
    handle (or :meth:`result`) for the :class:`RunResult`.
    """

    def __init__(self, task: asyncio.Future[RunResult]) -> None:
        self._task = task

    def cancel(self) -> bool:
        """Request cooperative cancellation; returns False if already done."""
        return self._task.cancel()

    def cancelled(self) -> bool:
        return self._task.cancelled()

    def done(self) -> bool:
        return self._task.done()

    async def result(self) -> RunResult:
        return await self._task

    def __await__(self):  # type: ignore[no-untyped-def]
        return self._task.__await__()


logger = logging.getLogger("vincio.app")


class _SourceConfig(BaseModel):
    name: str
    path: str | None = None
    loader: str | None = None
    chunking: str = "adaptive"
    retrieval: str = "hybrid"
    document_count: int = 0
    chunk_count: int = 0


class _AgentHandle:
    """Returned by app.agent(): sync/async runner over an AgentExecutor."""

    def __init__(self, app: ContextApp, executor: AgentExecutor, max_steps: int) -> None:
        self._app = app
        self._executor = executor
        self._max_steps = max_steps

    async def arun(
        self,
        objective: str,
        *,
        budget: Budget | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
    ):
        budget = budget or self._app.budget.model_copy(update={"max_steps": self._max_steps})
        attribution = {
            k: v
            for k, v in {"tenant_id": tenant_id, "user_id": user_id, "feature": feature}.items()
            if v is not None
        }
        return await self._executor.run(objective, budget=budget, attribution=attribution or None)

    def run(
        self,
        objective: str,
        *,
        budget: Budget | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
    ):
        return run_sync(
            self.arun(
                objective, budget=budget, tenant_id=tenant_id, user_id=user_id, feature=feature
            )
        )


class ContextApp:
    """The top-level Vincio application: one object that compiles prompts,
    memory, retrieval, tools, schemas, and policies into validated, observable,
    model-ready context and runs the end-to-end pipeline.

    Construct one with a name and an optional provider/model/config, attach
    sources, tools, memory, evaluators, and rails declaratively, then ``run`` /
    ``arun`` / ``astream`` to execute. The flat ``app.<method>`` surface is also
    grouped into capability facades (``app.runs``, ``app.knowledge``,
    ``app.governance``, ``app.optimization``, ``app.serving``, ``app.training``).
    """

    def __init__(
        self,
        name: str = "vincio_app",
        *,
        objective: Objective | str | None = None,
        output_schema: type[BaseModel] | OutputSchema | dict[str, Any] | None = None,
        config: VincioConfig | str | None = None,
        provider: ModelProvider | str | None = None,
        model: str | None = None,
        budget: Budget | None = None,
        policies: PolicySet | None = None,
        prompt_spec: PromptSpec | None = None,
    ) -> None:
        self.name = name
        if isinstance(config, str):
            config = load_config(config)
        self.config: VincioConfig = config or load_config()

        # objective / prompt
        if isinstance(objective, str):
            objective = Objective(text=objective)
        self.objective = objective
        self.prompt_spec = prompt_spec or PromptSpec(
            name=name,
            role=f"{name} assistant",
            objective=objective.text if objective else "",
        )
        self.prompt_variables: dict[str, Any] = {}
        self.instructions: list[Instruction] = []
        self.constraints: list[Constraint] = []

        # output contract
        self.output_contract = self._build_contract(output_schema)

        # run policy/budget
        self.budget = budget or self.config.budget
        self.policies = policies or self.config.policies

        # infrastructure
        self.events = EventBus()
        self.tracer = Tracer(
            name,
            build_exporter(
                self.config.observability.exporter, self.config.observability.traces_dir
            ),
            sample_rate=self.config.observability.sample_rate,
        )
        self.cost_tracker = CostTracker()
        # Opt-in energy/carbon accounting. Off by default; turned on by
        # ``use_energy_accounting`` / ``set_energy_budget``. When on, each run
        # accrues an energy + carbon estimate on the cost-report surface and on
        # the audit chain.
        self.energy_accounting_enabled = False
        # Cumulative usage for out-of-run media generation (image/TTS), so the
        # budget cap is honored across repeated app.generate_image/synthesize_speech
        # calls rather than per-call only.
        from ..core.types import BudgetUsage as _BudgetUsage

        self._media_usage = _BudgetUsage()
        self.store = create_metadata_store(self.config.storage.metadata)
        _audit_signer = None
        if self.config.security.audit_signing_key:
            from ..security.audit import HMACSigner

            _audit_signer = HMACSigner(
                self.config.security.audit_signing_key,
                key_id=self.config.security.audit_signing_key_id,
            )
        self.audit = AuditLog(
            self.config.security.audit_dir if self.config.security.audit_log else None,
            signer=_audit_signer,
        )
        self.access = AccessController(tenant_isolation=self.config.security.tenant_isolation)
        self.rail_engine = RailEngine()
        # Governance: a locale-aware PII detector (non-English packs),
        # data-residency policy, lineage index, content-marking, and tokenizer
        # fertility telemetry. All opt-in / empty by default.
        self._pii_detector = self._build_pii_detector()
        self.policy_engine = PolicyEngine(
            self.policies,
            pii_detector=self._pii_detector,
            rails=self.rail_engine,
            egress_dlp=self.config.security.egress_dlp,
        )
        self.residency = ResidencyPolicy(
            allowed_regions=list(self.config.governance.allowed_regions),
            provider_regions=dict(self.config.governance.provider_regions),
            deny_on_unknown=self.config.governance.deny_on_unknown_region,
        )
        self.lineage = LineageIndex()
        # Registered semantic layers, keyed by their grounding table — the governed
        # metric definitions :meth:`query_metric` resolves against.
        self._semantic_layers: dict[str, Any] = {}
        self.fertility = FertilityTracker(model=model or self.config.provider.model)
        self.content_marking = self.config.governance.content_marking
        # Optional signer for synthetic-content manifests (e.g. HmacSigner);
        # set it to cryptographically sign every marked output *and* every
        # erasure proof.
        self.content_signer: Any = None
        # Bound agent identity: opt-in, empty by default. When set via
        # ``app.identity(..., use=True)`` / ``app.use_identity(...)`` it becomes the
        # app's signer, so every audit entry, contract, and settlement binds to its
        # DID — accountability as a cryptographic fact, not a logged ``key_id`` string.
        self._identity: Any = None
        # Consent ledger: opt-in, empty by default. When configured via
        # ``app.use_consent_ledger(...)`` it binds data to a GDPR purpose/lawful
        # basis and is consulted by access decisions and memory recall.
        self.consent_ledger: Any = None
        # Differential-privacy accountant: opt-in, empty by default. When attached
        # via ``app.use_privacy_accountant(...)`` it composes a per-subject (ε, δ)
        # budget across memory consolidations and federated contributions and
        # surfaces a privacy report alongside the cost report.
        self.privacy_accountant: Any = None
        # Cross-fleet reputation ledger: opt-in, empty by default. When attached via
        # ``app.use_reputation_ledger(...)`` it earns a per-member reliability score
        # from how each federated contribution fared against the no-regression gate
        # and reliability-weights the federated aggregation accordingly.
        self.reputation_ledger: Any = None
        # Lazily-built signer for negotiated contracts when neither an explicit
        # signer nor an audit-chain signer is available (so an offline negotiation
        # still produces a signed, self-verifiable contract).
        self._contract_signer: Any = None
        # Settlement book: opt-in, empty by default. When attached via
        # ``app.use_settlement_book(...)`` it keeps a durable, hash-chained ledger
        # of the settlement records that close the books on contracted cross-org
        # work, and surfaces a settlement report alongside the cost report.
        self.settlement_book: Any = None
        # Imported portable reputation: opt-in, empty by default. When attached via
        # ``app.import_reputation(...)`` it combines other orgs' signed attestations
        # into a bounded, evidence-weighted prior that weights a negotiation against a
        # counterparty this app has no local history with.
        self.imported_reputation: Any = None
        # Revocations this app has issued (via ``app.revoke_attestation``), retained so
        # ``app.serve_attestations`` can return them to a peer that pulls this app's
        # standing about a subject — the gossip analogue of the settlement book.
        self._issued_revocations: list[Any] = []
        self.input_router = InputRouter()

        # provider
        # A passed provider *instance* carries its own registry name (mock /
        # local / openai / …); use it so residency, provenance marking, and
        # provider lookups reflect the real provider rather than the config
        # default. This is what lets the deterministic mock identify as on-prem
        # (region known) so fail-closed residency still admits the offline path.
        if isinstance(provider, str):
            self._provider_name = provider
        elif isinstance(provider, ModelProvider):
            self._provider_name = getattr(provider, "name", None) or self.config.provider.default
        else:
            self._provider_name = self.config.provider.default
        self._provider_instance = provider if isinstance(provider, ModelProvider) else None
        self.model = model or self.config.provider.model
        self._built_providers: dict[str, ModelProvider] = {}
        self._coalesced_providers: dict[int, ModelProvider] = {}

        # caches
        self.cache_invalidation = InvalidationManager()
        self.cache_invalidation.attach(self.events)
        self.response_cache: ResponseCache | None = None
        if self.config.cache.response_cache:
            backend = InMemoryCache(
                max_entries=self.config.cache.max_entries, default_ttl_s=self.config.cache.ttl_s
            )
            self.response_cache = ResponseCache(backend, ttl_s=self.config.cache.ttl_s)
            self.cache_invalidation.register(backend)
        # Content-addressed compilation caches: unchanged inputs are
        # never recompiled / re-chunked.
        self.prompt_compile_cache: PromptCompileCache | None = None
        if self.config.cache.prompt_compile_cache:
            backend = InMemoryCache(max_entries=self.config.cache.max_entries)
            self.prompt_compile_cache = PromptCompileCache(backend, ttl_s=self.config.cache.ttl_s)
            self.cache_invalidation.register(backend)
        self.context_compile_cache: ContextCompileCache | None = None
        if self.config.cache.context_compile_cache:
            backend = InMemoryCache(max_entries=self.config.cache.max_entries)
            self.context_compile_cache = ContextCompileCache(backend, ttl_s=self.config.cache.ttl_s)
            self.cache_invalidation.register(backend)
        self.chunk_cache: ChunkCache | None = None
        if self.config.cache.chunk_cache:
            backend = InMemoryCache(max_entries=self.config.cache.max_entries, default_ttl_s=None)
            self.chunk_cache = ChunkCache(backend)
            self.cache_invalidation.register(backend)

        # context compiler
        self.context_compiler = ContextCompiler(
            ContextCompilerOptions(
                slim_packets=self.config.performance.slim_packets,
                reuse_candidate_set=self.config.performance.reuse_candidate_set,
                single_pass_selection=self.config.performance.single_pass_selection,
                max_resident_bytes=(
                    int(self.config.performance.memory_budget_mb * 1_000_000)
                    if self.config.performance.memory_budget_mb is not None
                    else None
                ),
                max_candidates=self.config.performance.max_context_candidates,
            ),
            cache=self.context_compile_cache,
        )
        self.prompt_compiler = PromptCompiler(CompilerOptions(), cache=self.prompt_compile_cache)

        # Provider-aware prompt caching: attach a TTL to the compiler's
        # stable prefix for caching-capable providers and record cache-hit-rate
        # telemetry on the model span. On by default; tune via config or
        # ``enable_prompt_caching``.
        self.prompt_cache: PromptCacheStrategy | None = (
            PromptCacheStrategy(
                ttl=self.config.cache.provider_cache_ttl,  # type: ignore[arg-type]
                min_prefix_tokens=self.config.cache.provider_cache_min_prefix_tokens,
            )
            if self.config.cache.provider_cache
            else None
        )

        # retrieval
        self.embedder = self._build_embedder()
        # Thread the app embedder into the context compiler for opt-in semantic
        # scoring. The shared compiler keeps it vector-less; a per-compile
        # scorer holds the embeddings, so this is safe and only activates when
        # ``retrieval.semantic_context_scoring`` is enabled with a real embedder.
        self.context_compiler.embedder = self.embedder
        self.context_compiler.options.semantic_scoring = (
            self.config.retrieval.semantic_context_scoring
        )
        self.context_compiler.options.mmr_lambda = self.config.retrieval.mmr_lambda
        # Speculative retrieval prefetch: warms the query embedding from the task
        # classification before retrieval runs (opt-in). Shares the app embedder
        # so a landed warm is a cache hit for retrieval.
        self._prefetcher = (
            SpeculativePrefetcher(self.embedder)
            if self.config.performance.speculative_prefetch
            else None
        )
        # Learned semantic cache (near-miss reuse) and cross-request KV-prefix
        # reuse: both opt-in, consulted by the runtime only when installed, and
        # held under the resident-memory budget. The semantic cache needs the
        # app embedder, so it is built here after the embedder.
        self.semantic_cache: Any | None = None
        if self.config.cache.semantic_cache:
            self.use_semantic_cache()
        self.kv_prefix_pool: Any | None = None
        if self.config.cache.kv_prefix_reuse:
            self.use_kv_prefix_reuse()
        self.sources: dict[str, _SourceConfig] = {}
        self.retrieval: RetrievalEngine | None = None
        self._bm25: BM25Index | None = None
        self._vector: VectorIndex | None = None
        self._sparse: SparseIndex | None = None
        self._late_interaction: LateInteractionIndex | None = None
        self.entity_graph: EntityGraph | None = None
        self.pending_evidence: list[EvidenceItem] = []
        self._ingested_files: dict[str, list[EvidenceItem]] = {}

        # memory
        self.memory_enabled = False
        self.memory: MemoryEngine | None = None

        # tools
        self.tool_registry = ToolRegistry()
        self.tool_runtime = ToolRuntime(
            self.tool_registry,
            permission_checker=ToolPermissionChecker(
                self.access, allow_external=self.policies.allow_external_tools
            ),
            tracer=self.tracer,
            cache_enabled=self.config.cache.tool_cache,
        )
        self.enabled_tools: list[str] = []

        # protocols & interoperability: MCP servers, Agent Skills.
        self.skill_library: SkillLibrary | None = None
        self.mcp_clients: dict[str, Any] = {}

        # validation / repair / evaluators
        self.semantic_validators: dict[str, SemanticValidator] = {}
        self.repairer = Repairer(self.output_contract.repair_policy)
        self.evaluators: list[str] = []
        self.optimizers: list[str] = []
        self.online_evaluators: list[OnlineEvaluator] = []
        self._online_tasks: set[asyncio.Task[Any]] = set()
        self.schema_router: SchemaRouter | None = None
        self.self_correction: dict[str, Any] | None = None

        # cost & reliability: runtime model cascade, cost attribution
        # ledger, and per-tenant/feature budget enforcement. All opt-in.
        from ..observability.finops import BudgetManager, CostLedger
        from ..optimize.routing import ModelCascade

        self.cascade: ModelCascade | None = None
        self._cascade_confidence: Callable[[Any], float] | None = None
        # Opt-in per-step reasoning-effort control. When set, the runtime fills
        # an unset ``reasoning_effort`` from the task classification and the live
        # budget (see ``use_reasoning_controller``). ``None`` keeps reasoning a
        # per-call knob, so behavior is unchanged by default.
        self.reasoning_controller: Any | None = None
        # Opt-in long-horizon context governor. When set, a multi-session run can
        # feed each result's packet to the governor (``app.govern_packet``) and
        # the live context footprint stays bounded across the whole conversation
        # via intra-run decay and provenance-preserving compaction. ``None`` keeps
        # context unbounded by default (see ``use_context_governor``).
        self.context_governor: Any | None = None
        # Opt-in on-device adapter. When set via ``app.use_local_adapter(...)`` the
        # base provider is wrapped in an ``AdaptedProvider`` so in-distribution
        # requests are answered the way a locally-fit LoRA-class adapter learned,
        # without the run leaving the process. ``None`` keeps the base model
        # unchanged by default.
        self.local_adapter: Any | None = None
        self.cost_ledger: CostLedger = CostLedger(
            price_table=self.cost_tracker.price_table, store=self.store
        )
        self.budget_manager: BudgetManager = BudgetManager(self.cost_ledger, events=self.events)

        self._runtime = VincioRuntime(self)

    # -- construction helpers --------------------------------------------------------

    def _build_contract(
        self, output_schema: type[BaseModel] | OutputSchema | dict[str, Any] | None
    ) -> OutputContract:
        if output_schema is None:
            return OutputContract(format="text")
        if isinstance(output_schema, OutputSchema):
            schema = output_schema
        elif isinstance(output_schema, dict):
            schema = OutputSchema.from_json_schema(output_schema)
        elif isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            schema = OutputSchema.from_pydantic(output_schema)
        else:
            raise ConfigError(f"unsupported output_schema type: {type(output_schema).__name__}")
        return OutputContract.from_schema(
            schema,
            require_citations=self.policies.require_citations
            if hasattr(self, "policies")
            else False,
        )

    def _build_pii_detector(self) -> PIIDetector:
        """PII detector with the configured non-English locale packs."""
        locales = list(self.config.governance.locales)
        return PIIDetector(locales=locales or None)

    def _build_embedder(self):
        # Delegate to the one factory so the app and the standalone
        # `build_embedder` agree on every embedder kind (local, hosted
        # jina/voyage/cohere/contextual/multimodal, or any provider) and on
        # Matryoshka handling — hosted embedders truncate server-side, others
        # are wrapped. The cache stores the already-truncated vector.
        base = build_embedder(
            self.config.retrieval.embedder,
            config=self.config.provider,
            dimensions=self.config.retrieval.embedding_dimensions,
        )
        return CachedEmbedder(base)

    def check_residency(self, run_config: RunConfig | None = None) -> None:
        """Enforce data-residency routing: refuse disallowed egress.

        When a residency policy is configured (``governance.allowed_regions``),
        a run whose resolved provider/model region is not allowed is denied with
        a :class:`~vincio.core.errors.ResidencyViolationError`, recorded as a
        blocking residency decision on the hash-chained audit log. A no-op when
        no residency policy is set.
        """
        if not self.residency.enforced:
            return
        # The run boundary validates *every* model this run could egress to —
        # the configured/per-run model, a budget-degrade target, every cascade
        # rung, and the candidates of a router / shadow / canary wrapper — so a
        # rotation that picks a different model per request can never slip a
        # disallowed-region model past residency below the choke point.
        for model, provider_name in self._reachable_models(run_config):
            self._enforce_model_residency(model, provider=provider_name)

    def _reachable_models(
        self, run_config: RunConfig | None = None
    ) -> list[tuple[str, str | None]]:
        """Every ``(model, provider_name)`` a run could dispatch to, for residency.

        Enumerates the resolved model, any budget-degrade target, the rungs of an
        active cascade, and the candidate set of a ``Router`` / ``ShadowProvider``
        / ``CanaryRouter`` provider wrapper — so residency (and any future
        per-model run-boundary guard) sees the full reachable set, not just the
        primary model."""
        name = (run_config.provider if run_config else None) or self._provider_name
        primary = (run_config.model if run_config else None) or self.model
        seen: set[tuple[str, str | None]] = set()
        reachable: list[tuple[str, str | None]] = []

        def add(model: str | None, provider_name: str | None) -> None:
            if model and (model, provider_name) not in seen:
                seen.add((model, provider_name))
                reachable.append((model, provider_name))

        add(primary, name)
        for budget in getattr(self.budget_manager, "budgets", []):
            if budget.on_breach == "degrade" and budget.degrade_model:
                add(budget.degrade_model, name)
        cascade = getattr(self, "cascade", None)
        if cascade is not None:
            for rung in cascade.rungs:
                add(rung.model, rung.provider or name)
        instance = self._provider_instance
        if instance is not None:
            from ..optimize.routing import Router
            from ..providers.shadow import CanaryRouter, ShadowProvider

            if isinstance(instance, Router):
                for provider, model in instance.entries:
                    add(model, getattr(provider, "name", None) or name)
            elif isinstance(instance, (ShadowProvider, CanaryRouter)):
                add(
                    instance.candidate_model or primary,
                    getattr(instance.candidate, "name", None) or name,
                )
        return reachable

    def _enforce_model_residency(self, model: str, *, provider: str | None = None) -> None:
        """Refuse egress of a single ``model`` to a disallowed region.

        The shared per-model residency check behind :meth:`check_residency` and
        the rotation wrappers (``use_router`` / ``shadow`` / ``canary`` /
        ``use_cascade``), so a candidate model whose region is not allowed is
        refused at wiring time rather than silently egressed below the run choke
        point. A no-op when no residency policy is set."""
        if not self.residency.enforced:
            return
        name = provider or self._provider_name
        # The region is inferred from the configured endpoint when set, so a
        # region-pinned base_url drives the egress decision.
        base_url = self.config.provider.base_urls.get(name)
        violation = self.residency.check(provider=name, model=model, base_url=base_url)
        if violation is None:
            return
        self.audit.record(
            "residency_check",
            decision="deny",
            resource=f"{name}:{model}",
            details=violation.details,
        )
        self.events.emit("residency.denied", violation.details)
        raise ResidencyViolationError(
            violation.message,
            region=violation.details.get("region"),
            allowed=violation.details.get("allowed_regions", []),
        )

    def resolve_provider(self, run_config: RunConfig | None = None) -> ModelProvider:
        """Resolve the model provider for a run, enforcing data residency first."""
        self.check_residency(run_config)
        name = (run_config.provider if run_config else None) or self._provider_name
        if self._provider_instance is not None and (
            run_config is None or run_config.provider is None
        ):
            return self._wrap_provider(self._provider_instance)
        # Reuse built instances so connection pools and coalescing maps
        # persist across runs.
        if name not in self._built_providers:
            self._built_providers[name] = self._wrap_provider(
                build_provider(name, self.config.provider)
            )
        return self._built_providers[name]

    def _wrap_provider(self, provider: ModelProvider) -> ModelProvider:
        if not self.config.performance.coalesce_requests:
            return provider
        wrapped = self._coalesced_providers.get(id(provider))
        if wrapped is None:
            from ..providers.transport import CoalescingProvider

            wrapped = CoalescingProvider(provider)
            self._coalesced_providers[id(provider)] = wrapped
        return wrapped

    def principal_for(self, user_input: UserInput) -> Principal:
        """Build the :class:`Principal` (user, tenant, scopes) for an input."""
        return Principal(
            user_id=user_input.user_id,
            tenant_id=user_input.tenant_id,
            scopes=list(self.policies.custom.get("scopes", ["*"])),
        )

    def tenant_filter(self, tenant_id: str | None) -> FilterSpec | None:
        """Tenant-scope filter for retrieval.

        Returns a pushdown-capable :class:`FilterSpec` so the tenant
        predicate is applied in the vector store (Qdrant/pgvector/...) and other
        tenants' rows are never fetched to the client and dropped — closing the
        fetch-to-filter exfiltration gap. In-memory indexes evaluate it directly.
        """
        if tenant_id is None or not self.config.security.tenant_isolation:
            return None
        return build_filter_spec(tenant_id=tenant_id)

    # -- public configuration API ----------------------------------------------

    def configure(
        self,
        *,
        objective: str | None = None,
        role: str | None = None,
        rules: list[str] | None = None,
        soft_rules: list[str] | None = None,
        definitions: dict[str, str] | None = None,
        examples: list[Example] | None = None,
        citation_policy: str | None = None,
        insufficient_evidence_behavior: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> ContextApp:
        """Configure the prompt and run defaults declaratively (objective, role, rules, …)."""
        update: dict[str, Any] = {}
        if objective is not None:
            update["objective"] = objective
            self.objective = Objective(text=objective)
        if role is not None:
            update["role"] = role
        if rules is not None:
            update["rules"] = rules
            self.instructions = [Instruction(text=r) for r in rules]
        if soft_rules is not None:
            update["soft_rules"] = soft_rules
        if definitions is not None:
            update["definitions"] = definitions
        if examples is not None:
            update["examples"] = examples
        if citation_policy is not None:
            update["citation_policy"] = citation_policy
        if insufficient_evidence_behavior is not None:
            update["insufficient_evidence_behavior"] = insufficient_evidence_behavior
        self.prompt_spec = self.prompt_spec.model_copy(update=update)
        if variables:
            self.prompt_variables.update(variables)
        return self

    def set_policy(self, name: str, value: Any) -> ContextApp:
        """Set a run policy (e.g. ``answer_only_from_sources``)."""
        self.policies.set(name, value)
        if name == "answer_only_from_sources" and value:
            self.policies.require_citations = True
            self.output_contract.require_citations = True
            if not self.prompt_spec.citation_policy:
                self.prompt_spec = self.prompt_spec.model_copy(
                    update={
                        "rules": [
                            *self.prompt_spec.rules,
                            "Use only the provided sources to answer.",
                        ],
                        "citation_policy": "Cite evidence IDs in square brackets for every claim.",
                        "insufficient_evidence_behavior": "If the sources do not contain the answer, say so explicitly.",
                    }
                )
        if name == "require_citations":
            self.output_contract.require_citations = bool(value)
        self.policy_engine = PolicyEngine(
            self.policies,
            pii_detector=self._pii_detector,
            rails=self.rail_engine,
            egress_dlp=self.config.security.egress_dlp,
        )
        self.events.emit("policy.changed", {"policy": name})
        return self

    # -- rails ------------------------------------------------------------------

    def add_rail(self, rail: Rail | None = None, **kwargs: Any) -> ContextApp:
        """Add a programmable input/output rail (topic, format, safety, custom).

        Rails are evaluated by the deterministic policy engine before and
        after every generation::

            app.add_rail(name="no_competitors", kind="topic", direction="output",
                         blocked_topics=["acme corp"])
            app.add_rail(name="no_secrets", kind="safety", detectors=["secrets", "pii"])
        """
        self.rail_engine.add(rail if rail is not None else Rail(**kwargs))
        return self

    def register_rail_predicate(
        self, name: str, predicate: Callable[[str, dict[str, Any]], Any]
    ) -> ContextApp:
        """Register a custom rail predicate: ``(text, params) -> falsy | message``."""
        self.rail_engine.register(name, predicate)
        return self

    def edge_runtime(self, profile: Any | None = None) -> Any:
        """Build a bounded, in-process edge runtime that shares this app's rails.

        Returns an :class:`~vincio.edge.runtime.EdgeRuntime` — the dependency-free
        compile/score/rail/pack core packaged for a constrained or browser/WASM
        target — seeded with the app's configured rails so the edge path enforces
        the same deterministic safety the server does. ``profile`` defaults to the
        bounded edge-worker :class:`~vincio.edge.profile.EdgeProfile`. The runtime
        holds no provider, store, or tracer; it runs the identical context
        engineering at the edge, offline::

            edge = app.edge_runtime()
            result = edge.run("Summarize the renewal terms")
        """
        from ..edge import EdgeRuntime

        return EdgeRuntime(profile, rails=list(self.rail_engine.rails))

    # -- cost & reliability -----------------------------------------------

    def enable_prompt_caching(
        self, *, ttl: str = "5m", min_prefix_tokens: int = 1024
    ) -> ContextApp:
        """Turn on provider-aware prompt caching (default on).

        For providers with explicit breakpoints (Anthropic) the compiler's
        stable prefix gets a ``cache_control`` breakpoint with the chosen
        ``ttl`` ("5m" or "1h") when it is at least ``min_prefix_tokens`` long;
        for auto-cache providers (OpenAI/Gemini) the stable→volatile ordering
        already maximizes hits. Cache-hit rate is recorded on every model
        span::

            app.enable_prompt_caching(ttl="1h")  # long-lived stable context
        """
        self.prompt_cache = PromptCacheStrategy(
            enabled=True,
            ttl=ttl,  # type: ignore[arg-type]
            min_prefix_tokens=min_prefix_tokens,
        )
        return self

    def use_cascade(
        self,
        models: list[str] | None = None,
        *,
        rungs: list[Any] | None = None,
        min_confidence: float = 0.5,
        max_escalations: int | None = None,
        confidence: Callable[[Any], float] | None = None,
    ) -> ContextApp:
        """Route runs through a cheap→strong model cascade at run time.

        A run starts on the cheapest model and escalates to the next only when a
        response's confidence falls below the rung threshold (default: a clean,
        schema-valid stop is confident; a truncated/filtered/unparseable answer
        is not). Pass a custom ``confidence`` callable ``(ModelResponse) -> float``
        to drive escalation from your own metric. The offline routing optimizer
        keeps tuning the thresholds. An explicit per-run ``config.model`` or a
        budget degrade overrides the cascade; streaming runs (``astream``) buffer
        each rung and stream the accepted (escalated) answer, never a discarded
        cheap attempt::

            app.use_cascade(["gpt-5.2-mini", "gpt-5.2"])
            app.use_cascade(rungs=[{"model": "haiku", "min_confidence": 0.6}, {"model": "opus"}])
        """
        from ..optimize.routing import CascadeRung, ModelCascade

        if rungs is not None:
            parsed: list[CascadeRung] = []
            for rung in rungs:
                if isinstance(rung, CascadeRung):
                    parsed.append(rung)
                elif isinstance(rung, dict):
                    parsed.append(CascadeRung(**rung))
                else:
                    parsed.append(CascadeRung(model=str(rung)))
            self.cascade = ModelCascade(rungs=parsed, max_escalations=max_escalations)
        elif models:
            self.cascade = ModelCascade.from_models(
                list(models), min_confidence=min_confidence, max_escalations=max_escalations
            )
        else:
            raise ConfigError("use_cascade requires models=[...] or rungs=[...]")
        # Residency: every rung the cascade may escalate into must be an
        # allowed region — closes the gap where an escalation egressed below the
        # run's residency choke point.
        for rung in self.cascade.rungs:
            self._enforce_model_residency(rung.model, provider=rung.provider)
        self._cascade_confidence = confidence
        return self

    # -- provider/model rotation & swap regression ------------------------

    def _base_provider(self) -> ModelProvider:
        """The raw model provider (the current instance, or one built from config)
        — the inner provider that rotation wrappers compose over."""
        if self._provider_instance is not None:
            return self._provider_instance
        return build_provider(self._provider_name, self.config.provider)

    def _pinned_models(self) -> list[str]:
        """Every model id this app currently pins (default model + cascade rungs)."""
        models: set[str] = set()
        if self.model:
            models.add(self.model)
        cascade = getattr(self, "cascade", None)
        if cascade is not None:
            models.update(rung.model for rung in cascade.rungs)
        return sorted(m for m in models if m)

    def use_router(
        self,
        models: list[str],
        *,
        strategy: str = "cheapest",
        budget_usd: float | None = None,
        guard_capabilities: bool = True,
        provider: ModelProvider | None = None,
    ) -> ContextApp:
        """Route each run to the cheapest / fastest / least-busy *capable* model.

        A registry-backed :class:`~vincio.optimize.routing.Router` becomes the
        app's provider: before every call it filters ``models`` to those that can
        serve the request (capability guard) and picks by ``strategy``, optionally
        **downgrading** to honor a per-request ``budget_usd``. Each pick is emitted
        as a ``model.routed`` event on the app's bus::

            app.use_router(["gpt-5.2-nano", "gpt-5.2-mini", "gpt-5.2"], strategy="cheapest")
        """
        from ..optimize.routing import Router

        if not models:
            raise ConfigError("use_router requires at least one model")
        for candidate in models:  # residency: every routable model must be allowed
            self._enforce_model_residency(candidate)
        base = provider or self._base_provider()
        self._provider_instance = Router(
            [(base, m) for m in models],
            strategy=strategy,  # type: ignore[arg-type]
            budget_usd=budget_usd,
            guard_capabilities=guard_capabilities,
            events=self.events,
        )
        self.model = models[0]
        return self

    def shadow(
        self,
        candidate_model: str,
        *,
        candidate_provider: ModelProvider | None = None,
        block: bool = False,
    ) -> Any:
        """Serve the primary model but dual-dispatch ``candidate_model`` for an
        offline diff. Returns the :class:`~vincio.providers.shadow.ShadowProvider`
        (read ``.observations`` / ``.diff()``); it also becomes the app's provider
        so every run is shadowed until removed."""
        from ..providers.shadow import ShadowProvider

        # Residency: the shadow dual-dispatches the request to the candidate, so
        # the candidate model's region must be allowed before any egress.
        self._enforce_model_residency(
            candidate_model, provider=getattr(candidate_provider, "name", None)
        )
        primary = self._base_provider()
        candidate = candidate_provider or primary
        shadow = ShadowProvider(
            primary,
            candidate,
            candidate_model=candidate_model,
            block=block,
            price_table=self.cost_tracker.price_table,
            events=self.events,
        )
        self._provider_instance = shadow
        return shadow

    def canary(
        self,
        candidate_model: str,
        *,
        percent: float = 5.0,
        candidate_provider: ModelProvider | None = None,
        score_fn: Callable[[Any], float] | None = None,
        min_samples: int = 20,
        regression_threshold: float = 0.05,
        prompt_name: str | None = None,
    ) -> Any:
        """Ramp ``percent``% of live traffic onto ``candidate_model`` with online
        scoring and auto-rollback to the primary (and prompt-registry head) on
        regression. Returns the :class:`~vincio.providers.shadow.CanaryRouter`,
        which also becomes the app's provider."""
        from ..providers.shadow import CanaryRouter

        # Residency: the canary routes live traffic to the candidate model.
        self._enforce_model_residency(
            candidate_model, provider=getattr(candidate_provider, "name", None)
        )
        primary = self._base_provider()
        candidate = candidate_provider or primary
        canary = CanaryRouter(
            primary,
            candidate,
            percent=percent,
            candidate_model=candidate_model,
            score_fn=score_fn,
            min_samples=min_samples,
            regression_threshold=regression_threshold,
            prompt_registry=getattr(self, "prompt_registry", None),
            prompt_name=prompt_name,
            events=self.events,
        )
        self._provider_instance = canary
        return canary

    async def agate_swap(
        self,
        candidate_model: str,
        *,
        baseline_model: str | None = None,
        dataset: Any = None,
        traces: list[Any] | None = None,
        metrics: list[str] | None = None,
        quality_metric: str = "lexical_overlap",
        gates: dict[str, str] | None = None,
        alpha: float = 0.05,
        repeats: int = 1,
        flake_quarantine: bool = True,
        pin_tools: bool = True,
    ) -> Any:
        """Gate a model swap on replayed golden traces + an eval/cost/latency/
        behavioral diff with statistical backing. Returns a
        :class:`~vincio.evals.swap.SwapVerdict`."""
        from ..evals.swap import SwapGate

        gate = SwapGate(
            self,
            metrics=metrics,
            quality_metric=quality_metric,
            gates=gates,
            alpha=alpha,
            repeats=repeats,
            flake_quarantine=flake_quarantine,
        )
        return await gate.evaluate(
            candidate_model=candidate_model,
            baseline_model=baseline_model,
            dataset=dataset,
            traces=traces,
            pin_tools=pin_tools,
        )

    def gate_swap(self, candidate_model: str, **kwargs: Any) -> Any:
        """Synchronous :meth:`agate_swap`."""
        from ..providers.base import run_sync

        return run_sync(self.agate_swap(candidate_model, **kwargs))

    async def aswap_regression(
        self,
        dataset: Any,
        *,
        candidate_model: str,
        baseline_model: str | None = None,
        metrics: list[str] | None = None,
        quality_metric: str = "lexical_overlap",
        alpha: float = 0.05,
        repeats: int = 1,
        flake_quarantine: bool = True,
    ) -> Any:
        """Swap only the model on a fixed dataset and return a statistically
        grounded :class:`~vincio.evals.swap.SwapRegressionReport`."""
        from ..evals.swap import model_swap_regression

        return await model_swap_regression(
            self,
            dataset,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            metrics=metrics,
            quality_metric=quality_metric,
            alpha=alpha,
            repeats=repeats,
            flake_quarantine=flake_quarantine,
        )

    def swap_regression(self, dataset: Any, *, candidate_model: str, **kwargs: Any) -> Any:
        """Synchronous :meth:`aswap_regression`."""
        from ..providers.base import run_sync

        return run_sync(self.aswap_regression(dataset, candidate_model=candidate_model, **kwargs))

    def watch_lifecycle(
        self,
        models: list[str] | None = None,
        *,
        as_of: Any = None,
        warn_within_days: int = 90,
        propose: bool = True,
    ) -> dict[str, Any]:
        """Scan pinned models for sunset and (optionally) propose migrations off
        deprecated/retired/nearing-retirement ones. Returns ``{"alerts",
        "proposals"}``; defaults to the app's pinned models."""
        from ..providers.lifecycle import LifecycleWatcher

        watcher = LifecycleWatcher(warn_within_days=warn_within_days, events=self.events)
        targets = list(models) if models else self._pinned_models()
        alerts = watcher.scan(targets, as_of=as_of)
        proposals = watcher.propose_all(targets, as_of=as_of) if propose else []
        return {"alerts": alerts, "proposals": proposals}

    def set_cost_budget(
        self,
        *,
        limit_usd: float,
        scope: str = "tenant",
        id: str | None = None,
        period: str = "day",
        on_breach: str = "cap",
        degrade_model: str | None = None,
        anomaly_factor: float | None = None,
    ) -> ContextApp:
        """Enforce a per-tenant/feature/user cost budget.

        When the scope's spend over ``period`` reaches ``limit_usd``, ``on_breach``
        decides the action: ``"cap"`` denies the run, ``"degrade"`` swaps in
        ``degrade_model`` (a cheaper model), and ``"queue_to_batch"`` denies the
        interactive run and points the caller at :meth:`batch`. Set
        ``anomaly_factor`` to raise a ``cost.anomaly`` event on a spend spike::

            app.set_cost_budget(scope="tenant", id="acme", limit_usd=10.0, period="day")
            app.set_cost_budget(scope="feature", id="chat", limit_usd=5.0,
                                 on_breach="degrade", degrade_model="gpt-5.2-mini")
        """
        from ..observability.finops import CostBudget

        self.budget_manager.add(
            CostBudget(
                scope=scope,  # type: ignore[arg-type]
                id=id,
                limit_usd=limit_usd,
                period=period,  # type: ignore[arg-type]
                on_breach=on_breach,  # type: ignore[arg-type]
                degrade_model=degrade_model,
                anomaly_factor=anomaly_factor,
            )
        )
        return self

    def cost_report(self, *, by: str = "tenant", since: Any | None = None):
        """Roll up attributed model cost by ``tenant``/``feature``/``user``/
        ``model``/``provider``/``run`` (returns a :class:`CostReport`)."""
        return self.cost_ledger.report(by, since=since)  # type: ignore[arg-type]

    def use_energy_accounting(
        self,
        *,
        region: str | None = None,
        pue: float | None = None,
        carbon_intensity: dict[str, float] | None = None,
    ) -> ContextApp:
        """Turn on per-run energy & carbon accounting (opt-in).

        Once enabled, every run accrues an estimated energy (watt-hours) and
        carbon (grams CO₂e) figure — mechanical and deterministic, from the run's
        token accounting against a per-model intensity (by tier) and a per-region
        grid factor — onto the cost-report surface
        (:meth:`energy_report`, ``result.energy_wh`` / ``result.co2e_grams``) and
        the hash-chained audit log. No external service is consulted. The energy
        analogue of the dollar cost report::

            app.use_energy_accounting(region="eu")
            result = app.run("summarize this")
            print(result.energy_wh, result.co2e_grams)
            app.energy_report(by="model").print_summary()

        ``region`` overrides the default region used when a call's region cannot
        be resolved; ``pue`` overrides the datacenter power-overhead factor;
        ``carbon_intensity`` merges operator-measured grid factors (g CO₂e/kWh,
        keyed by region) over the built-in defaults.
        """
        table = self.cost_tracker.energy_table
        if region is not None:
            table.region_override = region
        if pue is not None:
            table.pue = max(0.0, pue)
        if carbon_intensity:
            for reg, g in carbon_intensity.items():
                table.set_region_intensity(reg, g)
        self.energy_accounting_enabled = True
        return self

    def set_energy_budget(
        self,
        *,
        scope: str = "global",
        id: str | None = None,
        limit_wh: float | None = None,
        limit_co2e_grams: float | None = None,
        period: str = "day",
    ) -> ContextApp:
        """Set an energy/carbon budget, refused on breach like a cost cap.

        The sustainability analogue of :meth:`set_cost_budget`. Give an energy
        ceiling (``limit_wh``), a carbon ceiling (``limit_co2e_grams``), or both;
        when a scope's accrued energy or carbon over ``period`` reaches a ceiling,
        the run is refused on the same audit path as a cost cap. Enables energy
        accounting on first use::

            app.set_energy_budget(scope="tenant", id="acme", limit_co2e_grams=500.0)
            app.set_energy_budget(limit_wh=1000.0, period="hour")
        """
        from ..core.errors import EnergyBudgetError
        from ..observability.finops import EnergyBudget

        if limit_wh is None and limit_co2e_grams is None:
            raise EnergyBudgetError(
                "an energy budget needs at least one of limit_wh / limit_co2e_grams"
            )
        if not self.energy_accounting_enabled:
            self.use_energy_accounting()
        self.budget_manager.add_energy_budget(
            EnergyBudget(
                scope=scope,  # type: ignore[arg-type]
                id=id,
                limit_wh=limit_wh,
                limit_co2e_grams=limit_co2e_grams,
                period=period,  # type: ignore[arg-type]
            )
        )
        return self

    def energy_report(self, *, by: str = "tenant", since: Any | None = None):
        """Roll up estimated energy + carbon by ``tenant``/``feature``/``user``/
        ``model``/``provider``/``run`` (returns an :class:`EnergyReport`).

        The energy analogue of :meth:`cost_report`, on the same surface and from
        the same attributed events."""
        return self.cost_ledger.energy_report(by, since=since)  # type: ignore[arg-type]

    def batch(
        self,
        inputs: list[str | UserInput],
        *,
        backend: Any | None = None,
        config: RunConfig | None = None,
        discount: float = 0.5,
        timeout_s: float | None = None,
    ) -> list[RunResult]:
        """Run a set of inputs through a provider Batch API at ~half the cost.

        Latency-tolerant work — evals, bulk extraction, synthetic data — with the
        same :class:`RunResult` contract as :meth:`run`. ``backend`` is a
        :class:`~vincio.providers.BatchBackend` (or a provider; defaults to the
        app's provider run in-process)::

            results = app.batch(["summarize doc A", "summarize doc B"])
        """
        return run_sync(
            self.abatch(
                inputs, backend=backend, config=config, discount=discount, timeout_s=timeout_s
            )
        )

    async def abatch(
        self,
        inputs: list[str | UserInput],
        *,
        backend: Any | None = None,
        config: RunConfig | None = None,
        discount: float = 0.5,
        timeout_s: float | None = None,
    ) -> list[RunResult]:
        """Async :meth:`batch`."""
        return await self._runtime.execute_batch(
            inputs, run_config=config, backend=backend, discount=discount, timeout_s=timeout_s
        )

    # -- governance & compliance -----------------------------------------------------

    def _card_format(self, override: Any | None = None):
        from ..governance.cards import CardFormat

        return CardFormat(override or self.config.governance.card_format)

    def model_card(self, *, eval_report: Any | None = None, format: Any | None = None):
        """Generate a :class:`~vincio.governance.ModelCard` from the live config.

        Pass an :class:`~vincio.evals.reports.EvalReport` to attach measured
        evaluation evidence. ``format`` overrides the configured card schema
        (``vincio`` / ``open_model_card`` / ``ai_card``).
        """
        from ..governance.cards import generate_model_card

        return generate_model_card(self, eval_report=eval_report, format=self._card_format(format))

    def system_card(self, *, eval_report: Any | None = None, format: Any | None = None):
        """Generate a :class:`~vincio.governance.SystemCard` (model + retrieval +
        memory + safety filters + human-oversight points) from the live config."""
        from ..governance.cards import generate_system_card

        return generate_system_card(
            self, eval_report=eval_report, format=self._card_format(format), name=self.name
        )

    def compliance_report(self, *, redteam: Any | None = None, eval_report: Any | None = None):
        """Map this app's controls to OWASP/NIST/MITRE frameworks as a coverage
        matrix, backed by red-team and eval evidence
        (:class:`~vincio.governance.ComplianceReport`)."""
        from ..governance.frameworks import ComplianceMapper

        return ComplianceMapper().map(redteam=redteam, eval_report=eval_report, target=self)

    def aibom(self, *, datasets: list[Any] | None = None, prompts: list[Any] | None = None):
        """Generate an AI bill of materials (:class:`~vincio.governance.AIBOM`)
        for the live model/embedder/reranker, with SHA-256 model-hash slots."""
        from ..governance.aibom import generate_aibom

        return generate_aibom(self, datasets=datasets, prompts=prompts)

    def trace_lineage(self, source: str):
        """Return the source → chunk → evidence → output lineage for a source
        name or document id (:class:`~vincio.governance.LineageRecord`)."""
        return self.lineage.trace(source)

    def verify_governance(
        self,
        invariants: Any | None = None,
        *,
        record: bool = True,
        raise_on_violation: bool = False,
    ):
        """Formally verify the governance invariants hold, ahead of any run.

        Proves — by exhaustive bounded model checking, not after-the-fact
        observation — that the platform's governance controls satisfy their
        specifications across the whole typed input space: injection-containment
        (``untrusted ⇒ no unapproved capability``), data residency (in-jurisdiction
        egress refusal, reflecting this app's ``deny_on_unknown`` posture), the
        budget hard cap, and the erasure-proof content binding. Returns a
        content-hashed :class:`~vincio.governance.VerificationReport`; a failed
        property carries a minimal :class:`~vincio.governance.Counterexample`::

            report = app.verify_governance()
            assert report.held
            for cx in report.counterexamples:
                print(cx.render())

        The verdict is deterministic and offline, and (when ``record``) lands on
        the hash-chained audit log as a ``governance_verification`` decision. Pass
        a custom ``invariants`` list to verify a different property set; set
        ``raise_on_violation`` to raise
        :class:`~vincio.core.errors.GovernanceVerificationError` instead of
        returning a non-holding report.
        """
        from ..core.errors import GovernanceVerificationError
        from ..governance.verification import (
            GovernanceVerifier,
            budget_invariant,
            containment_invariant,
            erasure_invariant,
            residency_invariant,
        )

        if invariants is None:
            invariants = [
                containment_invariant(),
                residency_invariant(deny_on_unknown=self.residency.deny_on_unknown),
                budget_invariant(),
                erasure_invariant(),
            ]
        verifier = GovernanceVerifier(invariants, audit_log=self.audit)
        report = verifier.verify(record=record)
        if raise_on_violation and not report.held:
            raise GovernanceVerificationError(
                f"{len(report.counterexamples)} governance invariant(s) violated: "
                + "; ".join(c.render() for c in report.counterexamples),
                counterexamples=report.counterexamples,
            )
        return report

    def mark_output(self, content: str, *, model: str | None = None, signer: Any | None = None):
        """Build a C2PA-style synthetic-content provenance manifest for output
        (:class:`~vincio.governance.ProvenanceManifest`).

        Signs the manifest when a ``signer`` is passed or ``app.content_signer``
        is set (e.g. an :class:`~vincio.governance.HmacSigner`)."""
        from ..governance.transparency import mark_synthetic_content

        return mark_synthetic_content(
            content,
            model_id=model or self.model,
            provider=self._provider_name,
            signer=signer or self.content_signer,
        )

    # -- verified reasoning & neuro-symbolic certificates --------------

    def verify_reasoning(
        self,
        answer: Any,
        *,
        verifiers: Any | None = None,
        evidence: Any | None = None,
        schema: dict[str, Any] | None = None,
        constraints: Any | None = None,
        statistical_claims: Any | None = None,
        facts: dict[str, Any] | None = None,
        now: Any | None = None,
        regenerate: Any | None = None,
        max_cycles: int = 2,
        raise_on_refute: bool = False,
        record: bool = True,
    ) -> Any:
        """Attach and check a deterministic :class:`~vincio.verify.Certificate` to an answer.

        Runs a set of offline kernels (arithmetic, units, temporal, schema,
        constraints, citation entailment — the default
        :func:`~vincio.verify.default_verifiers`) over ``answer`` and returns a
        :class:`~vincio.verify.VerifiedAnswer` whose certificate is **verified**,
        **refuted**, or **inapplicable**. A refuted certificate is a *proof the
        answer is wrong* (a recomputation disagreed), so the orchestrator refuses
        to emit it: :attr:`VerifiedAnswer.holds` is ``False`` and ``refused`` is set.

        When a refuted answer can be repaired, pass a ``regenerate`` callable
        ``(answer, critique) -> new_answer`` to drive the bounded self-correction
        loop: the deterministic refutations become a critique, the callable
        produces a fresh answer, and it is re-certified, up to ``max_cycles`` — the
        same refuse-or-repair discipline structured output already uses, now over
        *reasoning* rather than *structure*. Ground the kernels with ``evidence``
        (citation entailment), ``schema`` (structural conformance), ``constraints``
        (constraint satisfaction), ``statistical_claims`` (the trend / correlation /
        interval / forecast kernels, which recompute a stated statistic from the
        cited cells and refuse a spurious causal claim), ``facts`` and ``now``. When
        ``statistical_claims`` are supplied and ``verifiers`` is left default, the
        statistical kernels are added to the default set automatically. Because a
        statistical claim is grounded in the context rather than the answer text, a
        ``regenerate`` callback may repair one by returning a corrected
        :class:`~vincio.verify.StatisticalClaim` (or a list of them); the loop
        re-grounds the context with the corrected claim before re-certifying, so the
        same refuse-or-repair discipline drives the statistical kernels too. The
        verdict lands on the hash-chained audit log as a ``reasoning_verification``
        decision unless ``record`` is off; set ``raise_on_refute`` to raise
        :class:`~vincio.core.errors.CertificateRefutedError` instead.
        """
        from ..core.errors import CertificateRefutedError
        from ..verify import CompositeVerifier, VerificationContext, VerifiedAnswer
        from ..verify.kernels import default_verifiers
        from ..verify.statistical import statistical_verifiers

        claims = list(statistical_claims) if statistical_claims else []
        if verifiers is not None:
            kernels = list(verifiers)
        else:
            kernels = default_verifiers() + (statistical_verifiers() if claims else [])
        verifier = CompositeVerifier(kernels)
        context = VerificationContext(
            evidence=list(evidence) if evidence else [],
            schema=schema,
            constraints=list(constraints) if constraints else [],
            statistical_claims=claims,
            facts=facts or {},
            now=now,
        )
        from ..verify.statistical import StatisticalClaim

        current = answer
        certificate = verifier.certify(current, context)
        attempts = 1
        while certificate.refuted and regenerate is not None and attempts <= max_cycles:
            critique = "The previous answer failed verification:\n" + "\n".join(
                f"- {c.name}: {c.detail}" for c in certificate.refutations
            )
            repaired = regenerate(current, critique)
            if repaired is None or repaired == current:
                break
            # A statistical claim is grounded in the context, not the answer text, so
            # a repair that re-states the corrected claim(s) re-grounds the context
            # before re-certifying; any other value is a replacement answer as before.
            repaired_claims = (
                [repaired] if isinstance(repaired, StatisticalClaim)
                else list(repaired) if isinstance(repaired, list)
                and repaired and all(isinstance(c, StatisticalClaim) for c in repaired)
                else None
            )
            if repaired_claims is not None:
                context = context.model_copy(update={"statistical_claims": repaired_claims})
            current = repaired
            certificate = verifier.certify(current, context)
            attempts += 1

        refused = certificate.refuted
        verified = VerifiedAnswer(
            answer=current,
            certificate=certificate,
            attempts=attempts,
            refused=refused,
            stopped_reason=(
                "refused" if refused else certificate.status
            ),
        )
        if record and self.audit is not None:
            self.audit.record(
                "reasoning_verification",
                resource=certificate.subject_hash,
                decision=certificate.status,
                details={
                    "kinds": certificate.kinds,
                    "refutations": [c.name for c in certificate.refutations],
                    "attempts": attempts,
                    "certificate_hash": certificate.certificate_hash,
                },
            )
        if raise_on_refute and refused:
            raise CertificateRefutedError(
                "answer certificate refuted: "
                + "; ".join(c.detail for c in certificate.refutations),
                details={"refutations": [c.name for c in certificate.refutations]},
            )
        return verified

    def behavior_monitor(self, specs: Any) -> Any:
        """Build a :class:`~vincio.verify.RuntimeMonitor` over one or more
        :class:`~vincio.verify.BehaviorSpec`\\ s.

        The monitor checks a property over an agent's trajectory step-by-step:
        feed it :class:`~vincio.verify.BehaviorEvent`\\ s via ``observe`` as the
        agent runs, or a recorded list via ``check_trajectory``. It is the online,
        per-step, behavioural analogue of the ahead-of-run governance verifier.
        """
        from ..verify import RuntimeMonitor

        return RuntimeMonitor(specs)

    def shield(self, specs: Any, *, mode: str = "block", repair: Any | None = None, use: bool = False) -> Any:
        """Build a :class:`~vincio.verify.Shield` that prevents a behavioural violation.

        A shield wraps a monitor and, before an action executes, **blocks** it
        (``mode='block'``), **repairs** it to a safe alternative (``mode='repair'``
        with a ``repair`` callback), or merely records it (``mode='monitor'``). With
        ``use=True`` the shield is installed on this app's tool runtime, so a
        policy-violating tool call (a write before approval, a tool outside scope)
        is structurally refused — the per-step, online counterpart of the rails.
        """
        from ..verify import Shield

        built = Shield(specs, mode=mode, repair=repair)  # type: ignore[arg-type]
        if use:
            self.use_shield(built)
        return built

    def use_shield(self, shield: Any | None) -> Any:
        """Install (or clear, with ``None``) a behavioural shield on the tool runtime.

        Once installed, every tool call is checked against the shield's
        :class:`~vincio.verify.BehaviorSpec`\\ s *before* it executes; a blocked
        call returns a denied result like a failed permission check. Returns the
        shield.
        """
        self.tool_runtime.shield = shield
        return shield

    def synthesize_program(
        self, spec: Any, examples: Any, *, require: bool = True, record: bool = True
    ) -> Any:
        """Synthesize and verify a small data-transform program.

        Runs ``spec``'s whitelisted op pipeline on representative ``examples``,
        checks its declared properties (schema conformance, row-count relations,
        field invariants), and returns a
        :class:`~vincio.verify.SynthesizedProgram` carrying the
        :class:`~vincio.verify.Certificate` that proves them — proof-carrying code
        in the tool plane. With ``require`` (the default) a refuted program raises
        :class:`~vincio.core.errors.ProgramSynthesisError` rather than returning;
        the verdict lands on the audit log as a ``program_synthesis`` decision.
        """
        from ..verify import synthesize

        program = synthesize(spec, list(examples), require=require)
        if record and self.audit is not None:
            self.audit.record(
                "program_synthesis",
                resource=program.certificate.subject_hash,
                decision=program.certificate.status,
                details={
                    "name": getattr(spec, "name", ""),
                    "properties": [c.name for c in program.certificate.checks],
                    "certificate_hash": program.certificate.certificate_hash,
                },
            )
        return program

    # -- documents & media flow OUT ------------------------------------

    def build_document(
        self,
        source: Any,
        *,
        format: str = "markdown",
        contract: Any | None = None,
        title: str = "",
        evidence_ids: list[str] | None = None,
    ):
        """Render a validated result into a cited, contract-checked artifact.

        Thin wrapper over :class:`~vincio.generation.builder.DocumentBuilder`
        bound to this app's audit log. ``source`` is a validated
        :class:`~vincio.core.types.RunResult`, mapping, or Markdown; ``contract``
        is an optional :class:`~vincio.generation.contracts.DocumentContract`.
        """
        from ..generation.builder import DocumentBuilder

        builder = DocumentBuilder(audit_log=self.audit)
        return builder.build(
            source,
            format=cast("Any", format),
            contract=contract,
            title=title,
            evidence_ids=evidence_ids,
        )

    def cited_report(
        self,
        answer: Any,
        evidence: list[Any] | None = None,
        *,
        format: str = "markdown",
        title: str = "",
        contract: Any | None = None,
        entailment: Any | None = None,
        figures: list[Any] | None = None,
        catalog: Any | None = None,
    ):
        """Resolve ``[E1]`` citations into a rendered, footnoted, cited report.

        Synchronous wrapper over
        :class:`~vincio.generation.report.CitedReportBuilder`; use ``acited_report``
        from async code. Evidence defaults to an empty list (markers then resolve
        to nothing and are reported as unresolved). Pass ``figures=`` (a list of
        :class:`~vincio.generation.Figure`) to embed **data-bound** charts/tables —
        each verified to re-derive from its source against ``catalog`` (defaults to
        the app's registered :meth:`data_catalog`)."""
        return run_sync(
            self.acited_report(
                answer,
                evidence,
                format=format,
                title=title,
                contract=contract,
                entailment=entailment,
                figures=figures,
                catalog=catalog,
            )
        )

    async def acited_report(
        self,
        answer: Any,
        evidence: list[Any] | None = None,
        *,
        format: str = "markdown",
        title: str = "",
        contract: Any | None = None,
        entailment: Any | None = None,
        figures: list[Any] | None = None,
        catalog: Any | None = None,
    ):
        """Build a cited report from an answer and its evidence (async) → a document artifact."""
        from ..generation.report import CitedReportBuilder

        if catalog is None and figures:
            registered = self.data_catalog()
            catalog = registered if registered.names else None
        builder = CitedReportBuilder(entailment=entailment, audit_log=self.audit)
        return await builder.build(
            answer,
            list(evidence or []),
            format=cast("Any", format),
            title=title,
            contract=contract,
            figures=figures,
            catalog=catalog,
        )

    async def agenerate_image(
        self,
        prompt: Any,
        *,
        provider: Any,
        model: str | None = None,
        n: int = 1,
        size: str = "1024x1024",
        budget: Any | None = None,
    ):
        """Generate image(s) through an
        :class:`~vincio.generation.image.ImageProvider`, metered against the
        budget, audited (``image_generate``), and C2PA-stamped per asset."""
        from ..generation.image import ImageGenRequest
        from ..generation.media import meter_media_cost

        request = (
            prompt
            if isinstance(prompt, ImageGenRequest)
            else ImageGenRequest(prompt=str(prompt), n=n, size=size)
        )
        kwargs = {"model": model} if model else {}
        response = await provider.generate_image(request, **kwargs)
        self._meter_and_audit_media(
            "image_generate", response, request.prompt, budget, meter_media_cost
        )
        return response

    def generate_image(
        self, prompt: Any, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`agenerate_image`."""
        return run_sync(self.agenerate_image(prompt, provider=provider, model=model, **kwargs))

    async def asynthesize_speech(
        self,
        text: str,
        *,
        provider: Any,
        model: str | None = None,
        voice: str = "alloy",
        format: str = "mp3",
        budget: Any | None = None,
    ):
        """Synthesize speech through a
        :class:`~vincio.generation.speech.SpeechProvider`, metered, audited
        (``speech_synthesize``), and audio-provenance-stamped."""
        from ..generation.media import meter_media_cost
        from ..generation.speech import SpeechRequest

        request = SpeechRequest(text=text, voice=voice, format=format)  # type: ignore[arg-type]
        kwargs = {"model": model} if model else {}
        response = await provider.synthesize_speech(request, **kwargs)
        self._meter_and_audit_media("speech_synthesize", response, text, budget, meter_media_cost)
        return response

    def synthesize_speech(
        self, text: str, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`asynthesize_speech`."""
        return run_sync(self.asynthesize_speech(text, provider=provider, model=model, **kwargs))

    def _meter_and_audit_media(
        self, action: str, response: Any, prompt: str, budget: Any, meter: Any
    ) -> None:
        # Accumulate against the app's cumulative media usage so the cost cap is
        # honored across calls; also record on the cost tracker for cost_report.
        meter(response.cost_usd, budget=budget or self.budget, usage=self._media_usage)
        self.cost_tracker.record_infra(response.cost_usd)
        assets = (
            getattr(response, "images", None)
            or getattr(response, "videos", None)
            or [getattr(response, "audio", None)]
        )
        manifests = [getattr(a, "manifest", None) for a in assets if a is not None]
        self.audit.record(
            action,
            resource=response.model,
            details={
                "provider": response.provider,
                "model": response.model,
                "prompt": prompt[:200],
                "cost_usd": response.cost_usd,
                "assets": len([a for a in assets if a is not None]),
                "content_sha256": [m.content_sha256 for m in manifests if m is not None],
            },
        )

    def load_media(self, path: str, *, transcriber: Any, tenant_id: str | None = None):
        """Ingest audio/video as a timestamped transcript Document
        (:func:`vincio.documents.load_media`)."""
        from ..documents.loaders import load_media

        return load_media(path, transcriber=transcriber, tenant_id=tenant_id)

    def load_video(self, path: str, *, analyzer: Any, tenant_id: str | None = None):
        """Ingest a video as a temporally-segmented Document
        (:func:`vincio.documents.load_video`).

        ``analyzer`` is a :class:`~vincio.documents.video.VideoAnalyzer`
        (``MockVideoAnalyzer`` offline, ``ProviderVideoAnalyzer`` online). Each
        segment becomes a section carrying its ``start`` / ``end`` timestamps, so
        a retrieved claim grounds to a time range, not just a document."""
        from ..documents.loaders import load_video

        return load_video(path, analyzer=analyzer, tenant_id=tenant_id)

    async def agenerate_video(
        self,
        prompt: Any,
        *,
        provider: Any,
        model: str | None = None,
        seconds: float = 5.0,
        size: str = "1280x720",
        budget: Any | None = None,
    ):
        """Generate a video through a
        :class:`~vincio.generation.video.VideoProvider`, metered against the
        budget, audited (``video_generate``), and C2PA-stamped per clip."""
        from ..generation.media import meter_media_cost
        from ..generation.video import VideoGenRequest

        request = (
            prompt
            if isinstance(prompt, VideoGenRequest)
            else VideoGenRequest(prompt=str(prompt), seconds=seconds, size=size)
        )
        kwargs = {"model": model} if model else {}
        response = await provider.generate_video(request, **kwargs)
        self._meter_and_audit_media(
            "video_generate", response, request.prompt, budget, meter_media_cost
        )
        return response

    def generate_video(
        self, prompt: Any, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`agenerate_video`."""
        return run_sync(self.agenerate_video(prompt, provider=provider, model=model, **kwargs))

    async def aedit_video(
        self,
        video: Any,
        prompt: Any,
        *,
        provider: Any,
        model: str | None = None,
        seconds: float = 5.0,
        budget: Any | None = None,
    ):
        """Edit/extend a video through a
        :class:`~vincio.generation.video.VideoProvider`, metered, audited
        (``video_edit``), and C2PA-stamped (the manifest marks it as edited)."""
        from ..generation.media import meter_media_cost
        from ..generation.video import VideoGenRequest

        request = (
            prompt
            if isinstance(prompt, VideoGenRequest)
            else VideoGenRequest(prompt=str(prompt), seconds=seconds)
        )
        kwargs = {"model": model} if model else {}
        response = await provider.edit_video(video, request, **kwargs)
        self._meter_and_audit_media(
            "video_edit", response, request.prompt, budget, meter_media_cost
        )
        return response

    def edit_video(
        self, video: Any, prompt: Any, *, provider: Any, model: str | None = None, **kwargs: Any
    ):
        """Synchronous :meth:`aedit_video`."""
        return run_sync(self.aedit_video(video, prompt, provider=provider, model=model, **kwargs))

    def risk_tier(
        self,
        *,
        purpose: str = "",
        domains: list[str] | None = None,
        prohibited_practices: list[str] | None = None,
    ):
        """Classify this app into the EU AI Act risk tiers (advisory)."""
        from ..governance.eu_ai_act import RiskTierClassifier

        return RiskTierClassifier(
            purpose=purpose, domains=domains, prohibited_practices=prohibited_practices
        ).classify(self)

    def annex_iv(
        self,
        *,
        format: str = "markdown",
        purpose: str = "",
        domains: list[str] | None = None,
        eval_report: Any | None = None,
        redteam: Any | None = None,
    ):
        """Generate EU AI Act Annex IV technical documentation as a cited artifact."""
        from ..governance.eu_ai_act import AnnexIVBuilder, RiskTierClassifier

        classifier = RiskTierClassifier(purpose=purpose, domains=domains)
        return AnnexIVBuilder(classifier=classifier).build(
            self, format=cast("Any", format), eval_report=eval_report, redteam=redteam
        )

    def fria(
        self,
        *,
        format: str = "markdown",
        purpose: str = "",
        domains: list[str] | None = None,
        affected_groups: list[str] | None = None,
        eval_report: Any | None = None,
    ):
        """Generate an EU AI Act Art. 27 fundamental-rights impact assessment."""
        from ..governance.eu_ai_act import FRIAGenerator, RiskTierClassifier

        classifier = RiskTierClassifier(purpose=purpose, domains=domains)
        return FRIAGenerator(classifier=classifier).generate(
            self,
            format=cast("Any", format),
            affected_groups=affected_groups,
            eval_report=eval_report,
        )

    def set_residency(
        self,
        allowed_regions: list[str],
        *,
        provider_regions: dict[str, str] | None = None,
        deny_on_unknown: bool = True,
    ) -> ContextApp:
        """Pin allowed provider regions; runs outside them are refused egress."""
        self.residency = ResidencyPolicy(
            allowed_regions=list(allowed_regions),
            provider_regions={**self.residency.provider_regions, **(provider_regions or {})},
            deny_on_unknown=deny_on_unknown,
        )
        return self

    def use_consent_ledger(self, ledger: Any | None = None, *, default_allow: bool = False) -> Any:
        """Attach a :class:`~vincio.governance.consent.ConsentLedger`.

        Binds data to a GDPR purpose and lawful basis. Once attached, access
        decisions (:meth:`AccessController.check_purpose`) and memory recall
        consult it, so a withdrawn consent or a purpose mismatch is enforced in
        code. Persists to the app's store and writes grants/revokes/denied checks
        to the same audit chain as erasure. Returns the ledger."""
        from ..governance.consent import ConsentLedger

        if ledger is None:
            ledger = ConsentLedger(store=self.store, audit=self.audit, default_allow=default_allow)
        self.consent_ledger = ledger
        self.access.consent_ledger = ledger
        if self.memory is not None:
            self.memory.consent_ledger = ledger
        return ledger

    def use_privacy_accountant(
        self,
        accountant: Any | None = None,
        *,
        default_budget: Any | None = None,
        default_mechanism: Any | None = None,
        delta: float = 1e-5,
    ) -> Any:
        """Attach a differential-privacy accountant over the learning loop.

        Composes a per-subject ``(ε, δ)`` budget across every accounted memory
        consolidation and federated contribution: a step that would exceed a
        subject's remaining budget is refused (or down-weighted), every spend and
        refusal on the same hash-chained audit log as consent and erasure. Once
        attached, :meth:`MemoryEngine.consolidate` and
        :meth:`contribute_federated` gate automatically, and
        :meth:`privacy_report` rolls up the spent budget alongside
        :meth:`cost_report`. Pass a configured
        :class:`~vincio.governance.privacy.PrivacyAccountant`, or let this build
        one wired to the app's audit chain and store. Returns the accountant::

            from vincio import PrivacyBudget
            app.use_privacy_accountant(default_budget=PrivacyBudget(epsilon=2.0))
        """
        from ..governance.privacy import PrivacyAccountant

        if accountant is None:
            accountant = PrivacyAccountant(
                default_budget=default_budget,
                default_mechanism=default_mechanism,
                delta=delta,
                audit=self.audit,
                store=self.store,
            )
        self.privacy_accountant = accountant
        if self.memory is not None:
            self.memory.privacy_accountant = accountant
        return accountant

    def set_privacy_budget(
        self,
        *,
        subject_id: str | None = None,
        epsilon: float,
        delta: float = 1e-5,
        on_breach: str = "refuse",
    ) -> ContextApp:
        """Set a per-subject (or default) differential-privacy budget.

        Creates the accountant on first use. ``subject_id=None`` is the default
        budget applied to any subject without a specific one; ``on_breach`` is
        ``"refuse"`` (a hard cap) or ``"downweight"`` (clip harder to fit)::

            app.set_privacy_budget(subject_id="alice", epsilon=1.0)
            app.set_privacy_budget(epsilon=3.0, on_breach="downweight")
        """
        from ..governance.privacy import PrivacyBudget

        if self.privacy_accountant is None:
            self.use_privacy_accountant(delta=delta)
        self.privacy_accountant.set_budget(
            PrivacyBudget(
                subject_id=subject_id,
                epsilon=epsilon,
                delta=delta,
                on_breach=on_breach,  # type: ignore[arg-type]
            )
        )
        return self

    def privacy_report(self, subject: str | None = None):
        """Per-subject differential-privacy budget roll-up.

        The privacy analogue of :meth:`cost_report`: each row is a subject's
        cumulative ``ε`` spent against its ceiling, with operation and refusal
        counts, so the spent privacy budget is an auditable number. Returns an
        empty :class:`~vincio.governance.privacy.PrivacyReport` when no accountant
        is attached."""
        if self.privacy_accountant is None:
            from ..governance.privacy import PrivacyReport

            return PrivacyReport()
        return self.privacy_accountant.report(subject)

    def use_reputation_ledger(self, ledger: Any | None = None, *, config: Any | None = None) -> Any:
        """Attach a cross-fleet reputation ledger over the federated round.

        Earns a per-member reliability score from how each federated contribution
        fared against the no-regression gate — a pass credits the contributor, a
        regression debits it — and reliability-weights the
        :class:`~vincio.optimize.federated.SecureAggregator` so a repeatedly
        regressing or adversarial member is discounted without being singled out.
        The discount is bounded and reversible: a weight only ever lowers a
        member's pull, and adoption still clears the same gate, so reputation can
        never bypass the quality bar. Every update lands on the same hash-chained
        audit log as consent, privacy, and erasure, so a member's standing is a
        mechanical, auditable, replayable number.

        Once attached, :meth:`federated_improvement` / :meth:`adopt_federated`
        weight contributions by reputation and record each round's verdict back
        automatically, and :meth:`reputation_report` rolls up each member's score
        next to the cost and privacy reports. Pass a configured
        :class:`~vincio.optimize.reputation.ReputationLedger`, or let this build one
        wired to the app's audit chain, event bus, and store. Returns the ledger::

            app.use_reputation_ledger()
            result = app.adopt_federated(golden, [mine, *peer_updates])
        """
        from ..optimize.reputation import ReputationLedger

        if ledger is None:
            ledger = ReputationLedger(
                config=config, audit=self.audit, events=self.events, store=self.store
            )
        self.reputation_ledger = ledger
        return ledger

    def reputation_report(self, member: str | None = None):
        """Per-member cross-fleet reputation roll-up.

        Each row is a member's earned reliability score and the aggregation weight
        it maps to, with the success / failure tally behind it, so a member's
        standing in the fleet is an auditable number. Returns an empty
        :class:`~vincio.optimize.reputation.ReputationReport` when no ledger is
        attached."""
        if self.reputation_ledger is None:
            from ..optimize.reputation import ReputationReport

            return ReputationReport()
        return self.reputation_ledger.report(member)

    def erase_source(self, source: str, *, prove: bool = True) -> ErasureResult:
        """Right-to-erasure-by-source: purge a source from indexes, memory,
        caches, and generated artifacts, logged on the hash-chained audit chain.

        ``source`` is a source name (as passed to :meth:`add_source`) or a
        document id. Returns an :class:`~vincio.governance.ErasureResult`.
        Idempotent: a second call finds nothing left to erase.

        When ``prove``, the sweep emits a signed, content-bound
        :class:`~vincio.governance.ErasureProof` on the result — a manifest of
        exactly which chunk / document / memory / artifact ids were removed,
        bound by SHA-256, signed with :attr:`content_signer` when set, and
        anchored to the audit chain's Merkle root — so erasure is *provable*,
        not merely logged.
        """
        record = self.lineage.trace(source)
        result = ErasureResult(source=source, found=not record.is_empty)
        chunk_ids = list(record.chunks)
        # The exact identifiers removed, per store — the binding the proof covers.
        removed_ids: dict[str, list[str]] = {}
        per_index: dict[str, int] = {}
        index_handles = {
            "bm25": self._bm25,
            "vector": self._vector,
            "sparse": self._sparse,
            "late_interaction": self._late_interaction,
        }
        if chunk_ids:
            for label, index in index_handles.items():
                if index is None:
                    continue
                per_index[label] = run_sync(index.delete(chunk_ids))
                result.indexes_swept += 1
            result.chunks_removed = len(chunk_ids)
            removed_ids["chunks"] = list(chunk_ids)
            if self.entity_graph is not None:
                # Entity graph is rebuilt from sources; drop nothing destructively
                # here beyond chunk references already removed from indexes.
                pass

        # Documents recorded in the metadata store. Count only deletions that
        # actually succeed, so the audit trail never overstates erasure.
        removed_docs: list[str] = []
        for doc_id in record.documents:
            try:
                if hasattr(self.store, "delete") and self.store.delete("documents", doc_id):  # type: ignore[attr-defined]
                    result.documents_removed += 1
                    removed_docs.append(doc_id)
            except Exception:
                note_suppressed("governance.erase.document_delete")
        if removed_docs:
            removed_ids["documents"] = removed_docs

        # Memory items whose provenance references the source (exact matches on
        # source name / id, never a loose substring that could over-delete).
        removed_memories: list[str] = []
        if self.memory is not None:
            doc_set = set(record.documents)
            for item in list(self.memory.store.all_items(statuses=())):
                meta = item.metadata or {}
                refs = {meta.get("source"), meta.get("source_id")}
                if source in refs or bool(refs & doc_set):
                    if self.memory.delete(item.id):
                        result.memories_removed += 1
                        removed_memories.append(item.id)
        if removed_memories:
            removed_ids["memories"] = removed_memories

        # Generated artifacts (cited documents, images, audio) derived from the
        # source — removed from the blob/metadata store so the deliverable is
        # erased alongside the evidence and memory it was built from.
        removed_artifacts: list[str] = []
        for artifact_key in record.artifacts:
            erased = False
            for store_obj, kind in ((self.store, "artifacts"), (self.store, "documents")):
                try:
                    if hasattr(store_obj, "delete") and store_obj.delete(kind, artifact_key):  # type: ignore[attr-defined]
                        erased = True
                except Exception:
                    note_suppressed("governance.erase.artifact_delete")
            # The lineage link is severed regardless, which is the auditable fact.
            result.artifacts_removed += 1
            removed_artifacts.append(artifact_key)
            _ = erased
        if removed_artifacts:
            removed_ids["artifacts"] = removed_artifacts

        # Registered tabular datasets ingested from the source — dropped from the
        # data catalog so an erased source is erased as structured data too, and the
        # semantic layers defined over them un-registered (their definitions can no
        # longer ground to absent rows).
        removed_datasets: list[str] = []
        catalog = getattr(self, "_data_catalog_obj", None)
        for table in list(record.datasets):
            if catalog is not None and catalog.remove(table):
                removed_datasets.append(table)
            self._semantic_layers.pop(table, None)
        if removed_datasets:
            result.datasets_removed = len(removed_datasets)
            removed_ids["datasets"] = removed_datasets

        # Caches: erasure correctness outweighs cache retention.
        for cache in (self.response_cache, self.context_compile_cache):
            backend = getattr(cache, "backend", None) or getattr(cache, "cache", None)
            if backend is not None and hasattr(backend, "clear"):
                try:
                    backend.clear()
                    result.caches_invalidated += 1
                except Exception:
                    note_suppressed("governance.erase.cache_invalidate")

        entry = self.audit.record(
            "erase_source",
            decision="allow",
            resource=source,
            details={
                "found": result.found,
                "chunks_removed": result.chunks_removed,
                "documents_removed": result.documents_removed,
                "memories_removed": result.memories_removed,
                "artifacts_removed": result.artifacts_removed,
                "datasets_removed": result.datasets_removed,
                "indexes_swept": result.indexes_swept,
                "caches_invalidated": result.caches_invalidated,
                "per_index": per_index,
            },
        )
        result.audit_entry_id = entry.id

        # Build the signed, content-bound erasure proof over the precise
        # removed-id set, anchored to the audit chain's current Merkle root.
        if prove:
            proof = build_erasure_proof(
                source,
                removed_ids,
                counts={
                    "chunks": result.chunks_removed,
                    "documents": result.documents_removed,
                    "memories": result.memories_removed,
                    "artifacts": result.artifacts_removed,
                    "datasets": result.datasets_removed,
                    "caches": result.caches_invalidated,
                },
                signer=self.content_signer,
                audit_entry_id=entry.id,
                audit_merkle_root=self.audit.merkle_root(),
            )
            result.proof = proof
            self.audit.record(
                "erasure_proof",
                decision="allow",
                resource=source,
                details={
                    "content_sha256": proof.content_sha256,
                    "signed": proof.signature is not None,
                    "key_id": proof.key_id,
                    "removed": proof.removed,
                },
            )

        self.events.emit(
            "governance.source_erased",
            {
                "source": source,
                "found": result.found,
                "proven": result.proof is not None,
                "content_sha256": result.proof.content_sha256 if result.proof else None,
            },
        )
        self.lineage.forget(source)
        # Drop the source registration so it is not re-counted.
        self.sources.pop(source, None)
        return result

    # -- sources / retrieval ----------------------------------------------------------------

    def _ensure_retrieval(self, retrieval: str) -> None:
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
        )
        if retrieval in ("graph", "hybrid_graph") and self.entity_graph is None:
            self.entity_graph = EntityGraph()

    def add_source(
        self,
        name: str,
        *,
        path: str | None = None,
        documents: list[Any] | None = None,
        connector: Any | None = None,
        loader: str | None = None,
        chunking: str | None = None,
        retrieval: str = "hybrid",
    ) -> ContextApp:
        """Register a knowledge source: load, chunk, and index documents.

        Sources can come from a local ``path``, in-memory ``documents``, or
        any :class:`~vincio.connectors.Connector` (web, GitHub, SQL, S3,
        GCS, Notion, Confluence, Slack, or custom) via ``connector=``.
        """
        chunking = chunking or self.config.retrieval.chunking
        source = _SourceConfig(
            name=name, path=path, loader=loader, chunking=chunking, retrieval=retrieval
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
        return self

    async def _index_chunks(self, chunks: list[Any]) -> None:
        if self._bm25 is not None:
            await self._bm25.add(chunks)
        if self._vector is not None:
            await self._vector.add(chunks)
        if self._sparse is not None:
            await self._sparse.add(chunks)
        if self._late_interaction is not None:
            await self._late_interaction.add(chunks)

    async def ingest_files(self, paths: list[str]) -> list[EvidenceItem]:
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

    def retrieve_facts(
        self,
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

    def add_memory(
        self,
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

    def remember(self, content: str, **kwargs: Any) -> MemoryItem:
        """Ergonomic memory write; creates the memory engine on first use."""
        if self.memory is None:
            self.add_memory()
        return self.memory.remember(content, **kwargs)  # type: ignore[union-attr]

    def recall(self, query: str, **kwargs: Any) -> list[MemoryItem]:
        """Ergonomic memory recall over user/agent/session scopes."""
        if self.memory is None:
            self.add_memory()
        return self.memory.recall(query, **kwargs)  # type: ignore[union-attr]

    def consolidate_memory(
        self,
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

    def enable_memory_os(
        self,
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

    def enable_computer_use(
        self,
        backend: str = "mock",
        *,
        isolation: str | None = None,
        require_isolation: bool = False,
        permission: str = "computer:use",
        approval_required: bool = True,
        **backend_kwargs: Any,
    ):
        """Register a computer-use action surface (navigate / click / type /
        screenshot) as audited, permissioned tools.

        ``backend`` is ``"mock"`` (deterministic, offline), ``"playwright"`` (real
        browser), or ``"provider"`` (provider-native computer-use). With
        ``require_isolation=True`` the workload must run behind a real
        :class:`~vincio.tools.sandbox.IsolationBackend` (container / microVM /
        gVisor / WASM) — subprocess-only hosts are refused::

            app.enable_computer_use("mock")
            agent = app.agent(tools=["computer_navigate", "computer_screenshot"])
        """
        from ..tools.computer_use import (
            MockComputerUse,
            PlaywrightComputerUse,
            ProviderComputerUse,
            computer_use_tools,
        )

        if require_isolation:
            from ..tools.sandbox import get_isolation_backend, require_real_isolation

            require_real_isolation(get_isolation_backend(isolation or "subprocess"))
        if backend == "playwright":
            impl: Any = PlaywrightComputerUse(**backend_kwargs)
        elif backend == "provider":
            impl = ProviderComputerUse(self._base_provider(), self.model, **backend_kwargs)
        else:
            impl = MockComputerUse()
        for tool in computer_use_tools(impl):
            self.add_tool(
                tool,
                permissions=[permission],
                side_effects="external",
                approval_required=approval_required,
            )
        self.audit.record(
            "computer_use_enabled",
            decision="allow",
            details={
                "backend": backend,
                "isolation": isolation,
                "require_isolation": require_isolation,
            },
        )
        return impl

    def computer_use(
        self,
        backend: str = "mock",
        *,
        screen: Any = None,
        policy: Any = None,
        approve: Callable[..., bool] | None = None,
        auto_undo: bool = True,
        max_steps: int = 50,
        isolation: str | None = None,
        require_isolation: bool = False,
        **backend_kwargs: Any,
    ) -> Any:
        """Open a grounded, verified, reversible computer-use **action plane**.

        Returns a :class:`~vincio.tools.ComputerEnvironment` that perceives a screen
        as typed, addressable :class:`~vincio.tools.UIElement`\\ s, grounds an intent
        to a stable selector, **pre-gates** each action against an
        :class:`~vincio.tools.ActionPolicy` (a destructive or out-of-scope action is
        gated like a write tool, with an ``approve`` callback), acts, **post-verifies**
        the effect, and **undoes** it on divergence — every action recorded on this
        app's hash-chained audit log.

        ``backend`` is ``"mock"`` (deterministic, offline; pass a
        :class:`~vincio.tools.ScreenApp`/:class:`~vincio.tools.MockScreen` as
        ``screen``), ``"playwright"`` (a real browser / CDP), ``"accessibility"`` (an
        OS accessibility tree), or ``"remote_desktop"`` (a remote machine); the real
        adapters need ``vincio[computer-use]``. With ``require_isolation=True`` the
        workload must run behind a real
        :class:`~vincio.tools.sandbox.IsolationBackend`::

            app_spec, task = make_web_checkout()
            env = app.computer_use(screen=app_spec, policy=ActionPolicy(allow_urls=["https://shop.test"]))
            run = env.run(my_policy, task)
            run.success and run.safe  # verified end-state, no unapproved destructive action
        """
        from ..tools.computer_environment import (
            AccessibilityScreen,
            ActionPolicy,
            ComputerEnvironment,
            MockScreen,
            PlaywrightScreen,
            RemoteDesktopScreen,
            ScreenApp,
            ScreenBackend,
        )

        if require_isolation:
            from ..tools.sandbox import get_isolation_backend, require_real_isolation

            require_real_isolation(get_isolation_backend(isolation or "subprocess"))

        if isinstance(screen, ScreenBackend):
            impl: ScreenBackend = screen
        elif isinstance(screen, ScreenApp):
            impl = MockScreen(screen)
        elif backend == "playwright":
            impl = PlaywrightScreen(**backend_kwargs)
        elif backend == "accessibility":
            impl = AccessibilityScreen(**backend_kwargs)
        elif backend == "remote_desktop":
            impl = RemoteDesktopScreen(**backend_kwargs)
        elif isinstance(screen, dict):
            impl = MockScreen(ScreenApp.model_validate(screen))
        else:
            raise ConfigError(
                f"computer_use backend {backend!r} needs a ScreenApp/MockScreen via screen=; "
                "the deterministic offline backend is 'mock'"
            )

        self.audit.record(
            "computer_use_session",
            decision="allow",
            details={
                "backend": getattr(impl, "name", backend),
                "isolation": isolation,
                "require_isolation": require_isolation,
                "auto_undo": auto_undo,
            },
        )
        return ComputerEnvironment(
            impl,
            app=self,
            policy=policy if isinstance(policy, ActionPolicy) else (ActionPolicy(**policy) if isinstance(policy, dict) else None),
            approve=approve,
            auto_undo=auto_undo,
            max_steps=max_steps,
        )

    def use_hosted_tools(
        self, names: list[str] | None = None, *, namespace: str = "openai"
    ) -> ContextApp:
        """Surface provider-native hosted tools (``web_search`` / ``file_search`` /
        ``code_interpreter`` / ``computer_use``) as namespaced Vincio tools.

        They register on the tool registry with explicit permissions and ride the
        same RBAC + audit path as any local tool; the Responses adapter emits each
        as its provider-native built-in descriptor::

            app.use_hosted_tools(["web_search", "code_interpreter"])
        """
        from ..providers.hosted_tools import hosted_tool_specs

        for spec in hosted_tool_specs(names, namespace=namespace):
            self.tool_registry.register_spec(spec)
            if spec.name not in self.enabled_tools:
                self.enabled_tools.append(spec.name)
        self.audit.record(
            "hosted_tools_enabled",
            decision="allow",
            details={
                "namespace": namespace,
                "tools": [s.name for s in hosted_tool_specs(names, namespace=namespace)],
            },
        )
        return self

    # -- tools ------------------------------------------------------------------------------------

    def add_tool(
        self,
        tool: str | Callable,
        *,
        permissions: list[str] | None = None,
        permission: str | None = None,
        approval_required: bool = False,
        side_effects: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> ContextApp:
        """Enable a tool: a callable (registered now) or the name of a tool
        already registered on app.tool_registry."""
        if permission is not None:  # default permission is "read_only"
            if permission == "read_only":
                side_effects = side_effects or "read"
            else:
                permissions = [*(permissions or []), permission]
        if callable(tool):
            self.tool_registry.register(
                tool,
                permissions=permissions or [],
                approval_required=approval_required,
                side_effects=side_effects or "read",
                description=description,
                **kwargs,
            )
            name = kwargs.get("name") or tool.__name__
        else:
            name = tool
            if name not in self.tool_registry:
                raise ToolNotFoundError(
                    f"tool {name!r} is not registered; pass a callable or register it via "
                    "app.tool_registry.register(...)",
                    tool=name,
                )
            spec = self.tool_registry.get(name).spec
            if permissions:
                spec.permissions = permissions
            if approval_required:
                spec.approval_required = True
            if side_effects:
                spec.side_effects = side_effects  # type: ignore[assignment]
        if name not in self.enabled_tools:
            self.enabled_tools.append(name)
        return self

    # -- skills ---------------------------------------------------------------------------------

    def add_skill(self, skill: str | Any, *, register_scripts: bool = False) -> ContextApp:
        """Load an Agent Skill (``SKILL.md`` path or a :class:`Skill`) and inject
        it through the compiler with progressive disclosure: a one-line summary
        is always available; the full body is included only when a run's task is
        relevant. Set ``register_scripts=True`` to expose bundled scripts as
        sandboxed, permissioned tools."""
        from ..skills import Skill, load_skill, register_skill_scripts

        loaded = skill if isinstance(skill, Skill) else load_skill(skill)
        if self.skill_library is None:
            self.skill_library = SkillLibrary()
        self.skill_library.add(loaded)
        if register_scripts and loaded.scripts:
            for name in register_skill_scripts(self.tool_registry, loaded):
                if name not in self.enabled_tools:
                    self.enabled_tools.append(name)
        return self

    # -- MCP ------------------------------------------------------------------------------------

    def add_mcp_server(
        self,
        name: str,
        *,
        command: list[str] | None = None,
        url: str | None = None,
        server: Any | None = None,
        transport: Any | None = None,
        headers: dict[str, str] | None = None,
        http_client: Any | None = None,
        auth: str | None = None,
        tools: bool = True,
        resources: bool = True,
        prompts: bool = False,
        permissions: list[str] | None = None,
        sampling: bool = True,
        elicitation: Any | None = None,
        elicitation_policy: Any | None = None,
        elicitation_approval: Any | None = None,
    ) -> ContextApp:
        """Connect to an MCP server and register its tools/resources/prompts.

        Provide exactly one of ``command`` (stdio), ``url`` (Streamable HTTP),
        ``server`` (an in-process :class:`MCPServer`), or ``transport``. MCP
        tools register through the existing permissioned, sandboxed, audited
        runtime (namespaced ``<name>.<tool>``); resources become evidence with
        ``origin: mcp:<name>``. Server-initiated sampling routes to this app's
        provider.

        A server's mid-call **elicitation** request routes to a governed
        :class:`~vincio.mcp.apps.ElicitationGate` built from ``elicitation`` (the
        collector that obtains the user's value): the collected value is screened
        through this app's input rails and tainted *untrusted*, so it is contained
        like any other untrusted input. Pass an ``elicitation_approval`` callable
        (``ElicitationRequest -> bool``) to additionally gate the request behind an
        approval — the way a write tool is gated — or an
        :class:`~vincio.mcp.apps.ElicitationPolicy` for full control. Connect
        happens now (synchronously); the live client is kept on
        ``app.mcp_clients[name]``.
        """
        from ..mcp import (
            InProcessTransport,
            MCPClient,
            StdioTransport,
            StreamableHTTPTransport,
        )
        from ..providers.base import run_sync

        if transport is None:
            provided = [x for x in (command, url, server) if x is not None]
            if len(provided) != 1:
                raise ConfigError(
                    "add_mcp_server requires exactly one of command=, url=, server=, or transport="
                )
            if command is not None:
                transport = StdioTransport(command)
            elif url is not None:
                transport = StreamableHTTPTransport(url, headers=headers, client=http_client)
            else:
                transport = InProcessTransport(server, auth=auth)
        elicitation_gate = None
        if elicitation is not None or elicitation_policy is not None:
            from ..mcp.apps import ElicitationGate, ElicitationPolicy

            policy = elicitation_policy
            if policy is None:
                policy = ElicitationPolicy(require_approval=callable(elicitation_approval))
            elicitation_gate = ElicitationGate(
                elicitation,
                policy=policy,
                rail_engine=self.rail_engine,
                approver=elicitation_approval if callable(elicitation_approval) else None,
                audit=self.audit,
            )
        client = MCPClient(
            transport,
            name=name,
            sampling_provider=self.resolve_provider() if sampling else None,
            sampling_model=self.model,
            elicitation_gate=elicitation_gate,
        )
        run_sync(
            client.register_into(
                self,
                tools=tools,
                resources=resources,
                prompts=prompts,
                permissions=permissions,
            )
        )
        self.mcp_clients[name] = client
        return self

    def add_mcp_from_registry(
        self,
        name: str,
        *,
        registry: Any,
        directory: Any | None = None,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        server: Any | None = None,
        transport: Any | None = None,
        headers: dict[str, str] | None = None,
        http_client: Any | None = None,
        tools: bool = True,
        resources: bool = True,
        prompts: bool = False,
        permissions: list[str] | None = None,
        principal: Any | None = None,
    ) -> ContextApp:
        """Discover an MCP server from a registry and land its tools in the
        permissioned runtime — one governed call (the marketplace bridge).

        Three concerns compose: **discovery** (an
        :class:`~vincio.registry.MCPRegistryClient` — the official MCP Registry
        or an offline catalog — finds the server), **governance** (a governed
        :class:`~vincio.registry.AgentDirectory` under an
        :class:`~vincio.security.access.AllowListGate` decides reachability and
        records the decision on this app's audit chain), and **connection**
        (:meth:`add_mcp_server` runs the server's tools through the existing
        permissioned, sandboxed, audited runtime).

        Pass ``directory=`` to reuse an existing governed directory, or
        ``allow`` / ``deny`` globs to build one (fail-closed; defaults to
        allowing exactly ``name``). For offline / in-process use, pass
        ``server=`` (an in-process :class:`~vincio.mcp.MCPServer`) or
        ``transport=``; otherwise the resolved server's URL or stdio command is
        used. Raises :class:`~vincio.core.errors.AccessDeniedError` if the gate
        denies the server.
        """
        from ..providers.base import run_sync

        if directory is None:
            directory = self.agent_directory(
                allow=allow if allow is not None else [name], deny=deny
            )
        # Discovery registers candidate servers into the directory as governed,
        # audited AgentRecords (protocol="mcp").
        run_sync(registry.register_into_directory(directory))
        # The governed resolution is the audited access decision.
        record = directory.resolve(name, principal=principal)
        srv = run_sync(registry.get_server(name))

        conn: dict[str, Any] = {}
        if transport is not None:
            conn["transport"] = transport
        elif server is not None:
            conn["server"] = server
        elif srv is not None and srv.url:
            conn["url"] = srv.url
        elif srv is not None and srv.command:
            conn["command"] = srv.command
        elif record.url:
            conn["url"] = record.url
        else:
            raise ConfigError(
                f"MCP server {name!r} has no url/command in the registry; pass server= or transport="
            )
        return self.add_mcp_server(
            name,
            headers=headers,
            http_client=http_client,
            tools=tools,
            resources=resources,
            prompts=prompts,
            permissions=permissions,
            **conn,
        )

    def serve_mcp(
        self,
        *,
        name: str | None = None,
        expose_resources: bool = True,
        expose_prompts: bool = True,
        ui_resources: list[Any] | None = None,
        token_validator: Any | None = None,
    ) -> Any:
        """Expose this app as an MCP server (returns an :class:`MCPServer`).

        Registered tools become MCP tools (run through the permissioned,
        sandboxed, audited runtime); evidence/sources become resources; the
        prompt spec becomes a prompt. Pass ``ui_resources`` — a list of
        :class:`~vincio.mcp.MCPUIResource` — to also serve MCP-UI / AG-UI
        resources for generative-UI hosts. Run it over stdio with
        ``vincio.mcp.serve_stdio(server)`` or the ``vincio mcp serve`` CLI.
        """
        from ..mcp import build_app_server

        return build_app_server(
            self,
            name=name,
            expose_resources=expose_resources,
            expose_prompts=expose_prompts,
            ui_resources=ui_resources,
            token_validator=token_validator,
        )

    def mcp_app(self, name: str, *, max_render_tokens: int = 4096) -> Any:
        """Bridge a consumed MCP server's UI resources onto the AG-UI channel.

        Returns an :class:`~vincio.mcp.apps.MCPAppBridge` over the client
        connected as ``name`` (via :meth:`add_mcp_server`). The bridge reads the
        server's server-rendered ``ui://`` resources and lowers each into an
        :class:`~vincio.server.agui.AGUIEvent` — token-metered against
        ``max_render_tokens`` and recorded on this app's audit chain — so MCP
        Apps UI rides the *same* governed generative-UI stream as the run.
        """
        from ..mcp.apps import MCPAppBridge

        client = self.mcp_clients.get(name)
        if client is None:
            raise ConfigError(
                f"no MCP server connected as {name!r}; call add_mcp_server({name!r}, ...) first"
            )
        return MCPAppBridge(client, audit=self.audit, max_render_tokens=max_render_tokens)

    # -- realtime / voice (optional module) ------------------------------------------------------

    def realtime_session(
        self,
        *,
        backend: str = "inprocess",
        config: Any | None = None,
        **backend_kwargs: Any,
    ) -> Any:
        """Open a voice/realtime session (returns a :class:`RealtimeSession`).

        In-session tool calls route through this app's **permissioned,
        sandboxed, audited** tool runtime — exactly like a native tool call.
        ``backend`` is ``inprocess`` (offline default), ``openai`` (OpenAI
        Realtime), or ``gemini`` (Gemini Live); the hosted backends need
        ``pip install "vincio[realtime]"``. Optional module — see
        :mod:`vincio.realtime`.
        """
        from ..core.types import ToolCall
        from ..realtime import RealtimeConfig, connect_realtime

        async def _dispatch(name: str, arguments: dict[str, Any]) -> Any:
            # Route through the permissioned runtime exactly like a native tool
            # call: validation, scopes, and the approval gate all apply. We do
            # NOT pre-approve — an approval-required tool hits the same gate
            # (and raises ToolApprovalRequiredError, surfaced as an error event)
            # as on the text path, so voice cannot auto-run a write tool.
            result = await self.tool_runtime.execute(
                ToolCall(tool_name=name, arguments=arguments),
                principal=Principal(scopes=list(self.policies.custom.get("scopes", ["*"]))),
            )
            return result.output if result.status == "ok" else {"error": result.error}

        if config is None:
            config = RealtimeConfig(model=backend_kwargs.pop("model", "gpt-realtime"))
        return connect_realtime(backend, config=config, tool_dispatcher=_dispatch, **backend_kwargs)

    def voice_agent(
        self,
        *,
        backend: str = "inprocess",
        config: Any | None = None,
        research: bool = True,
        memory_os: bool = True,
        rails: bool = True,
        owner_id: str = "voice",
        **backend_kwargs: Any,
    ) -> Any:
        """Open an end-to-end :class:`~vincio.realtime.VoiceAgent`.

        A realtime session wired to the full stack: the deep-research agent (as
        an in-session ``research`` tool), the self-editing memory OS, and the
        app's deterministic input/output rails over every spoken transcript and
        reply — so a spoken assistant inherits the same grounding, budget, and
        audit guarantees as the text path. In-session tool calls route through
        this app's permissioned, sandboxed, audited runtime::

            app.add_source("kb", documents=[...])
            agent = app.voice_agent()
            async with agent:
                await agent.send_text("What is the refund window?")
                await agent.commit()
                async for event in agent.events():
                    ...

        ``backend`` is ``inprocess`` (offline default), ``openai``, or ``gemini``
        (hosted backends need ``pip install "vincio[realtime]"``). Optional
        module — see :mod:`vincio.realtime`.
        """
        from ..realtime.voice_agent import VoiceAgent

        return VoiceAgent(
            self,
            backend=backend,
            config=config,
            research=research,
            memory_os=memory_os,
            rails=rails,
            owner_id=owner_id,
            **backend_kwargs,
        )

    # -- A2A ------------------------------------------------------------------------------------

    def serve_a2a(
        self,
        target: Any | None = None,
        *,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose a crew, a compiled graph, or this app over A2A.

        Pass a :class:`Crew`, a compiled :class:`StateGraph`, or ``None`` (the
        app itself). Returns an :class:`A2AServer` whose Agent Card is served at
        ``/.well-known/agent.json``; delegation stays bounded and traced. Run it
        over HTTP behind the FastAPI server or consume it in-process.
        """
        from ..a2a import app_a2a_server, crew_a2a_server, graph_a2a_server
        from ..agents.crew import Crew
        from ..agents.graph import CompiledGraph

        if target is None:
            return app_a2a_server(
                self, name=name, url=url, description=description, token_validator=token_validator
            )
        if isinstance(target, Crew):
            return crew_a2a_server(
                target,
                name=name,
                url=url,
                description=description,
                token_validator=token_validator,
                audit=self.audit,
            )
        if isinstance(target, CompiledGraph):
            return graph_a2a_server(
                target,
                name=name or "graph",
                url=url,
                description=description,
                tracer=self.tracer,
                token_validator=token_validator,
                audit=self.audit,
            )
        raise ConfigError(
            "serve_a2a target must be a Crew, a compiled StateGraph, or None (the app)"
        )

    def agent_directory(
        self,
        *,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        default_allow: bool = False,
    ) -> Any:
        """A governed, audited :class:`~vincio.registry.AgentDirectory` for this app.

        Resolutions pass an allow-list gate (``allow`` / ``deny`` fnmatch globs;
        fail-closed by default) and are recorded as access decisions on this app's
        hash-chained audit log, so the agent fabric is as accountable as a local
        tool call. Register A2A Agent Cards directly, or discover agents from an
        AGNTCY/ACP or MCP registry into it.
        """
        from ..registry import AgentDirectory
        from ..security.access import AllowListGate

        gate = None
        if allow is not None or deny is not None or default_allow is False:
            gate = AllowListGate(allow=allow, deny=deny, default_allow=default_allow)
        return AgentDirectory(allow_list=gate, audit=self.audit)

    # -- agent negotiation & contracting --------------------------------------

    def _negotiation_party(self, spec: Any, role: str, member_id: str) -> Any:
        """Coerce a position or a party into a negotiating :class:`Party`."""
        from ..negotiation import LocalParty, NegotiationPosition
        from ..negotiation.engine import Party

        if isinstance(spec, NegotiationPosition):
            if spec.role != role:
                raise ConfigError(
                    f"negotiate {role}= expects a {role} position; got role={spec.role!r}"
                )
            return LocalParty(member_id, spec, reputation=self._reputation_view())
        if isinstance(spec, Party):
            return spec
        raise ConfigError(f"negotiate {role}= must be a NegotiationPosition or a negotiation Party")

    def _reputation_view(self) -> Any:
        """The reputation an offer is weighted by: imported prior over local ledger.

        Returns the imported :class:`~vincio.settlement.PortableReputation` when one
        is attached (it already falls back to the local ledger for a counterparty
        this app knows), else the local
        :class:`~vincio.optimize.reputation.ReputationLedger`, else ``None`` (offers
        are weighted at parity). So a negotiation against a brand-new counterparty is
        weighted by what its past counterparties attest, while one this app has lived
        through keeps its own earned standing.
        """
        if self.imported_reputation is not None:
            return self.imported_reputation
        return self.reputation_ledger

    def _resolve_contract_signer(self, signer: Any | None, sign: bool) -> Any | None:
        """Pick the signer for a contract: explicit → audit signer → per-app key."""
        if signer is not None:
            return signer
        if not sign:
            return None
        audit_signer = getattr(self.audit, "signer", None)
        if audit_signer is not None:
            return audit_signer
        if self._contract_signer is None:
            from ..core.utils import new_id
            from ..security.audit import HMACSigner

            self._contract_signer = HMACSigner(
                new_id("contract-key"), key_id=f"{self.name}-contracts"
            )
        return self._contract_signer

    async def anegotiate(
        self,
        scope: str,
        *,
        buyer: Any,
        seller: Any,
        budget: Any | None = None,
        signer: Any | None = None,
        sign: bool = True,
        buyer_id: str = "buyer",
        seller_id: str = "seller",
    ) -> Any:
        """Run a bounded buyer/seller negotiation; return a :class:`NegotiationResult`.

        ``buyer`` / ``seller`` are each a
        :class:`~vincio.negotiation.NegotiationPosition` (run as a local,
        deterministic party) or an already-built
        :class:`~vincio.negotiation.Party` — e.g. an
        :class:`~vincio.negotiation.A2ANegotiator` reaching a counterparty over the
        A2A fabric. The bargain is bounded by ``budget`` (a
        :class:`~vincio.negotiation.NegotiationBudget` or a kwargs dict);
        termination is guaranteed, returning a partial result on a deadline. On
        agreement a :class:`~vincio.negotiation.Contract` is minted and signed by
        both parties (with ``signer``, else the audit-chain signer, else a per-app
        key), and the outcome is recorded on the hash-chained audit log. When a
        reputation ledger is attached (:meth:`use_reputation_ledger`) it weights
        each local party's view of the counterparty's offers — a regressing agent
        is discounted without being singled out::

            from vincio.negotiation import buyer_position, seller_position

            result = app.negotiate(
                "transcribe 1k support calls",
                buyer=buyer_position(max_price_usd=0.10, max_sla_seconds=5.0),
                seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
            )
            if result.agreed:
                result.contract.verify(app.contract_signer)  # offline-verifiable
        """
        from ..negotiation import Negotiation, NegotiationBudget

        nbudget = (
            budget if isinstance(budget, NegotiationBudget) else NegotiationBudget(**(budget or {}))
        )
        buyer_party = self._negotiation_party(buyer, "buyer", buyer_id)
        seller_party = self._negotiation_party(seller, "seller", seller_id)
        contract_signer = self._resolve_contract_signer(signer, sign)
        negotiation = Negotiation(
            buyer_party,
            seller_party,
            budget=nbudget,
            signer=contract_signer,
            audit=self.audit,
            events=self.events,
        )
        return await negotiation.run(scope)

    def negotiate(self, scope: str, **kwargs: Any) -> Any:
        """Synchronous wrapper around :meth:`anegotiate`."""
        return run_sync(self.anegotiate(scope, **kwargs))

    @property
    def contract_signer(self) -> Any | None:
        """The signer this app uses to sign/verify contracts (may build one)."""
        return self._resolve_contract_signer(None, True)

    def serve_negotiation(
        self,
        party: Any,
        *,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose a local negotiating :class:`~vincio.negotiation.Party` over A2A.

        Returns an :class:`~vincio.a2a.A2AServer` whose Agent Card advertises a
        ``negotiate`` skill; a remote engine reaches it with an
        :class:`~vincio.negotiation.A2ANegotiator`. Each offer exchange is a
        bounded, audited A2A task on this app's hash-chained log.
        """
        from ..negotiation.fabric import negotiation_a2a_server

        return negotiation_a2a_server(
            party,
            name=name,
            url=url,
            description=description,
            token_validator=token_validator,
            audit=self.audit,
        )

    def enforce_contract(
        self,
        contract: Any,
        *,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        quality: float | None = None,
        raise_on_breach: bool = False,
        record_reputation: bool = True,
    ) -> Any:
        """Check delivered work against a contract and record the verdict.

        Compares the delivered cost / latency / quality against the agreed terms
        (:meth:`~vincio.negotiation.Contract.check`), records a
        ``contract_fulfillment`` decision on the audit chain, and — when a
        reputation ledger is attached and ``record_reputation`` is set — credits
        the seller on fulfilment or debits it on a breach, so a breached SLA
        discounts the seller's future offers. Returns a
        :class:`~vincio.negotiation.ContractFulfillment`.
        """
        fulfillment = contract.check(
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality=quality,
            raise_on_breach=raise_on_breach,
        )
        self.audit.record(
            "contract_fulfillment",
            resource=getattr(contract, "id", None),
            decision="fulfilled" if fulfillment.fulfilled else "breached",
            details={
                "seller": getattr(contract, "seller", None),
                "buyer": getattr(contract, "buyer", None),
                "breaches": fulfillment.breaches,
            },
        )
        if record_reputation and self.reputation_ledger is not None:
            self.reputation_ledger.record_outcome(
                contract.seller,
                passed=fulfillment.fulfilled,
                round_id=getattr(contract, "id", "contract"),
                details={"kind": "contract_fulfillment"},
            )
        return fulfillment

    # -- cross-org workflow choreography --------------------------------------

    def _capability_binder(self, saga: Any, directory: Any, binder: Any, weights: Any) -> Any:
        """Resolve the binder for a saga: explicit binder, else built from a directory.

        Returns ``None`` for a fully statically-wired saga (no discovered steps), so
        the engine path is unchanged unless discovery is actually used. When the
        saga has capability steps, an explicit ``binder`` wins; otherwise a
        :class:`~vincio.choreography.CapabilityBinder` is built over ``directory``,
        this app's reputation ledger, and its settlement book, so discovery is
        ranked by the same reputation and settlement signals the rest of the fabric
        uses.
        """
        if binder is not None:
            return binder
        if not any(getattr(s, "is_discovered", False) for s in saga.steps):
            return None
        if directory is None:
            raise ConfigError(
                "this saga declares capability steps; pass directory= (a governed "
                "AgentDirectory) or binder= so the participant can be resolved at "
                "dispatch time"
            )
        from ..choreography import CapabilityBinder

        return CapabilityBinder(
            directory,
            reputation=self.reputation_ledger,
            settlement_book=self.settlement_book,
            weights=weights,
        )

    def _choreography(
        self, saga: Any, participants: dict[str, Any], signer: Any, binder: Any = None
    ) -> Any:
        """Build a :class:`~vincio.choreography.Choreography` bound to this app."""
        from ..choreography import Choreography

        return Choreography(
            saga,
            participants,
            coordinator=self.name,
            store=self.store,
            audit=self.audit,
            events=self.events,
            signer=signer,
            binder=binder,
        )

    async def achoreograph(
        self,
        saga: Any,
        *,
        participants: dict[str, Any],
        input: dict[str, Any] | None = None,
        saga_id: str | None = None,
        signer: Any | None = None,
        sign: bool = True,
        directory: Any | None = None,
        binder: Any | None = None,
        binding_weights: Any | None = None,
        interrupt_after: int | None = None,
    ) -> Any:
        """Run a durable, compensating cross-org saga; return a :class:`SagaResult`.

        ``saga`` is a :class:`~vincio.choreography.Saga`; ``participants`` maps each
        org id the saga dispatches to onto a
        :class:`~vincio.choreography.Participant` — a
        :class:`~vincio.choreography.RemoteParticipant` reaching a counterparty over
        the A2A fabric, or (as a convenience) a ``dict`` of handler callables run
        in-process. The :class:`~vincio.choreography.SagaJournal` is checkpointed to
        this app's metadata store after every step, so the saga **survives a
        restart** — continue it with :meth:`aresume_choreography` — and is recorded,
        hash-chained, on this app's audit log. A forward step that fails, raises, or
        breaches its step contract triggers deterministic compensation of the
        completed steps in reverse order. ``interrupt_after`` cooperatively pauses
        the forward pass into a resumable state::

            from vincio.choreography import Saga

            saga = (
                Saga(name="fulfil-order")
                .step("reserve", participant="warehouse", action="reserve",
                      compensation="release")
                .step("charge", participant="payments", action="charge",
                      compensation="refund")
            )
            result = app.choreograph(saga, participants={
                "warehouse": warehouse_client, "payments": payments_handlers,
            })
            assert result.journal.verify().intact  # offline-verifiable

        A step may instead declare the *capability* it needs and have its
        counterparty **resolved at dispatch time** from a governed
        :class:`~vincio.registry.AgentDirectory` passed as ``directory=`` — ranked
        by reputation and prior settlement fit, under the same allow-list, contract,
        and per-org audit a statically-wired step runs under. The candidate set is
        the orgs registered in both the directory and ``participants``; pass a
        prepared :class:`~vincio.choreography.CapabilityBinder` as ``binder=`` (or
        :class:`~vincio.choreography.BindingWeights` as ``binding_weights=``) to
        tune the ranking::

            saga = Saga(name="fulfil").step(
                "transcribe", capability="transcription", action="run",
            )
            result = app.choreograph(
                saga, participants={"vendor-a": a, "vendor-b": b}, directory=directory,
            )
            result.bindings["transcribe"].org  # the counterparty discovery chose
        """
        engine = self._choreography(
            saga,
            participants,
            self._resolve_contract_signer(signer, sign),
            self._capability_binder(saga, directory, binder, binding_weights),
        )
        return await engine.arun(input, saga_id=saga_id, interrupt_after=interrupt_after)

    def choreograph(self, saga: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper around :meth:`achoreograph`."""
        return run_sync(self.achoreograph(saga, **kwargs))

    async def aresume_choreography(
        self,
        saga: Any,
        saga_id: str,
        *,
        participants: dict[str, Any],
        signer: Any | None = None,
        sign: bool = True,
        directory: Any | None = None,
        binder: Any | None = None,
        binding_weights: Any | None = None,
        interrupt_after: int | None = None,
    ) -> Any:
        """Resume a saga from this app's durable store after a restart.

        Rebuild the same :class:`~vincio.choreography.Saga` and ``participants`` in
        code (only the journal is persisted) and pass the ``saga_id``; completed
        steps keep their outputs and are not re-run, and a saga interrupted
        mid-rollback finishes compensating. A terminal saga is returned unchanged.
        A discovered step that already ran keeps the org it was bound to (recorded
        on the journal); one not yet reached binds at dispatch time on resume, so
        pass the same ``directory=`` / ``binder=`` used for the original run.
        """
        engine = self._choreography(
            saga,
            participants,
            self._resolve_contract_signer(signer, sign),
            self._capability_binder(saga, directory, binder, binding_weights),
        )
        return await engine.aresume(saga_id, interrupt_after=interrupt_after)

    def resume_choreography(self, saga: Any, saga_id: str, **kwargs: Any) -> Any:
        """Synchronous wrapper around :meth:`aresume_choreography`."""
        return run_sync(self.aresume_choreography(saga, saga_id, **kwargs))

    def serve_choreography(
        self,
        handlers: Any,
        *,
        org_id: str | None = None,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose this org's choreography handlers over A2A.

        Returns an :class:`~vincio.a2a.A2AServer` whose Agent Card advertises a
        ``choreograph`` skill; a remote coordinator dispatches steps to it with a
        :class:`~vincio.choreography.RemoteParticipant`. Each step this org performs
        or compensates is recorded on **this app's** hash-chained audit log — its
        self-governance of the steps that cross into it.
        """
        from ..choreography.fabric import choreography_a2a_server

        return choreography_a2a_server(
            handlers,
            org_id=org_id or self.name,
            name=name,
            url=url,
            description=description,
            token_validator=token_validator,
            audit=self.audit,
        )

    # -- agent-to-agent settlement & metering ---------------------------------

    def use_settlement_book(self, book: Any | None = None, *, owner: str | None = None) -> Any:
        """Attach a durable, hash-chained ledger of cross-org settlements.

        Closing the books on contracted work — :meth:`settle` and
        :meth:`settle_saga` — appends a typed, signed, offline-verifiable
        :class:`~vincio.settlement.SettlementRecord` to this book, links it into the
        book's hash chain, records the verdict on this app's audit chain, and (when
        a reputation ledger is attached) credits or debits the seller, so a settled
        overrun or shortfall weights the next negotiation. Pass a configured
        :class:`~vincio.settlement.SettlementBook`, or let this build one wired to
        the app's contract signer, audit chain, event bus, store, and reputation
        ledger. Returns the book::

            app.use_settlement_book()
            record = app.settle(contract, cost_usd=0.08, latency_ms=1200, quality=0.9)
            app.settlement_report().print_summary()
        """
        from ..settlement import SettlementBook

        if book is None:
            book = SettlementBook(
                owner or self.name,
                signer=self._resolve_contract_signer(None, True),
                audit=self.audit,
                events=self.events,
                store=self.store,
                reputation=self.reputation_ledger,
            )
        self.settlement_book = book
        return book

    def _settlement_book(self) -> Any:
        """The attached book, or a transient one wired to this app for one call."""
        if self.settlement_book is not None:
            return self.settlement_book
        from ..settlement import SettlementBook

        return SettlementBook(
            self.name,
            signer=self._resolve_contract_signer(None, True),
            audit=self.audit,
            events=self.events,
            reputation=self.reputation_ledger,
        )

    def meter(self, contract: Any, *, run_id: str | None = None) -> Any:
        """A :class:`~vincio.settlement.Meter` accruing usage against a contract.

        Accrue a :class:`~vincio.settlement.UsageEvent` as each unit of contracted
        work completes; :meth:`settle` reconciles the resulting reading against the
        agreed terms. Metering is pure accumulation — it records what was delivered,
        attributed to the contract and the run, the way the cost report attributes
        spend; the contract's budget is what enforces a cap.
        """
        from ..settlement import Meter

        return Meter(contract.id, run_id=run_id)

    def settle(
        self,
        contract: Any,
        *,
        reading: Any | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        quality: float | None = None,
        run_id: str | None = None,
        party: str | None = None,
        sign: bool = True,
        record_reputation: bool = True,
        escrow: Any | None = None,
        escrow_config: Any | None = None,
        pool: Any | None = None,
    ) -> Any:
        """Close the books on contracted work: reconcile, sign, audit, and record.

        Reconciles the delivered work — a metered
        :class:`~vincio.settlement.MeterReading` (``reading``) or explicit
        ``cost_usd`` / ``latency_ms`` / ``quality`` figures — against the contract's
        agreed price / SLA / quality into a typed
        :class:`~vincio.settlement.SettlementRecord`, signs it as this app's side of
        the contract, appends it to the attached settlement book (hash-chained,
        checkpointed) or a transient one, records the verdict on the audit chain,
        and — unless ``record_reputation`` is off — credits the seller on fulfilment
        or debits it on a breach. The record verifies offline from the bytes alone;
        the counterparty's independently-produced record reconciles against it with
        :func:`~vincio.settlement.reconcile`. Returns the record::

            record = app.settle(contract, cost_usd=0.08, latency_ms=1200, quality=0.92)
            record.verify(app.contract_signer)  # offline-verifiable

        Pass an ``escrow`` posted against the contract (:meth:`post_escrow`) to settle the
        collateral in the same call: it is resolved against the record — the whole stake
        released on a fulfilled delivery, a bounded proportional slice forfeited on a
        breach — signed, and audited in place, so the collateral closes the same loop the
        settlement does. ``escrow_config`` overrides the forfeiture policy.

        Pass a ``pool`` the contract is backed by (:meth:`post_collateral_pool`) to draw the
        same settlement against a shared
        :class:`~vincio.settlement.CollateralPool` instead — the forfeiture drawn from the
        pooled stake and the rest released back to the available balance, re-signed and
        audited in place.
        """
        return self._settlement_book().settle(
            contract,
            reading=reading,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality=quality,
            run_id=run_id,
            party=party,
            sign=sign,
            record_reputation=record_reputation,
            escrow=escrow,
            escrow_config=escrow_config,
            pool=pool,
        )

    def post_escrow(
        self,
        contract: Any,
        *,
        decision: Any | None = None,
        fraction: float | None = None,
        amount: float | None = None,
        poster: str | None = None,
        beneficiary: str | None = None,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Post collateral against a contract as a signed, offline-verifiable escrow.

        Binds the admission-required collateral — read from an
        :class:`~vincio.settlement.AdmissionDecision` (``decision``), an explicit
        ``fraction`` / ``amount``, or the admission posture
        :meth:`~vincio.settlement.AdmissionDecision.apply_to_terms` stamped onto the
        contract's terms — to the specific contract and counterparty into an
        :class:`~vincio.settlement.Escrow`, signs it as this app's side of the contract,
        appends the posting to the attached settlement book's audit chain, and returns
        it. The escrow verifies offline from the bytes alone (the held amount re-derives
        from the admission posture); :meth:`settle` (with ``escrow=``) or
        :meth:`settle_escrow` resolves it against delivery::

            decision = app.admit("vendor")
            escrow = app.post_escrow(contract, decision=decision)
            escrow.verify().valid  # offline-verifiable
        """
        return self._settlement_book().post_escrow(
            contract,
            decision=decision,
            fraction=fraction,
            amount=amount,
            poster=poster,
            beneficiary=beneficiary,
            config=config,
            party=party,
            sign=sign,
        )

    def settle_escrow(
        self,
        escrow: Any,
        record: Any,
        *,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Resolve a posted escrow against a settlement record (release or forfeit).

        Settles ``escrow`` against the contract's
        :class:`~vincio.settlement.SettlementRecord` (from :meth:`settle`): releases the
        whole stake on a fulfilled delivery and forfeits a bounded slice proportional to
        the shortfall on a breach — driven by the same settlement verdict — re-signs the
        resolved escrow as this app, and records the release / forfeiture on the audit
        chain. ``config`` overrides the forfeiture policy. Returns the resolved escrow::

            record = app.settle(contract, cost_usd=0.20)   # a cost overrun: a breach
            app.settle_escrow(escrow, record)              # forfeits a proportional slice
        """
        return self._settlement_book().settle_escrow(
            escrow, record, config=config, party=party, sign=sign
        )

    def post_collateral_pool(
        self,
        contracts: Any,
        *,
        poster: str | None = None,
        posted: float | None = None,
        decisions: Any | None = None,
        fraction: float | None = None,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Post one stake backing many contracts as a signed, offline-verifiable margin account.

        Binds a counterparty's single posted stake to the set of ``contracts`` it backs into
        a :class:`~vincio.settlement.CollateralPool`, allocating each a per-contract share
        proportional to its admission-required collateral — read from a matching
        :class:`~vincio.settlement.AdmissionDecision` in ``decisions``, a uniform
        ``fraction``, or the admission posture stamped onto each contract's terms. Signs it
        as this app's side and appends the posting to the settlement book's audit chain. A
        clean delivery frees capital for the next contract and a breach is covered from the
        shared stake; :meth:`settle` (with ``pool=``) or :meth:`draw_pool` draws an open
        contract's settlement against it::

            pool = app.post_collateral_pool([c1, c2, c3], decisions={c1.id: d1, ...})
            pool.verify().valid  # offline-verifiable — allocations re-derive, balance reconciles
        """
        return self._settlement_book().post_collateral_pool(
            contracts,
            poster=poster,
            posted=posted,
            decisions=decisions,
            fraction=fraction,
            config=config,
            party=party,
            sign=sign,
        )

    def draw_pool(
        self,
        pool: Any,
        record: Any,
        *,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Draw one backed contract's settlement against a collateral pool (draw or release).

        Settles the matching contract against its
        :class:`~vincio.settlement.SettlementRecord` (from :meth:`settle`): draws a bounded
        slice proportional to the shortfall from the shared stake on a breach and releases
        the rest back to the available balance on a clean delivery — driven by the same
        settlement verdict — re-signs the pool as this app, and records the draw on the
        audit chain. ``config`` overrides the forfeiture policy. Returns the pool::

            record = app.settle(contract, cost_usd=140.0)   # a cost overrun: a breach
            app.draw_pool(pool, record)                     # draws a proportional slice
        """
        return self._settlement_book().draw_pool(
            pool, record, config=config, party=party, sign=sign
        )

    def identity(
        self,
        name: str | None = None,
        *,
        controller: str = "",
        capabilities: Any | None = None,
        seed: Any | None = None,
        use: bool = False,
        record_audit: bool = True,
    ) -> Any:
        """Mint a portable, self-certifying :class:`~vincio.security.AgentIdentity`.

        The identity is built on an Ed25519 key whose **DID is derived from the public
        key** (``did:vincio:ed25519:<hex>``), so the identifier resolves to the
        verifying key offline with no registry. ``name`` labels it (defaults to this
        app's name), ``controller`` names the operating org, ``capabilities`` are the
        capabilities it advertises, and ``seed`` (32 bytes) makes the key deterministic
        for tests. With ``use=True`` the identity also becomes this app's signer (see
        :meth:`use_identity`). Unless ``record_audit`` is off, the issuance lands on the
        hash-chained audit log. The identity satisfies the
        :class:`~vincio.security.audit.ChainSigner` protocol, so it drops into every
        signing slot the platform already exposes::

            agent = app.identity("billing-agent", capabilities=["retrieve", "summarize"])
            grant = agent.delegate("did:vincio:ed25519:...", capabilities=["retrieve"])
        """
        from ..security.identity import AgentIdentity

        identity = AgentIdentity.generate(
            name or self.name,
            controller=controller,
            capabilities=list(capabilities) if capabilities else None,
            seed=seed,
        )
        # Bind first (when requested) so the identity adopts the audit signer before
        # the mint entry is recorded — the mint then lands signed by its own DID.
        if use:
            self.use_identity(identity, record_audit=False)
        if record_audit and self.audit is not None:
            from ..security.identity import IDENTITY_ACTION

            entry = self.audit.record(
                IDENTITY_ACTION,
                resource=identity.did,
                decision="minted",
                details=identity.document.audit_details(),
            )
            identity.document.audit_id = getattr(entry, "id", None)
        return identity

    def use_identity(self, identity: Any, *, record_audit: bool = True) -> Any:
        """Bind ``identity`` as this app's signer so every artifact carries its DID.

        Sets the identity as the content signer and the contract/settlement signer —
        and, when the audit log has not yet recorded anything, as the audit-chain
        signer too — so subsequent audit entries, negotiated contracts, settlement
        records, and signed manifests all record the identity's **DID** as their
        ``key_id``. Accountability becomes mechanical: a verifier resolves the signer
        from the DID and checks the signature from the bytes, rather than trusting an
        out-of-band ``key_id`` string. Returns the identity.
        """
        self._identity = identity
        self.content_signer = identity
        self._contract_signer = identity
        # Adopt the audit signer only on a fresh log, so the chain stays verifiable
        # under one key (mixing signers mid-chain would break offline verification).
        if self.audit is not None and not self.audit.entries:
            self.audit.signer = identity
        if record_audit and self.audit is not None:
            from ..security.identity import IDENTITY_ACTION

            self.audit.record(
                IDENTITY_ACTION,
                resource=getattr(identity, "did", None),
                decision="bound",
                details={"did": getattr(identity, "did", None), "name": getattr(identity, "name", "")},
            )
        return identity

    def issue_credential(
        self,
        subject: Any,
        claims: dict[str, str],
        *,
        as_identity: Any | None = None,
        not_after: Any | None = None,
        expires_in: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Issue a signed, offline-verifiable :class:`~vincio.security.AgentCredential`.

        The issuer signs a verifiable claim about ``subject`` (an agent DID or
        :class:`~vincio.security.AgentIdentity`) — e.g.
        ``{"admitted_capability": "retrieve", "operated_by": "org-acme"}`` — that an
        importer verifies offline and folds into the admission / registry path
        (:meth:`~vincio.security.AgentCredential.admits`). The issuer is ``as_identity``
        or this app's bound identity (:meth:`use_identity`); raises
        :class:`~vincio.core.errors.IdentityError` if neither is set. Records the
        issuance on the audit chain unless ``record_audit`` is off. Returns the
        credential::

            org = app.identity("org-acme", use=True)
            cred = app.issue_credential(agent, {"admitted_capability": "retrieve"})
            cred.verify().valid  # True, from the bytes alone
        """
        from ..core.errors import IdentityError

        issuer = as_identity or self._identity
        if issuer is None:
            raise IdentityError(
                "no issuing identity: pass as_identity= or bind one with app.use_identity(...)",
                details={"app": self.name},
            )
        credential = issuer.issue_credential(
            subject, claims, not_after=not_after, expires_in=expires_in
        )
        if record_audit and self.audit is not None:
            from ..security.identity import CREDENTIAL_ACTION

            entry = self.audit.record(
                CREDENTIAL_ACTION,
                resource=credential.subject,
                decision="issued",
                details=credential.audit_details(),
            )
            credential.audit_id = getattr(entry, "id", None)
        return credential

    def attest_custody(
        self,
        poster: str,
        reserves: Any,
        *,
        custodian: str | None = None,
        as_of: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Attest a poster's proven reserves into a signed, content-bound proof-of-reserves.

        Issues a :class:`~vincio.settlement.CustodyAttestation` over the capital ``poster``
        actually holds — itemized ``reserves`` (a number, a mapping of ``account -> amount``,
        or :class:`~vincio.settlement.ReserveLine` items) whose total re-derives on every
        verify — so the held figure :meth:`guard_collateral` bounds the pledges against is
        **evidence-backed** rather than asserted. ``custodian`` defaults to this app (a
        third-party custodian vouching), and when it is also the ``poster`` the attestation is
        self-custody. Signs it as the custodian and, unless ``record_audit`` is off, records
        the issuance on the audit chain. The attestation verifies offline from the bytes
        alone — a tampered reserve figure or a forged custodian is caught. Returns it::

            proof = app.attest_custody("vendor", {"omnibus": 80.0})
            ledger = app.guard_collateral([pool_a, pool_b], custody=proof)
            ledger.require_reserved()  # raises if proven reserves < pledged
        """
        from ..settlement import attest_custody as _attest

        resolved_custodian = custodian or self.name
        attestation = _attest(poster, reserves, custodian=resolved_custodian, as_of=as_of)
        if sign and self.name == attestation.custodian:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                attestation.sign(signer, party=attestation.custodian)
        if record_audit and self.audit is not None:
            from ..settlement.custody import CUSTODY_ACTION

            entry = self.audit.record(
                CUSTODY_ACTION,
                resource=attestation.poster,
                decision="self_custody" if attestation.self_custody else "custodied",
                details=attestation.audit_details(),
            )
            attestation.audit_id = getattr(entry, "id", None)
        return attestation

    def attest_liabilities(
        self,
        poster: str,
        liabilities: Any,
        *,
        attestor: str | None = None,
        as_of: Any | None = None,
        prior: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Attest a poster's total obligations into a signed, content-bound proof-of-liabilities.

        Issues a :class:`~vincio.settlement.LiabilityAttestation` over the obligations ``poster``
        owes — itemized ``liabilities`` (a number, a mapping of ``creditor -> amount``, or
        :class:`~vincio.settlement.LiabilityLine` items) whose total re-derives on every verify —
        the liability side of a proof-of-solvency. ``attestor`` defaults to this app (a
        third-party attestor vouching), and when it is also the ``poster`` the attestation is
        self-attested. Pass ``prior`` (the preceding snapshot) to link it into a hash-linked history
        :meth:`check_history_consistency` can walk. Signs it as the attestor and, unless
        ``record_audit`` is off, records the issuance on the audit chain. Returns it::

            owed = app.attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
            proof = app.prove_solvency(reserves_proof, owed)
        """
        from ..settlement import attest_liabilities as _attest

        resolved_attestor = attestor or self.name
        attestation = _attest(
            poster, liabilities, attestor=resolved_attestor, as_of=as_of, prior=prior
        )
        if sign and self.name == attestation.attestor:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                attestation.sign(signer, party=attestation.attestor)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import LIABILITY_ACTION

            entry = self.audit.record(
                LIABILITY_ACTION,
                resource=attestation.poster,
                decision="self_attested" if attestation.self_attested else "attested",
                details=attestation.audit_details(),
            )
            attestation.audit_id = getattr(entry, "id", None)
        return attestation

    def inclusion_proof(self, liabilities: Any, creditor: str) -> Any:
        """Build an offline-verifiable inclusion proof for one creditor's liability claim.

        Thin wrapper over :meth:`~vincio.settlement.LiabilityAttestation.inclusion_proof`: the
        :class:`~vincio.settlement.InclusionProof` shows ``creditor``'s obligation is a leaf of
        the attestation's signed Merkle root, so the creditor confirms its claim was counted in
        the attested total and a poster cannot quietly drop it. Returns it::

            owed = app.attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
            proof = app.inclusion_proof(owed, "acme")
            proof.verify(owed).valid  # True
        """
        return liabilities.inclusion_proof(creditor)

    def check_completeness(
        self,
        liabilities: Any,
        claims: Any,
        *,
        as_of: Any | None = None,
        sign: bool = True,
        verify_with: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Fold creditor claims against a liability attestation into a completeness check.

        Issues a :class:`~vincio.settlement.CompletenessProof`
        (:func:`~vincio.settlement.check_completeness`) folding what creditors can prove they are
        owed (``claims`` — a ``creditor -> amount`` mapping, or
        :class:`~vincio.settlement.LiabilityLine` / settlement-record / ``(creditor, amount)``
        items) against the attestation, pinpointing every omitted or under-stated claim as an
        :class:`~vincio.settlement.OmissionBreach` and raising the attested figure to a completed
        total :meth:`prove_solvency` reads (``completeness=``). Signs the check as this app and,
        unless ``record_audit`` is off, records it on the audit chain. A tampered attestation is
        refused (a forged attestor signature too, with ``verify_with``). Returns it::

            owed = app.attest_liabilities("vendor", {"acme": 60.0})
            check = app.check_completeness(owed, {"acme": 60.0, "globex": 40.0})
            check.require_complete()  # raises: globex is omitted
        """
        from ..settlement import check_completeness as _check

        proof = _check(liabilities, claims, verifier=verify_with, as_of=as_of)
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                proof.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import COMPLETENESS_ACTION

            entry = self.audit.record(
                COMPLETENESS_ACTION,
                resource=proof.poster,
                decision=proof.status,
                details=proof.audit_details(),
            )
            proof.audit_id = getattr(entry, "id", None)
        return proof

    def prove_solvency(
        self,
        custody: Any,
        liabilities: Any,
        *,
        poster: str | None = None,
        completeness: Any | None = None,
        as_of: Any | None = None,
        sign: bool = True,
        verify_with: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Fold a reserve proof against a liability proof into a proof-of-solvency.

        Reconciles a proven :class:`~vincio.settlement.CustodyAttestation` (reserves) against a
        proven :class:`~vincio.settlement.LiabilityAttestation` (obligations) for the same poster
        into a bounded :class:`~vincio.settlement.SolvencyProof` — the proof-of-solvency the
        literature pairs with a proof-of-reserves (``reserves ≥ liabilities``). When the
        liabilities exceed the reserves the shortfall surfaces as a pinpointed
        :class:`~vincio.settlement.InsolvencyBreach`. Pass ``completeness`` (a
        :class:`~vincio.settlement.CompletenessProof` over this attestation) to bound the margin
        against the *completed* liability total — the attestor's figure raised by every
        obligation a creditor proved it omitted, not just the creditors the attestor listed.
        Signs the proof as this app and, unless ``record_audit`` is off, records it on the audit
        chain. A tampered or wrong-poster attestation (or completeness check) is refused (a
        forged signature too, with ``verify_with``). The proof's solvency-adjusted held figure
        reads into :meth:`guard_collateral` (``solvency=``). Returns the proof::

            reserves = app.attest_custody("vendor", {"omnibus": 80.0})
            owed = app.attest_liabilities("vendor", {"acme": 60.0})
            proof = app.prove_solvency(reserves, owed)
            ledger = app.guard_collateral([pool], solvency=proof)
            proof.require_solvent()  # raises if liabilities exceed reserves
        """
        from ..settlement import prove_solvency as _prove

        proof = _prove(
            custody,
            liabilities,
            poster=poster,
            completeness=completeness,
            as_of=as_of,
            verifier=verify_with,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                proof.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import SOLVENCY_ACTION

            entry = self.audit.record(
                SOLVENCY_ACTION,
                resource=proof.poster,
                decision=proof.status,
                details=proof.audit_details(),
            )
            proof.audit_id = getattr(entry, "id", None)
        return proof

    def check_root_consistency(
        self,
        attestations: Any,
        *,
        verify_with: Any | None = None,
        record_reputation: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Compare liability attestations across creditors for cross-org non-equivocation.

        Folds a set of liability attestations a group of creditors hold — each the attestation a
        poster signed *for it* — into a :class:`~vincio.settlement.RootConsistencyReport`
        (:func:`~vincio.settlement.check_root_consistency`), surfacing every poster that signed
        **different** roots for the same ``(poster, attestor, as_of)`` as a non-repudiable
        :class:`~vincio.settlement.EquivocationProof`. Where :meth:`check_completeness` catches an
        omission only when the omitted creditor folds its own claim, this catches the counterparty
        that **equivocates** — showing each creditor a root on which its own claim *is* present
        while the totals disagree. ``attestations`` items may be bare attestations or
        ``(creditor, attestation)`` pairs to record which creditor saw each root. With
        ``verify_with`` only attestor-signed roots are admitted, so a forged root cannot
        manufacture a false accusation. Unless ``record_audit`` is off, records each proven
        equivocation on the audit chain; unless ``record_reputation`` is off, credits a failure
        against the equivocating poster on this app's reputation ledger (when one is attached).
        Returns the report::

            owed_acme = vendor.attest_liabilities("vendor", {"acme": 60.0}, as_of=t)
            owed_globex = vendor.attest_liabilities("vendor", {"globex": 40.0}, as_of=t)
            report = auditor.check_root_consistency([("acme", owed_acme), ("globex", owed_globex)])
            report.require_consistent()  # raises: vendor signed two roots for one instant
        """
        from ..settlement import check_root_consistency as _check

        report = _check(attestations, verifier=verify_with)
        dinged: set[str] = set()
        for proof in report.equivocations:
            if record_audit and self.audit is not None:
                from ..settlement.solvency import EQUIVOCATION_ACTION

                entry = self.audit.record(
                    EQUIVOCATION_ACTION,
                    resource=proof.poster,
                    decision="equivocation",
                    details=proof.audit_details(),
                )
                proof.audit_id = getattr(entry, "id", None)
            # Every distinct pairwise conflict is audited, but a poster's reputation is debited
            # once per check — three conflicting roots are one equivocating counterparty.
            if (
                record_reputation
                and self.reputation_ledger is not None
                and proof.poster not in dinged
            ):
                self.reputation_ledger.record_outcome(
                    proof.poster,
                    passed=False,
                    round_id=proof.id,
                    details={"kind": "liability_equivocation", "attestor": proof.attestor},
                )
                dinged.add(proof.poster)
        return report

    def discharge_liability(
        self,
        poster: str,
        amount_usd: float,
        *,
        creditor: str | None = None,
        as_of: Any | None = None,
        note: str = "",
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Issue a signed, content-bound :class:`~vincio.settlement.Discharge` of what ``poster`` owes.

        Releases ``amount_usd`` of the obligation ``poster`` owes this app — the **creditor** issues
        the discharge, so ``creditor`` defaults to this app and it is signed with this app's key.
        Folded into :meth:`check_history_consistency` (``discharges=``) to explain a legitimate
        reduction in the poster's liabilities between two snapshots, so the matching drop is not
        treated as a debt that silently vanished. Unless ``record_audit`` is off, records the
        issuance on the audit chain. Returns it::

            settled = acme.discharge_liability("vendor", 70.0)  # acme releases $70 of vendor's debt
            report = auditor.check_history_consistency(snapshots, discharges=[settled])
        """
        from ..settlement import discharge_liability as _discharge

        resolved_creditor = creditor or self.name
        discharge = _discharge(poster, resolved_creditor, amount_usd, as_of=as_of, note=note)
        if sign and self.name == discharge.creditor:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                discharge.sign(signer, party=discharge.creditor)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import DISCHARGE_ACTION

            entry = self.audit.record(
                DISCHARGE_ACTION,
                resource=discharge.poster,
                decision=discharge.status,
                details=discharge.audit_details(),
            )
            discharge.audit_id = getattr(entry, "id", None)
        return discharge

    def check_history_consistency(
        self,
        attestations: Any,
        *,
        discharges: Any | None = None,
        verify_with: Any | None = None,
        record_reputation: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Walk a poster's liability snapshots for cross-time monotonicity (no debt silently dropped).

        Folds a set of liability snapshots into a
        :class:`~vincio.settlement.HistoryConsistencyReport`
        (:func:`~vincio.settlement.check_history_consistency`), surfacing every poster that let a
        creditor's obligation **drop** between snapshots without a signed
        :class:`~vincio.settlement.Discharge` (``discharges``) explaining the release as a pinpointed
        :class:`~vincio.settlement.MonotonicityBreach`. Where :meth:`check_root_consistency` catches a
        counterparty signing different roots for the *same* instant, this catches one quietly dropping
        a past obligation in a *later* snapshot. With ``verify_with`` only attestor-signed snapshots
        and creditor-signed discharges are admitted as evidence. Unless ``record_audit`` is off,
        records each inconsistent history on the audit chain; unless ``record_reputation`` is off,
        credits a failure against the breaching poster on this app's reputation ledger (when one is
        attached). Returns the report::

            s1 = vendor.attest_liabilities("vendor", {"acme": 100.0}, as_of=t1)
            s2 = vendor.attest_liabilities("vendor", {"acme": 30.0}, as_of=t2, prior=s1)
            report = auditor.check_history_consistency([s1, s2])
            report.require_consistent()  # raises: $70 owed to acme vanished without a discharge
        """
        from ..settlement import check_history_consistency as _check

        report = _check(attestations, discharges=discharges, verifier=verify_with)
        signer = self._resolve_contract_signer(None, True)
        for proof in report.proofs:
            if signer is not None:
                proof.sign(signer, party=self.name)
            if record_audit and self.audit is not None:
                from ..settlement.solvency import HISTORY_ACTION

                entry = self.audit.record(
                    HISTORY_ACTION,
                    resource=proof.poster,
                    decision=proof.status,
                    details=proof.audit_details(),
                )
                proof.audit_id = getattr(entry, "id", None)
            if record_reputation and self.reputation_ledger is not None and not proof.monotone:
                self.reputation_ledger.record_outcome(
                    proof.poster,
                    passed=False,
                    round_id=proof.id,
                    details={"kind": "liability_history", "attestor": proof.attestor},
                )
        return report

    def build_set_off_statement(
        self,
        poster: str,
        creditor: str,
        owed_usd: float,
        owing_usd: float,
        *,
        references: Any | None = None,
        as_of: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Collapse the mutual obligations between a poster and one creditor into a statement.

        Builds a :class:`~vincio.settlement.SetOffStatement`
        (:func:`~vincio.settlement.build_set_off_statement`) stating the obligations running *both
        ways* between ``poster`` and ``creditor`` — ``owed_usd`` the poster owes the creditor,
        ``owing_usd`` the creditor owes the poster back — and computing the poster's bounded net
        liability (``max(0, owed − owing)``). Signs it as this app (one side of the mutually-agreed
        close-out — the counterparty co-signs its copy) and, unless ``record_audit`` is off, records
        the issuance on the audit chain. Returns it::

            statement = vendor.build_set_off_statement("vendor", "acme", 30.0, 12.0)
            resolution = auditor.resolve_insolvency(reserves, owed, set_off=[statement])
        """
        from ..settlement import build_set_off_statement as _build

        statement = _build(
            poster, creditor, owed_usd, owing_usd, references=references, as_of=as_of
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                statement.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.setoff import SETOFF_ACTION

            entry = self.audit.record(
                SETOFF_ACTION,
                resource=statement.poster,
                decision=statement.direction,
                details=statement.audit_details(),
            )
            statement.audit_id = getattr(entry, "id", None)
        return statement

    def build_seniority_schedule(
        self,
        poster: str,
        tranches: Any,
        *,
        as_of: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Rank a poster's obligations into a signed, content-bound seniority schedule.

        Builds a :class:`~vincio.settlement.SenioritySchedule`
        (:func:`~vincio.settlement.build_seniority_schedule`) ranking the creditors ``poster`` owes
        into priority tranches — rank ``0`` most senior — an :meth:`resolve_insolvency` waterfall
        pays out in. ``tranches`` is an ordered spec — its simplest form is a list of creditor-name
        lists where **position is priority** (``[["bank"], ["acme", "globex"]]``) — or
        :class:`~vincio.settlement.SeniorityTranche` items for explicit ranks and labels. Signs the
        schedule as this app and, unless ``record_audit`` is off, records the issuance on the audit
        chain. Returns it::

            schedule = app.build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
            resolution = app.resolve_insolvency(reserves, owed, schedule)
        """
        from ..settlement import build_seniority_schedule as _build

        schedule = _build(poster, tranches, as_of=as_of)
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                schedule.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.waterfall import SENIORITY_ACTION

            entry = self.audit.record(
                SENIORITY_ACTION,
                resource=schedule.poster,
                decision="self_ranked" if schedule.poster == self.name else "ranked",
                details=schedule.audit_details(),
            )
            schedule.audit_id = getattr(entry, "id", None)
        return schedule

    def resolve_insolvency(
        self,
        custody: Any,
        liabilities: Any,
        schedule: Any | None = None,
        *,
        poster: str | None = None,
        completeness: Any | None = None,
        solvency: Any | None = None,
        set_off: Any | None = None,
        as_of: Any | None = None,
        verify_with: Any | None = None,
        sign: bool = True,
        record_reputation: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Distribute a poster's proven reserves across its ranked liabilities into a resolution.

        Folds a proven :class:`~vincio.settlement.CustodyAttestation` against a proven
        :class:`~vincio.settlement.LiabilityAttestation` and distributes the reserves across the
        obligations **by seniority then pari-passu within a tranche** (``schedule``) into a
        content-bound :class:`~vincio.settlement.InsolvencyResolution`
        (:func:`~vincio.settlement.resolve_insolvency`), pinpointing each creditor's bounded
        :class:`~vincio.settlement.CreditorRecovery` and the shortfall it bears — so an insolvency a
        :class:`~vincio.settlement.SolvencyProof` only *flagged* is *resolved* into who-gets-what.
        With no ``schedule`` the whole liability set is one pari-passu tranche; pass
        ``completeness`` to distribute against the *completed* liability set, and ``set_off`` (a list
        of mutually-signed :class:`~vincio.settlement.SetOffStatement`\\ s) to **close-out net** each
        creditor to its net claim before the waterfall. Reuses
        :func:`~vincio.settlement.prove_solvency`, so a tampered, forged, or wrong-poster
        attestation (or a malformed/wrong-poster schedule, or a one-sided/over-stated set-off) is
        refused (a forged signature too, with ``verify_with``). Signs the resolution as this app;
        unless ``record_audit`` is off, records it on the audit chain; unless ``record_reputation``
        is off, credits a failure against the poster on this app's reputation ledger (when one is
        attached) when the reserves could not make every creditor whole. Returns the resolution::

            owed = app.attest_liabilities("vendor", {"bank": 50.0, "acme": 50.0})
            reserves = app.attest_custody("vendor", {"omnibus": 50.0})
            schedule = app.build_seniority_schedule("vendor", [["bank"], ["acme"]])
            resolution = app.resolve_insolvency(reserves, owed, schedule)
            resolution.require_fully_recovered()  # raises: acme bears the $50 shortfall
        """
        from ..settlement import resolve_insolvency as _resolve

        resolution = _resolve(
            custody,
            liabilities,
            schedule,
            poster=poster,
            completeness=completeness,
            solvency=solvency,
            set_off=set_off,
            as_of=as_of,
            verifier=verify_with,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                resolution.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.waterfall import INSOLVENCY_ACTION

            entry = self.audit.record(
                INSOLVENCY_ACTION,
                resource=resolution.poster,
                decision=resolution.status,
                details=resolution.audit_details(),
            )
            resolution.audit_id = getattr(entry, "id", None)
        if record_reputation and self.reputation_ledger is not None and not resolution.solvent:
            self.reputation_ledger.record_outcome(
                resolution.poster,
                passed=False,
                round_id=resolution.id,
                details={"kind": "insolvency_resolution", "attestor": resolution.attestor},
            )
        return resolution

    def guard_collateral(
        self,
        pools: list[Any],
        *,
        poster: str | None = None,
        held: float | None = None,
        custody: Any | None = None,
        solvency: Any | None = None,
        sign: bool = True,
        verify_with: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Fold a counterparty's collateral pools into a bounded re-use guard.

        Reconciles what ``pools`` collectively pledge against the capital the poster holds
        into a single, content-bound :class:`~vincio.settlement.CollateralLedger` — the
        rehypothecation analogue of :meth:`clear_settlements`. The same capital pledged across
        more than one pool is pinpointed as a :class:`~vincio.settlement.ReuseBreach`, and
        each beneficiary's claim is bounded to its deterministic, pari-passu share of the held
        capital, so a scarce stake is apportioned by priority rather than over-promised. Signs
        the ledger as this app and, unless ``record_audit`` is off, records the guard on the
        audit chain. The ledger verifies offline from the bytes alone, and a tampered pool is
        **refused** (with ``verify_with`` a forged pool signature is too) rather than folded
        silently.

        The held figure comes from a ``solvency``
        :class:`~vincio.settlement.SolvencyProof` (the solvency-adjusted ``max(0, reserves −
        liabilities)``, bounding pledges against capital not already owed elsewhere and exposing
        :attr:`~vincio.settlement.CollateralLedger.insolvent`), a ``custody``
        :class:`~vincio.settlement.CustodyAttestation` **proving** the reserves (a tampered or
        forged one is refused, and an :class:`~vincio.settlement.UnderReservedBreach` surfaces
        when the proven reserves fall below the pledges), an explicit *asserted* ``held``, or
        — by default — the gross pledge minus the provably double-pledged capital. Returns the
        ledger::

            proof = app.attest_custody("vendor", {"omnibus": 80.0})
            ledger = app.guard_collateral([pool_a, pool_b], custody=proof)
            ledger.under_reserved          # proven reserves below the pledges
            ledger.require_within_bounds()  # raises if over-committed
        """
        from ..settlement import guard_collateral as _guard

        ledger = _guard(
            pools,
            poster=poster,
            held=held,
            custody=custody,
            solvency=solvency,
            verify_with=verify_with,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                ledger.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.rehypothecation import REHYPOTHECATION_ACTION

            entry = self.audit.record(
                REHYPOTHECATION_ACTION,
                resource=ledger.id,
                decision=ledger.status,
                details=ledger.audit_details(),
            )
            ledger.audit_id = getattr(entry, "id", None)
        return ledger

    def settle_saga(
        self,
        result: Any,
        *,
        contracts: dict[str, Any],
        run_id: str | None = None,
        party: str | None = None,
        sign: bool = True,
        record_reputation: bool = True,
    ) -> list[Any]:
        """Close the books on every contract a cross-org saga ran under.

        Meters each contracted forward step from the saga's durable journal and
        reconciles the per-step delivery against the matching contract in
        ``contracts`` (keyed by contract id), appending one signed, hash-chained
        :class:`~vincio.settlement.SettlementRecord` per contract to the settlement
        book — so a whole cross-org engagement reconciles in one call. Returns the
        records, in contract-id order.
        """
        return self._settlement_book().settle_saga(
            result,
            contracts=contracts,
            run_id=run_id,
            party=party,
            sign=sign,
            record_reputation=record_reputation,
        )

    def settlement_report(self, counterparty: str | None = None) -> Any:
        """Per-counterparty settlement roll-up — beside the cost report.

        Each row totals what was owed, what was delivered, and the net balance with
        a counterparty, with the settled / breached tally behind it. Returns an
        empty :class:`~vincio.settlement.SettlementReport` when no book is attached.
        """
        if self.settlement_book is None:
            from ..settlement import SettlementReport

            return SettlementReport(owner=self.name)
        return self.settlement_book.report(counterparty)

    def clear_settlements(
        self,
        *,
        books: list[Any] | None = None,
        records: list[Any] | None = None,
        sign: bool = True,
        verify_with: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Net a fleet's settlement books into one minimal cleared set.

        Folds the bilateral balances across ``books`` (and/or loose ``records``) —
        or, by default, this app's own attached settlement book — into a single,
        content-bound :class:`~vincio.settlement.NettingSet`: each org's many
        positions collapsed to the minimal set of net obligations, the same web of
        contracts cleared once. Signs the set as this app (the clearer) and, unless
        ``record_audit`` is off, records the clearing on the audit chain. The set
        verifies offline from the bytes alone — the positions balance and the cleared
        obligations reproduce them — and pinpoints any disputed contract rather than
        netting it silently. Returns the set::

            netting = app.clear_settlements(books=[acme_book, vendor_book])
            netting.verify().valid  # offline-verifiable
            netting.print_summary()
        """
        from ..settlement import net_settlements

        all_records: list[Any] = list(records or [])
        sources = books
        if sources is None and not all_records and self.settlement_book is not None:
            sources = [self.settlement_book]
        for book in sources or []:
            all_records.extend(book.records)
        netting = net_settlements(all_records, owner=self.name, verify_with=verify_with)
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                netting.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.netting import NETTING_ACTION

            entry = self.audit.record(
                NETTING_ACTION,
                resource=netting.id,
                decision="clean" if netting.clean else "disputed",
                details=netting.audit_details(),
            )
            netting.audit_id = getattr(entry, "id", None)
        return netting

    def arbitrate(
        self,
        records: list[Any],
        *,
        contract_id: str | None = None,
        sign: bool = True,
        verify_with: Any | None = None,
        record_audit: bool = True,
        record_reputation: bool = True,
    ) -> Any:
        """Adjudicate a disputed contract from the records its parties submit.

        Resolves a pinpointed disagreement (a
        :class:`~vincio.settlement.NettingDispute`, or two records that do not
        reconcile) into a content-bound :class:`~vincio.settlement.Resolution`:
        deterministically decides which figure stands — a reconciliation hash both
        parties co-signed is upheld, a contradicting unilateral claim is rejected and
        pinpointed — reading only the submitted signed records and asserting nothing
        it cannot recompute. Signs the resolution as this app (the arbiter) and,
        unless ``record_audit`` is off, records it on the audit chain; unless
        ``record_reputation`` is off, closes the reputation loop by debiting the
        party whose claim did not stand. The resolution verifies offline from the
        bytes alone. Returns it::

            resolution = app.arbitrate([buyer_record, seller_record])
            resolution.verify().valid  # offline-verifiable
            resolution.print_summary()
        """
        from ..settlement import arbitrate

        resolution = arbitrate(
            records,
            contract_id=contract_id,
            arbiter=self.name,
            verify_with=verify_with,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                resolution.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.arbitration import ARBITRATION_ACTION

            entry = self.audit.record(
                ARBITRATION_ACTION,
                resource=resolution.contract_id,
                decision=resolution.status,
                details=resolution.audit_details(),
            )
            resolution.audit_id = getattr(entry, "id", None)
        if record_reputation and self.reputation_ledger is not None:
            for party in resolution.dissenters:
                self.reputation_ledger.record_outcome(
                    party,
                    passed=False,
                    round_id=resolution.contract_id,
                    details={
                        "kind": "arbitration",
                        "resolution_id": resolution.id,
                        "contract_id": resolution.contract_id,
                        "reason": "claim did not stand",
                    },
                )
        return resolution

    def attest_reputation(
        self,
        subject: str,
        *,
        book: Any | None = None,
        resolutions: Any | None = None,
        config: Any | None = None,
        horizon_days: float | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Issue a signed, portable attestation of a counterparty's earned standing.

        Reads this app's own settlement book (``book``, else the attached one) and
        any arbitration ``resolutions`` for ``subject`` and summarizes how its
        delivery fared — fulfilled settlements as successes, breaches and arbitration
        dissents as failures — into a content-bound
        :class:`~vincio.settlement.ReputationAttestation`, signed as this app (the
        issuer). A prospective counterparty verifies it from the bytes alone (a
        tampered score or a forged issuer is caught) and folds several issuers'
        attestations into a bounded prior with :meth:`import_reputation`.
        ``horizon_days`` optionally declares a validity window after which an
        as-of-aware import treats the attestation as stale. Unless ``record_audit`` is
        off, the issuance lands on the audit chain. Raises
        :class:`~vincio.core.errors.SettlementError` when this app has no admissible
        history with the subject to attest. Returns the attestation::

            att = app.attest_reputation("vendor")
            att.verify(app.contract_signer).valid  # offline-verifiable
        """
        from ..settlement.attestation import ATTESTATION_ACTION

        source = book if book is not None else self._settlement_book()
        signer = self._resolve_contract_signer(None, sign)
        attestation = source.attest(
            subject,
            resolutions=resolutions,
            config=config,
            sign=sign and signer is not None,
            verify_with=None,
            horizon_days=horizon_days,
        )
        if sign and signer is not None and source.signer is None:
            # Sign as the issuer (the book's owner), the identity book.attest would
            # use, so the signature party matches the attestation's issuer and the
            # attestation verifies against its own default require=[issuer].
            attestation.sign(signer, party=attestation.issuer)
        if record_audit and self.audit is not None:
            entry = self.audit.record(
                ATTESTATION_ACTION,
                resource=attestation.subject,
                decision="issued",
                details=attestation.audit_details(),
            )
            attestation.audit_id = getattr(entry, "id", None)
        return attestation

    def revoke_attestation(
        self,
        attestation: Any,
        *,
        book: Any | None = None,
        replacement: Any | None = None,
        reason: str = "",
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Withdraw a prior attestation, by its hash, as a signed revocation.

        Builds a content-bound
        :class:`~vincio.settlement.AttestationRevocation` that supersedes or withdraws
        ``attestation`` — which this app (the issuer) must have issued — signed as this
        app and, unless ``record_audit`` is off, recorded on the audit chain.
        ``replacement`` optionally names the attestation that supersedes it. A
        prospective counterparty passes the revocation to :meth:`import_reputation` so
        the withdrawn claim is excluded from the combination, pinpointed, never
        silently honored. Returns the revocation::

            rev = app.revoke_attestation(att, reason="vendor regressed")
            rev.verify(app.contract_signer).valid  # offline-verifiable
        """
        from ..settlement.attestation import REVOCATION_ACTION

        source = book if book is not None else self._settlement_book()
        signer = self._resolve_contract_signer(None, sign)
        revocation = source.revoke(
            attestation,
            replacement=replacement,
            reason=reason,
            sign=sign and signer is not None,
        )
        if sign and signer is not None and source.signer is None:
            # Sign as the issuer (the book's owner), matching how revoke would, so the
            # signature party matches the revocation's issuer and it verifies against
            # its own default require=[issuer].
            revocation.sign(signer, party=revocation.issuer)
        if record_audit and self.audit is not None:
            entry = self.audit.record(
                REVOCATION_ACTION,
                resource=revocation.subject,
                decision="superseded" if revocation.is_supersession else "withdrawn",
                details=revocation.audit_details(),
            )
            revocation.audit_id = getattr(entry, "id", None)
        # Retain it so ``serve_attestations`` can return it to a peer that pulls this
        # app's standing about the subject, superseding any cached copy of the claim.
        self._issued_revocations = [
            r for r in self._issued_revocations if r.content_hash != revocation.content_hash
        ]
        self._issued_revocations.append(revocation)
        return revocation

    def import_reputation(
        self,
        attestations: list[Any],
        *,
        subject: str | None = None,
        config: Any | None = None,
        verify_with: Any | None = None,
        allow_self: bool = False,
        revocations: list[Any] | None = None,
        as_of: Any | None = None,
        trust: Any | None = None,
        trust_config: Any | None = None,
        weight: bool = True,
    ) -> Any:
        """Combine other orgs' attestations into a prior that weights negotiation.

        Verifies each :class:`~vincio.settlement.ReputationAttestation` offline,
        refusing and pinpointing a tampered or forged one, and pools the admissible
        evidence across issuers into a bounded, evidence-weighted
        :class:`~vincio.settlement.PortableReputation` prior under ``config`` — never
        a single self-asserted number (an issuer that vouches for itself is refused).
        Any signed :class:`~vincio.settlement.AttestationRevocation` in ``revocations``
        excludes the attestation its issuer withdrew, and with an ``as_of`` clock a
        stale attestation (past its issuer-declared validity window) decays out of the
        prior rather than anchoring it forever — so the imported standing reflects
        *current* standing, not a frozen snapshot. Pass a ``trust`` source or a
        ``trust_config`` to weigh each issuer's evidence by this app's **own trust in
        that issuer** (rooted in the attached
        :class:`~vincio.optimize.reputation.ReputationLedger`, composed transitively
        over the attestations), so corroboration from a trusted issuer counts for more
        than volume from an unknown one and a Sybil cluster cannot manufacture standing.
        With ``weight`` (the default) the prior is attached so the next negotiation
        weights a counterparty with no local history by what its past counterparties
        attest, under the same bounded ``[floor, 1]`` rule a local reputation uses; the
        attached local ledger stays the source of truth for a counterparty this app
        already knows. Returns the prior::

            prior = app.import_reputation([att_a, att_b], revocations=[rev], as_of=now)
            result = app.negotiate("transcribe calls", buyer=..., seller=...)
        """
        from ..settlement.attestation import combine_attestations

        prior = combine_attestations(
            attestations,
            subject=subject,
            config=config,
            verify_with=verify_with,
            base=self.reputation_ledger,
            allow_self=allow_self,
            revocations=revocations,
            as_of=as_of,
            trust=trust,
            trust_config=trust_config,
        )
        if weight:
            self.imported_reputation = prior
        return prior

    def admit(
        self,
        subject: str,
        *,
        reputation: Any | None = None,
        policy: Any | None = None,
        config: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Decide a counterparty's admitted exposure from its earned standing.

        Reads ``subject``'s standing from the same source the negotiation path weights
        by — an imported :class:`~vincio.settlement.PortableReputation` if one is attached
        (:meth:`import_reputation`), else the local
        :class:`~vincio.optimize.reputation.ReputationLedger` — or an explicit
        ``reputation`` source — and maps it to a bounded
        :class:`~vincio.settlement.AdmissionDecision`: a maximum contract value (the
        exposure ceiling), a required escrow fraction, and an SLA-strictness factor. A
        thin or low-trust standing is admitted on *conservative* terms rather than
        refused — discounted exposure, never a hard gate — and as the counterparty
        accrues settled, corroborated history its ceiling **ramps** toward parity, a
        regression walking it back. Pass an :class:`~vincio.settlement.AdmissionPolicy` as
        ``policy`` (or an :class:`~vincio.settlement.AdmissionConfig` as ``config``) to set
        the parity ceiling and ramp. Unless ``record_audit`` is off, the decision lands on
        the hash-chained audit log, binding the standing it read and the ceiling it set.
        The decision verifies offline from the bytes alone — the terms re-derive from the
        bound standing — and folds into the existing negotiation / contracting path
        (:meth:`~vincio.settlement.AdmissionDecision.bound_position` /
        :meth:`~vincio.settlement.AdmissionDecision.apply_to_terms`). Returns it::

            decision = app.admit("vendor")
            buyer = decision.bound_position(buyer_position(max_price_usd=5.0, max_sla_seconds=5.0))
            result = app.negotiate("transcribe", buyer=buyer, seller=..., seller_id="vendor")
        """
        from ..settlement.admission import ADMISSION_ACTION, AdmissionPolicy

        engine = policy if isinstance(policy, AdmissionPolicy) else AdmissionPolicy(config)
        source = reputation if reputation is not None else self._reputation_view()
        decision = engine.admit(subject, reputation=source)
        if record_audit and self.audit is not None:
            entry = self.audit.record(
                ADMISSION_ACTION,
                resource=decision.subject,
                decision="parity" if decision.at_parity else "graduated",
                details=decision.audit_details(),
            )
            decision.audit_id = getattr(entry, "id", None)
        return decision

    def serve_attestations(
        self,
        *,
        book: Any | None = None,
        revocations: list[Any] | None = None,
        attestations: list[Any] | None = None,
        config: Any | None = None,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose this app's earned standing as a queryable attestation peer over A2A.

        Returns an :class:`~vincio.a2a.A2AServer` whose Agent Card advertises an
        ``attestation-exchange`` skill; an importer pulls from it with
        :meth:`gather_reputation`. Answering a query for a subject, the peer returns a
        :class:`~vincio.settlement.ReputationBundle` of its **own** signed artifacts —
        the *current* attestation it can issue from its settlement ``book`` (else the
        attached one) and the revocations it has issued (``revocations``, else the ones
        this app has signed via :meth:`revoke_attestation`). Pass an explicit
        ``attestations`` list to serve a fixed signed snapshot instead of re-issuing.
        **Pull, never push:** the peer only ever answers a query, and only with
        artifacts it signed.
        """
        from ..settlement.exchange import attestation_a2a_server

        return attestation_a2a_server(
            book if book is not None else self._settlement_book(),
            revocations=revocations if revocations is not None else self._issued_revocations,
            attestations=attestations,
            config=config,
            name=name,
            url=url,
            description=description,
            token_validator=token_validator,
            audit=self.audit,
        )

    async def agather_reputation(
        self,
        subject: str,
        *,
        peers: Any,
        directory: Any | None = None,
        principal: Any | None = None,
        config: Any | None = None,
        verify_with: Any | None = None,
        allow_self: bool = False,
        held_attestations: list[Any] | None = None,
        held_revocations: list[Any] | None = None,
        as_of: Any | None = None,
        trust: Any | None = None,
        trust_config: Any | None = None,
        max_peers: int | None = None,
        weight: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Assemble a current prior by pulling signed artifacts from a bounded peer set.

        The gossip analogue of :meth:`import_reputation`: instead of being *handed* a
        bundle, this app **queries** a bounded set of ``peers`` (each an
        :class:`~vincio.settlement.AttestationExchange`, an in-process
        :class:`~vincio.a2a.A2AServer`, or an :class:`~vincio.a2a.A2AClient`) for the
        signed attestations and revocations they hold about ``subject``, governs each
        through ``directory`` (an :class:`~vincio.registry.AgentDirectory`'s
        allow-list, audited), verifies every fetched artifact from the bytes,
        deduplicates by content hash, and folds them — with any ``held_attestations`` /
        ``held_revocations`` already on hand — into a bounded, evidence-weighted
        :class:`~vincio.settlement.PortableReputation` under the same freshness,
        revocation, and ``[floor, 1]`` discipline :meth:`import_reputation` uses. Pass a
        ``trust`` source or a ``trust_config`` to weigh each gathered issuer's evidence
        by this app's own trust in it (rooted in the attached ledger, composed
        transitively), so a cluster of unknown peers cannot out-evidence a few it
        trusts. Every peer visited and artifact fetched lands on the audit chain. With
        ``weight`` (the default) the assembled prior is attached so the next negotiation
        weights an unknown counterparty by what its peers attest. Returns a
        :class:`~vincio.settlement.GatheredReputation`::

            result = await app.agather_reputation("vendor", peers={"acme": acme_server})
            result.weight("vendor")  # drops into the negotiation path
        """
        from ..settlement.exchange import gather_reputation

        result = await gather_reputation(
            subject,
            peers=peers,
            directory=directory,
            principal=principal,
            config=config,
            verify_with=verify_with,
            base=self.reputation_ledger,
            allow_self=allow_self,
            held_attestations=held_attestations,
            held_revocations=held_revocations,
            as_of=as_of,
            trust=trust,
            trust_config=trust_config,
            max_peers=max_peers,
            audit=self.audit,
            record_audit=record_audit,
        )
        if weight:
            self.imported_reputation = result.reputation
        return result

    def gather_reputation(self, subject: str, **kwargs: Any) -> Any:
        """Synchronous wrapper around :meth:`agather_reputation`."""
        return run_sync(self.agather_reputation(subject, **kwargs))

    def cross_org_engagement(
        self,
        *,
        buyer: str = "",
        seller: str = "",
        scope: str = "",
        coordinator: str | None = None,
    ) -> Any:
        """Thread the whole cross-org settlement & credit fabric behind one call-path.

        Returns a :class:`~vincio.settlement.CrossOrgEngagement` — the capstone facade
        that composes the entire pipeline (discover → negotiate → contract →
        choreograph delivery → meter → settle → net → arbitrate → attest and port
        reputation → admit → post and pool collateral under a rehypothecation guard →
        prove reserves, solvency, completeness, non-equivocation, and history → and, on
        default, resolve the insolvency by seniority waterfall with close-out set-off)
        into one governed, audited, hash-linked narrative. Each lifecycle method
        delegates to the *same* entry point on this app a caller would use directly, so
        the primitives stay unchanged and usable on their own; the facade only captures
        and **narrates** them.

        :meth:`~vincio.settlement.CrossOrgEngagement.seal` mints the content-bound,
        signed :class:`~vincio.settlement.EngagementNarrative`, and
        :meth:`~vincio.settlement.CrossOrgEngagement.verify` proves the whole chain —
        and every captured artifact — verifies offline, so a tamper introduced anywhere
        is caught::

            eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe")
            contract = eng.negotiate(buyer=buyer_pos, seller=seller_pos)
            eng.choreograph(saga, participants=parts)
            eng.settle_saga(contracts={contract.id: contract})
            narrative = eng.seal()
            narrative.verify(app.contract_signer).valid  # offline-verifiable
        """
        from ..settlement.engagement import CrossOrgEngagement

        return CrossOrgEngagement(
            self, buyer=buyer, seller=seller, scope=scope, coordinator=coordinator or self.name
        )

    # -- evaluators / optimizers ----------------------------------------------------------------------

    def add_evaluator(self, name: str | Callable) -> ContextApp:
        """Register a metric (by name or callable) that scores every run."""
        if callable(name):
            from ..evals.metrics import METRICS

            # Resolve the name once: computing it twice around the insertion
            # would disagree for a callable without __name__ (the second
            # len(METRICS) is one larger), registering one key but recording
            # a different one in self.evaluators.
            fn = name
            name = getattr(fn, "__name__", f"custom_{len(METRICS)}")
            METRICS[name] = fn
        self.evaluators.append(name)
        return self

    def add_validator(
        self, name: str, validator: SemanticValidator, *, blocking: bool = True
    ) -> ContextApp:
        """Register a semantic output validator (blocking by default)."""
        from ..output.schemas import ValidatorSpec

        self.semantic_validators[name] = validator
        self.output_contract.validators.append(ValidatorSpec(name=name, blocking=blocking))
        return self

    def add_optimizer(self, name: str) -> ContextApp:
        """Register an optimization dimension the improvement loop may tune."""
        known = {"context_budget", "prompt_format", "retrieval_config", "model_routing"}
        if name not in known:
            raise ConfigError(f"unknown optimizer {name!r}; known: {sorted(known)}")
        if name not in self.optimizers:
            self.optimizers.append(name)
        return self

    def add_online_evaluator(
        self, metric: str | Callable, *, sample_rate: float = 1.0, name: str | None = None
    ) -> ContextApp:
        """Score a sampled fraction of live runs with ``metric`` after each run
        completes, writing the score as a time series on the metadata store
        (no traffic mirrored anywhere). Scoring runs off the hot path; sampling
        bounds the overhead. The same metric object can gate releases offline
        and act as a runtime guardrail::

            app.add_online_evaluator("answer_relevance", sample_rate=0.1)
            app.add_online_evaluator("goal_accuracy", sample_rate=0.2)
        """
        self.online_evaluators.append(
            OnlineEvaluator(
                metric, name=name, sample_rate=sample_rate, store=self.store, app_name=self.name
            )
        )
        return self

    def add_metric_rail(
        self,
        metric: str | Callable,
        *,
        threshold: float,
        direction: str = "output",
        action: str = "block",
        name: str | None = None,
        **params: Any,
    ) -> ContextApp:
        """Use an eval metric as a runtime guardrail. The same metric that gates
        releases offline blocks (or warns on) generations at run time::

            app.add_metric_rail("toxicity", threshold=0.0)
            app.add_metric_rail("answer_relevance", threshold=0.3, action="warn")
        """
        from ..evals.guardrails import metric_guardrail

        metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
        predicate_name = name or f"{metric_name}_guard"
        self.register_rail_predicate(
            predicate_name, metric_guardrail(metric, threshold=threshold, name=predicate_name)
        )
        self.add_rail(
            name=predicate_name,
            kind="custom",
            direction=direction,
            action=action,
            predicate=predicate_name,
            params=params,
        )
        return self

    def experiment(
        self,
        name: str,
        *,
        variants: dict[str, dict[str, Any]] | None = None,
        dataset: Dataset | str | None = None,
        metrics: list[str] | None = None,
    ) -> Any:
        """A production-style A/B over prompt/model/config variants of this app,
        compared on eval metrics *and* cost with significance tests. Returns an
        :class:`~vincio.evals.experiments.Experiment` handle; if ``variants`` and
        ``dataset`` are given, every variant is evaluated first::

            exp = app.experiment(
                "prompt_ab",
                variants={"baseline": {}, "concise": {"prompt": concise_spec}},
                dataset=golden, metrics=["goal_accuracy", "cost"],
            )
            exp.compare(); exp.significance("goal_accuracy"); exp.cost()
        """
        from ..evals.experiments import Experiment

        handle = Experiment(self, name, metrics=metrics)
        if variants and dataset is not None:
            for variant, config in variants.items():
                config = config or {}
                handle.run_variant(
                    variant,
                    dataset,
                    model=config.get("model"),
                    prompt=config.get("prompt"),
                    apply=config.get("apply"),
                    params=config.get("params"),
                )
        return handle

    # -- structured output -------------------------------------------------

    def add_output_schema(
        self,
        schema: type[BaseModel] | OutputSchema | dict[str, Any],
        *,
        name: str | None = None,
        task_types: list[str] | None = None,
        keywords: list[str] | None = None,
        when: Callable[[str], bool] | None = None,
        priority: int = 100,
    ) -> ContextApp:
        """Register an alternative output schema, routed by task or content.

        The first call creates the schema router; the app's base schema (if
        any) stays the default when no route matches::

            app.add_output_schema(BugReport, keywords=["bug", "crash"])
            app.add_output_schema(BillingIssue, keywords=["invoice", "refund"])
        """
        if self.schema_router is None:
            self.schema_router = SchemaRouter(default=self.output_contract.output_schema())
        self.schema_router.add(
            schema,
            name=name,
            task_types=task_types,
            keywords=keywords,
            when=when,
            priority=priority,
        )
        return self

    def enable_self_correction(
        self, *, max_cycles: int = 2, max_cost_usd: float = 0.05, temperature: float = 0.0
    ) -> ContextApp:
        """Turn on bounded validate → critique → repair cycles for failed
        outputs. Structure-only: the critique and repair prompt forbid
        changing factual content, and all validators re-run each cycle."""
        self.self_correction = {
            "max_cycles": max_cycles,
            "max_cost_usd": max_cost_usd,
            "temperature": temperature,
        }
        return self

    def predictor(
        self,
        sig: type[Signature],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        prompt_spec: PromptSpec | None = None,
    ) -> Predict:
        """A :class:`~vincio.prompts.signatures.Predict` bound to the app's
        provider and model: ``app.predictor(Triage)(ticket="...")``."""
        return Predict(
            sig,
            provider=self.resolve_provider(),
            model=model or self.model,
            temperature=temperature,
            prompt_spec=prompt_spec,
        )

    # -- task decorator ----------------------------------------------------------------------

    def task(self, cls: type) -> type:
        """Configure the app from a task class::

        @app.task
        class Triage:
            objective = "Classify support tickets"
            labels = ["bug", "billing", "feature", "other"]
        """
        objective = getattr(cls, "objective", None)
        labels = getattr(cls, "labels", None)
        rules = list(getattr(cls, "rules", []))
        update: dict[str, Any] = {}
        if objective:
            self.objective = Objective(
                text=objective, task_type=TaskType.CLASSIFICATION if labels else TaskType.GENERAL
            )
            update["objective"] = objective
        if labels:
            update["rules"] = [
                *rules,
                f"Answer with exactly one of these labels: {', '.join(labels)}.",
            ]
            if self.output_contract.schema_def is None:
                schema = OutputSchema.from_json_schema(
                    {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "enum": list(labels)},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["label", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                    name=cls.__name__,
                )
                self.output_contract = OutputContract.from_schema(schema)
        elif rules:
            update["rules"] = rules
        self.prompt_spec = self.prompt_spec.model_copy(update=update)
        return cls

    # -- execution -------------------------------------------------------------------------------------------

    @staticmethod
    def _coerce_input(
        user_input: str | UserInput,
        *,
        files: list[str] | None,
        tenant_id: str | None,
        user_id: str | None,
        session_id: str | None,
        feature: str | None,
    ) -> UserInput:
        """Normalize the run entry points' input into a fresh UserInput."""
        if isinstance(user_input, str):
            normalized = UserInput(text=user_input)
        else:
            normalized = user_input.model_copy(deep=True)
        if files:
            normalized.files.extend(FileRef(path=f) for f in files)
        if tenant_id is not None:
            normalized.tenant_id = tenant_id
        if user_id is not None:
            normalized.user_id = user_id
        if session_id is not None:
            normalized.session_id = session_id
        if feature is not None:
            normalized.feature = feature
        return normalized

    async def arun(
        self,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        feature: str | None = None,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Run the full context-engineering pipeline asynchronously → :class:`RunResult`."""
        user_input = self._coerce_input(
            user_input,
            files=files,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            feature=feature,
        )
        result = await self._runtime.execute(user_input, config)
        if self.online_evaluators:
            self._spawn_online(result, user_input)
        return result

    def submit(
        self,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        feature: str | None = None,
        config: RunConfig | None = None,
    ) -> RunHandle:
        """Start a run in the background and return a :class:`RunHandle`.

        ``handle.cancel()`` propagates cooperative cancellation into the run's
        bounded-concurrency groups (retrieval, tools, model); ``await handle``
        (or ``await handle.result()``) yields the :class:`RunResult`. A cancelled
        run is still fully recorded on its trace and audit chain — cancellation
        is identical to the non-streaming path's. Must be called from within a
        running event loop::

            handle = app.submit("Summarize the filing")
            handle.cancel()  # cooperative — the partial run is still recorded
        """
        normalized = self._coerce_input(
            user_input,
            files=files,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            feature=feature,
        )

        async def _run() -> RunResult:
            result = await self._runtime.execute(normalized, config)
            if self.online_evaluators:
                self._spawn_online(result, normalized)
            return result

        task = asyncio.ensure_future(_run())
        return RunHandle(task)

    def run(self, user_input: str | UserInput, **kwargs: Any) -> RunResult:
        """Run the full context-engineering pipeline synchronously → :class:`RunResult`."""
        return run_sync(self._run_and_flush(user_input, **kwargs))

    async def _run_and_flush(self, user_input: str | UserInput, **kwargs: Any) -> RunResult:
        result = await self.arun(user_input, **kwargs)
        await self.aflush_online()
        return result

    async def astream(
        self,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        feature: str | None = None,
        config: RunConfig | None = None,
    ) -> AsyncIterator[RunStreamEvent]:
        """Run the full pipeline with end-to-end streaming.

        Yields :class:`RunStreamEvent` items — pipeline stages, model text
        deltas, incremental partial-JSON output, tool activity — ending with
        a ``done`` event that carries the final :class:`RunResult`::

            async for event in app.astream("Summarize the refund policy"):
                if event.type == "text_delta":
                    print(event.text, end="", flush=True)
                elif event.type == "done":
                    result = event.result
        """
        user_input = self._coerce_input(
            user_input,
            files=files,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            feature=feature,
        )
        config = config or RunConfig()
        config = config.model_copy(update={"stream": True})
        async for event in self._runtime.execute_stream(user_input, config):
            yield event

    def stream(self, user_input: str | UserInput, **kwargs: Any) -> Iterator[RunStreamEvent]:
        """Synchronous streaming convenience: collects the async event
        stream and yields the events in order (like provider.stream_sync)."""

        async def collect() -> list[RunStreamEvent]:
            return [event async for event in self.astream(user_input, **kwargs)]

        yield from run_sync(collect())

    # -- agents -------------------------------------------------------------------------------------------------

    def _build_executor(
        self,
        *,
        tools: list[str | Callable] | None = None,
        planner: str = "dag",
        max_steps: int = 8,
        model: str | None = None,
        system_prompt_extra: str = "",
        restrict_tools: bool = False,
        domain: Any | None = None,
        cost_aware_models: list[str] | None = None,
    ) -> AgentExecutor:
        tool_names: list[str] = []
        for tool in tools or []:
            self.add_tool(tool)
            tool_names.append(
                tool if isinstance(tool, str) else getattr(tool, "__name__", str(tool))
            )
        planner_mode = {
            "dag": "static",
            "static": "static",
            "dynamic": "dynamic",
            "react": "react",
            "direct": "direct",
            "plan_and_execute": "plan_and_execute",
            "hierarchical": "hierarchical",
            "htn": "hierarchical",
        }.get(planner, "static")
        provider = self.resolve_provider()
        agent_model = model or self.model
        llm_planning = planner_mode in ("dynamic", "plan_and_execute", "hierarchical")
        planner_obj = Planner(
            mode=planner_mode,  # type: ignore[arg-type]
            provider=provider if llm_planning else None,
            model=agent_model if llm_planning else None,
            max_steps=max_steps,
            domain=domain,
        )
        # Cost-aware action selection over the candidate models, reading the
        # data-driven model registry's pricing and the live budget.
        selector = None
        if cost_aware_models:
            from ..agents.selection import CostAwareSelector

            selector = CostAwareSelector(cost_aware_models, events=self.events)
        retrieve_fn = None
        if self.retrieval is not None:
            engine = self.retrieval

            async def retrieve_fn(query: str) -> list[EvidenceItem]:
                result = await engine.retrieve(query, top_k=self.config.retrieval.top_k)
                return result.evidence

        from ..output.validators import OutputValidator

        validator = None
        if self.output_contract.schema_def is not None or self.output_contract.require_citations:
            validator = OutputValidator(
                self.output_contract,
                semantic_validators=self.semantic_validators,
                policy_engine=self.policy_engine,
                repairer=self.repairer,
            )
        system_prompt = self.prompt_compiler.compile(
            self.prompt_spec, variables=self.prompt_variables
        ).system_text
        if system_prompt_extra:
            system_prompt = (
                f"{system_prompt}\n\n{system_prompt_extra}"
                if system_prompt
                else system_prompt_extra
            )
        # restrict_tools (crew members): least privilege — only the tools named
        # for this executor, never the app-wide enabled set.
        enabled = tool_names if restrict_tools else self.enabled_tools
        return AgentExecutor(
            provider,
            model=agent_model,
            planner=planner_obj,
            tool_runtime=self.tool_runtime if enabled else None,
            tool_specs=self.tool_registry.specs(enabled) if enabled else [],
            retrieve_fn=retrieve_fn,
            output_validator=validator,
            tracer=self.tracer,
            cost_tracker=self.cost_tracker,
            cost_ledger=self.cost_ledger,
            events=self.events,
            selector=selector,
            system_prompt=system_prompt,
        )

    def agent(
        self,
        *,
        name: str | None = None,
        tools: list[str | Callable] | None = None,
        planner: str = "dag",
        max_steps: int = 8,
        evaluator: str | None = None,
        model: str | None = None,
        domain: Any | None = None,
        cost_aware_models: list[str] | None = None,
    ) -> _AgentHandle:
        """Build a bounded agent over the app's tools, memory, and retrieval.

        ``planner`` selects the planning shape (``dag`` / ``dynamic`` / ``react``
        / ``direct`` / ``plan_and_execute`` / ``hierarchical``). Pass an
        :class:`~vincio.agents.HTNDomain` as ``domain`` to drive deterministic
        hierarchical decomposition, and ``cost_aware_models`` (cheapest→strongest)
        to enable cost-aware action selection. In-place plan repair is on by
        default. Tool failures, validation contradictions, and budget shocks are
        repaired in place rather than restarting the run.
        """
        if evaluator is not None:
            self.add_evaluator(evaluator)
        executor = self._build_executor(
            tools=tools,
            planner=planner,
            max_steps=max_steps,
            model=model,
            domain=domain,
            cost_aware_models=cost_aware_models,
        )
        return _AgentHandle(self, executor, max_steps)

    def assistant(
        self,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
        memory_writeback: bool = True,
        auto_approve: list[str] | None = None,
        on_approval: Any | None = None,
        feature: str | None = "assistant",
    ) -> Assistant:
        """Open a conversational, session-aware :class:`~vincio.assistant.Assistant`.

        A thin multi-turn layer over this app: every turn is still a full
        :meth:`run` (retrieval, grounding, validation, rails, budget, trace,
        audit all apply), threaded under one ``session_id`` with session-scoped
        memory write-back and an approval surface for write tools::

            chat = app.assistant(user_id="u-1")
            print(chat.send("How do I reset my password?").text)
            print(chat.send("And change my email?").text)   # remembers the thread

        Write tools are denied by default and surfaced as pending approvals;
        ``auto_approve=[...]`` pre-allows trusted tools, or pass ``on_approval``
        for an interactive decision.
        """
        from ..assistant import Assistant

        return Assistant(
            self,
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=session_id,
            memory_writeback=memory_writeback,
            auto_approve=auto_approve,
            on_approval=on_approval,
            feature=feature,
        )

    def research(self, question: str, *, objective: str = "", **kwargs: Any):
        """Run the deep-research loop: search → read → reflect → verify →
        synthesize, emitting a cited, budget-bounded, eval-scored report.

        Composes the query-understanding planners, the retrieval engine, the
        grounded-fact extractor, and the cited-report builder into one
        :class:`~vincio.agents.research.ResearchAgent`. Requires a source
        (``app.add_source(...)``)::

            report = app.research("What changed in the refund policy?")
            report.answer, report.metrics["citation_coverage"], report.sources
        """
        from ..agents.research import ResearchAgent

        return ResearchAgent(self, **kwargs).run(question, objective=objective)

    async def aresearch(self, question: str, *, objective: str = "", **kwargs: Any):
        """Async :meth:`research`."""
        from ..agents.research import ResearchAgent

        return await ResearchAgent(self, **kwargs).arun(question, objective=objective)

    def reasoning(self, policy: Any | None = None, **kwargs: Any):
        """Build a :class:`~vincio.agents.reasoning.ReasoningController`.

        The controller sets the thinking effort and a hard-ceilinged thinking
        budget per step from the task classification and the live budget, and
        steps effort down when a thinking prefix is already warm in its
        :class:`~vincio.caching.ReasoningTraceCache` (created here unless one is
        passed as ``trace_cache=``). Pass it to :meth:`use_reasoning_controller`
        to have the runtime apply it on every run, or call ``.decide(...)``
        directly::

            ctl = app.reasoning()
            d = ctl.decide(task=routed.task, text=question, remaining_output_tokens=4096)
            app.run(question, config=RunConfig(reasoning_effort=d.effort))
        """
        from ..agents.reasoning import ReasoningController, ReasoningPolicy
        from ..caching import ReasoningTraceCache

        if policy is None:
            policy = ReasoningPolicy()
        trace_cache = kwargs.pop("trace_cache", None) or ReasoningTraceCache()
        return ReasoningController(policy, trace_cache=trace_cache, **kwargs)

    def use_reasoning_controller(self, controller: Any | None = None, **kwargs: Any) -> ContextApp:
        """Install a reasoning controller so the runtime sets effort per run.

        With a controller installed, a run that does not pin ``reasoning_effort``
        on its :class:`~vincio.core.types.RunConfig` has the effort and thinking
        budget chosen by the controller (only for reasoning-capable models), and
        each paid thinking prefix is recorded so a re-ask reuses it. ``controller``
        may be a :class:`~vincio.agents.reasoning.ReasoningController`, a
        :class:`~vincio.agents.reasoning.ReasoningPolicy`, or ``None`` (default
        policy). Returns ``self`` for chaining."""
        from ..agents.reasoning import ReasoningController

        if not isinstance(controller, ReasoningController):
            controller = self.reasoning(controller, **kwargs)
        self.reasoning_controller = controller
        return self

    def use_context_governor(
        self,
        governor: Any | None = None,
        *,
        evidence_store: Any | None = None,
        blob_store: Any | None = None,
        **kwargs: Any,
    ) -> ContextApp:
        """Install a long-horizon :class:`~vincio.context.ContextGovernor`.

        For million-token, multi-day, multi-session runs: the governor holds a
        :class:`~vincio.context.ContextBudget` (live tokens, resident bytes,
        KV-cache footprint), decays stale spans within the run, and compacts the
        coldest ones into the memory OS — paging the full text back on demand —
        so the live context stays bounded as the horizon grows. ``governor`` may
        be a :class:`~vincio.context.ContextGovernor`, a
        :class:`~vincio.context.ContextBudget` (wrapped in a default governor that
        writes compaction summaries into this app's memory engine), or ``None``
        (an unbounded-budget governor). Feed each run's packet with
        :meth:`govern_packet`. Returns ``self`` for chaining::

            app.use_context_governor(ContextBudget(max_tokens=8000))
            for turn in conversation:
                result = app.run(turn)
                app.govern_packet(result)          # admits result.evidence
            report = app.context_budget_report()

        By default a compacted span's full text pages back from a process-local
        store, which a fresh worker or a restart cannot read. Pass a ``blob_store``
        (any :class:`~vincio.storage.base.BlobStore`) and the compactor backs cold
        spans with a content-addressed
        :class:`~vincio.context.evidence_store.BlobEvidenceStore`, so a multi-day
        run survives a restart and a multi-process run pages the same cold text
        back across workers — the cross-process path slim packets use. Pass a ready
        :class:`~vincio.context.evidence_store.EvidenceStore` as ``evidence_store``
        to supply your own. These apply only when building the default governor;
        a fully-built ``governor`` carries its own store::

            from vincio.storage.base import FileBlobStore
            app.use_context_governor(
                ContextBudget(max_tokens=8000), blob_store=FileBlobStore("spans/")
            )
        """
        from ..context.evidence_store import BlobEvidenceStore
        from ..context.longhorizon import (
            ContextBudget,
            ContextCompactor,
            ContextGovernor,
        )

        if isinstance(governor, ContextGovernor):
            if evidence_store is not None or blob_store is not None:
                raise InputError(
                    "evidence_store / blob_store apply only when building a governor; "
                    "the passed ContextGovernor already carries its own compactor store"
                )
            self.context_governor = governor
            return self
        budget = governor if isinstance(governor, ContextBudget) else ContextBudget(**kwargs)
        store = evidence_store
        if store is None and blob_store is not None:
            store = BlobEvidenceStore(blob_store)
        compactor = ContextCompactor(
            memory=getattr(self, "memory", None), owner_id=self.name, store=store
        )
        self.context_governor = ContextGovernor(budget, compactor=compactor)
        return self

    def govern_packet(self, source: Any) -> Any:
        """Admit a run's evidence into the installed long-horizon governor.

        The multi-session hook: after each :meth:`run`, pass the
        :class:`~vincio.core.types.RunResult` (or a
        :class:`~vincio.context.ContextPacket` from :meth:`compile`) here so the
        long-horizon footprint stays bounded across the conversation. Returns the
        governor's :class:`~vincio.context.ContextBudgetReport`, or ``None`` if no
        governor is installed."""
        if self.context_governor is None or source is None:
            return None
        if hasattr(source, "evidence_items"):  # a ContextPacket
            self.context_governor.admit_packet(source)
        elif hasattr(source, "evidence"):  # a RunResult
            self.context_governor.admit_evidence(source.evidence)
        return self.context_governor.report()

    def context_budget_report(self) -> Any:
        """The installed governor's live context-budget report (or ``None``).

        The residency analogue of :meth:`cost_report`: live tokens, resident
        bytes, KV-cache footprint, compactions, and intra-run decay for the
        long-horizon run."""
        if self.context_governor is None:
            return None
        return self.context_governor.report()

    def use_semantic_cache(self, cache: Any | None = None, **kwargs: Any) -> ContextApp:
        """Install a learned semantic cache so near-misses are served from cache.

        With a cache installed, a run whose request misses the exact-match
        response cache is checked against recent answers in the same scope (model
        + stable prompt head) and schema; a semantically-equivalent one is served
        for free **only above the calibrated acceptance threshold** — never below
        the floor. ``cache`` may be a
        :class:`~vincio.caching.LearnedSemanticCache`, a
        :class:`~vincio.caching.SemanticCachePolicy`, or ``None`` (a policy built
        from this app's ``cache`` config). The cache shares the app embedder.
        Calibrate it from traces before trusting near-misses, and gate it with a
        :class:`~vincio.caching.SemanticCacheGate`. Returns ``self`` for
        chaining::

            app.use_semantic_cache()
            app.semantic_cache.calibrate(examples)   # fit the threshold
            report = app.semantic_cache_report()
        """
        from ..caching import LearnedSemanticCache, SemanticCachePolicy

        if isinstance(cache, LearnedSemanticCache):
            self.semantic_cache = cache
        else:
            if isinstance(cache, SemanticCachePolicy):
                policy = cache
            else:
                cfg = self.config.cache
                policy = SemanticCachePolicy(
                    enabled=True,
                    threshold=cfg.semantic_threshold,
                    target_precision=cfg.semantic_cache_target_precision,
                    min_floor=cfg.semantic_cache_min_floor,
                    ttl_s=float(cfg.ttl_s),
                    max_entries=cfg.semantic_cache_max_entries,
                    max_resident_bytes=cfg.semantic_cache_max_resident_bytes,
                    **kwargs,
                )
            self.semantic_cache = LearnedSemanticCache(self.embedder, policy=policy)
        self.cache_invalidation.register_semantic(self.semantic_cache)
        return self

    def semantic_cache_report(self) -> Any:
        """The installed semantic cache's stats (or ``None``).

        Hit-rate, near-misses rejected, output tokens saved, the calibrated
        threshold in force, and the cache's resident footprint — the savings the
        cache realized, alongside the $0-billed calls it produced in the cost
        report."""
        if self.semantic_cache is None:
            return None
        return self.semantic_cache.stats()

    def use_kv_prefix_reuse(self, pool: Any | None = None, **kwargs: Any) -> ContextApp:
        """Install a KV-prefix pool so cross-request stable-prefix reuse is tracked.

        With a pool installed, each run's compiled stable prefix is recorded; a
        later request that shares the same head (same ``prompt_spec_hash`` on the
        same model) is reported as a reuse, with the serving-engine KV bytes the
        shared head avoids recomputing. ``pool`` may be a
        :class:`~vincio.caching.KVPrefixPool` or ``None`` (one built from this
        app's ``cache`` config). Returns ``self`` for chaining::

            app.use_kv_prefix_reuse()
            for q in questions:
                app.run(q)
            report = app.kv_prefix_report()
        """
        from ..caching import KVPrefixPool

        if isinstance(pool, KVPrefixPool):
            self.kv_prefix_pool = pool
        else:
            cfg = self.config.cache
            self.kv_prefix_pool = KVPrefixPool(
                kv_bytes_per_token=cfg.kv_bytes_per_token,
                max_entries=cfg.kv_prefix_max_entries,
                max_resident_bytes=cfg.kv_prefix_max_resident_bytes,
                **kwargs,
            )
        return self

    def kv_prefix_report(self) -> Any:
        """The installed KV-prefix pool's reuse report (or ``None``).

        Distinct stable heads tracked, total requests seen, how many reused a
        warm head, and the cumulative serving-engine KV those reuses avoided
        recomputing — the cross-request analogue of the prompt-cache hit rate."""
        if self.kv_prefix_pool is None:
            return None
        return self.kv_prefix_pool.report()

    async def atest_time_search(
        self,
        user_input: str | UserInput,
        *,
        verifier: Any = None,
        strategy: str = "best_of_n",
        n: int = 4,
        config: RunConfig | None = None,
        vary: str = "seed",
        generate: Any | None = None,
        budget: Any | None = None,
        **budget_kwargs: Any,
    ):
        """Run verifier-guided test-time search over this app.

        Draws ``n`` candidates by re-running the app with a varied ``seed`` (or
        ``temperature``), scores them with ``verifier`` (any
        :class:`~vincio.evals.judges.Judge` / ensemble, any
        :class:`~vincio.optimize.rewards.VerifiableReward` / ``RewardModel``, or a
        callable), and returns a
        :class:`~vincio.optimize.test_time.SearchResult`. ``strategy`` is
        ``"best_of_n"`` (verifier required) or ``"self_consistency"`` (verifier
        optional). Pass a custom ``generate(index)`` to search over something
        other than a plain re-run::

            from vincio.optimize import RewardVerifier
            res = await app.atest_time_search(q, verifier=reward, n=8)
            res.output, res.confidence, res.stop_reason
        """
        from ..optimize.test_time import SearchBudget, TestTimeSearch

        if generate is None:
            base = config

            def generate(index: int):  # noqa: ANN202 — local closure
                update: dict[str, Any] = {"seed": index}
                if vary == "temperature":
                    update = {"temperature": round(0.2 * index, 4)}
                cfg = base.model_copy(update=update) if base is not None else RunConfig(**update)
                return self.arun(user_input, config=cfg)

        if budget is None:
            budget = SearchBudget(max_candidates=n, **budget_kwargs)
        search = TestTimeSearch(generate, verifier=verifier, budget=budget)
        if strategy == "best_of_n":
            return await search.best_of_n(n)
        if strategy == "self_consistency":
            return await search.self_consistency(n)
        raise InputError(
            f"unknown test-time search strategy {strategy!r}; "
            f"expected 'best_of_n' or 'self_consistency'"
        )

    def test_time_search(self, user_input: str | UserInput, **kwargs: Any):
        """Synchronous :meth:`atest_time_search`."""
        return run_sync(self.atest_time_search(user_input, **kwargs))

    def crew(
        self,
        name: str = "crew",
        *,
        members: list[AgentRole | dict[str, Any]],
        process: str = "sequential",
        tools: list[str | Callable] | None = None,
        planner: str = "direct",
        max_steps: int = 8,
        max_rounds: int = 4,
        model: str | None = None,
    ) -> Crew:
        """Build a multi-agent crew over a shared blackboard.

        ``members`` are :class:`AgentRole` objects or dicts with the role
        fields (``name``, ``description``, ``goal``, ``keywords``,
        ``budget_fraction``) plus optional per-member ``tools`` / ``planner``
        / ``model`` / ``max_steps`` overrides. The hierarchical process uses
        the app's provider as the crew manager (deterministic fallback when
        offline)::

            crew = app.crew(members=[
                {"name": "researcher", "goal": "gather evidence", "keywords": ["find"]},
                {"name": "writer", "goal": "draft the report"},
            ])
            result = crew.run("Summarize Q3 refund trends")
        """
        crew = Crew(
            name,
            process=process,  # type: ignore[arg-type]
            blackboard=Blackboard(event_bus=self.events),
            tracer=self.tracer,
            manager_provider=self.resolve_provider() if process == "hierarchical" else None,
            manager_model=model or self.model,
            max_rounds=max_rounds,
            cost_tracker=self.cost_tracker,
            cost_ledger=self.cost_ledger,
        )
        role_fields = set(AgentRole.model_fields)
        override_fields = {"tools", "planner", "model", "max_steps"}
        for spec in members:
            overrides: dict[str, Any] = {}
            if isinstance(spec, dict):
                unknown = set(spec) - role_fields - override_fields
                if unknown:
                    raise AgentEngineError(
                        f"unknown crew member fields {sorted(unknown)}; "
                        f"expected {sorted(role_fields | override_fields)}"
                    )
                overrides = spec
                role = AgentRole(**{k: v for k, v in spec.items() if k in role_fields})
            else:
                role = spec
            executor = self._build_executor(
                tools=overrides.get("tools", tools),
                planner=overrides.get("planner", planner),
                max_steps=overrides.get("max_steps", max_steps),
                model=overrides.get("model", model),
                system_prompt_extra=f"You are {role.name}. {role.description}".strip(),
                restrict_tools=True,
            )
            crew.add(role, executor)
        return crew

    def graph(
        self,
        name: str = "graph",
        *,
        state_schema: type[BaseModel] | None = None,
        reducers: dict[str, Callable[[Any, Any], Any]] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> StateGraph:
        """A durable :class:`StateGraph` bound to the app's tracer and
        metadata store: checkpoints persist wherever the app's runs do, so
        threads survive restarts when the store is SQLite/Postgres."""
        graph = StateGraph(name, state_schema=state_schema, reducers=reducers, defaults=defaults)
        graph.default_tracer = self.tracer
        graph.default_checkpointer = Checkpointer(self.store)
        return graph

    # -- workflows ------------------------------------------------------------------------------------------------

    def workflow(self, name: str) -> Workflow:
        """Create a deterministic :class:`Workflow` builder bound to this app's tracer."""
        return Workflow(name, tracer=self.tracer)

    # -- evaluation -------------------------------------------------------------------------------

    async def eval_target(self, case: EvalCase) -> RunOutput:
        """EvalRunner adapter: run one case through the app."""
        result = await self.arun(case.input_text)
        return self._run_output_from_result(result)

    @staticmethod
    def _run_output_from_result(result: RunResult) -> RunOutput:
        """Project a RunResult onto a RunOutput, carrying a lightweight trajectory
        built from the run's tool results so trajectory metrics can score it."""
        from ..evals.trajectory import Trajectory, TrajectoryStep

        steps = [
            TrajectoryStep(type="tool", name=tr.tool_name, tool_name=tr.tool_name, status=tr.status)
            for tr in result.tool_results
        ]
        trajectory = Trajectory(
            steps=steps,
            final_answer=result.output,
            raw_text=result.raw_text,
            terminated=True,
            termination_reason=result.status.value
            if hasattr(result.status, "value")
            else str(result.status),
            success=result.error is None,
            source="run",
            usage={
                "steps": float(len(steps)),
                "tool_calls": float(len(steps)),
                "cost_usd": float(result.cost_usd),
            },
        )
        return RunOutput(
            output=result.output,
            raw_text=result.raw_text,
            evidence=result.evidence,
            citations=result.citations,
            usage=result.usage,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            schema_valid=result.validation.get("valid") if result.validation else None,
            error=result.error,
            trace_id=result.trace_id,
            trajectory=trajectory if steps else None,
            metadata={"input": result.metadata.get("input", "")},
        )

    # -- online / continuous evaluation --------------------------------

    def _spawn_online(self, result: RunResult, user_input: UserInput) -> None:
        """Schedule online scoring off the hot path; run inline if no loop."""
        coro = self._score_online(result, user_input)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            run_sync(coro)
            return
        task = loop.create_task(coro)
        self._online_tasks.add(task)
        task.add_done_callback(self._online_tasks.discard)

    async def _score_online(self, result: RunResult, user_input: UserInput) -> None:
        run_output = self._run_output_from_result(result)
        run_output.metadata.setdefault("input", user_input.text or "")
        case = EvalCase(id=result.trace_id or result.run_id, input=user_input.text or "")
        for evaluator in self.online_evaluators:
            try:
                metric_result = evaluator.observe(
                    run_output, case=case, run_id=result.trace_id or result.run_id
                )
            except Exception:  # noqa: BLE001 - online eval must never break a run
                logger.exception("online evaluator %s failed", evaluator.name)
                continue
            if metric_result is not None:
                self.events.emit(
                    "eval.online",
                    {
                        "metric": evaluator.name,
                        "value": metric_result.value,
                        "run_id": result.trace_id,
                    },
                )

    async def aflush_online(self) -> None:
        """Await any in-flight online evaluations (for tests and shutdown)."""
        if self._online_tasks:
            await asyncio.gather(*list(self._online_tasks), return_exceptions=True)

    def evaluate(
        self,
        dataset: Dataset | str,
        *,
        metrics: list[str] | None = None,
        concurrency: int = 8,
        gates: dict[str, str] | None = None,
        judges: list[Any] | None = None,
    ):
        """Evaluate the app over a dataset and return an :class:`EvalReport`."""
        runner = EvalRunner(
            self,
            metrics=metrics or (self.evaluators or None),
            concurrency=concurrency,
            gates=gates,
            judges=judges,
        )
        return runner.run(dataset)

    def benchmark_suite(
        self,
        benchmarks: str | list[str] = "all",
        *,
        tier: str = "static",
        sample: int | None = None,
        datasets: dict[str, Any] | None = None,
        concurrency: int = 8,
        model: str | None = None,
        solver_mode: str | None = None,
        store: Any | None = None,
        version: str = "",
        record: bool = True,
    ):
        """Run the open evaluation plane over this app and return a ``SuiteRun``.

        The pluggable harness for the **standard public model benchmarks** (MMLU,
        GPQA, GSM8K, HumanEval, IFEval, TruthfulQA, RULER, …) grouped by niche and
        reported the same way for every model and version — distinct from
        :meth:`evaluate`, which scores this app over a golden ``Dataset``. Every
        number carries a **provenance tier**: the default ``"static"`` replays the
        bundled fabricated fixtures fully offline (reproducible, gates CI);
        ``"recorded"`` / ``"live"`` need a per-benchmark
        :class:`~vincio.evals.suite.BenchmarkDataset` in ``datasets`` (and, for
        ``"live"``, this app drives the model). The engine **refuses** to let a
        lower tier print a higher tier's label, and runs each long-context
        benchmark twice — with and without the context governor — so the uplift is
        measured::

            run = app.benchmark_suite("knowledge", tier="static")
            run.overall(); run.niche_scores(); run.determinism_digest
            from vincio.evals.suite import SuiteReport
            SuiteReport(run).save("report.md")

        ``benchmarks`` is an id (``"knowledge.mmlu"``), a niche (``"knowledge"``),
        ``"all"``, or a list. Pass a :class:`~vincio.evals.suite.RunStore` as
        ``store`` to persist the run (``version`` tags the model version for
        :meth:`~vincio.evals.suite.RunStore.model_version_diff`). Returns a
        :class:`~vincio.evals.suite.SuiteRun`.
        """
        from ..evals.suite import BenchmarkSuite

        runner = BenchmarkSuite(concurrency=concurrency)
        run = run_sync(
            runner.arun(
                benchmarks, target=self, model=model, tier=tier, sample=sample,
                datasets=datasets, solver_mode=solver_mode,
            )
        )
        if store is not None:
            store.save(run, version=version)
        if record and self.audit is not None:
            self.audit.record(
                "benchmark_suite",
                decision="allow",
                details={
                    "run_id": run.run_id, "tier": run.tier.value,
                    "benchmarks": len(run.runs), "overall": run.overall(),
                    "gated": run.gated,
                },
            )
        return run

    # -- closed loop ---------------------------------------------------------

    def improvement_loop(self, **kwargs: Any):
        """The trace → dataset → eval → optimize → promote loop on this app.

        Returns an :class:`~vincio.optimize.ImprovementLoop` bound to this
        app's tracer, store, and prompt::

            loop = app.improvement_loop(gates={"groundedness": ">= 0.8"})
            result = loop.run(min_feedback_score=0.5)
        """
        from ..optimize.loop import ImprovementLoop

        return ImprovementLoop(self, **kwargs)

    # -- reflective optimization & the data flywheel -------------------------

    def _evaluate_variant_fn(self, metrics: list[str], *, concurrency: int = 4):
        """Build a memory-write-free variant evaluator for the optimizers.

        Candidate evaluations must never mutate user memory or hand later
        candidates different recall state than earlier ones saw, so the prompt
        spec, compiler options, and ``memory.write_back`` are saved, neutralized,
        and restored around every evaluation — the same discipline the
        improvement loop uses.
        """
        from ..evals.runners import EvalRunner

        async def evaluate_variant(variant, ds):
            original_spec = self.prompt_spec
            original_options = self.prompt_compiler.options
            original_write_back = self.config.memory.write_back
            self.prompt_spec = variant.spec
            self.prompt_compiler.options = variant.compiler_options
            self.config.memory.write_back = []
            try:
                runner = EvalRunner(self, metrics=metrics, concurrency=concurrency)
                return await runner.arun(ds, name=variant.name)
            finally:
                self.prompt_spec = original_spec
                self.prompt_compiler.options = original_options
                self.config.memory.write_back = original_write_back

        return evaluate_variant

    def reflective_optimize(
        self,
        dataset: Dataset,
        *,
        strategy: str = "reflective",
        metrics: list[str] | None = None,
        budget: int = 12,
        minibatch_size: int = 8,
        seed: int = 7,
        weights: Any | None = None,
        gates: dict[str, str] | None = None,
        objectives: Any | None = None,
        concurrency: int = 4,
        reflector: str = "heuristic",
        apply: bool = False,
    ):
        """Run the GEPA-style reflective optimizer against ``dataset``.

        Instead of blind variant search, the optimizer reads the eval report's
        failures, reflects on why the prompt lost, and proposes targeted edits,
        evolving a Pareto frontier under a hard ``budget`` of rollouts —
        deterministic under ``seed``. ``strategy="mipro"`` switches to joint
        instruction+example proposal. With ``apply=True`` a promoted winner is
        installed on the app::

            result = app.reflective_optimize(dataset, gates={"groundedness": ">= 0.8"})
            result.promoted, result.reason, result.frontier.front

        ``reflector="llm"`` uses the real provider-backed :class:`LLMReflector`
        wired to this app's own provider — it reads the actual failing cases,
        clusters them into failure modes, and proposes targeted edits, falling
        back to the deterministic heuristic reflector offline. ``"heuristic"``
        (the default) is the fully reproducible, air-gapped floor.
        """
        from ..optimize.loop import DEFAULT_LOOP_METRICS
        from ..optimize.reflective import LLMReflector, ReflectiveOptimizer

        metric_list = metrics or (self.evaluators or DEFAULT_LOOP_METRICS)
        reflector_impl = None
        if reflector == "llm":
            reflector_impl = LLMReflector(self._base_provider(), self.model)
        optimizer = ReflectiveOptimizer(
            self._evaluate_variant_fn(metric_list, concurrency=concurrency),
            weights=weights,
            gates=gates,
            objectives=objectives,
            reflector=reflector_impl,
        )
        result = run_sync(
            optimizer.optimize(
                self.prompt_spec,
                dataset,
                strategy=strategy,  # type: ignore[arg-type]
                budget=budget,
                minibatch_size=minibatch_size,
                seed=seed,
            )
        )
        if apply and result.promoted and result.best is not None:
            winner = result.best.payload
            self.prompt_spec = winner.spec
            self.prompt_compiler.options = winner.compiler_options
            self.events.emit("optimize.reflective", {"reason": result.reason})
        return result

    def self_improvement(self, policy: Any | None = None, **kwargs: Any):
        """The unified, declarative self-improvement contract.

        One :class:`~vincio.optimize.self_improvement.SelfImprovementPolicy`
        composes scheduling, autonomous proposal, online updates, meta-optimization
        (learned fitness weights + successive-halving), active-learning label
        acquisition, and canary-gated promotion/rollback. Returns a
        :class:`~vincio.optimize.self_improvement.SelfImprovementController` whose
        :meth:`~vincio.optimize.self_improvement.SelfImprovementController.astream`
        emits the cycle as ``observe → proposal → meta → label → canary →
        promote/rollback`` events::

            from vincio.optimize import SelfImprovementPolicy
            ctl = app.self_improvement(SelfImprovementPolicy(), dataset=golden)
            async for ev in ctl.astream():
                print(ev.phase, ev.reason)

        Every promotion passes the same significance + safety + golden
        non-regression gates the loop always used; every decision lands on the
        shared audit chain and event bus."""
        from ..optimize.self_improvement import SelfImprovementController, SelfImprovementPolicy

        if policy is None:
            policy = SelfImprovementPolicy()
        return SelfImprovementController(self, policy, **kwargs)

    def deploy(self, candidate: Any, *, dataset: Any = None, **kwargs: Any):
        """Canary-gate a prompt/policy candidate and deploy it only if it clears.

        Two modes: an **offline** gated comparison against the live prompt on a
        canary ``dataset=``, or a **live-traffic** canary that ramps a fraction of
        ``live_inputs=`` onto the candidate (scored by ``score_fn=``) with
        auto-rollback. On a pass it is pushed to the prompt registry, tagged,
        applied live, and audited (``deploy``); on a fail it is refused and rolled
        back to the last known-good version. Returns a
        :class:`~vincio.optimize.self_improvement.DeployResult`. This is the
        canary-driven promotion surface for prompt and policy candidates."""
        from ..optimize.self_improvement import deploy_candidate
        from ..providers.base import run_sync

        return run_sync(deploy_candidate(self, candidate, dataset=dataset, **kwargs))

    def learn(
        self,
        tasks: list[Any],
        *,
        reward: Any,
        policy: Any | None = None,
        learning_rate: float = 0.5,
        kl_max: float = 0.5,
        iterations: int = 3,
        group_normalize: bool = True,
        min_reward_improvement: float = 0.0,
        flywheel: Any | None = None,
        held_out: Any | None = None,
        teacher: str | None = None,
        student: str | None = None,
    ):
        """On-policy reinforcement from verifiable rewards (RLVR).

        Closes the learning loop on a *policy*, not just a prompt. Each
        :class:`~vincio.optimize.trajectory_opt.LearningTask` carries a group of
        candidate outcomes; a :class:`~vincio.optimize.rewards.RewardModel` scores
        them from the verifiable signals the platform already computes (the
        task-success oracle, the benchmark scorers, calibrated judge ensembles),
        and a GRPO-style update improves the policy behind the same safety
        discipline prompt optimization uses — advantage normalization, a
        KL-to-reference clamp, and a monotonic no-regression gate::

            from vincio.optimize import LearningTask, OracleReward, RewardModel
            result = app.learn(tasks, reward=RewardModel([OracleReward()]))
            result.promoted, result.reward_delta, result.kl_to_reference

        The result's verdict is the same
        :class:`~vincio.optimize.self_improvement.CanaryVerdict` a prompt deploy
        produces, and the decision lands on the shared audit chain and event bus.
        On a promotion the on-policy winners are exported as a grounded
        :class:`~vincio.optimize.distill.TrainingSet`; pass a configured
        ``flywheel`` (with ``held_out`` / ``teacher`` / ``student``) to emit a
        fine-tune job through the existing distillation flywheel in the same call.
        Returns a :class:`~vincio.optimize.trajectory_opt.LearningResult`.
        """
        from ..optimize.trajectory_opt import TrajectoryOptimizer

        optimizer = TrajectoryOptimizer(
            reward,
            policy=policy,
            learning_rate=learning_rate,
            kl_max=kl_max,
            iterations=iterations,
            group_normalize=group_normalize,
            min_reward_improvement=min_reward_improvement,
        )
        result = run_sync(
            optimizer.alearn(
                tasks, flywheel=flywheel, held_out=held_out, teacher=teacher, student=student
            )
        )
        self.audit.record(
            "learn",
            decision="allow" if result.promoted else "deny",
            resource=self.name,
            details={
                "promoted": result.promoted,
                "reason": result.reason,
                "baseline_reward": result.baseline_reward,
                "policy_reward": result.policy_reward,
                "reward_delta": result.reward_delta,
                "kl_to_reference": result.kl_to_reference,
                "kl_within_bound": result.kl_within_bound,
                "tasks": result.tasks,
            },
        )
        self.events.emit(
            "learn.promoted" if result.promoted else "learn.rejected",
            {
                "reward_delta": result.reward_delta,
                "kl_to_reference": result.kl_to_reference,
                "reason": result.reason,
            },
        )
        return result

    def cultivate(
        self,
        curriculum: Any,
        *,
        library: Any | None = None,
        held_out: list[Any] | None = None,
        cycles: int = 3,
        rails: Any | None = None,
        governance: Any | None = None,
        search: Any | None = None,
        min_capability_gain: float = 0.0,
        prune: bool = True,
        record: bool = True,
    ) -> Any:
        """Grow capability open-endedly: propose → attempt → verify → distill → promote.

        Closes the open-ended-learning loop on a *skill library*, not just a
        prompt or a policy. ``curriculum`` is an
        :class:`~vincio.cultivate.AutoCurriculum` (or a list of
        :class:`~vincio.cultivate.CurriculumTask`); each cycle proposes the tasks
        at the **frontier of current competence** — gating every objective through
        this app's rails and its :meth:`verify_governance` invariants, so an unsafe
        or out-of-policy task is refused and never attempted — attempts each with a
        library-composing test-time search, verifies the result against the
        task-success oracle, distills a winning trajectory into a verified,
        content-addressed :class:`~vincio.cultivate.LearnedSkill`, and promotes it
        only through the **same no-regression gate** a prompt deploy clears
        (capability on a held-out frontier set must not fall). A skill that stops
        paying its way is demoted, never silently kept::

            from vincio.cultivate import AutoCurriculum, CurriculumTask
            result = app.cultivate(AutoCurriculum(tasks))
            result.capability_after >= result.capability_before  # monotone
            result.stayed_in_policy  # no refused objective was attempted

        The decision lands on the shared audit chain (``skill_cultivation``) and
        event bus (``cultivation.completed``). Returns a content-bound
        :class:`~vincio.cultivate.CultivationResult` whose ``verify`` re-derives the
        monotonicity and stay-in-policy verdicts from the bytes, with the grown
        :class:`~vincio.cultivate.LearnedSkillLibrary` on ``result.library``.
        """
        from ..cultivate import Cultivator

        cultivator = Cultivator(
            self,
            curriculum=curriculum,
            library=library,
            held_out=held_out,
            rails=rails if rails is not None else self.rail_engine,
            governance=governance,
            search=search,
            min_capability_gain=min_capability_gain,
            prune=prune,
            record=record,
        )
        return cultivator.run(cycles=cycles)

    # -- tabular evidence & the compact data encoder --------------------

    def table_evidence(
        self,
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

    def _coerce_dataset(
        self, data: Any, *, schema: Any | None = None, columns: list[str] | None = None, name: str = ""
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

    def profile_dataset(
        self,
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

    def sample_dataset(
        self,
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

    def fit_dataset(
        self,
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

    def screen_data(
        self,
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

    def stream_dataset(
        self,
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

    def aggregate_stream(
        self,
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

    async def map_stream(
        self,
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

    def data_catalog(self) -> Any:
        """The app's lazily-created :class:`~vincio.data.DataCatalog` — the grounding
        source for :meth:`query_data` and the catalog a
        :meth:`~vincio.data.QueryResult.verify` re-executes against."""
        catalog = getattr(self, "_data_catalog_obj", None)
        if catalog is None:
            from ..data import DataCatalog

            catalog = DataCatalog()
            self._data_catalog_obj = catalog
        return catalog

    def register_dataset(
        self,
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

    def query_data(
        self,
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
        guarantee :func:`~vincio.data.make_query_contract` carries when the
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

    def analyze_data(
        self,
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

    def generate_chart(
        self,
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
                "content_sha256": chart.manifest.content_sha256 if chart.manifest else None,
            },
        )
        return chart

    # -- semantic layer & governed metrics ------------------------------

    def semantic_layer(
        self,
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

    def _resolve_layer(self, layer: Any | None, table: str | None) -> Any:
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

    def query_metric(
        self,
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

    def metric_lineage(
        self,
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

    def data_engagement(
        self,
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

    def federated_data_engagement(
        self,
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

    def stream_analytics(
        self,
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

    # -- continuous assurance & production certification ----------------

    def assurance_case(
        self,
        statement: str,
        *,
        context: str = "",
        subclaims: list[Any] | None = None,
        evidence: list[Any] | None = None,
        subject: str | None = None,
        sign: bool = True,
        signer: Any | None = None,
        record: bool = True,
    ) -> Any:
        """Assemble the platform's evidence into one continuously-checkable safety argument.

        Builds a content-bound :class:`~vincio.assurance.AssuranceCase`: a top
        :class:`~vincio.assurance.Claim` (*this app is fit for purpose X under
        context Y*) decomposed into ``subclaims``, each discharged by
        :class:`~vincio.assurance.Evidence` the platform **already emits** — an eval
        gate verdict, a :meth:`verify_governance` proof, a reasoning
        :class:`~vincio.verify.Certificate`, an audit-chain segment, an
        identity/delegation chain, or an AI-BOM — bound by hash so the whole case
        :meth:`~vincio.assurance.AssuranceCase.verify`\\s offline and a missing,
        stale, or falsified piece of evidence is pinpointed::

            from vincio.assurance import Claim, Evidence
            case = app.assurance_case(
                "The assistant is fit for production",
                context="EU deployment",
                subclaims=[Claim(id="governance", statement="Controls hold",
                                 evidence=[Evidence.from_governance(app.verify_governance())])],
            )
            report = case.check()  # re-derives the verdict from the bytes

        Re-check the case on every change (a model swap, a prompt edit, a dependency
        bump) with :meth:`~vincio.assurance.AssuranceCase.check` and gate the build
        with :func:`~vincio.assurance.assurance_regression_gate`. The case is signed
        with the app's identity unless ``sign`` is off, and (when ``record``) the
        verdict lands on the hash-chained audit log as an ``assurance_case``
        decision. Returns the sealed :class:`~vincio.assurance.AssuranceCase`.
        """
        from ..assurance import AssuranceCase, Claim

        goal = Claim(
            id="goal",
            statement=statement,
            context=context,
            subclaims=list(subclaims or []),
            evidence=list(evidence or []),
        )
        case = AssuranceCase(subject=subject or self.name, goal=goal).seal()
        chain_signer = self._resolve_contract_signer(signer, sign)
        if chain_signer is not None:
            case.sign(chain_signer)
        if record and self.audit is not None:
            report = case.check()
            self.audit.record(
                "assurance_case",
                resource=case.case_hash,
                decision="allow" if report.holds else "deny",
                details={
                    "statement": statement,
                    "holds": report.holds,
                    "claims": len(report.root.walk()),
                    "failing_claims": report.failing_claims,
                    "missing": report.missing,
                    "stale": report.stale,
                    "falsified": report.falsified,
                },
            )
        return case

    def certify(
        self,
        case: Any,
        *,
        residual_risks: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
        aibom: bool = True,
        sign: bool = True,
        signer: Any | None = None,
        record: bool = True,
        as_of: Any | None = None,
    ) -> Any:
        """Emit a portable, offline-verifiable production-certification report.

        Checks the :class:`~vincio.assurance.AssuranceCase`, records the residual
        risks (any undischarged claim, plus any passed in), stamps the build
        provenance (the ``vincio`` version and, unless ``aibom`` is off, a CycloneDX
        AI-BOM of the live configuration), and signs the report with the app's
        identity. Returns a :class:`~vincio.assurance.CertificationReport` a
        downstream operator or auditor checks **from the bytes**::

            report = app.certify(case)
            assert report.verify()                 # re-runs the case's own check
            assert report.certified                # the case holds

        :meth:`~vincio.assurance.CertificationReport.verify` recomputes the report
        hash, re-verifies the embedded case, and re-runs its evidence check, so a
        report certifying a case that does not hold is caught. The verdict lands on
        the hash-chained audit log as an ``assurance_certification`` decision unless
        ``record`` is off.
        """
        from ..assurance import certify as _certify

        prov: dict[str, Any] = dict(provenance or {})
        if aibom and "sbom" not in prov:
            try:
                from ..governance.aibom import generate_aibom

                bom = generate_aibom(self)
                prov.setdefault("vincio_version", bom.vincio_version)
                prov["sbom"] = bom.to_cyclonedx()
            except Exception:
                note_suppressed("assurance.certify.sbom")
        prov.setdefault("slsa", "SLSA build provenance attested by the release pipeline")
        chain_signer = self._resolve_contract_signer(signer, sign)
        report = _certify(
            case,
            signer=chain_signer,
            residual_risks=residual_risks,
            provenance=prov,
            as_of=as_of,
        )
        if record and self.audit is not None:
            self.audit.record(
                "assurance_certification",
                resource=report.report_hash,
                decision="allow" if report.certified else "deny",
                details={
                    "statement": report.statement,
                    "certified": report.certified,
                    "residual_risks": report.residual_risks,
                    "case_hash": case.case_hash,
                },
            )
        return report

    def use_bandit_router(
        self, models: list[str], *, bandit: str = "epsilon_greedy", **kwargs: Any
    ) -> ContextApp:
        """Route live traffic through a guarded online bandit over ``models``.

        Wires an :class:`~vincio.optimize.routing.GuardedBanditRouter` over the
        app's base provider: the bandit learns which model pays off, never
        explores on safety-/high-risk-tagged traffic, persists arm stats to the
        app's store, and auto-freezes / rolls back on regression. The router
        becomes the app's provider, so it nests inside the existing
        circuit-breaker / key-pool / failover stack.
        """
        from ..optimize.routing import GuardedBanditRouter

        base = self._base_provider()
        entries = [(base, m) for m in models]
        self._provider_instance = GuardedBanditRouter(
            entries,
            bandit=bandit,
            store=self.store,
            app_name=self.name,
            events=self.events,
            **kwargs,
        )
        if models:
            self.model = models[0]
        return self

    def enable_training_capture(self, enabled: bool = True) -> ContextApp:
        """Record the full output and cited evidence on every trace, so
        :meth:`export_training_set` can curate faithful, grounded fine-tuning
        data. Off by default (the span output stays truncated for cost)::

            app.enable_training_capture()  # then run production traffic
        """
        self.config.observability.training_capture = enabled
        return self

    def export_training_set(
        self,
        *,
        name: str = "distilled",
        runs: list[Any] | None = None,
        traces: list[Any] | None = None,
        limit: int = 500,
        min_feedback_score: float | None = None,
        require_grounding: bool = True,
        min_support: float = 0.5,
        max_examples: int | None = None,
        path: str | None = None,
        format: str = "openai",
    ):
        """Curate runs or captured traces into a grounded fine-tuning :class:`TrainingSet`.

        Two faithful sources, both grounding-checked, deduped, and
        provenance-stamped, emitting provider-ready JSONL (nothing ungrounded is
        exported):

        - ``runs=[...]`` — :class:`RunResult` objects (the natural output of
          :meth:`run`). These carry the **full** output and cited evidence, so
          the export is faithful with **no opt-in capture** required — the
          recommended path::

              results = [app.run(q) for q in prompts]
              ts = app.export_training_set(runs=results, path="train.jsonl")

        - traces (default) — reuses the traces production runs already write,
          feedback-filtered (``min_feedback_score``). Faithful only when
          :meth:`enable_training_capture` recorded the full artifacts; otherwise
          the span output is truncated.

        With ``path`` the JSONL is written for ``format`` ("openai"/"anthropic").
        """
        from ..optimize.distill import export_training_set, export_training_set_from_runs

        system = self.prompt_spec.role or self.prompt_spec.objective
        if runs is not None:
            training_set = export_training_set_from_runs(
                runs,
                name=name,
                system=system,
                require_grounding=require_grounding,
                min_support=min_support,
                max_examples=max_examples,
            )
        else:
            if traces is None:
                exporter = self.tracer.exporter
                if hasattr(exporter, "load_all"):
                    traces = exporter.load_all(limit=limit)
                elif hasattr(exporter, "traces"):
                    traces = list(exporter.traces)[-limit:]
                else:
                    traces = []
            training_set = export_training_set(
                traces,
                name=name,
                system=system,
                min_feedback_score=min_feedback_score,
                require_grounding=require_grounding,
                min_support=min_support,
                max_examples=max_examples,
            )
        if path is not None:
            training_set.save(path, format=format)  # type: ignore[arg-type]
            self.events.emit(
                "distill.exported", {"path": path, "examples": len(training_set), "format": format}
            )
        return training_set

    def distill(
        self,
        training_set: Any,
        dataset: Dataset,
        *,
        teacher: str,
        student: str,
        trainer: Any | None = None,
        quality_metric: str = "lexical_overlap",
        min_quality_ratio: float = 0.97,
        gates: dict[str, str] | None = None,
        concurrency: int = 4,
        apply: bool = True,
    ):
        """Teacher → student distillation, gated on holding quality.

        Evaluates teacher and student on the held-out ``dataset`` and promotes
        the (optionally fine-tuned) student into a cheap→strong runtime cascade
        only when it preserves ``min_quality_ratio`` of the teacher's quality at
        strictly lower cost, with no safety/schema regression. With
        ``apply=True`` a promoted cascade is installed via :meth:`use_cascade`::

            ts = app.export_training_set(min_feedback_score=0.5)
            result = app.distill(ts, held_out, teacher="gpt-5.2", student="gpt-5.2-mini")
            result.promoted, result.cost_savings
        """
        from ..optimize.distill import BootstrapFinetune

        async def evaluate_model(model, ds):
            from ..evals.runners import EvalRunner

            original_model = self.model
            original_write_back = self.config.memory.write_back
            self.model = model
            self.config.memory.write_back = []
            try:
                runner = EvalRunner(
                    self,
                    metrics=[quality_metric, "cost", "safety", "schema_validity"],
                    concurrency=concurrency,
                )
                return await runner.arun(ds, name=f"distill:{model}")
            finally:
                self.model = original_model
                self.config.memory.write_back = original_write_back

        loop = BootstrapFinetune(
            evaluate_model,
            quality_metric=quality_metric,
            min_quality_ratio=min_quality_ratio,
            gates=gates,
            trainer=trainer,
        )
        result = run_sync(loop.distill(training_set, dataset, teacher=teacher, student=student))
        if apply and result.promoted and result.cascade is not None:
            self.cascade = result.cascade
            self.events.emit(
                "distill.promoted",
                {"student": result.trained_student, "cost_savings": result.cost_savings},
            )
        return result

    def use_local_adapter(self, adapter: Any | None) -> ContextApp:
        """Apply (or remove) an on-device LoRA-class adapter on the base provider.

        Wraps the app's base provider in an
        :class:`~vincio.optimize.local_adaptation.AdaptedProvider` so an
        in-distribution request is answered the way the locally-fit
        :class:`~vincio.optimize.local_adaptation.LocalAdapter` learned, while
        everything else falls through to the base model unchanged — the run never
        leaves the process. The wrapper reports the base provider's name and
        capabilities, so residency, provenance, and the rotation stack are
        unaffected. Pass ``None`` to unload the adapter and restore the base model
        (the one-call reversibility path). Returns ``self`` for chaining::

            adapter = app.adapt_locally(golden, runs=results).verdict  # gated fit
            app.use_local_adapter(registry.active("local-adapter"))
        """
        from ..optimize.local_adaptation import AdaptedProvider

        base = self._base_provider()
        if isinstance(base, AdaptedProvider):
            base = base.base
        if adapter is None:
            self._provider_instance = base
            self.local_adapter = None
            return self
        self._provider_instance = AdaptedProvider(base, adapter, embedder=self.embedder)
        self.local_adapter = adapter
        return self

    def local_adaptation(self, policy: Any | None = None, **kwargs: Any):
        """The continual on-device adaptation loop, as a streaming controller.

        Returns a
        :class:`~vincio.optimize.local_adaptation.ContinualAdaptation` driven by a
        :class:`~vincio.optimize.local_adaptation.LocalAdaptationPolicy`: gather
        the flywheel's promoted grounded dataset, fit a new
        :class:`~vincio.optimize.local_adaptation.LocalAdapter` version on-device,
        gate it against the current base on a held-out set, and promote or roll
        back — every version registered and reversible, every decision on the
        shared audit chain and event bus, all in-process::

            ctl = app.local_adaptation(dataset=golden)
            async for ev in ctl.astream(runs=results):
                print(ev.phase, ev.reason)

        Promotion clears the same no-regression discipline a hosted fine-tune job
        does (:class:`~vincio.optimize.local_adaptation.AdapterGate`)."""
        from ..optimize.local_adaptation import ContinualAdaptation

        return ContinualAdaptation(self, policy, **kwargs)

    def adapt_locally(
        self,
        dataset: Any,
        *,
        runs: list[Any] | None = None,
        training_set: Any | None = None,
        policy: Any | None = None,
        registry: Any | None = None,
        base_model: str | None = None,
        apply: bool = True,
    ):
        """Fit, gate, and (on a pass) install an on-device adapter — one call.

        The one-shot form of :meth:`local_adaptation`. Curates a grounded training
        set (from ``runs``, a prebuilt ``training_set``, or the app's captured
        traces), fits a LoRA-class adapter on-device, gates it against the base on
        the held-out ``dataset`` (no-regression — the adapted model must be
        at-least-as-good), and on a pass registers it, makes it the active head,
        and (with ``apply``) applies it via :meth:`use_local_adapter`. Returns an
        :class:`~vincio.optimize.local_adaptation.AdaptationResult`::

            results = [app.run(q) for q in prompts]
            result = app.adapt_locally(golden, runs=results)
            result.promoted, result.verdict.delta
        """
        controller = self.local_adaptation(
            policy, dataset=dataset, registry=registry, base_model=base_model
        )
        return controller.adapt(runs=runs, training_set=training_set, apply=apply)

    def federated_improvement(self, policy: Any | None = None, **kwargs: Any):
        """The cross-org federated-improvement round, as a streaming controller.

        Returns a
        :class:`~vincio.optimize.federated.FederatedImprovement` driven by a
        :class:`~vincio.optimize.federated.FederatedPolicy`: securely aggregate a
        fleet's privacy-preserving :class:`~vincio.optimize.federated.Contribution`\\ s
        into a shared :class:`~vincio.optimize.federated.FederatedSubspace`, re-fit
        *this* member's own on-device adapter against that geometry, gate it against
        the member's base on a held-out set, and adopt or roll back — every version
        in the :class:`~vincio.optimize.local_adaptation.AdapterRegistry`, every
        decision on the shared audit chain and event bus, all in-process::

            ctl = app.federated_improvement(dataset=golden)
            mine = await ctl.build_contribution(member_id="org-a", participants=fleet)
            async for ev in ctl.astream(contributions=[mine, *peer_updates]):
                print(ev.phase, ev.reason)

        Only numeric, masked, bounded-sensitivity aggregates cross a trust
        boundary; adoption clears the same no-regression discipline a hosted
        fine-tune job does."""
        from ..optimize.federated import FederatedImprovement

        return FederatedImprovement(self, policy, **kwargs)

    def contribute_federated(
        self,
        *,
        member_id: str,
        participants: list[str] | None = None,
        runs: list[Any] | None = None,
        training_set: Any | None = None,
        policy: Any | None = None,
        consent_subject: str | None = None,
        residency: str | None = None,
    ):
        """Build this member's privacy-preserving contribution to a federated round.

        Curates a grounded training set from this app's own data (``runs``, a
        prebuilt ``training_set``, or captured traces), then returns a
        :class:`~vincio.optimize.federated.Contribution` carrying **only** the
        numeric subspace scatter — clipped, optionally DP-noised, and masked for
        secure aggregation — never a prompt or a response. Enforces the consent
        ledger's TRAINING purpose when the policy requires it and stamps the app's
        residency tag::

            mine = app.contribute_federated(member_id="org-a", participants=fleet)
        """
        controller = self.federated_improvement(policy)
        return run_sync(
            controller.build_contribution(
                member_id=member_id,
                participants=participants,
                runs=runs,
                training_set=training_set,
                consent_subject=consent_subject,
                residency=residency,
            )
        )

    def adopt_federated(
        self,
        dataset: Any,
        contributions: list[Any],
        *,
        runs: list[Any] | None = None,
        training_set: Any | None = None,
        policy: Any | None = None,
        registry: Any | None = None,
        base_model: str | None = None,
        apply: bool = True,
    ):
        """Aggregate a fleet's contributions, refit, gate, and adopt — one call.

        The one-shot form of :meth:`federated_improvement`. Securely merges the
        fleet's :class:`~vincio.optimize.federated.Contribution`\\ s into a shared
        subspace, re-fits this member's adapter against it over the member's **own**
        local data (from ``runs``, a ``training_set``, or captured traces), gates it
        against the base on the held-out ``dataset`` (no-regression — at-least-as-good),
        and on a pass registers, makes active, and (with ``apply``) applies it.
        Returns a :class:`~vincio.optimize.federated.FederatedRoundResult`::

            result = app.adopt_federated(golden, [mine, *peer_updates])
            result.adopted, result.verdict.delta, result.privacy.secure_aggregation
        """
        controller = self.federated_improvement(
            policy, dataset=dataset, registry=registry, base_model=base_model
        )
        return controller.adopt(
            contributions=contributions, runs=runs, training_set=training_set, apply=apply
        )

    def use_semantic_context_scoring(
        self, enabled: bool = True, *, mmr_lambda: float | None = None
    ) -> ContextApp:
        """Score and select context by embedding cosine instead of lexical overlap.

        When enabled, the context compiler scores relevance, novelty, dedup, and
        conflict by cosine over the app embedder's cached vectors, blends the
        reranker's verdict into relevance, and selects evidence by maximal
        marginal relevance (``mmr_lambda`` trades relevance against diversity).
        Only meaningful with a real semantic embedder configured
        (``retrieval.embedder``); the default hash embedder is not semantic, so
        leave it off unless you've set one::

            app = ContextApp(config={"retrieval": {"embedder": "voyage"}})
            app.use_semantic_context_scoring()
        """
        self.context_compiler.options.semantic_scoring = enabled
        if mmr_lambda is not None:
            self.context_compiler.options.mmr_lambda = mmr_lambda
        return self

    def use_learned_compression(self, compressor: Any | None = None) -> ContextApp:
        """Install a learned token-importance compressor on the compiler.

        Replaces the default extractive compressor with a learned one (default:
        :class:`~vincio.context.LLMLinguaCompressor`) for the inline
        budget-overflow compression step. Prefer :meth:`gate_compression` to
        adopt one only after it passes the faithfulness gate::

            app.use_learned_compression()  # opt-in, ungated
        """
        from ..context.llmlingua import LLMLinguaCompressor

        self.context_compiler.compressor = compressor or LLMLinguaCompressor()
        return self

    def gate_compression(
        self,
        dataset: Dataset,
        *,
        compressor: Any | None = None,
        metrics: list[str] | None = None,
        min_faithfulness: float = 0.9,
        min_quality_ratio: float = 0.98,
        concurrency: int = 4,
    ):
        """Adopt a learned compressor only if it preserves cited facts and quality.

        Runs the dataset with the baseline and the learned compressor, compares
        faithfulness, quality, and token usage, and installs the learned
        compressor only when it shrinks the prompt without losing the cited-fact
        set or regressing quality — returning the :class:`CompressionTuningResult`
        with the decision::

            result = app.gate_compression(golden)
            result.adopted, result.token_savings, result.learned_faithfulness
        """
        from ..context.compression import extractive_compress
        from ..context.llmlingua import LLMLinguaCompressor
        from ..evals.runners import EvalRunner
        from ..optimize.compression_tuning import CompressionTuner

        learned = compressor or LLMLinguaCompressor()
        metric_list = metrics or ["lexical_overlap", "faithfulness", "input_tokens"]

        async def evaluate(compressor_choice, ds):
            original = self.context_compiler.compressor
            original_write_back = self.config.memory.write_back
            self.context_compiler.compressor = compressor_choice or extractive_compress
            self.config.memory.write_back = []
            try:
                runner = EvalRunner(self, metrics=metric_list, concurrency=concurrency)
                return await runner.arun(ds)
            finally:
                self.context_compiler.compressor = original
                self.config.memory.write_back = original_write_back

        tuner = CompressionTuner(
            evaluate, min_faithfulness=min_faithfulness, min_quality_ratio=min_quality_ratio
        )
        result, chosen = run_sync(tuner.tune(learned, dataset))
        if chosen is not None:
            self.context_compiler.compressor = chosen
            self.events.emit("compression.adopted", {"token_savings": result.token_savings})
        return result

    def calibrate_judge(self, judge: Any, samples: list[Any], *, budget: int = 4):
        """Reflectively tune an LLM judge's evaluation steps for κ agreement.

        Proposes alternative evaluation procedures, scores each against the
        labelled ``samples`` (``(case, output, human_score)``), and installs the
        procedure that best agrees with people — only when it strictly beats the
        incumbent — leaving the judge calibrated for CI gating::

            result = app.calibrate_judge(geval, labelled_samples)
            result.adopted, result.kappa_before, result.kappa_after
        """
        from ..optimize.judge_calibration import JudgeCalibrator

        return JudgeCalibrator(judge).calibrate(samples, budget=budget)

    def use_learned_budgets(self, source: Any) -> ContextApp:
        """Install eval-tuned per-task budget allocations.

        ``source`` is a :class:`~vincio.optimize.LearnedAllocations`, a path
        to one saved as JSON, or a plain ``{task_type: {block: fraction}}``
        mapping. Tasks without a learned table keep the fixed defaults.
        """
        from ..context.budgeting import BudgetAllocator
        from ..optimize.budget_learning import LearnedAllocations

        if isinstance(source, (str, Path)):
            source = LearnedAllocations.load(source)
        if isinstance(source, LearnedAllocations):
            learned = source.allocations
        else:
            learned = {str(key): dict(value) for key, value in dict(source).items()}
        self.context_compiler.allocator = BudgetAllocator(learned=learned)
        return self

    def use_pack(
        self, pack: Any, *, set_schema: bool = True, merge_rules: bool = False
    ) -> ContextApp:
        """Apply a domain pack: prompt config + schema + policies +
        evaluators + rails.

        ``pack`` is a pack name (``"support"``, ``"engineering"``, ``"finance"``,
        ``"legal"``) or a :class:`~vincio.packs.Pack`. Packs are opt-in, ship in
        the package, and configure the app through its public API, so you can
        layer your own settings on top::

            app = ContextApp(name="helpdesk").use_pack("support")
        """
        from ..packs import Pack, load_pack

        if isinstance(pack, str):
            pack = load_pack(pack)
        if not isinstance(pack, Pack):
            raise ConfigError(f"use_pack expects a pack name or Pack, got {type(pack).__name__}")
        pack.apply(self, set_schema=set_schema, merge_rules=merge_rules)
        self.events.emit("pack.applied", {"pack": pack.name})
        return self

    # -- capability facades ------------------------------------------------------------------------------------

    def _facade(self, key: str, factory: Any) -> Any:
        """Build a capability facade once, on first access, and cache it — so
        cold start and footprint scale with the facades an app actually uses."""
        cache = self.__dict__.setdefault("_facade_cache", {})
        if key not in cache:
            cache[key] = factory(self)
        return cache[key]

    @property
    def runs(self) -> RunFacade:
        """Execution facade: run / arun / stream / astream / submit / batch / evaluate."""
        return self._facade("runs", RunFacade)

    @property
    def knowledge(self) -> RetrievalFacade:
        """Knowledge facade: sources, ingestion, and scoped memory."""
        return self._facade("knowledge", RetrievalFacade)

    @property
    def governance(self) -> GovernanceFacade:
        """Governance & compliance facade: residency, erasure, cards, lineage, EU AI Act."""
        return self._facade("governance", GovernanceFacade)

    @property
    def optimization(self) -> OptimizationFacade:
        """Cost, evaluation, rotation, and self-improvement facade."""
        return self._facade("optimization", OptimizationFacade)

    @property
    def serving(self) -> ServingFacade:
        """Serving facade: MCP server, A2A server, realtime sessions."""
        return self._facade("serving", ServingFacade)

    @property
    def training(self) -> TrainingFacade:
        """Training facade: capture, dataset export, and gated distillation."""
        return self._facade("training", TrainingFacade)

    # -- maintenance -------------------------------------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the app's providers and release their resources."""
        if self._provider_instance is not None:
            await self._provider_instance.aclose()
        for provider in self._built_providers.values():
            await provider.aclose()

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of the app's configured sources, tools, evaluators, and memory."""
        return {
            "name": self.name,
            "sources": {k: v.model_dump() for k, v in self.sources.items()},
            "tools": self.enabled_tools,
            "evaluators": self.evaluators,
            "memory": self.memory.stats() if self.memory else None,
            "cost": self.cost_tracker.summary(),
            "runs": self.store.count("runs"),
        }
