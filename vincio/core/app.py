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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..lager import LagerEngine

from pydantic import BaseModel

from ..caching.base import InMemoryCache
from ..caching.compilation import ChunkCache, ContextCompileCache, PromptCompileCache
from ..caching.invalidation import InvalidationManager
from ..caching.layers import ResponseCache
from ..context.anchors import AnchorSet
from ..context.compiler import ContextCompiler, ContextCompilerOptions
from ..evals.online import OnlineEvaluator
from ..governance.fertility import FertilityTracker
from ..governance.lineage import LineageIndex
from ..governance.residency import ResidencyPolicy
from ..input.routers import InputRouter
from ..memory.engine import MemoryEngine
from ..observability import build_exporter
from ..observability.costs import CostTracker
from ..observability.traces import Tracer
from ..output.repair import Repairer
from ..output.routing import SchemaRouter
from ..output.schemas import OutputContract, OutputSchema
from ..output.validators import SemanticValidator
from ..prompts.compiler import CompilerOptions, PromptCompiler
from ..prompts.templates import PromptSpec
from ..providers import build_provider
from ..providers.base import ModelProvider
from ..providers.cache_strategy import PromptCacheStrategy
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
from ..retrieval.sparse import SparseIndex
from ..security.access import AccessController, Principal
from ..security.audit import AuditLog
from ..security.pii import PIIDetector
from ..security.policy import PolicyEngine
from ..security.rails import RailEngine
from ..skills.library import SkillLibrary
from ..storage.base import build_metadata_store
from ..tools.permissions import ToolPermissionChecker
from ..tools.registry import ToolRegistry
from ..tools.runtime import ToolRuntime
from ._app_config import _ConfigVerbs
from ._app_crossorg import _CrossOrgVerbs
from ._app_data import _DataVerbs
from ._app_governance import _GovernanceVerbs
from ._app_knowledge import _KnowledgeVerbs
from ._app_media import _MediaVerbs
from ._app_optimize import _OptimizeVerbs
from ._app_runs import _RunVerbs
from ._app_serving import _ServingVerbs
from ._app_settlement import _SettlementVerbs
from ._app_support import (  # noqa: F401  # RunHandle is re-exported through
    RunHandle,  # __all__ (vincio/__init__.py imports it from here); the private
    _AgentHandle,  # support classes stay importable from vincio.core.app for
    _SourceConfig,  # compatibility with pre-split callers.
)
from .config import VincioConfig, load_config
from .errors import (
    ConfigError,
    ResidencyViolationError,
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
    Document,
    EvidenceItem,
    Instruction,
    Objective,
    PolicySet,
    RunConfig,
    UserInput,
)

__all__ = ["ContextApp", "RunHandle"]


logger = logging.getLogger("vincio.app")


class ContextApp(
    _ConfigVerbs,
    _GovernanceVerbs,
    _MediaVerbs,
    _KnowledgeVerbs,
    _ServingVerbs,
    _CrossOrgVerbs,
    _SettlementVerbs,
    _RunVerbs,
    _OptimizeVerbs,
    _DataVerbs,
):
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
        self._init_core(
            name=name,
            config=config,
            objective=objective,
            output_schema=output_schema,
            budget=budget,
            policies=policies,
            prompt_spec=prompt_spec,
        )
        self._init_infrastructure(name=name, model=model)
        self._init_provider(provider=provider, model=model)
        self._init_caches()
        self._init_compilers()
        self._init_retrieval_memory()
        self._init_tools_protocols()
        self._init_validation()
        self._init_optimization()

        self._runtime = VincioRuntime(self)

    # -- construction phases (called once, in order, by __init__) --------------------

    def _init_core(
        self,
        *,
        name: str,
        config: VincioConfig | str | None,
        objective: Objective | str | None,
        output_schema: type[BaseModel] | OutputSchema | dict[str, Any] | None,
        budget: Budget | None,
        policies: PolicySet | None,
        prompt_spec: PromptSpec | None,
    ) -> None:
        """Phase 1: identity, objective/prompt, output contract, and run policy/budget."""
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

    def _init_infrastructure(
        self,
        *,
        name: str,
        model: str | None,
    ) -> None:
        """Phase 2: events, tracing, cost, storage, audit, governance, and ledgers."""
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
        self.store = build_metadata_store(self.config.storage.metadata)
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

    def _init_provider(
        self,
        *,
        provider: ModelProvider | str | None,
        model: str | None,
    ) -> None:
        """Phase 3: provider identity/instance and the built-provider registries."""
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

    def _init_caches(
        self,
    ) -> None:
        """Phase 4: response and compilation caches."""
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

    def _init_compilers(
        self,
    ) -> None:
        """Phase 5: context/prompt compilers and provider-aware prompt caching."""
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

    def _init_retrieval_memory(
        self,
    ) -> None:
        """Phase 6: embedder, retrieval indexes, prefetch/caches, sources, and memory."""
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
        # Task-frame anchors: an always-on compact brief of PRD/spec/brand docs
        # injected as pinned evidence every run (populated by add_source(anchor=True)).
        self.anchors: AnchorSet = AnchorSet()
        self.retrieval: RetrievalEngine | None = None
        self._bm25: BM25Index | None = None
        self._vector: VectorIndex | None = None
        self._sparse: SparseIndex | None = None
        self._late_interaction: LateInteractionIndex | None = None
        self.entity_graph: EntityGraph | None = None
        # LAGER: when attached (use_lager), the lazy reasoning-driven evidence
        # loop replaces top-k retrieval for this app's runs. _source_documents
        # keeps each source's loaded documents so use_lager() can ingest the
        # registered corpus (purged by erase_source alongside everything else).
        self.lager_engine: LagerEngine | None = None
        self._source_documents: dict[str, list[Document]] = {}
        self.pending_evidence: list[EvidenceItem] = []
        self._ingested_files: dict[str, list[EvidenceItem]] = {}

        # memory
        self.memory_enabled = False
        self.memory: MemoryEngine | None = None

    def _init_tools_protocols(
        self,
    ) -> None:
        """Phase 7: tool registry/runtime, skills, and MCP clients."""
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
        # the governed browsing session, when app.use_web_search() enables it.
        self.web_browser: Any = None

    def _init_validation(
        self,
    ) -> None:
        """Phase 8: validators, repair, evaluators, and self-correction slots."""
        # validation / repair / evaluators
        self.semantic_validators: dict[str, SemanticValidator] = {}
        self.repairer = Repairer(self.output_contract.repair_policy)
        self.evaluators: list[str] = []
        self.optimizers: list[str] = []
        self.online_evaluators: list[OnlineEvaluator] = []
        self._online_tasks: set[asyncio.Task[Any]] = set()
        self.schema_router: SchemaRouter | None = None
        self.self_correction: dict[str, Any] | None = None

    def _init_optimization(
        self,
    ) -> None:
        """Phase 9: cascade, governors, adapters, and the cost ledger/budget manager."""
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
