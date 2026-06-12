"""ContextApp: the Vincio public API.

::

    from vincio import ContextApp

    app = ContextApp(name="docs_qa")
    app.add_source("docs", path="./docs", retrieval="hybrid")
    answer = app.run("How do I configure SSO?")
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..agents.executor import AgentExecutor
from ..agents.planner import Planner
from ..caching.base import InMemoryCache
from ..caching.compilation import ChunkCache, ContextCompileCache, PromptCompileCache
from ..caching.invalidation import InvalidationManager
from ..caching.layers import ResponseCache
from ..context.compiler import ContextCompiler, ContextCompilerOptions
from ..documents.loaders import load_directory, load_document
from ..evals.datasets import Dataset, EvalCase
from ..evals.metrics import RunOutput
from ..evals.runners import EvalRunner
from ..input.routers import InputRouter
from ..memory.engine import MemoryEngine
from ..memory.policies import MemoryWritePolicy
from ..memory.stores import SQLiteMemoryStore
from ..observability import build_exporter
from ..observability.costs import CostTracker
from ..observability.traces import Tracer
from ..output.repair import Repairer
from ..output.schemas import OutputContract, OutputSchema
from ..output.validators import SemanticValidator
from ..prompts.compiler import CompilerOptions, PromptCompiler
from ..prompts.templates import PromptSpec
from ..providers import build_provider
from ..providers.base import ModelProvider, run_sync
from ..retrieval.chunking import chunk_document
from ..retrieval.embeddings import CachedEmbedder, LocalHashEmbedder, ProviderEmbedder
from ..retrieval.engine import RetrievalEngine
from ..retrieval.graph_retrieval import EntityGraph
from ..retrieval.indexes import BM25Index, SearchFilter, VectorIndex, build_filter
from ..retrieval.rerankers import build_reranker
from ..security.access import AccessController, Principal
from ..security.audit import AuditLog
from ..security.policy import PolicyEngine
from ..storage.base import create_metadata_store
from ..tools.permissions import ToolPermissionChecker
from ..tools.registry import ToolRegistry
from ..tools.runtime import ToolRuntime
from ..workflows.engine import Workflow
from .config import VincioConfig, load_config
from .errors import ConfigError, ToolNotFoundError
from .events import EventBus
from .runtime import VincioRuntime
from .types import (
    Budget,
    Constraint,
    EvidenceItem,
    Example,
    FileRef,
    Instruction,
    Objective,
    PolicySet,
    RunConfig,
    RunResult,
    RunStreamEvent,
    TaskType,
    UserInput,
)

__all__ = ["ContextApp"]


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

    async def arun(self, objective: str, *, budget: Budget | None = None):
        budget = budget or self._app.budget.model_copy(update={"max_steps": self._max_steps})
        return await self._executor.run(objective, budget=budget)

    def run(self, objective: str, *, budget: Budget | None = None):
        return run_sync(self.arun(objective, budget=budget))


class ContextApp:
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
            build_exporter(self.config.observability.exporter, self.config.observability.traces_dir),
            sample_rate=self.config.observability.sample_rate,
        )
        self.cost_tracker = CostTracker()
        self.store = create_metadata_store(self.config.storage.metadata)
        self.audit = AuditLog(
            self.config.security.audit_dir if self.config.security.audit_log else None
        )
        self.access = AccessController(tenant_isolation=self.config.security.tenant_isolation)
        self.policy_engine = PolicyEngine(self.policies)
        self.input_router = InputRouter()

        # provider
        self._provider_name = (
            provider if isinstance(provider, str) else None
        ) or self.config.provider.default
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
        # Content-addressed compilation caches (0.2): unchanged inputs are
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
            ContextCompilerOptions(slim_packets=self.config.performance.slim_packets),
            cache=self.context_compile_cache,
        )
        self.prompt_compiler = PromptCompiler(CompilerOptions(), cache=self.prompt_compile_cache)

        # retrieval
        self.embedder = self._build_embedder()
        self.sources: dict[str, _SourceConfig] = {}
        self.retrieval: RetrievalEngine | None = None
        self._bm25: BM25Index | None = None
        self._vector: VectorIndex | None = None
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
            permission_checker=ToolPermissionChecker(self.access, allow_external=self.policies.allow_external_tools),
            tracer=self.tracer,
            cache_enabled=self.config.cache.tool_cache,
        )
        self.enabled_tools: list[str] = []

        # validation / repair / evaluators
        self.semantic_validators: dict[str, SemanticValidator] = {}
        self.repairer = Repairer(self.output_contract.repair_policy)
        self.evaluators: list[str] = []
        self.optimizers: list[str] = []

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
        return OutputContract.from_schema(schema, require_citations=self.policies.require_citations if hasattr(self, "policies") else False)

    def _build_embedder(self):
        kind = self.config.retrieval.embedder
        if kind == "local":
            return CachedEmbedder(LocalHashEmbedder())
        provider = build_provider(kind, self.config.provider)
        return CachedEmbedder(ProviderEmbedder(provider))

    def resolve_provider(self, run_config: RunConfig | None = None) -> ModelProvider:
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
        return Principal(
            user_id=user_input.user_id,
            tenant_id=user_input.tenant_id,
            scopes=list(self.policies.custom.get("scopes", ["*"])),
        )

    def tenant_filter(self, tenant_id: str | None) -> SearchFilter | None:
        if tenant_id is None or not self.config.security.tenant_isolation:
            return None
        return build_filter(tenant_id=tenant_id)

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
        self.policies.set(name, value)
        if name == "answer_only_from_sources" and value:
            self.policies.require_citations = True
            self.output_contract.require_citations = True
            if not self.prompt_spec.citation_policy:
                self.prompt_spec = self.prompt_spec.model_copy(
                    update={
                        "rules": [*self.prompt_spec.rules, "Use only the provided sources to answer."],
                        "citation_policy": "Cite evidence IDs in square brackets for every claim.",
                        "insufficient_evidence_behavior": "If the sources do not contain the answer, say so explicitly.",
                    }
                )
        if name == "require_citations":
            self.output_contract.require_citations = bool(value)
        self.policy_engine = PolicyEngine(self.policies)
        self.events.emit("policy.changed", {"policy": name})
        return self

    # -- sources / retrieval ----------------------------------------------------------------

    def _ensure_retrieval(self, retrieval: str) -> None:
        if self._bm25 is None:
            self._bm25 = BM25Index()
        if self._vector is None:
            self._vector = VectorIndex(self.embedder)
        indexes: list[Any]
        if retrieval == "bm25":
            indexes = [self._bm25]
        elif retrieval in ("dense", "vector"):
            indexes = [self._vector]
        else:  # hybrid / hybrid_graph
            indexes = [self._bm25, self._vector]
        reranker = build_reranker(self.config.retrieval.reranker)
        self.retrieval = RetrievalEngine(
            indexes,
            reranker=reranker,
            candidate_multiplier=self.config.retrieval.candidate_multiplier,
        )
        if retrieval in ("graph", "hybrid_graph") and self.entity_graph is None:
            self.entity_graph = EntityGraph()

    def add_source(
        self,
        name: str,
        *,
        path: str | None = None,
        documents: list[Any] | None = None,
        loader: str | None = None,
        chunking: str | None = None,
        retrieval: str = "hybrid",
    ) -> ContextApp:
        """Register a knowledge source: load, chunk, and index documents."""
        chunking = chunking or self.config.retrieval.chunking
        source = _SourceConfig(
            name=name, path=path, loader=loader, chunking=chunking, retrieval=retrieval
        )
        self._ensure_retrieval(retrieval)
        docs = list(documents or [])
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
            self.store.save("documents", {"id": document.id, "title": document.title, "source": name, "uri": document.source_uri})
        if all_chunks:
            run_sync(self._index_chunks(all_chunks))
            if self.entity_graph is not None:
                self.entity_graph.add_chunks(all_chunks)
        source.document_count = len(docs)
        source.chunk_count = len(all_chunks)
        self.sources[name] = source
        return self

    async def _index_chunks(self, chunks: list[Any]) -> None:
        if self._bm25 is not None:
            await self._bm25.add(chunks)
        if self._vector is not None:
            await self._vector.add(chunks)

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

    # -- memory ---------------------------------------------------------------------------------

    def add_memory(
        self,
        *,
        scope: str = "user",
        strategy: str = "semantic",
        store: Any | None = None,
    ) -> ContextApp:
        if store is None:
            metadata_url = self.config.storage.metadata
            if metadata_url.startswith("sqlite"):
                from ..storage.base import parse_storage_url

                _scheme, location = parse_storage_url(metadata_url)
                store = SQLiteMemoryStore(Path(location).with_name("memory.db"))
            else:
                from ..memory.stores import InMemoryMemoryStore

                store = InMemoryMemoryStore()
        self.memory = MemoryEngine(
            store,
            write_policy=MemoryWritePolicy(min_confidence=self.config.memory.min_confidence),
            decay_lambda=self.config.memory.decay_lambda,
            min_confidence=self.config.memory.min_confidence,
            graph_enabled=strategy in ("semantic_graph", "graph"),
        )
        self.memory_enabled = self.config.memory.enabled
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

    # -- evaluators / optimizers ----------------------------------------------------------------------

    def add_evaluator(self, name: str | Callable) -> ContextApp:
        if callable(name):
            from ..evals.metrics import METRICS

            METRICS[getattr(name, "__name__", f"custom_{len(METRICS)}")] = name
            name = getattr(name, "__name__", f"custom_{len(METRICS)}")
        self.evaluators.append(name)
        return self

    def add_validator(self, name: str, validator: SemanticValidator, *, blocking: bool = True) -> ContextApp:
        from ..output.schemas import ValidatorSpec

        self.semantic_validators[name] = validator
        self.output_contract.validators.append(ValidatorSpec(name=name, blocking=blocking))
        return self

    def add_optimizer(self, name: str) -> ContextApp:
        known = {"context_budget", "prompt_format", "retrieval_config", "model_routing"}
        if name not in known:
            raise ConfigError(f"unknown optimizer {name!r}; known: {sorted(known)}")
        if name not in self.optimizers:
            self.optimizers.append(name)
        return self

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
            self.objective = Objective(text=objective, task_type=TaskType.CLASSIFICATION if labels else TaskType.GENERAL)
            update["objective"] = objective
        if labels:
            update["rules"] = [*rules, f"Answer with exactly one of these labels: {', '.join(labels)}."]
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

    async def arun(
        self,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        config: RunConfig | None = None,
    ) -> RunResult:
        if isinstance(user_input, str):
            user_input = UserInput(text=user_input)
        else:
            user_input = user_input.model_copy(deep=True)
        if files:
            user_input.files.extend(FileRef(path=f) for f in files)
        if tenant_id is not None:
            user_input.tenant_id = tenant_id
        if user_id is not None:
            user_input.user_id = user_id
        if session_id is not None:
            user_input.session_id = session_id
        return await self._runtime.execute(user_input, config)

    def run(self, user_input: str | UserInput, **kwargs: Any) -> RunResult:
        return run_sync(self.arun(user_input, **kwargs))

    async def astream(
        self,
        user_input: str | UserInput,
        *,
        files: list[str] | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
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
        if isinstance(user_input, str):
            user_input = UserInput(text=user_input)
        else:
            user_input = user_input.model_copy(deep=True)
        if files:
            user_input.files.extend(FileRef(path=f) for f in files)
        if tenant_id is not None:
            user_input.tenant_id = tenant_id
        if user_id is not None:
            user_input.user_id = user_id
        if session_id is not None:
            user_input.session_id = session_id
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

    def agent(
        self,
        *,
        name: str | None = None,
        tools: list[str | Callable] | None = None,
        planner: str = "dag",
        max_steps: int = 8,
        evaluator: str | None = None,
        model: str | None = None,
    ) -> _AgentHandle:
        for tool in tools or []:
            self.add_tool(tool)
        planner_mode = {"dag": "static", "static": "static", "dynamic": "dynamic", "react": "react", "direct": "direct"}.get(planner, "static")
        provider = self.resolve_provider()
        agent_model = model or self.model
        planner_obj = Planner(
            mode=planner_mode,  # type: ignore[arg-type]
            provider=provider if planner_mode == "dynamic" else None,
            model=agent_model if planner_mode == "dynamic" else None,
            max_steps=max_steps,
        )
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
        if evaluator is not None:
            self.add_evaluator(evaluator)
        executor = AgentExecutor(
            provider,
            model=agent_model,
            planner=planner_obj,
            tool_runtime=self.tool_runtime if self.enabled_tools else None,
            tool_specs=self.tool_registry.specs(self.enabled_tools) if self.enabled_tools else [],
            retrieve_fn=retrieve_fn,
            output_validator=validator,
            tracer=self.tracer,
            cost_tracker=self.cost_tracker,
            system_prompt=self.prompt_compiler.compile(
                self.prompt_spec, variables=self.prompt_variables
            ).system_text,
        )
        return _AgentHandle(self, executor, max_steps)

    # -- workflows ------------------------------------------------------------------------------------------------

    def workflow(self, name: str) -> Workflow:
        return Workflow(name, tracer=self.tracer)

    # -- evaluation -------------------------------------------------------------------------------

    async def eval_target(self, case: EvalCase) -> RunOutput:
        """EvalRunner adapter: run one case through the app."""
        text = case.input_text
        result = await self.arun(text)
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
        )

    def evaluate(
        self,
        dataset: Dataset | str,
        *,
        metrics: list[str] | None = None,
        concurrency: int = 8,
        gates: dict[str, str] | None = None,
        judges: list[Any] | None = None,
    ):
        runner = EvalRunner(
            self,
            metrics=metrics or (self.evaluators or None),
            concurrency=concurrency,
            gates=gates,
            judges=judges,
        )
        return runner.run(dataset)

    # -- maintenance -------------------------------------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._provider_instance is not None:
            await self._provider_instance.aclose()
        for provider in self._built_providers.values():
            await provider.aclose()

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sources": {k: v.model_dump() for k, v in self.sources.items()},
            "tools": self.enabled_tools,
            "evaluators": self.evaluators,
            "memory": self.memory.stats() if self.memory else None,
            "cost": self.cost_tracker.summary(),
            "runs": self.store.count("runs"),
        }
