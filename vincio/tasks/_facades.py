"""Task-shaped constructors and their facades — the one-line front door.

Each constructor here configures a :class:`~vincio.core.app.ContextApp` with sane
governed defaults using the *same* public builder calls a caller would make by
hand, then wraps it in a thin facade exposing one task-shaped verb. Because the
configuration is the verbose path — :meth:`~vincio.core.app.ContextApp.add_source`
/ :meth:`~vincio.core.app.ContextApp.set_policy` /
:meth:`~vincio.core.app.ContextApp.add_evaluator` / ``output_schema`` /
:meth:`~vincio.core.app.ContextApp.add_tool` — the one-liner **lowers to the exact
same governed run packet** as writing those calls out longhand. Nothing new
happens; retrieval, grounding, validation, rails, budgets, tracing, and the audit
chain all apply unchanged, and ``facade.app`` is the escape hatch to every deep
method for the complex case.

The constructors are deliberately small and purely compositional, in the proven
:class:`~vincio.assistant.Assistant` / :class:`~vincio.settlement.CrossOrgEngagement`
/ :class:`~vincio.data.DataEngagement` mold. They are a *top layer*, not a new
capability.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..assistant import ApprovalRecord
from ..stability import experimental

if TYPE_CHECKING:
    from ..assistant import Assistant
    from ..core.app import ContextApp
    from ..core.config import VincioConfig
    from ..core.types import RunResult
    from ..providers.base import ModelProvider
    from ..tools.permissions import ApprovalRequest

__all__ = [
    "Extractor",
    "Evaluation",
    "RagTask",
    "ToolAgent",
    "chat",
    "evaluation",
    "extractor",
    "rag",
    "tool_agent",
]

# The grounded-RAG default evaluators — groundedness then citation accuracy, the
# same pair (and order) the canonical six-call RAG path adds. Turning grounding
# into "the evidence says so", measured.
DEFAULT_RAG_EVALUATORS: tuple[str, str] = ("groundedness", "citation_accuracy")


# --------------------------------------------------------------------------- #
# Shared lowering helpers — the verbose builder calls, written once.
# --------------------------------------------------------------------------- #
def _resolve_app(
    app: ContextApp | None,
    *,
    name: str,
    provider: ModelProvider | str | None,
    model: str | None,
    output_schema: Any | None = None,
    config: VincioConfig | str | None = None,
) -> ContextApp:
    """Return the app to configure: the supplied escape-hatch app, or a fresh one.

    When ``app`` is supplied it is used as-is (the escape hatch — provider / model
    / config are taken to already be set), with only an explicit ``output_schema``
    layered on so a typed task can re-shape an existing app's contract.
    """
    from ..core.app import ContextApp

    if app is not None:
        if output_schema is not None:
            # Mirror the constructor's normalization exactly so the contract is
            # byte-for-byte what ``ContextApp(output_schema=...)`` would build.
            app.output_contract = app._build_contract(output_schema)
        return app
    return ContextApp(
        name=name,
        provider=provider,
        model=model,
        output_schema=output_schema,
        config=config,
    )


def _apply_persona(
    app: ContextApp,
    *,
    role: str | None,
    objective: str | None,
    rules: Sequence[str] | None,
) -> None:
    """Apply an optional role / objective / rules persona via ``app.configure``."""
    if role is None and objective is None and rules is None:
        return
    app.configure(
        role=role,
        objective=objective,
        rules=list(rules) if rules is not None else None,
    )


def _register_tools(
    app: ContextApp,
    tools: Sequence[str | Callable[..., Any]],
    writes: Sequence[str | Callable[..., Any]],
) -> None:
    """Enable read tools and approval-gated write tools (least privilege)."""
    for tool in tools:
        app.add_tool(tool)
    for tool in writes:
        app.add_tool(tool, approval_required=True, side_effects="write")


def _add_one_source(
    app: ContextApp, name: str, spec: Any, *, chunking: str | None, retrieval: str
) -> None:
    if isinstance(spec, (str, Path)):
        app.add_source(name, path=str(spec), chunking=chunking, retrieval=retrieval)
    elif isinstance(spec, Mapping):
        kwargs: dict[str, Any] = dict(spec)
        kwargs.setdefault("chunking", chunking)
        kwargs.setdefault("retrieval", retrieval)
        app.add_source(name, **kwargs)
    elif isinstance(spec, Sequence):
        app.add_source(name, documents=list(spec), chunking=chunking, retrieval=retrieval)
    else:
        raise TypeError(
            f"source {name!r} must be a path, a list of documents, or an add_source "
            f"kwargs mapping; got {type(spec).__name__}"
        )


def _add_sources(app: ContextApp, sources: Any, *, chunking: str | None, retrieval: str) -> None:
    """Register knowledge sources from the ergonomic ``sources`` argument.

    Accepts a single path, a list of paths and/or in-memory documents, or a
    ``name -> spec`` mapping (spec = a path, a document list, or add_source
    kwargs). A single path indexes one ``"docs"`` source — exactly the canonical
    ``add_source("docs", path=..., chunking=..., retrieval=...)`` call.
    """
    if sources is None:
        return
    if isinstance(sources, (str, Path)):
        app.add_source("docs", path=str(sources), chunking=chunking, retrieval=retrieval)
        return
    if isinstance(sources, Mapping):
        for src_name, spec in sources.items():
            _add_one_source(app, str(src_name), spec, chunking=chunking, retrieval=retrieval)
        return
    if isinstance(sources, Sequence):
        paths = [s for s in sources if isinstance(s, (str, Path))]
        documents = [s for s in sources if not isinstance(s, (str, Path))]
        if documents:
            app.add_source("docs", documents=documents, chunking=chunking, retrieval=retrieval)
        single = len(paths) == 1 and not documents
        for index, path in enumerate(paths):
            src_name = "docs" if single else f"docs_{index}"
            app.add_source(src_name, path=str(path), chunking=chunking, retrieval=retrieval)
        return
    raise TypeError(
        "sources must be a path, a list of paths/documents, or a name->spec mapping; "
        f"got {type(sources).__name__}"
    )


# --------------------------------------------------------------------------- #
# Facades — a thin, transparent wrapper with one verb and a `.app` escape hatch.
# --------------------------------------------------------------------------- #
class _TaskBase:
    """Shared base for the one-line task facades.

    Holds the fully-configured :class:`~vincio.core.app.ContextApp` on
    :attr:`app` — the escape hatch to every deep method — and a ``feature`` label
    each subclass stamps on its runs for observability. Subclasses add exactly one
    task-shaped verb that lowers to :meth:`~vincio.core.app.ContextApp.run`;
    nothing is shadowed, and ``.app`` reaches everything the verbose path can.
    """

    feature: str = "task"

    def __init__(self, app: ContextApp) -> None:
        self.app = app

    def __repr__(self) -> str:
        return f"{type(self).__name__}(app={self.app.name!r})"


class RagTask(_TaskBase):
    """A one-line grounded-RAG question answerer over a governed ``ContextApp``.

    Built by :func:`rag`. :meth:`ask` (or calling the instance) runs a full,
    grounded, cited, eval-scored :meth:`~vincio.core.app.ContextApp.run` — the
    exact packet the canonical six-call verbose form compiles. ``.app`` is the
    escape hatch to retrieval tuning, rails, memory, and every other deep method.
    """

    feature = "rag"

    def ask(self, question: str, **kwargs: Any) -> RunResult:
        """Answer ``question`` grounded in the configured sources, with citations."""
        kwargs.setdefault("feature", self.feature)
        return self.app.run(question, **kwargs)

    async def aask(self, question: str, **kwargs: Any) -> RunResult:
        """Async :meth:`ask`."""
        kwargs.setdefault("feature", self.feature)
        return await self.app.arun(question, **kwargs)

    def __call__(self, question: str, **kwargs: Any) -> RunResult:
        return self.ask(question, **kwargs)


class Extractor(_TaskBase):
    """A one-line typed extractor: text in, a validated Pydantic object out.

    Built by :func:`extractor`. :meth:`extract` (or calling the instance) runs a
    full structured-output :meth:`~vincio.core.app.ContextApp.run` against the
    schema's contract — provider-native constrained decoding, streaming
    validation, and bounded self-correction all apply — and returns the validated
    object; :meth:`run` returns the whole :class:`~vincio.core.types.RunResult`.
    """

    feature = "extract"

    def extract(self, text: str, **kwargs: Any) -> Any:
        """Extract the schema's typed, validated object from ``text``."""
        return self.run(text, **kwargs).output

    async def aextract(self, text: str, **kwargs: Any) -> Any:
        """Async :meth:`extract`."""
        return (await self.arun(text, **kwargs)).output

    def run(self, text: str, **kwargs: Any) -> RunResult:
        """Run the extraction and return the whole :class:`RunResult`."""
        kwargs.setdefault("feature", self.feature)
        return self.app.run(text, **kwargs)

    async def arun(self, text: str, **kwargs: Any) -> RunResult:
        """Async :meth:`run`."""
        kwargs.setdefault("feature", self.feature)
        return await self.app.arun(text, **kwargs)

    def __call__(self, text: str, **kwargs: Any) -> Any:
        return self.extract(text, **kwargs)


class ToolAgent(_TaskBase):
    """A one-line approval-gated tool-using agent over a governed ``ContextApp``.

    Built by :func:`tool_agent`. :meth:`run` runs the governed model+tool loop:
    read tools run freely, while write tools are **denied by default** and
    surfaced as pending approvals (:attr:`pending_approvals`) — a one-shot reply
    can never silently run a write tool. :meth:`approve` pre-allows a tool by
    name; ``.app`` is the escape hatch to planners, budgets, and rails.
    """

    feature = "tool_agent"

    def __init__(self, app: ContextApp, *, auto_approve: Sequence[str] = ()) -> None:
        super().__init__(app)
        self._auto_approve: set[str] = set(auto_approve)
        self._approvals: list[ApprovalRecord] = []
        # Install the approval surface, chaining to any callback already
        # configured so an app-level policy still has the final say.
        permissions = app.tool_runtime.permissions
        self._prior_callback = permissions.approval_callback
        permissions.approval_callback = self._resolve_approval

    async def _resolve_approval(self, request: ApprovalRequest) -> bool:
        if request.tool in self._auto_approve:
            self._approvals.append(
                ApprovalRecord(
                    tool=request.tool,
                    arguments=dict(request.arguments),
                    status="approved",
                    reason="pre-approved",
                )
            )
            return True
        if self._prior_callback is not None:
            granted = await self._prior_callback(request)
            self._approvals.append(
                ApprovalRecord(
                    tool=request.tool,
                    arguments=dict(request.arguments),
                    status="approved" if granted else "denied",
                    reason="app policy",
                )
            )
            return granted
        # No standing decision: surface it and do not run the tool this turn.
        self._approvals.append(
            ApprovalRecord(
                tool=request.tool,
                arguments=dict(request.arguments),
                status="pending",
                reason="awaiting approval",
            )
        )
        return False

    def approve(self, tool: str) -> ToolAgent:
        """Pre-approve a write tool by name for subsequent runs."""
        self._auto_approve.add(tool)
        return self

    def revoke(self, tool: str) -> ToolAgent:
        """Remove a tool's standing approval."""
        self._auto_approve.discard(tool)
        return self

    def run(self, task: str, **kwargs: Any) -> RunResult:
        """Run the governed model+tool loop on ``task``."""
        self._approvals = []
        kwargs.setdefault("feature", self.feature)
        return self.app.run(task, **kwargs)

    async def arun(self, task: str, **kwargs: Any) -> RunResult:
        """Async :meth:`run`."""
        self._approvals = []
        kwargs.setdefault("feature", self.feature)
        return await self.app.arun(task, **kwargs)

    def __call__(self, task: str, **kwargs: Any) -> RunResult:
        return self.run(task, **kwargs)

    @property
    def approvals(self) -> list[ApprovalRecord]:
        """Every tool-approval decision made during the most recent run."""
        return list(self._approvals)

    @property
    def pending_approvals(self) -> list[ApprovalRecord]:
        """Write tools from the most recent run still awaiting a decision."""
        return [record for record in self._approvals if record.status == "pending"]


class Evaluation(_TaskBase):
    """A one-line offline evaluation over a governed ``ContextApp``.

    Built by :func:`evaluation`. :meth:`run` (or calling the instance) evaluates
    the bound dataset with the configured metrics and gates — the same
    :meth:`~vincio.core.app.ContextApp.evaluate` a caller would invoke, bundled
    with its dataset and gates so the common case is one call.
    """

    feature = "evaluation"

    def __init__(
        self,
        app: ContextApp,
        *,
        dataset: Any | None = None,
        gates: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(app)
        self.dataset = dataset
        self.gates = dict(gates) if gates is not None else None

    @property
    def metrics(self) -> list[str]:
        """The metric names this evaluation will score (the app's evaluators)."""
        return list(self.app.evaluators)

    def run(self, dataset: Any | None = None, *, gates: Mapping[str, str] | None = None) -> Any:
        """Evaluate ``dataset`` (or the bound one) with the configured metrics/gates."""
        target = dataset if dataset is not None else self.dataset
        if target is None:
            raise ValueError(
                "evaluation needs a dataset: pass evaluation(dataset=...) or .run(dataset)"
            )
        chosen_gates = gates if gates is not None else self.gates
        return self.app.evaluate(target, gates=dict(chosen_gates) if chosen_gates else None)

    def __call__(self, dataset: Any | None = None) -> Any:
        return self.run(dataset)


# --------------------------------------------------------------------------- #
# Constructors — the discoverable verbs. Each is one expression with governed
# defaults; pass app= to layer the task onto a fully-configured app.
# --------------------------------------------------------------------------- #
@experimental(since="5.3")
def rag(
    sources: Any | None = None,
    *,
    provider: ModelProvider | str | None = None,
    model: str | None = None,
    name: str = "rag",
    grounded: bool = True,
    evaluators: Sequence[str | Callable[..., Any]] = DEFAULT_RAG_EVALUATORS,
    role: str | None = None,
    objective: str | None = None,
    rules: Sequence[str] | None = None,
    output_schema: Any | None = None,
    chunking: str | None = None,
    retrieval: str = "hybrid",
    app: ContextApp | None = None,
    config: VincioConfig | str | None = None,
) -> RagTask:
    """Build a grounded-RAG question answerer in one expression.

    Indexes ``sources`` (a path, a list of paths/documents, or a ``name->spec``
    mapping), turns on grounding-only answering with citations, and adds the
    groundedness + citation-accuracy evaluators — the canonical six-call RAG path,
    as one call::

        answer = rag("./docs", provider=p, model=m).ask("What is the refund window?")
        print(answer.output, answer.citations, answer.eval_scores)

    ``grounded=False`` skips the answer-only policy, ``evaluators=()`` skips the
    default metrics, and ``app=`` layers the task onto a pre-configured app. The
    returned :class:`RagTask` lowers to the same packet as the verbose form, and
    ``.app`` is the escape hatch.
    """
    application = _resolve_app(
        app, name=name, provider=provider, model=model, output_schema=output_schema, config=config
    )
    _add_sources(application, sources, chunking=chunking, retrieval=retrieval)
    if grounded:
        application.set_policy("answer_only_from_sources", True)
    for evaluator in evaluators:
        application.add_evaluator(evaluator)
    _apply_persona(application, role=role, objective=objective, rules=rules)
    return RagTask(application)


@experimental(since="5.3")
def extractor(
    schema: Any,
    *,
    provider: ModelProvider | str | None = None,
    model: str | None = None,
    name: str = "extractor",
    role: str | None = None,
    objective: str | None = None,
    rules: Sequence[str] | None = None,
    app: ContextApp | None = None,
    config: VincioConfig | str | None = None,
) -> Extractor:
    """Build a typed structured-extraction task from a schema in one expression.

    ``schema`` is a Pydantic model, an :class:`~vincio.output.schemas.OutputSchema`,
    or a JSON-Schema dict; the returned :class:`Extractor` parses and validates
    every reply into it::

        get_ticket = extractor(TicketClassification, provider=p, model=m)
        ticket = get_ticket.extract("I was charged twice this month.")
        print(ticket.label, ticket.confidence)

    Lowers to the same contract a ``ContextApp(output_schema=...)`` builds;
    ``.app`` is the escape hatch (e.g. ``app.enable_self_correction()``).
    """
    application = _resolve_app(
        app, name=name, provider=provider, model=model, output_schema=schema, config=config
    )
    _apply_persona(application, role=role, objective=objective, rules=rules)
    return Extractor(application)


@experimental(since="5.3")
def tool_agent(
    tools: Sequence[str | Callable[..., Any]] = (),
    *,
    writes: Sequence[str | Callable[..., Any]] = (),
    approve: Sequence[str] = (),
    provider: ModelProvider | str | None = None,
    model: str | None = None,
    name: str = "agent",
    role: str | None = None,
    objective: str | None = None,
    rules: Sequence[str] | None = None,
    app: ContextApp | None = None,
    config: VincioConfig | str | None = None,
) -> ToolAgent:
    """Build an approval-gated tool-using agent in one expression.

    ``tools`` are read-only tools enabled freely; ``writes`` are write tools
    registered as approval-required, denied by default and surfaced as pending
    approvals (``approve=[...]`` pre-allows trusted ones)::

        agent = tool_agent(tools=[search_docs], writes=[create_ticket], provider=p, model=m)
        result = agent.run("Open a ticket for the duplicate charge")
        if agent.pending_approvals:
            agent.approve("create_ticket"); result = agent.run("yes, open it")

    Lowers to the governed model+tool loop of ``ContextApp.run``; ``.app`` is the
    escape hatch to planners, budgets, and rails.
    """
    application = _resolve_app(app, name=name, provider=provider, model=model, config=config)
    _register_tools(application, tools, writes)
    _apply_persona(application, role=role, objective=objective, rules=rules)
    return ToolAgent(application, auto_approve=approve)


@experimental(since="5.3")
def evaluation(
    dataset: Any | None = None,
    *,
    metrics: Sequence[str | Callable[..., Any]] = (),
    gates: Mapping[str, str] | None = None,
    provider: ModelProvider | str | None = None,
    model: str | None = None,
    name: str = "eval",
    role: str | None = None,
    objective: str | None = None,
    rules: Sequence[str] | None = None,
    app: ContextApp | None = None,
    config: VincioConfig | str | None = None,
) -> Evaluation:
    """Build an offline evaluation in one expression.

    Registers ``metrics`` (metric names or callables) as the app's evaluators and
    bundles the ``dataset`` and ``gates`` so scoring is one call::

        report = evaluation(golden, metrics=["groundedness", "citation_accuracy"],
                            gates={"groundedness": ">= 0.8"}, provider=p, model=m).run()
        print(report.gates, report.summary())   # gate verdicts + per-metric aggregates

    Lowers to :meth:`~vincio.core.app.ContextApp.evaluate`; ``.app`` is the escape
    hatch (judges, adaptive sampling, regression gates).
    """
    application = _resolve_app(app, name=name, provider=provider, model=model, config=config)
    for metric in metrics:
        application.add_evaluator(metric)
    _apply_persona(application, role=role, objective=objective, rules=rules)
    return Evaluation(application, dataset=dataset, gates=gates)


@experimental(since="5.3")
def chat(
    *,
    provider: ModelProvider | str | None = None,
    model: str | None = None,
    name: str = "chat",
    tools: Sequence[str | Callable[..., Any]] = (),
    writes: Sequence[str | Callable[..., Any]] = (),
    approve: Sequence[str] = (),
    user_id: str | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
    memory_writeback: bool = True,
    on_approval: Any | None = None,
    role: str | None = None,
    objective: str | None = None,
    rules: Sequence[str] | None = None,
    app: ContextApp | None = None,
    config: VincioConfig | str | None = None,
) -> Assistant:
    """Open a multi-turn, session-aware chat in one expression.

    A re-presentation of :meth:`~vincio.core.app.ContextApp.assistant`: every turn
    is a full governed run, threaded under one session with memory write-back and
    an approval surface for write tools::

        bot = chat(provider=p, model=m)
        print(bot.send("What's my refund window? My plan is Pro.").text)
        print(bot.send("Thanks!").text)   # remembers the thread

    ``tools`` / ``writes`` enable tools (writes approval-gated, ``approve=[...]``
    pre-allows); ``.app`` is the escape hatch.
    """
    application = _resolve_app(app, name=name, provider=provider, model=model, config=config)
    _register_tools(application, tools, writes)
    _apply_persona(application, role=role, objective=objective, rules=rules)
    return application.assistant(
        user_id=user_id,
        tenant_id=tenant_id,
        session_id=session_id,
        memory_writeback=memory_writeback,
        auto_approve=list(approve),
        on_approval=on_approval,
    )
