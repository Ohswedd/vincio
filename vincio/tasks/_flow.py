"""``Flow`` — the fluent, immutable context-engineering pipeline.

The Vincio answer to LCEL. A :class:`Flow` threads the steps of a grounded run —
retrieve → ground → call → validate → evaluate — as a *value*: every step returns
a **new** Flow (nothing mutates in place), so a flow can be branched, reused, and
passed around safely. Calling :meth:`Flow.run` lowers the whole pipeline to one
governed :meth:`~vincio.core.app.ContextApp.run` packet — so retrieval, grounding,
validation, rails, budgets, tracing, and the audit chain all apply unchanged, and
the one expression compiles byte-for-byte to the verbose builder form.

``flow.app`` materializes the configured :class:`~vincio.core.app.ContextApp` (the
escape hatch to every deep method). It is a pure top layer: a flow adds no new
behavior, it only *spells the builder calls fluently*.

Example::

    answer = (
        Flow(provider=p, model=m)
        .retrieve("./docs", chunking="adaptive")
        .ground()
        .evaluate("groundedness", "citation_accuracy")
        .run("What is the refund window for the Pro plan?")
    )
    print(answer.output, answer.citations)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ..stability import experimental

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..core.config import VincioConfig
    from ..core.types import RunResult
    from ..providers.base import ModelProvider

__all__ = ["Flow"]

# One pipeline step: a verb plus its keyword arguments, replayed as a builder
# call when the flow is lowered to a ContextApp.
_Step = tuple[str, dict[str, Any]]


def _apply_step(app: ContextApp, verb: str, params: dict[str, Any]) -> None:
    """Replay one declared step as the public builder call it stands for."""
    if verb == "retrieve":
        path = params.get("path")
        app.add_source(
            params.get("name", "docs"),
            path=str(path) if path is not None else None,
            documents=params.get("documents"),
            connector=params.get("connector"),
            loader=params.get("loader"),
            chunking=params.get("chunking"),
            retrieval=params.get("retrieval", "hybrid"),
        )
    elif verb == "ground":
        app.set_policy("answer_only_from_sources", bool(params.get("only_from_sources", True)))
    elif verb == "call":
        role = params.get("role")
        objective = params.get("objective")
        rules = params.get("rules")
        if role is not None or objective is not None or rules is not None:
            app.configure(
                role=role,
                objective=objective,
                rules=list(rules) if rules is not None else None,
            )
        model = params.get("model")
        if model is not None:
            app.model = model
    elif verb == "validate":
        schema = params.get("schema")
        if schema is not None:
            # Mirror ContextApp(output_schema=...) exactly so the contract is
            # byte-for-byte identical to the verbose form.
            app.output_contract = app._build_contract(schema)
        require_citations = params.get("require_citations")
        if require_citations is not None:
            app.set_policy("require_citations", bool(require_citations))
    elif verb == "evaluate":
        for metric in params.get("metrics", ()):
            app.add_evaluator(metric)
    else:  # pragma: no cover - guarded by the typed step constructors
        raise ValueError(f"unknown flow step {verb!r}")


@experimental(since="5.3")
class Flow:
    """An immutable, fluent pipeline that lowers to one governed run packet.

    Construct with a provider + model (a fresh app is built on first use) or wrap
    a fully-configured app with :meth:`over`. Chain :meth:`retrieve`,
    :meth:`ground`, :meth:`call`, :meth:`validate`, and :meth:`evaluate` — each
    returns a new Flow — then :meth:`run` (or :meth:`invoke`) to execute. The
    configured app is reachable via :attr:`app`, the escape hatch.
    """

    def __init__(
        self,
        *,
        provider: ModelProvider | str | None = None,
        model: str | None = None,
        name: str = "flow",
        output_schema: Any | None = None,
        app: ContextApp | None = None,
        config: VincioConfig | str | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._name = name
        self._output_schema = output_schema
        self._base_app = app
        self._config = config
        self._steps: tuple[_Step, ...] = ()
        self._built: ContextApp | None = None

    @classmethod
    def over(cls, app: ContextApp) -> Flow:
        """Start a flow over a pre-configured app (the escape hatch, inbound)."""
        return cls(app=app)

    # -- immutability --------------------------------------------------------
    def _extend(self, verb: str, **params: Any) -> Flow:
        """Return a new Flow with one more step; the receiver is unchanged."""
        clone: Flow = Flow.__new__(Flow)
        clone._provider = self._provider
        clone._model = self._model
        clone._name = self._name
        clone._output_schema = self._output_schema
        clone._base_app = self._base_app
        clone._config = self._config
        clone._steps = (*self._steps, (verb, params))
        clone._built = None
        return clone

    # -- steps ---------------------------------------------------------------
    def retrieve(
        self,
        path: str | None = None,
        *,
        documents: Sequence[Any] | None = None,
        name: str = "docs",
        chunking: str | None = None,
        retrieval: str = "hybrid",
        connector: Any | None = None,
        loader: str | None = None,
    ) -> Flow:
        """Add a knowledge source (``ContextApp.add_source``). Chain for several."""
        return self._extend(
            "retrieve",
            path=path,
            documents=list(documents) if documents is not None else None,
            name=name,
            chunking=chunking,
            retrieval=retrieval,
            connector=connector,
            loader=loader,
        )

    def ground(self, only_from_sources: bool = True) -> Flow:
        """Answer only from retrieved sources, with citations (the grounding policy)."""
        return self._extend("ground", only_from_sources=only_from_sources)

    def call(
        self,
        *,
        role: str | None = None,
        objective: str | None = None,
        rules: Sequence[str] | None = None,
        model: str | None = None,
    ) -> Flow:
        """Shape the generation: persona (role/objective/rules) and/or the model."""
        return self._extend(
            "call",
            role=role,
            objective=objective,
            rules=list(rules) if rules is not None else None,
            model=model,
        )

    def validate(self, schema: Any | None = None, *, require_citations: bool | None = None) -> Flow:
        """Constrain the output: a typed schema and/or required citations."""
        return self._extend("validate", schema=schema, require_citations=require_citations)

    def evaluate(self, *metrics: str | Callable[..., Any]) -> Flow:
        """Score every run with these evaluators (``ContextApp.add_evaluator``)."""
        return self._extend("evaluate", metrics=tuple(metrics))

    # -- lowering ------------------------------------------------------------
    def _build(self) -> ContextApp:
        """Materialize (once) the configured app by replaying the declared steps."""
        if self._built is not None:
            return self._built
        from ..core.app import ContextApp

        app = self._base_app
        if app is None:
            app = ContextApp(
                name=self._name,
                provider=self._provider,
                model=self._model,
                output_schema=self._output_schema,
                config=self._config,
            )
        for verb, params in self._steps:
            _apply_step(app, verb, params)
        self._built = app
        return app

    @property
    def app(self) -> ContextApp:
        """The configured :class:`~vincio.core.app.ContextApp` (built on first access)."""
        return self._build()

    @property
    def steps(self) -> list[str]:
        """The pipeline verbs declared so far, in order."""
        return [verb for verb, _ in self._steps]

    def run(self, user_input: str, **kwargs: Any) -> RunResult:
        """Lower the pipeline to one governed run and execute it on ``user_input``."""
        kwargs.setdefault("feature", "flow")
        return self._build().run(user_input, **kwargs)

    async def arun(self, user_input: str, **kwargs: Any) -> RunResult:
        """Async :meth:`run`."""
        kwargs.setdefault("feature", "flow")
        return await self._build().arun(user_input, **kwargs)

    def invoke(self, user_input: str, **kwargs: Any) -> RunResult:
        """Alias for :meth:`run` (familiar to LCEL users)."""
        return self.run(user_input, **kwargs)

    def __call__(self, user_input: str, **kwargs: Any) -> RunResult:
        return self.run(user_input, **kwargs)

    def __repr__(self) -> str:
        return f"Flow(steps={self.steps})"
