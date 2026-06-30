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

Design notes (why this stays SOLID as it grows):

* **One source of truth.** Each step lowers through :mod:`vincio.tasks._lowering`,
  the *same* helpers the task-shaped constructors use — so a flow and its facade
  twin emit identical builder calls by construction, never by parallel upkeep.
* **Open for extension.** A step is a typed, immutable :class:`_Step` object that
  knows how to :meth:`~_Step.apply` itself; adding a verb adds a step type, it
  does not grow a central ``if/elif`` dispatcher.
* **Immutable, robustly.** A flow carries its construction config in one frozen
  :class:`_FlowConfig` and an append-only tuple of steps; :meth:`_extend` threads
  exactly those two, so a new config field can never be silently dropped by a
  hand-written clone.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..stability import experimental
from . import _lowering as lower

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..core.config import VincioConfig
    from ..core.types import RunResult
    from ..providers.base import ModelProvider

__all__ = ["Flow"]


# --------------------------------------------------------------------------- #
# Steps — one immutable, self-applying value per pipeline verb. Each `apply`
# delegates to vincio.tasks._lowering, the single source of truth the facades
# share, so a flow lowers byte-for-byte to the verbose builder form.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Step:
    """A declared pipeline step: its verb and how to replay it as builder calls."""

    verb: str

    def apply(self, app: ContextApp) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _Retrieve(_Step):
    name: str = "docs"
    path: str | None = None
    documents: tuple[Any, ...] | None = None
    connector: Any | None = None
    loader: str | None = None
    chunking: str | None = None
    retrieval: str = "hybrid"

    def apply(self, app: ContextApp) -> None:
        lower.add_source(
            app,
            self.name,
            path=self.path,
            documents=list(self.documents) if self.documents is not None else None,
            connector=self.connector,
            loader=self.loader,
            chunking=self.chunking,
            retrieval=self.retrieval,
        )


@dataclass(frozen=True, slots=True)
class _Ground(_Step):
    only_from_sources: bool = True

    def apply(self, app: ContextApp) -> None:
        lower.apply_grounding(app, self.only_from_sources)


@dataclass(frozen=True, slots=True)
class _Call(_Step):
    role: str | None = None
    objective: str | None = None
    rules: tuple[str, ...] | None = None
    model: str | None = None

    def apply(self, app: ContextApp) -> None:
        lower.apply_persona(
            app,
            role=self.role,
            objective=self.objective,
            rules=self.rules,
        )
        if self.model is not None:
            app.model = self.model


@dataclass(frozen=True, slots=True)
class _Validate(_Step):
    schema: Any | None = None
    require_citations: bool | None = None

    def apply(self, app: ContextApp) -> None:
        if self.schema is not None:
            lower.apply_output_schema(app, self.schema)
        if self.require_citations is not None:
            lower.apply_require_citations(app, self.require_citations)


@dataclass(frozen=True, slots=True)
class _Evaluate(_Step):
    metrics: tuple[str | Callable[..., Any], ...] = ()

    def apply(self, app: ContextApp) -> None:
        lower.add_evaluators(app, self.metrics)


# --------------------------------------------------------------------------- #
# Config — the construction inputs, in one frozen value so cloning is total.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _FlowConfig:
    provider: ModelProvider | str | None = None
    model: str | None = None
    name: str = "flow"
    output_schema: Any | None = None
    base_app: ContextApp | None = None
    config: VincioConfig | str | None = None


@experimental(since="5.3")
class Flow:
    """An immutable, fluent pipeline that lowers to one governed run packet.

    Construct with a provider + model (a fresh app is built on first use) or wrap
    a fully-configured app with :meth:`over`. Chain :meth:`retrieve`,
    :meth:`ground`, :meth:`call`, :meth:`validate`, and :meth:`evaluate` — each
    returns a new Flow — then :meth:`run` (or :meth:`invoke`) to execute. The
    configured app is reachable via :attr:`app`, the escape hatch.
    """

    __slots__ = ("_config", "_steps", "_built")

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
        self._config = _FlowConfig(
            provider=provider,
            model=model,
            name=name,
            output_schema=output_schema,
            base_app=app,
            config=config,
        )
        self._steps: tuple[_Step, ...] = ()
        self._built: ContextApp | None = None

    @classmethod
    def over(cls, app: ContextApp) -> Flow:
        """Start a flow over a pre-configured app (the escape hatch, inbound)."""
        return cls(app=app)

    # -- immutability --------------------------------------------------------
    @classmethod
    def _with_steps(cls, config: _FlowConfig, steps: tuple[_Step, ...]) -> Flow:
        """Build a flow from its config + steps without re-parsing constructor args."""
        flow = cls.__new__(cls)
        flow._config = config
        flow._steps = steps
        flow._built = None
        return flow

    def _extend(self, step: _Step) -> Flow:
        """Return a new Flow with one more step; the receiver is unchanged."""
        return Flow._with_steps(self._config, (*self._steps, step))

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
            _Retrieve(
                verb="retrieve",
                name=name,
                path=path,
                documents=tuple(documents) if documents is not None else None,
                connector=connector,
                loader=loader,
                chunking=chunking,
                retrieval=retrieval,
            )
        )

    def ground(self, only_from_sources: bool = True) -> Flow:
        """Answer only from retrieved sources, with citations (the grounding policy)."""
        return self._extend(_Ground(verb="ground", only_from_sources=only_from_sources))

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
            _Call(
                verb="call",
                role=role,
                objective=objective,
                rules=tuple(rules) if rules is not None else None,
                model=model,
            )
        )

    def validate(self, schema: Any | None = None, *, require_citations: bool | None = None) -> Flow:
        """Constrain the output: a typed schema and/or required citations."""
        return self._extend(
            _Validate(verb="validate", schema=schema, require_citations=require_citations)
        )

    def evaluate(self, *metrics: str | Callable[..., Any]) -> Flow:
        """Score every run with these evaluators (``ContextApp.add_evaluator``)."""
        return self._extend(_Evaluate(verb="evaluate", metrics=tuple(metrics)))

    # -- lowering ------------------------------------------------------------
    def _build(self) -> ContextApp:
        """Materialize (once) the configured app by replaying the declared steps."""
        if self._built is not None:
            return self._built
        from ..core.app import ContextApp

        cfg = self._config
        app = cfg.base_app
        if app is None:
            app = ContextApp(
                name=cfg.name,
                provider=cfg.provider,
                model=cfg.model,
                output_schema=cfg.output_schema,
                config=cfg.config,
            )
        for step in self._steps:
            step.apply(app)
        self._built = app
        return app

    @property
    def app(self) -> ContextApp:
        """The configured :class:`~vincio.core.app.ContextApp` (built on first access)."""
        return self._build()

    @property
    def steps(self) -> list[str]:
        """The pipeline verbs declared so far, in order."""
        return [step.verb for step in self._steps]

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
