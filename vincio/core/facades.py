"""Capability facades over :class:`~vincio.core.app.ContextApp` (2.0).

Five milestones of additive growth grew ``ContextApp`` into a ~3000-line
god-object that couples every feature. 2.0 decomposes its public surface into
narrow, independently-testable capability facades — ``RunFacade``,
``RetrievalFacade``, ``GovernanceFacade``, ``OptimizationFacade``,
``ServingFacade``, ``TrainingFacade`` — each exposing one cohesive method group
and delegating to the app's implementation. They are constructed lazily (on
first access), so cold start and memory footprint scale with what an app
actually uses, and each facade can be exercised in isolation.

The flat ``app.<method>`` API is unchanged and fully supported; the facades are
the organizing structure that makes the surface navigable and decoupled. A
facade exposes exactly its allow-listed method group — reaching for a method
that belongs to another facade raises ``AttributeError``, so the boundaries are
real rather than cosmetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import ContextApp

__all__ = [
    "CapabilityFacade",
    "RunFacade",
    "RetrievalFacade",
    "GovernanceFacade",
    "OptimizationFacade",
    "ServingFacade",
    "TrainingFacade",
]


class CapabilityFacade:
    """Base facade: a narrow, delegating view over an app's method group."""

    _METHODS: frozenset[str] = frozenset()

    def __init__(self, app: ContextApp) -> None:
        self._app = app

    def __getattr__(self, name: str) -> Any:
        # Only the facade's own group is reachable; delegate to the app's impl.
        if name in type(self)._METHODS:
            return getattr(self._app, name)
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}; "
            f"it exposes {sorted(type(self)._METHODS)}"
        )

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | type(self)._METHODS)


class RunFacade(CapabilityFacade):
    """Execution: single-shot, streaming, background, and batch runs."""

    _METHODS = frozenset(
        {"run", "arun", "stream", "astream", "submit", "batch", "abatch", "evaluate", "eval_target"}
    )


class RetrievalFacade(CapabilityFacade):
    """Knowledge: sources, ad-hoc ingestion, and scoped memory."""

    _METHODS = frozenset(
        {"add_source", "ingest_files", "add_memory", "remember", "recall", "enable_memory_os"}
    )


class GovernanceFacade(CapabilityFacade):
    """Governance & compliance: residency, erasure, cards, lineage, EU AI Act."""

    _METHODS = frozenset(
        {
            "check_residency",
            "set_residency",
            "erase_source",
            "model_card",
            "system_card",
            "compliance_report",
            "aibom",
            "trace_lineage",
            "mark_output",
            "risk_tier",
            "annex_iv",
            "fria",
        }
    )


class OptimizationFacade(CapabilityFacade):
    """Cost, evaluation, rotation, and self-improvement."""

    _METHODS = frozenset(
        {
            "set_cost_budget",
            "cost_report",
            "gate_swap",
            "agate_swap",
            "swap_regression",
            "aswap_regression",
            "watch_lifecycle",
            "add_evaluator",
            "add_validator",
            "add_optimizer",
            "add_online_evaluator",
            "add_metric_rail",
            "experiment",
            "improvement_loop",
            "reflective_optimize",
            "use_bandit_router",
            "self_improvement",
            "continuous_improvement",
            "experiment_proposer",
            "gate_compression",
            "calibrate_judge",
            "use_learned_budgets",
            "use_learned_compression",
            "use_semantic_context_scoring",
            "use_cascade",
            "use_router",
            "shadow",
            "canary",
        }
    )


class ServingFacade(CapabilityFacade):
    """Serving surfaces: MCP server, A2A server, realtime sessions, and the
    canary-gated deploy surface (3.0)."""

    _METHODS = frozenset({"serve_mcp", "serve_a2a", "realtime_session", "deploy"})


class TrainingFacade(CapabilityFacade):
    """Training: trace capture, dataset export, and gated distillation."""

    _METHODS = frozenset({"enable_training_capture", "export_training_set", "distill"})
