"""2.0 capability facades: ContextApp's surface decomposed into narrow,
lazily-constructed, independently-testable views."""

from __future__ import annotations

import pytest

from vincio import ContextApp
from vincio.core.facades import (
    GovernanceFacade,
    OptimizationFacade,
    RetrievalFacade,
    RunFacade,
    ServingFacade,
    TrainingFacade,
)


def test_facades_are_lazy_and_cached():
    app = ContextApp(name="t")
    # Not built until first access (cold start scales with what is used).
    assert "_facade_cache" not in app.__dict__
    runs = app.runs
    assert isinstance(runs, RunFacade)
    assert app.runs is runs  # cached: same instance on re-access
    assert app.__dict__["_facade_cache"]["runs"] is runs


def test_facade_exposes_only_its_group():
    app = ContextApp(name="t")
    # GovernanceFacade exposes model_card; it must NOT leak run().
    assert callable(app.governance.model_card)
    with pytest.raises(AttributeError):
        _ = app.governance.run


def test_facade_delegates_to_app_implementation():
    app = ContextApp(name="t")
    # The facade method IS the app's bound method (delegation, not a copy).
    assert app.runs.run == app.run
    assert app.optimization.add_evaluator == app.add_evaluator
    assert app.training.distill == app.distill


def test_all_six_named_facades_present():
    app = ContextApp(name="t")
    assert isinstance(app.runs, RunFacade)
    assert isinstance(app.knowledge, RetrievalFacade)
    assert isinstance(app.governance, GovernanceFacade)
    assert isinstance(app.optimization, OptimizationFacade)
    assert isinstance(app.serving, ServingFacade)
    assert isinstance(app.training, TrainingFacade)


def test_facade_run_executes():
    app = ContextApp(name="t")
    result = app.runs.run("What is 2 + 2?")
    assert result.trace_id
    # Equivalent to the flat API.
    assert hasattr(result, "output")


def test_facade_is_testable_in_isolation():
    # A facade can be constructed over a minimal stand-in, not the whole app.
    class _Stub:
        def model_card(self, **kw):
            return {"ok": True}

    facade = GovernanceFacade(_Stub())  # type: ignore[arg-type]
    assert facade.model_card()["ok"] is True


def test_dir_includes_group_methods():
    app = ContextApp(name="t")
    listing = dir(app.serving)
    assert "serve_mcp" in listing
    assert "serve_a2a" in listing
