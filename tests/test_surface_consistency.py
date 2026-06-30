"""Gate: two-level public-surface consistency (vincio._surface).

``vincio.__all__`` is the frozen top-level contract; each public subpackage also
declares its own ``__all__``. This gate reconciles the two: it freezes the
classified subpackage surface in ``docs/reference/subpackage-surface.txt`` (so
any ``__all__`` change is a reviewed edit), proves every exported name resolves
to a live attribute (no dead surface like the symbols 6.0 removed), and pins the
handful of intentional top-level name collisions so a *new* one cannot slip in
silently.
"""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import vincio
from vincio import _surface

# The intentional top-level name collisions: a subpackage symbol whose name is
# also a *different* object in vincio.__all__. Each is a genuinely distinct class
# the frozen top level assigns to another owner, kept reachable here by deep
# import. A new collision must be added here deliberately (and will also move the
# frozen manifest), never appear by accident.
KNOWN_COLLISIONS = {
    ("caching", "CalibrationReport"),  # semantic-cache vs agents.world_model
    ("data", "Dataset"),  # tabular data.core vs evals.datasets
    ("retrieval", "QueryPlan"),  # retrieval.engine vs data.query
    ("verify", "Constraint"),  # verify.kernels vs core.types
}


def test_subpackage_surface_is_frozen():
    """The live subpackage surface must match the committed manifest.

    Any addition, removal, or TOP/DUP/SUB reclassification of a subpackage
    ``__all__`` symbol must be a deliberate edit to
    ``docs/reference/subpackage-surface.txt`` (regenerate with
    ``python -m vincio._surface --freeze``).
    """
    committed = _surface.load_surface()
    rendered = _surface.render_surface()
    if committed != rendered:
        committed_lines = set(committed.splitlines())
        rendered_lines = set(rendered.splitlines())
        added = sorted(rendered_lines - committed_lines)
        removed = sorted(committed_lines - rendered_lines)
        raise AssertionError(
            "subpackage surface drifted from the frozen manifest; if intentional, "
            "regenerate with `python -m vincio._surface --freeze` and review the diff. "
            f"added={added} removed={removed}"
        )


def test_no_dead_or_malformed_public_surface():
    """Every subpackage ``__all__`` resolves, with no duplicate/malformed entries."""
    problems = _surface.surface_problems()
    assert problems == [], "\n".join(problems)


def test_every_classification_tag_is_consistent():
    """Each tag matches reality: TOP is the same object, DUP a different one, SUB absent."""
    top = {name: getattr(vincio, name) for name in vincio.__all__ if name != "__version__"}
    for subpackage, rows in _surface.subpackage_surface().items():
        module = importlib.import_module(f"vincio.{subpackage}")
        for symbol, tag in rows:
            assert tag in {"TOP", "DUP", "SUB"}, (subpackage, symbol, tag)
            obj = getattr(module, symbol)
            if tag == "TOP":
                assert symbol in top and obj is top[symbol], (subpackage, symbol)
            elif tag == "DUP":
                assert symbol in top and obj is not top[symbol], (subpackage, symbol)
            else:
                assert symbol not in top, (subpackage, symbol)


def test_intentional_collisions_are_exactly_the_known_set():
    """The DUP collisions are exactly the documented set — a new one fails here."""
    dups = {
        (subpackage, symbol)
        for subpackage, rows in _surface.subpackage_surface().items()
        for symbol, tag in rows
        if tag == "DUP"
    }
    assert dups == KNOWN_COLLISIONS, (
        f"collision set changed: added={sorted(dups - KNOWN_COLLISIONS)} "
        f"removed={sorted(KNOWN_COLLISIONS - dups)}"
    )


def test_every_public_subpackage_imports_and_is_enumerated():
    """Every first-level public subpackage is importable offline and accounted for."""
    enumerated = set(_surface.public_subpackages())
    assert enumerated, "no public subpackages enumerated"
    for name in enumerated:
        assert not name.startswith("_")
        importlib.import_module(f"vincio.{name}")  # imports clean offline
    # Subpackages that declare an __all__ all appear in the surface.
    surfaced = set(_surface.subpackage_surface())
    assert surfaced <= enumerated


def test_render_is_idempotent():
    """Rendering the surface twice is byte-identical (deterministic ordering)."""
    assert _surface.render_surface() == _surface.render_surface()


# --- the gate bites: injected violations must be reported -------------------


def test_gate_detects_dead_symbol():
    fake = SimpleNamespace(Real=object())
    problems = _surface._module_problems("fake", ["Real", "Ghost"], fake)
    assert any("Ghost" in p and "dead surface" in p for p in problems)
    assert not any("Real" in p for p in problems)


def test_gate_detects_duplicate_entry():
    fake = SimpleNamespace(Thing=object())
    problems = _surface._module_problems("fake", ["Thing", "Thing"], fake)
    assert any("more than once" in p for p in problems)


def test_gate_detects_malformed_all():
    assert _surface._module_problems("fake", "not-a-list", object())
    fake = SimpleNamespace(Ok=object())
    problems = _surface._module_problems("fake", ["Ok", 123], fake)
    assert any("non-string" in p for p in problems)


# --- 6.0 regression guards: removals stay removed, additions stay added -----


def test_removed_dead_symbols_stay_removed():
    concurrency = importlib.import_module("vincio.core.concurrency")
    tokens = importlib.import_module("vincio.core.tokens")
    shapley = importlib.import_module("vincio.core.shapley")
    utils = importlib.import_module("vincio.core.utils")
    classifiers = importlib.import_module("vincio.input.classifiers")
    from vincio.memory.engine import MemoryEngine
    from vincio.prompts.templates import PromptSpec
    from vincio.retrieval.prefetch import SpeculativePrefetcher
    from vincio.retrieval.quantization import TwoStageIndex

    assert "race_with_timeout" not in concurrency.__all__
    assert not hasattr(concurrency, "race_with_timeout")
    assert "CallableTokenCounter" not in tokens.__all__
    assert not hasattr(tokens, "CallableTokenCounter")
    assert "ashapley_values" not in shapley.__all__
    assert not hasattr(shapley, "ashapley_values")
    assert "truncate_text" not in utils.__all__
    assert not hasattr(utils, "truncate_text")
    assert not hasattr(classifiers, "LLMTaskClassifier")
    assert not hasattr(PromptSpec, "build_ast")
    assert not hasattr(TwoStageIndex, "stats")
    assert "reranker" not in inspect.signature(SpeculativePrefetcher.__init__).parameters
    assert not hasattr(MemoryEngine, "for_tenant")


def test_two_public_exceptions_are_exported():
    errors = importlib.import_module("vincio.core.errors")
    for name in ("IdentityError", "GovernanceVerificationError"):
        assert name in errors.__all__, name
        cls = getattr(errors, name)
        assert issubclass(cls, errors.VincioError), name
