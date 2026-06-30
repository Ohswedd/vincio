"""Wire-or-retire: every public capability is reachable through a production path.

A standing internal audit found capabilities that were implemented and even public
but that nothing could reach — no ``app.*`` verb, no example, no internal caller.
6.3 wired each one (or, where a primitive is genuinely an advanced deep-import API,
documented it as such). This module is the *static* guard that keeps it wired, built
on the scan idiom :mod:`vincio._surface`, :mod:`vincio._error_contract`, and
:mod:`vincio._observable_failure` use, and folded into the ``hygiene`` VincioBench
family.

It holds a small, frozen ledger of the capabilities 6.3 acted on. Each entry names
the formerly-unhooked **symbol**, the module that **defines** it, the **reach** that
now makes it usable (an importable dotted path that must resolve to a live
attribute — an ``app.*`` verb, an engine method, a registration helper, a public
class member), and its **disposition**:

* ``"wired"`` — given a real production caller (an ``app.*`` verb or an internal
  call site). The guard requires both that the reach resolves *and* that the symbol
  is referenced by production code **outside its defining module**, so a capability
  cannot quietly become dead surface again.
* ``"advanced_api"`` — a deliberate deep-import primitive with no ``app.*`` verb
  (``ContextCompiler.compile_streaming`` / ``recompile`` / ``CompileStreamEvent``),
  documented as advanced API. The guard requires only that the reach resolves, since
  by design it has no internal caller.

The detector is proven to *bite* by ``tests/test_wire_or_retire.py``: an injected
check whose reach does not resolve, and an injected ``"wired"`` symbol with no
production caller, are each reported.

Run ``python -m vincio._wire_or_retire`` to reproduce the check offline.
"""

from __future__ import annotations

import ast
import importlib
import os.path
from dataclasses import dataclass

__all__ = [
    "WireCheck",
    "WIRE_CHECKS",
    "reachability_problems",
    "wire_or_retire_conformant",
]

_PACKAGE_DIR = os.path.dirname(__file__)
_THIS_FILE = os.path.abspath(__file__)


@dataclass(frozen=True)
class WireCheck:
    """One 6.3 wire-or-retire capability and the production reach that proves it.

    *symbol* is the formerly-unhooked name; *defining_module* is its package-relative
    source path (excluded from the production-reference scan); *reach* is a
    ``"module:Qual.attr"`` path that must resolve to a live attribute; *disposition*
    is ``"wired"`` (needs a production caller too) or ``"advanced_api"`` (reach only).
    """

    name: str
    symbol: str
    defining_module: str
    reach: str
    disposition: str


# The frozen ledger of capabilities 6.3 acted on. Adding a capability here is a
# reviewed edit; the guard then holds it reachable for good.
WIRE_CHECKS: tuple[WireCheck, ...] = (
    WireCheck(
        name="reasoning_retrieval",
        symbol="ReasoningRetriever",
        defining_module="retrieval/reasoning_retrieval.py",
        reach="vincio.core.app:ContextApp.retrieve_facts",
        disposition="wired",
    ),
    WireCheck(
        name="blob_evidence_store",
        symbol="BlobEvidenceStore",
        defining_module="context/evidence_store.py",
        reach="vincio.core.app:ContextApp.use_context_governor",
        disposition="wired",
    ),
    WireCheck(
        name="memory_consolidation",
        symbol="promote_aged_episodes",
        defining_module="memory/consolidation.py",
        reach="vincio.core.app:ContextApp.consolidate_memory",
        disposition="wired",
    ),
    WireCheck(
        name="token_counter_registry",
        symbol="register_token_counter",
        defining_module="core/tokens.py",
        reach="vincio.providers.base:register_provider_token_counters",
        disposition="wired",
    ),
    WireCheck(
        name="compile_streaming",
        symbol="compile_streaming",
        defining_module="context/compiler.py",
        reach="vincio.context.compiler:ContextCompiler.compile_streaming",
        disposition="advanced_api",
    ),
    WireCheck(
        name="recompile",
        symbol="recompile",
        defining_module="context/compiler.py",
        reach="vincio.context.compiler:ContextCompiler.recompile",
        disposition="advanced_api",
    ),
    WireCheck(
        name="compile_stream_event",
        symbol="CompileStreamEvent",
        defining_module="context/compiler.py",
        reach="vincio.context:CompileStreamEvent",
        disposition="advanced_api",
    ),
)

_VALID_DISPOSITIONS = frozenset({"wired", "advanced_api"})

_UNRESOLVED = object()


def _resolve(reach: str) -> object:
    """Resolve a ``"module:Qual.attr"`` reach to its attribute, or ``_UNRESOLVED``.

    Imports the module and walks the dotted attribute chain, so a reach naming an
    ``app.*`` verb, an engine method, or a public class member is confirmed to be a
    live attribute (not a removed or misspelled one).
    """
    module_name, _, qual = reach.partition(":")
    try:
        obj: object = importlib.import_module(module_name)
        for attr in qual.split("."):
            obj = getattr(obj, attr)
    except (ImportError, AttributeError):
        return _UNRESOLVED
    return obj


def _referenced_names(source: str) -> set[str]:
    """Every identifier *referenced* in one module: name loads, attribute access, imports.

    Collects ``ast.Name`` ids, ``ast.Attribute`` attrs (so ``consolidator.promote_aged_episodes``
    counts), and import alias names — so a symbol used as a call, an attribute, or an
    imported name is all detected. String literals are not identifiers and never match.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a parse error surfaces elsewhere
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _module_reference_sets() -> dict[str, set[str]]:
    """``abs_path -> referenced identifiers`` for every module under the package.

    The whole-tree reference index a ``"wired"`` capability is checked against: a
    symbol is wired iff some production module *other than its definition* references
    it. This guard module is itself excluded.
    """
    sets: dict[str, set[str]] = {}
    for root, _dirs, files in os.walk(_PACKAGE_DIR):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            abs_path = os.path.join(root, filename)
            if abs_path == _THIS_FILE:
                continue
            with open(abs_path, encoding="utf-8") as handle:
                sets[abs_path] = _referenced_names(handle.read())
    return sets


def reachability_problems(checks: tuple[WireCheck, ...] = WIRE_CHECKS) -> list[str]:
    """Every wire-or-retire violation as a sorted, one-line ``"<name>: ..."`` message.

    The always-on gate: an empty list means every listed capability resolves to a
    live reach and (for a ``"wired"`` one) is referenced by production code outside
    its defining module. Injectable *checks* so the detector can be proven to bite.
    """
    ref_sets = _module_reference_sets()
    problems: list[str] = []
    for check in checks:
        if check.disposition not in _VALID_DISPOSITIONS:
            problems.append(f"{check.name}: unknown disposition {check.disposition!r}")
            continue
        if _resolve(check.reach) is _UNRESOLVED:
            problems.append(
                f"{check.name}: reach {check.reach!r} does not resolve to a live attribute "
                "(the capability has no production entry point)"
            )
            continue
        if check.disposition == "wired":
            defining_abs = os.path.join(_PACKAGE_DIR, *check.defining_module.split("/"))
            referenced = any(
                check.symbol in names
                for path, names in ref_sets.items()
                if path != defining_abs
            )
            if not referenced:
                problems.append(
                    f"{check.name}: {check.symbol!r} has no production caller outside "
                    f"{check.defining_module} (dead public surface again — wire it or retire it)"
                )
    return sorted(problems)


def wire_or_retire_conformant() -> bool:
    """Whether every listed capability is reachable through a production path."""
    return not reachability_problems()


if __name__ == "__main__":  # pragma: no cover - dev tool
    import sys

    found = reachability_problems()
    if found:
        print("unhooked capabilities (give each an app.* verb / internal caller, or")
        print("retire it from the public surface):")
        print("\n".join(found))
        sys.exit(1)
    print("wire-or-retire conformant")
