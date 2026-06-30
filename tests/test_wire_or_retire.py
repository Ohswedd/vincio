"""Wire-or-retire (6.3): every public capability is reachable, and the guard bites.

6.3 wired five capabilities that were public but unreachable (no ``app.*`` verb, no
internal caller). These tests pin that each is reachable through its production path
and that ``vincio._wire_or_retire`` provably bites when a capability becomes dead
surface again.
"""

from __future__ import annotations

import inspect

from vincio import _wire_or_retire
from vincio.context import CompileStreamEvent
from vincio.context.compiler import ContextCompiler
from vincio.core.app import ContextApp
from vincio.memory.engine import MemoryEngine
from vincio.providers.base import ModelProvider, register_provider_token_counters


def test_live_tree_is_wire_or_retire_conformant():
    assert _wire_or_retire.reachability_problems() == []
    assert _wire_or_retire.wire_or_retire_conformant() is True


def test_ledger_covers_the_six_three_capabilities():
    names = {check.name for check in _wire_or_retire.WIRE_CHECKS}
    assert names == {
        "reasoning_retrieval",
        "blob_evidence_store",
        "memory_consolidation",
        "token_counter_registry",
        "compile_streaming",
        "recompile",
        "compile_stream_event",
    }


def test_gate_bites_on_unreachable_reach():
    check = _wire_or_retire.WireCheck(
        "fake", "X", "x/y.py", "vincio.core.app:ContextApp.no_such_verb", "wired"
    )
    problems = _wire_or_retire.reachability_problems((check,))
    assert problems and "does not resolve" in problems[0]


def test_gate_bites_on_dead_wired_symbol():
    check = _wire_or_retire.WireCheck(
        "dead",
        "NotReferencedAnywhereSymbol",
        "core/tokens.py",
        "vincio.core.app:ContextApp.retrieve_facts",
        "wired",
    )
    problems = _wire_or_retire.reachability_problems((check,))
    assert problems and "no production caller" in problems[0]


def test_advanced_api_disposition_needs_only_a_live_reach():
    # An advanced-API capability has no internal caller by design, so a resolving
    # reach is sufficient even though the symbol is referenced nowhere in production.
    check = _wire_or_retire.WireCheck(
        "adv",
        "NotReferencedAnywhereSymbol",
        "context/compiler.py",
        "vincio.context.compiler:ContextCompiler.compile_streaming",
        "advanced_api",
    )
    assert _wire_or_retire.reachability_problems((check,)) == []


def test_unknown_disposition_is_flagged():
    check = _wire_or_retire.WireCheck(
        "weird", "X", "x/y.py", "vincio.context:CompileStreamEvent", "sideways"
    )
    problems = _wire_or_retire.reachability_problems((check,))
    assert problems and "unknown disposition" in problems[0]


# The reach paths the ledger relies on, asserted directly so a rename is caught here
# too (not only through the AST reach resolution).


def test_reasoning_retrieval_reach_exists():
    assert callable(ContextApp.retrieve_facts)


def test_memory_consolidation_reach_exists():
    assert callable(ContextApp.consolidate_memory)
    assert callable(MemoryEngine.promote_aged_episodes)


def test_blob_evidence_store_reach_accepts_a_blob_store():
    params = inspect.signature(ContextApp.use_context_governor).parameters
    assert "blob_store" in params
    assert "evidence_store" in params


def test_token_counter_registry_reach_exists():
    assert callable(register_provider_token_counters)
    assert callable(ModelProvider.exact_token_counter)
    assert callable(ModelProvider.token_id_prefixes)


def test_compile_streaming_advanced_api_reach_exists():
    assert callable(ContextCompiler.compile_streaming)
    assert callable(ContextCompiler.recompile)
    assert issubclass(CompileStreamEvent, object)
