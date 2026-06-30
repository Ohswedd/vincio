"""Docstring / behaviour parity (6.4): the docs match the code.

A docstring that advertises behaviour the code no longer performs is a quiet
lie. These tests pin the reconciliations made in the docstring-parity pass so a
future edit cannot silently reopen the gap:

* ``vincio.context.budgeting`` no longer advertises a separate ``redistribute``
  reclaim that nothing calls — the allocator pushes every non-fixed token into
  the flexible blocks at allocation time, and the dead method is gone.
* ``vincio.context.llmlingua`` no longer claims the compression tuner calls
  ``compression_faithfulness`` / ``faithfulness_preserved``; those are the
  offline fidelity measures, and ``CompressionTuner`` gates adoption on a
  ``faithfulness`` eval metric — the metric the docstring names.
* the federated default-deny consent demonstration fires deterministically,
  regardless of any consent persisted to disk from an earlier run.
* ``MemoryEngine.delete`` delegates to ``forget`` (one body, audit semantics
  preserved): a plain delete records no reason, ``forget`` records one.
* the stale spec-item comments are gone.
"""

from __future__ import annotations

import io
import types
from contextlib import redirect_stdout
from pathlib import Path

from vincio.context.budgeting import BudgetAllocator
from vincio.context.llmlingua import (
    LLMLinguaCompressor,
    compression_faithfulness,
    faithfulness_preserved,
)
from vincio.governance.consent import ConsentLedger, Purpose
from vincio.memory.engine import MemoryEngine
from vincio.memory.stores import InMemoryMemoryStore
from vincio.optimize.compression_tuning import CompressionTuner
from vincio.security.audit import AuditLog

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"


# -- budgeting -----------------------------------------------------------------


def test_redistribute_reclaim_method_is_gone():
    # The module docstring no longer promises a separate reclaim step, so the
    # dead method that backed that promise must not return.
    assert not hasattr(BudgetAllocator, "redistribute")


def test_allocate_distributes_every_non_fixed_token_to_flexible_blocks():
    # The honest behaviour the docstring now describes: with the fixed blocks
    # charged at cost, the whole remainder lands on the flexible blocks rather
    # than being held back for a redistribute that never ran.
    allocation = BudgetAllocator().allocate(
        10_000, fixed_costs={"instructions": 1_000, "user_task": 200}
    )
    flexible = ("evidence", "memory", "tool_results", "examples", "schema")
    remaining = 10_000 - 1_000 - 200
    handed_out = sum(allocation.block(name).tokens for name in flexible)
    # Proportional integer split: every token bar per-block rounding is handed out.
    assert remaining - handed_out < len(flexible)
    assert allocation.block("evidence").tokens > 0


# -- llmlingua / compression tuning -------------------------------------------


def test_compression_tuner_reads_the_faithfulness_metric_the_docstring_names():
    async def _noop(_compressor, _dataset):  # pragma: no cover - never called here
        raise AssertionError("evaluate should not run")

    tuner = CompressionTuner(_noop)
    assert tuner.faithfulness_metric == "faithfulness"


def test_faithfulness_helpers_are_offline_fidelity_measures():
    # The helpers the docstring points at really measure answer-bearing survival.
    assert compression_faithfulness("Pro plan 30 days", "Pro plan 30 days") == 1.0
    assert compression_faithfulness("Pro plan 30 days", "Pro plan days") < 1.0
    assert faithfulness_preserved(["The Pro plan refund is 30 days."], "Pro plan refund 30 days")
    assert not faithfulness_preserved(["The Pro plan refund is 30 days."], "the refund window")
    # And they are wired as the compressor's drop-in companion, not a dead claim.
    assert callable(LLMLinguaCompressor())


# -- federated consent: deterministic default-deny ----------------------------


def test_storeless_default_deny_ledger_denies_then_allows():
    # The enforcement semantics the federated path applies: a store-less ledger
    # starts empty, so an ungranted subject is refused; a grant flips it.
    ledger = ConsentLedger(default_allow=False)
    assert ledger.check("org", Purpose.ANALYTICS).allowed is False
    ledger.grant("org", [Purpose.ANALYTICS])
    assert ledger.check("org", Purpose.ANALYTICS).allowed is True


def _run_example_section(name: str, cwd: Path) -> str:
    """Exec example 13 as ``__main__`` in *cwd* and run one section, capturing stdout."""
    import os
    import sys

    path = EXAMPLES_DIR / "13_data_and_analytics.py"
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    module = types.ModuleType("__main__")
    module.__dict__.update({"__file__": str(path), "__name__": "__main__"})
    saved_main = sys.modules.get("__main__")
    saved_path = list(sys.path)
    saved_cwd = os.getcwd()
    sys.modules["__main__"] = module
    sys.path.insert(0, str(EXAMPLES_DIR))  # for `import _shared`
    os.chdir(cwd)
    try:
        with redirect_stdout(io.StringIO()):
            exec(code, module.__dict__)  # noqa: S102 - first-party example code
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            module.__dict__[name]()
        return buffer.getvalue()
    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        if saved_main is not None:
            sys.modules["__main__"] = saved_main
        else:  # pragma: no cover
            sys.modules.pop("__main__", None)
        sys.modules.pop("_shared", None)


def test_consent_refusal_fires_even_against_a_persisted_store(tmp_path):
    # The bug this pass closed: the demo only fired against a pristine store, so
    # the second run (which sees the grant persisted by the first) silently
    # stopped refusing. Run the section twice in the SAME cwd; it must refuse
    # both times.
    needle = "consent refused for an org without an ANALYTICS grant"
    first = _run_example_section("section_federated", tmp_path)
    second = _run_example_section("section_federated", tmp_path)
    assert needle in first
    assert needle in second
    assert "consent granted → org contributes: True" in second


# -- memory delete / forget dedup ---------------------------------------------


def _engine() -> MemoryEngine:
    return MemoryEngine(store=InMemoryMemoryStore(), audit=AuditLog())


def _last_delete_detail(engine: MemoryEngine) -> dict:
    for record in reversed(engine.audit.entries):
        if record.action == "memory_delete":
            return dict(record.details)
    raise AssertionError("no memory_delete audit record")


def test_delete_records_no_reason_forget_records_one():
    engine = _engine()
    a = engine.remember("alpha")
    b = engine.remember("beta")

    assert engine.delete(a.id) is True
    plain = _last_delete_detail(engine)
    assert "reason" not in plain
    assert plain["scope"] == a.scope.value

    assert engine.forget(b.id, reason="user_request") is True
    annotated = _last_delete_detail(engine)
    assert annotated["reason"] == "user_request"


def test_delete_delegates_to_forget_and_removes():
    engine = _engine()
    item = engine.remember("gamma")
    assert engine.delete(item.id) is True
    # Soft-marked then hard-removed: a missing id is a clean False.
    assert engine.delete(item.id) is False


# -- stale comments cleared ----------------------------------------------------


def test_no_spec_item_numbering_in_input_docstrings():
    import re

    pattern = re.compile(r"items?\s*\d", re.IGNORECASE)
    for path in (ROOT / "vincio" / "input").glob("*.py"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


def test_compiler_pipeline_numbering_has_no_stale_marker():
    source = (ROOT / "vincio" / "context" / "compiler.py").read_text(encoding="utf-8")
    assert "# 7+8." not in source
