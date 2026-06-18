"""Ergonomics: notebook rich reprs and the interactive TUI."""

from __future__ import annotations

import vincio.notebook as nb
from vincio import ContextApp
from vincio.core.types import MemoryItem, RunResult
from vincio.providers import MockProvider
from vincio.tui import TUI, render_home, render_memory, render_trace


class _FakeTrace:
    id = "trace_abc"
    app_name = "demo"
    status = "ok"
    duration_ms = 42
    spans: list = []
    start_time = 0


def _scripted(commands):
    it = iter(commands)

    def _input(_prompt):
        try:
            return next(it)
        except StopIteration as exc:
            raise EOFError from exc

    return _input


# -- notebook -------------------------------------------------------------------


def test_run_result_markdown_and_html():
    app = ContextApp(name="t", provider=MockProvider(responder=lambda r: "hi"), model="mock-1")
    result = app.run("hello")
    md = nb.run_result_markdown(result)
    assert "status" in md and "cost_usd" in md
    assert "RunResult" in nb.run_result_html(result)


def test_enable_and_disable_rich_reprs():
    nb.enable_rich_reprs()
    try:
        assert hasattr(RunResult, "_repr_html_")
        item = MemoryItem(content="prefers concise answers")
        assert "MemoryItem" in item._repr_html_()
    finally:
        nb.disable_rich_reprs()
    assert "_repr_html_" not in RunResult.__dict__


def test_reprs_never_raise_on_partial_objects():
    # Defensive: a duck-typed trace should still render.
    assert "Trace" in nb.trace_html(_FakeTrace())


def test_run_result_repr_handles_none_numeric():
    class _R:
        status = "ok"
        output = "x"
        cost_usd = None  # would crash a bare ":.6f" format
        latency_ms = 0
        usage = None
        citations: list = []
        trace_id = ""
        eval_scores: dict = {}
        error = None

    assert "RunResult" in nb.run_result_html(_R())  # did not raise


def test_memory_and_hit_reprs_handle_none_numeric():
    class _M:
        id = "m1"
        scope = "user"
        type = "fact"
        confidence = None
        status = "active"
        content = "x"

    class _H:
        score = None
        source = "bm25"
        chunk = None

    assert "MemoryItem" in nb.memory_item_html(_M())
    assert "SearchHit" in nb.search_hit_html(_H())


# -- TUI ------------------------------------------------------------------------


def test_render_home_empty_and_populated():
    assert "no traces" in render_home([])
    populated = render_home([_FakeTrace()])
    assert "[0]" in populated and "demo" in populated


def test_render_memory_and_trace():
    assert "no memories" in render_memory([])
    item = MemoryItem(content="x")
    assert item.id in render_memory([item])
    assert "back" in render_trace(_FakeTrace())


def test_tui_loop_navigates_and_quits(tmp_path):
    tui = TUI(traces_dir=str(tmp_path / "traces"), memory_db=str(tmp_path / "m.db"))
    tui._safe_load_traces = lambda: [_FakeTrace()]
    outputs: list[str] = []
    final = tui.run(
        input_fn=_scripted(["0", "b", "m", "b", "q"]),
        output_fn=outputs.append,
    )
    joined = "\n".join(outputs)
    assert "trace_abc" in joined  # home listing
    assert "memory" in joined.lower()  # memory screen visited
    assert "bye" in joined
    assert final == "home"


def test_tui_handles_empty_store(tmp_path):
    tui = TUI(traces_dir=str(tmp_path / "none"), memory_db=str(tmp_path / "none.db"))
    outputs: list[str] = []
    tui.run(input_fn=_scripted(["q"]), output_fn=outputs.append)
    assert "no traces" in "\n".join(outputs)


def test_tui_memories_are_cached(tmp_path):
    # The memory store opens a sqlite connection per load; the TUI must cache
    # so repeated memory renders don't leak file descriptors.
    tui = TUI(traces_dir=str(tmp_path / "t"), memory_db=str(tmp_path / "m.db"))
    first = tui._safe_load_memories()
    second = tui._safe_load_memories()
    assert first is second  # cache hit, no reopen
