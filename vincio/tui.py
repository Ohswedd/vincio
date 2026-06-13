"""Interactive terminal inspector for runs, traces, and memory.

A dependency-free, keyboard-driven TUI: it lists the traces an app has
written, drills into a span tree, and browses stored memories. The screen
renderers are pure functions (they take data, return text), and the loop reads
commands through injectable ``input_fn`` / ``output_fn`` callables, so the whole
thing is unit-testable without a real terminal::

    from vincio.tui import TUI
    TUI(traces_dir=".vincio/traces", memory_db=".vincio/memory.db").run()

Launch it from the CLI with ``vincio tui``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["TUI", "render_home", "render_trace", "render_memory"]

_HOME_FOOTER = "commands: <number> open trace · m memory · r refresh · q quit"
_DETAIL_FOOTER = "commands: b back · q quit"


def render_home(traces: list[Any]) -> str:
    lines = ["═══ Vincio TUI · traces ═══"]
    if not traces:
        lines.append("  (no traces found)")
    for index, trace in enumerate(traces[:50]):
        status = getattr(trace, "status", "")
        mark = {"ok": "✓", "error": "✗"}.get(status, "•")
        lines.append(
            f"  [{index}] {mark} {getattr(trace, 'id', '?')}  "
            f"{getattr(trace, 'app_name', '')}  "
            f"{getattr(trace, 'duration_ms', 0)}ms  "
            f"{len(getattr(trace, 'spans', []) or [])} spans"
        )
    lines.append("")
    lines.append(_HOME_FOOTER)
    return "\n".join(lines)


def render_trace(trace: Any) -> str:
    if trace is None:
        return "trace not found\n" + _DETAIL_FOOTER
    try:
        from .observability.viewer import render_trace_text

        body = render_trace_text(trace)
    except Exception:  # noqa: BLE001 - fall back to a minimal rendering
        body = (
            f"trace {getattr(trace, 'id', '?')}  app={getattr(trace, 'app_name', '')}  "
            f"status={getattr(trace, 'status', '')}  {getattr(trace, 'duration_ms', 0)}ms"
        )
    return f"{body}\n\n{_DETAIL_FOOTER}"


def render_memory(items: list[Any]) -> str:
    lines = ["═══ Vincio TUI · memory ═══"]
    if not items:
        lines.append("  (no memories found)")
    for item in items[:50]:
        scope = getattr(getattr(item, "scope", None), "value", getattr(item, "scope", ""))
        mtype = getattr(getattr(item, "type", None), "value", getattr(item, "type", ""))
        content = str(getattr(item, "content", ""))
        lines.append(
            f"  {getattr(item, 'id', '?')}  [{scope}/{mtype}]  "
            f"conf={getattr(item, 'confidence', 0.0):.2f}  {content[:80]}"
        )
    lines.append("")
    lines.append(_DETAIL_FOOTER)
    return "\n".join(lines)


class TUI:
    def __init__(
        self, *, traces_dir: str = ".vincio/traces", memory_db: str = ".vincio/memory.db"
    ) -> None:
        self.traces_dir = traces_dir
        self.memory_db = memory_db
        self.traces: list[Any] = []
        self.memories: list[Any] | None = None  # loaded lazily, cached

    # -- data (failure-tolerant; an empty store is a normal state) --------------

    def _safe_load_traces(self) -> list[Any]:
        try:
            from .observability.exporters import JSONLExporter

            traces = JSONLExporter(self.traces_dir).load_all()
            return sorted(traces, key=lambda t: getattr(t, "start_time", None) or 0, reverse=True)
        except Exception:  # noqa: BLE001
            return []

    def _safe_load_memories(self) -> list[Any]:
        # Cached: the memory screen renders every loop iteration, and each
        # SQLiteMemoryStore opens (and must close) its own sqlite connection —
        # reopening per render would leak file descriptors over a long session.
        if self.memories is not None:
            return self.memories
        try:
            from .memory.stores import SQLiteMemoryStore

            store = SQLiteMemoryStore(self.memory_db)
            try:
                self.memories = store.all_items(statuses=())
            finally:
                store.close()
        except Exception:  # noqa: BLE001
            self.memories = []
        return self.memories

    # -- rendering --------------------------------------------------------------

    def _render(self, state: str, current: int | None) -> str:
        if state == "home":
            return render_home(self.traces)
        if state == "trace":
            trace = self.traces[current] if current is not None and current < len(self.traces) else None
            return render_trace(trace)
        if state == "memory":
            return render_memory(self._safe_load_memories())
        return ""

    # -- loop -------------------------------------------------------------------

    def run(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], Any] = print,
        max_steps: int = 100_000,
    ) -> str:
        """Run the interactive loop; returns the final state (useful for tests)."""
        self.traces = self._safe_load_traces()
        self.memories = None
        state: str = "home"
        current: int | None = None
        for _ in range(max_steps):
            output_fn(self._render(state, current))
            try:
                command = input_fn("> ").strip().lower()
            except EOFError:
                break
            if command in ("q", "quit", "exit"):
                output_fn("bye")
                break
            if state == "home":
                if command == "m":
                    state = "memory"
                elif command == "r":
                    self.traces = self._safe_load_traces()
                    self.memories = None  # force reload on next memory view
                elif command.isdigit() and int(command) < len(self.traces):
                    state, current = "trace", int(command)
            elif state in ("trace", "memory") and command == "b":
                state, current = "home", None
        return state
