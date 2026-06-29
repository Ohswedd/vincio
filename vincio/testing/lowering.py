"""Lowering signatures: prove two compile/run paths produce the same packet.

A *lowering signature* is a deterministic, structural projection of a compiled
context (or of a finished run) with the volatile identity stripped out — the
packet id, the trace id, timestamps, latency — so two call paths that lower to
the same governed packet produce equal signatures, and a behavioral fork shows
up as a diff.

Two layers build on one idea:

* :func:`selection_signature` canonicalizes a
  :class:`~vincio.context.compiler.CompiledContext` — the selected
  evidence/memory/tools, the excluded report, conflicts, the budget, and the
  token count. It is the harness the single-pass feature arena (5.2) uses to
  prove that turning the optimization on selects byte-identical context.
* :func:`result_signature` / :func:`run_signature` canonicalize a finished
  :class:`~vincio.core.types.RunResult` together with its persisted packet's
  ``spec_hash``, so the ergonomic front door (5.3) can prove a one-line task
  lowers to the same packet *and* the same result as the verbose builder form.

Everything here is deterministic and offline; it reads only what a run already
produced and persisted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..core.utils import to_jsonable

if TYPE_CHECKING:
    from ..core.app import ContextApp
    from ..core.types import RunResult

__all__ = ["selection_signature", "result_signature", "run_signature"]


def selection_signature(compiled: Any) -> dict[str, Any]:
    """Canonicalize a ``CompiledContext`` to everything selection could affect.

    Returns the selected evidence (id / text / relevance / token-cost), the
    memory ids, the tool names, the excluded report, the conflicts, the budget
    report, the token count, the resident footprint, and the packet's slim flag —
    excluding only the packet's own non-deterministic identity (id / timestamp).
    Two compiles whose signatures are equal selected byte-identical context.
    """
    ir = compiled.ir
    return {
        "evidence": [(e.id, e.text, round(e.relevance, 12), e.token_cost) for e in ir.evidence],
        "memory": [m.id for m in ir.memory],
        "tool_specs": [t.name for t in ir.tool_specs],
        "excluded": compiled.excluded_report,
        "conflicts": compiled.conflicts,
        "budget": compiled.budget_report,
        "tokens": compiled.token_count,
        "resident_bytes": compiled.resident_bytes,
        "slim": compiled.packet.slim,
    }


def result_signature(app: ContextApp, result: RunResult) -> dict[str, Any]:
    """Canonicalize a finished run to its packet fingerprint plus stable outputs.

    Reads the run's persisted context packet (by ``result.context_packet_id``)
    for the deterministic ``spec_hash`` and token count — the packet's byte
    fingerprint — and projects the result's stable, reproducible fields (output,
    raw text, citations, eval scores, status, excluded context, token usage, and
    cost), excluding the volatile identity (run / trace ids, latency, and
    timestamps). Two runs with equal signatures lowered to the same governed
    packet and produced the same result.
    """
    packet: dict[str, Any] = {}
    packet_id = result.context_packet_id
    if packet_id:
        stored = app.store.get("context_packets", packet_id)
        if stored is not None:
            packet = stored
    status = result.status
    return {
        "spec_hash": packet.get("spec_hash"),
        "packet_tokens": packet.get("token_count"),
        "output": to_jsonable(result.output),
        "raw_text": result.raw_text,
        "citations": list(result.citations),
        "eval_scores": {k: round(v, 9) for k, v in sorted(result.eval_scores.items())},
        "status": getattr(status, "value", str(status)),
        "excluded": to_jsonable(result.excluded_context),
        "usage_total_tokens": result.usage.total_tokens,
        "cost_usd": round(result.cost_usd, 9),
    }


def run_signature(
    app: ContextApp,
    user_input: str,
    *,
    runner: Callable[..., RunResult] | None = None,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Run ``app`` on ``user_input`` and return its :func:`result_signature`.

    A convenience for the byte-identical proof: drive a configured app once and
    capture its lowering signature. Pass ``runner`` to use a specific entry point
    (for example a task facade's verb); it defaults to ``app.run``.
    """
    run = runner if runner is not None else app.run
    result = run(user_input, **run_kwargs)
    return result_signature(app, result)
