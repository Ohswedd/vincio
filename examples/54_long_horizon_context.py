"""Long-horizon context engineering.

The regime where context engineering matters most is the one naïve accumulation
breaks: million-token, multi-day, multi-session agent runs where stale context
crowds out fresh signal ("context rot") and the resident footprint grows without
bound. Vincio composes its existing primitives — the footprint estimator, the
memory decay model, and the content-addressed evidence store's cross-process
``materialize()`` — into an explicit long-horizon governor.

Four steps, all offline and deterministic (no model required):

  1. RelevanceDecay: the memory subsystem's exponential decay applied *within a
     single run*, so a span admitted many steps ago loses weight before it
     crowds out fresh signal.
  2. ContextCompactor: hierarchical, provenance-preserving compaction that folds
     cold spans into a summary, pages their full text to a content-addressed
     store, and pages it back on demand — so detail is never lost, only moved.
  3. ContextGovernor: a per-run controller holding a context budget (tokens,
     residency, KV-cache footprint) the way the cost report holds a dollar
     budget — bounded as the horizon scales 10×.
  4. Runtime wiring: install the governor once, feed each run's result, and the
     live context stays bounded across the whole conversation.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    ContextBudget,
    ContextCompactor,
    ContextGovernor,
    RelevanceDecay,
)
from vincio.context.evidence_store import content_hash
from vincio.context.footprint import ENTRY_OVERHEAD_BYTES
from vincio.core.types import EvidenceItem, RunResult

NEEDLE = "The Pro plan refund window is exactly 30 days from the purchase date."


def _filler(i: int) -> str:
    return f"Filler observation {i}: telemetry, logs, metrics, traces, spans, and counters."


def relevance_decay() -> None:
    print("1. RelevanceDecay — stale spans lose weight to fresh signal of equal base")
    decay = RelevanceDecay(half_life_steps=8)
    print(f"   fresh (age 0)  → weight {decay.weight(0):.3f}")
    print(f"   one half-life  → weight {decay.weight(8):.3f}")
    print(f"   stale (age 40) → weight {decay.weight(40):.3f}")
    print(f"   a fact at relevance 0.5 admitted 40 steps ago now scores "
          f"{decay.decayed(0.5, 40):.3f} < {decay.decayed(0.5, 0):.3f} fresh")


def compactor_pages_back() -> None:
    print("\n2. ContextCompactor — fold cold spans, page the full text back on demand")
    compactor = ContextCompactor(summary_tokens=40)
    cold = [EvidenceItem(text=NEEDLE, source_id="needle"),
            *[EvidenceItem(text=_filler(i), source_id=f"f{i}") for i in range(5)]]
    # Admit through a governor so the spans carry run metadata, then compact.
    gov = ContextGovernor(ContextBudget(), compactor=compactor)
    spans = gov.admit_evidence(cold)
    summary, record = compactor.compact(spans)
    print(f"   {len(spans)} cold spans → 1 level-{summary.level} summary span "
          f"({record.tokens_before}→{record.tokens_after} tokens)")
    print(f"   provenance kept: {len(record.covered_hashes)} covered hashes, "
          f"source ids {record.source_ids[:3]}…")
    recovered = compactor.page_in([content_hash(NEEDLE)])
    print(f"   paged back on demand: {recovered[content_hash(NEEDLE)]!r}")


def _governed(horizon: int) -> ContextGovernor:
    gov = ContextGovernor(
        ContextBudget(max_tokens=400, max_resident_bytes=6000),
        compactor=ContextCompactor(summary_tokens=48),
        decay=RelevanceDecay(half_life_steps=8),
        keep_recent_spans=3,
    )
    gov.admit(NEEDLE, relevance=0.95, source_ids=["needle"])
    for i in range(horizon):
        gov.admit(_filler(i), relevance=0.5)
    return gov


def governor_bounds_the_horizon() -> None:
    print("\n3. ContextGovernor — footprint stays flat as the horizon grows 10×")
    small, large = _governed(20), _governed(200)  # a 10× horizon

    def naive(h: int) -> int:
        texts = [NEEDLE] + [_filler(i) for i in range(h)]
        return ENTRY_OVERHEAD_BYTES * len(texts) + sum(len(t.encode("utf-8")) for t in texts)

    governed_ratio = large.resident_bytes / small.resident_bytes
    naive_ratio = naive(200) / naive(20)
    print(f"   governed resident: {small.resident_bytes}B (1×) → {large.resident_bytes}B (10×) "
          f"= {governed_ratio:.2f}×")
    print(f"   naïve resident would grow {naive_ratio:.1f}× — context rot + budget blow-up")
    rep = large.report()
    print(f"   at 10×: {rep.compaction_count} compactions saved {rep.compacted_tokens_saved} tokens, "
          f"within_budget={rep.within_budget}")
    hits = large.recall("Pro plan refund window days purchase", top_k=3)
    print(f"   recall@10× still finds the needle (paged back): {any('30 days' in h for h in hits)}")
    decayed = sum(1 for e in large.excluded_report() if e['reason'] == 'intra_run_decay')
    print(f"   excluded-context report: {decayed} spans demoted by intra-run decay")


def runtime_wiring() -> None:
    print("\n4. Runtime wiring — install once, feed each run, stay bounded across sessions")
    app = ContextApp(name="lh_demo")  # deterministic mock provider, fully offline
    app.use_context_governor(ContextBudget(max_tokens=150))
    report = None
    for i in range(12):  # twelve turns of a long conversation
        result = RunResult(
            run_id=f"r{i}",
            status="succeeded",
            evidence=[EvidenceItem(text=_filler(i), source_id=f"s{i}", relevance=0.7)],
        )
        report = app.govern_packet(result)  # admits result.evidence
    assert report is not None
    print(f"   after 12 turns: live_tokens={report.live_tokens} ≤ budget 150, "
          f"within_budget={report.within_budget}")
    print(f"   context_budget_report() is the residency analogue of cost_report(): "
          f"{app.context_budget_report().span_count} live spans")


async def main() -> None:
    relevance_decay()
    compactor_pages_back()
    governor_bounds_the_horizon()
    runtime_wiring()


if __name__ == "__main__":
    asyncio.run(main())
