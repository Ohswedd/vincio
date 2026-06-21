"""Long-horizon context engineering.

Covers intra-run relevance decay, the provenance-preserving span compactor with
paging back from the content-addressed store, and the per-run context governor
that holds a tokens/residency/KV-cache budget as the horizon scales 10×, plus
the runtime/app wiring (``use_context_governor`` / ``govern_packet``).
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    ContextBudget,
    ContextCompactor,
    ContextGovernor,
    RelevanceDecay,
)
from vincio.context.evidence_store import InMemoryEvidenceStore, content_hash
from vincio.context.footprint import ENTRY_OVERHEAD_BYTES
from vincio.context.longhorizon import ContextBudgetReport, RunSpan
from vincio.core.types import EvidenceItem, MemoryScope, MemoryType, RunResult
from vincio.memory.engine import MemoryEngine
from vincio.providers import MockProvider

NEEDLE = "The Pro plan refund window is exactly 30 days from the purchase date."


def _filler(i: int) -> str:
    return f"Filler observation {i}: telemetry, logs, metrics, traces, spans, and counters."


# ---------------------------------------------------------------------------
# RelevanceDecay
# ---------------------------------------------------------------------------


class TestRelevanceDecay:
    def test_fresh_keeps_full_weight(self):
        d = RelevanceDecay(half_life_steps=8)
        assert d.weight(0) == 1.0
        assert d.decayed(0.8, 0) == pytest.approx(0.8)

    def test_half_life_halves_weight(self):
        d = RelevanceDecay(half_life_steps=8)
        assert d.weight(8) == pytest.approx(0.5)
        assert d.weight(16) == pytest.approx(0.25)

    def test_stale_demoted_below_fresh_of_equal_base(self):
        d = RelevanceDecay(half_life_steps=8)
        assert d.decayed(0.5, age_steps=40) < d.decayed(0.5, age_steps=0)

    def test_floor_clamps(self):
        d = RelevanceDecay(half_life_steps=4, floor=0.1)
        assert d.weight(1000) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# ContextCompactor
# ---------------------------------------------------------------------------


class TestContextCompactor:
    def test_compact_preserves_provenance_and_pages_back(self):
        store = InMemoryEvidenceStore()
        compactor = ContextCompactor(store=store, summary_tokens=20)
        spans = [
            RunSpan(text=NEEDLE, source_ids=["needle"], step=1),
            *[RunSpan(text=_filler(i), source_ids=[f"f{i}"], step=2 + i) for i in range(5)],
        ]
        summary, record = compactor.compact(spans)

        assert summary.kind == "summary"
        assert summary.level == 1
        # Provenance: the record carries source ids and the covered content hashes.
        assert "needle" in record.source_ids
        assert content_hash(NEEDLE) in record.covered_hashes
        # Paged back on demand: the exact original text returns from the store.
        recovered = compactor.page_in([content_hash(NEEDLE)])
        assert recovered[content_hash(NEEDLE)] == NEEDLE

    def test_hierarchical_compaction_raises_level_and_keeps_trail(self):
        compactor = ContextCompactor(summary_tokens=20)
        raw = [RunSpan(text=_filler(i), source_ids=[f"f{i}"], step=i) for i in range(6)]
        s1, _ = compactor.compact(raw[:3])
        s2, _ = compactor.compact(raw[3:])
        top, record = compactor.compact([s1, s2])
        assert top.level == 2
        # The leaf hashes survive two levels of compaction.
        assert content_hash(_filler(0)) in record.covered_hashes
        assert content_hash(_filler(5)) in record.covered_hashes

    def test_writes_summary_into_memory_with_provenance(self):
        memory = MemoryEngine()
        compactor = ContextCompactor(
            store=InMemoryEvidenceStore(), memory=memory, owner_id="s1", scope=MemoryScope.SESSION
        )
        _summary, record = compactor.compact(
            [RunSpan(text=NEEDLE, source_ids=["needle"], step=1),
             RunSpan(text=_filler(0), source_ids=["f0"], step=2)]
        )
        assert record.memory_id is not None
        item = memory.store.get(record.memory_id)
        assert item is not None
        assert item.type == MemoryType.SUMMARY
        assert item.metadata["origin"] == "context_compaction"
        assert "needle" in item.metadata["source_ids"]


# ---------------------------------------------------------------------------
# ContextGovernor
# ---------------------------------------------------------------------------


def _governed(horizon: int, **budget_kwargs) -> ContextGovernor:
    gov = ContextGovernor(
        ContextBudget(max_tokens=budget_kwargs.pop("max_tokens", 400),
                      max_resident_bytes=budget_kwargs.pop("max_resident_bytes", 6000),
                      **budget_kwargs),
        compactor=ContextCompactor(summary_tokens=48),
        decay=RelevanceDecay(half_life_steps=8),
        keep_recent_spans=3,
    )
    gov.admit(NEEDLE, relevance=0.95, source_ids=["needle"])
    for i in range(horizon):
        gov.admit(_filler(i), relevance=0.5)
    return gov


class TestContextGovernor:
    def test_holds_token_budget(self):
        gov = _governed(60, max_tokens=300, max_resident_bytes=None)
        assert gov.live_tokens <= 300
        assert gov.within_budget()

    def test_holds_resident_budget(self):
        gov = _governed(60, max_tokens=None, max_resident_bytes=5000)
        assert gov.resident_bytes <= 5000
        assert gov.within_budget()

    def test_footprint_bounded_as_horizon_grows_10x(self):
        small, large = _governed(20), _governed(200)
        ratio = large.resident_bytes / small.resident_bytes
        assert ratio <= 1.5  # flat, not the ~10× naïve growth
        assert large.within_budget()

    def test_recall_preserved_at_horizon_via_page_in(self):
        gov = _governed(200)
        hits = gov.recall("Pro plan refund window days purchase", top_k=3)
        assert any("30 days" in h for h in hits)
        assert gov.paged_in >= 1  # the needle was paged back from a summary

    def test_intra_run_decay_demotes_and_surfaces(self):
        gov = _governed(200)
        reasons = {e["reason"] for e in gov.excluded_report()}
        assert "intra_run_decay" in reasons
        assert "compacted_into_summary" in reasons

    def test_pinned_span_never_decays_or_compacts(self):
        gov = ContextGovernor(
            ContextBudget(max_tokens=120),
            compactor=ContextCompactor(summary_tokens=20),
            keep_recent_spans=1,
        )
        anchor = gov.admit("SYSTEM OBJECTIVE: resolve the customer ticket.", pinned=True, relevance=1.0)
        for i in range(40):
            gov.admit(_filler(i), relevance=0.5)
        live_ids = {s.id for s in gov.spans}
        assert anchor.id in live_ids  # the pinned anchor is never paged out
        assert gov.spans  # bounded
        anchor_span = next(s for s in gov.spans if s.id == anchor.id)
        assert anchor_span.effective_relevance == 1.0

    def test_evicts_when_no_compactor(self):
        gov = ContextGovernor(ContextBudget(max_tokens=80), keep_recent_spans=1)
        for i in range(30):
            gov.admit(_filler(i), relevance=0.5)
        assert gov.live_tokens <= 80
        assert any(e["reason"] == "context_budget_exceeded" for e in gov.excluded_report())

    def test_report_shape(self):
        gov = _governed(50)
        report = gov.report()
        assert isinstance(report, ContextBudgetReport)
        assert report.within_budget
        assert report.compaction_count >= 1
        assert report.live_tokens == gov.live_tokens
        assert report.resident_bytes == gov.resident_bytes
        d = report.as_dict()
        assert d["span_count"] == len(gov.spans)

    def test_resident_bytes_matches_footprint_formula(self):
        gov = ContextGovernor(ContextBudget())
        gov.admit("hello world", relevance=1.0)
        expected = ENTRY_OVERHEAD_BYTES + len(b"hello world")
        assert gov.resident_bytes == expected

    def test_kv_cache_estimate(self):
        gov = ContextGovernor(ContextBudget(kv_bytes_per_token=10))
        gov.admit("one two three four", relevance=1.0)
        assert gov.kv_cache_bytes == gov.live_tokens * 10

    def test_admit_evidence_from_run_result(self):
        gov = ContextGovernor(ContextBudget(max_tokens=200), compactor=ContextCompactor())
        items = [EvidenceItem(text=_filler(i), source_id=f"s{i}", relevance=0.6) for i in range(20)]
        gov.admit_evidence(items)
        assert gov.live_tokens <= 200
        assert gov.step == 20

    def test_naive_growth_is_linear_governed_is_flat(self):
        # The contrast the SLO protects: ungoverned footprint grows ~10×.
        def naive(h: int) -> int:
            texts = [NEEDLE] + [_filler(i) for i in range(h)]
            return ENTRY_OVERHEAD_BYTES * len(texts) + sum(len(t.encode("utf-8")) for t in texts)

        assert naive(200) / naive(20) > 5.0
        assert _governed(200).resident_bytes / _governed(20).resident_bytes < 1.5


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


class TestAppWiring:
    def _app(self) -> ContextApp:
        return ContextApp(name="lh", provider=MockProvider(default_text="ok"), model="mock-1")

    def test_use_context_governor_from_budget(self):
        app = self._app()
        out = app.use_context_governor(ContextBudget(max_tokens=200))
        assert out is app  # chainable
        assert isinstance(app.context_governor, ContextGovernor)
        assert app.context_governor.budget.max_tokens == 200

    def test_use_context_governor_kwargs(self):
        app = self._app()
        app.use_context_governor(max_tokens=150, max_resident_bytes=4000)
        assert app.context_governor.budget.max_tokens == 150
        assert app.context_governor.budget.max_resident_bytes == 4000

    def test_govern_packet_admits_run_result_evidence(self):
        app = self._app()
        app.use_context_governor(ContextBudget(max_tokens=150))
        report = None
        for i in range(12):
            rr = RunResult(
                run_id=f"r{i}",
                status="succeeded",
                evidence=[EvidenceItem(text=_filler(i), source_id=f"s{i}", relevance=0.7)],
            )
            report = app.govern_packet(rr)
        assert report is not None and report.within_budget
        assert app.context_governor.live_tokens <= 150

    def test_govern_packet_admits_context_packet(self):
        from vincio.context import ContextCompiler, ContextCompilerOptions
        from vincio.core.types import Objective, UserInput

        app = self._app()
        app.use_context_governor(ContextBudget(max_tokens=500))
        compiler = ContextCompiler(ContextCompilerOptions())
        import asyncio

        compiled = asyncio.run(
            compiler.compile(
                objective=Objective(text="refund question"),
                user_input=UserInput(text="what is the refund window?"),
                evidence=[EvidenceItem(text=NEEDLE, source_id="needle", relevance=0.9)],
            )
        )
        report = app.govern_packet(compiled.packet)
        assert report is not None
        assert app.context_governor.step >= 1

    def test_no_governor_returns_none(self):
        app = self._app()
        assert app.context_budget_report() is None
        assert app.govern_packet(None) is None
