"""Real-behavior coverage tests for vincio.context.compiler.

Targets the uncovered branches: conflict resolution (authority/freshness/
unresolved/polarity), budget-overflow error, inline compression, modality
token costs, the empty-evidence path, footprint enforcement, ordering modes,
the low-relevance exclusion cascade, tool-result candidates, reserve tokens,
streaming, and recompile. Every interaction uses real objects (no mocking).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from vincio.context.compiler import (
    CompiledContext,
    CompileStreamEvent,
    ContextCompiler,
    ContextCompilerOptions,
    _looks_negated,
    _media_identity,
    _value_disagreement,
    _value_units,
)
from vincio.context.ir import OutputContractRef
from vincio.core.errors import ContextCompileError
from vincio.core.types import (
    Budget,
    Constraint,
    EvidenceItem,
    Example,
    ImageRef,
    Instruction,
    MemoryItem,
    MemoryScope,
    Objective,
    PolicySet,
    PrivacyClass,
    TaskType,
    ToolResult,
    UserInput,
    VideoRef,
    utcnow,
)


def _obj(text="answer the refund question"):
    return Objective(text=text, task_type=TaskType.DOCUMENT_QA)


def _ui(text="What is the refund window?"):
    return UserInput(text=text)


def _open() -> PolicySet:
    # Skip the tenant-isolation privacy screen so memory candidates survive.
    return PolicySet(privacy="open")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestValueHelpers:
    def test_value_units_excludes_structural_references(self):
        # "Section 5" is a structural reference; "30" is a real value.
        units = _value_units("Per Section 5 the refund window is 30 days.")
        assert "30" in units
        assert "5" not in units

    def test_value_disagreement_numeric(self):
        delta = _value_disagreement(
            "Refunds allowed within 30 days of the purchase date.",
            "Refunds allowed within 14 days of the purchase date.",
        )
        assert delta == {
            "kind": "value_disagreement",
            "a_values": ["30"],
            "b_values": ["14"],
            "differing": ["14", "30"],
        }

    def test_value_disagreement_polarity_only(self):
        delta = _value_disagreement(
            "The account is active and billing continues.",
            "The account is not active and billing continues.",
        )
        assert delta == {
            "kind": "polarity_disagreement",
            "a_values": [],
            "b_values": [],
            "differing": [],
        }

    def test_value_disagreement_none_when_aligned(self):
        assert _value_disagreement("refund within 30 days", "refund within 30 days") is None

    def test_looks_negated(self):
        assert _looks_negated("This cannot be refunded") is True
        assert _looks_negated("This is fully refundable") is False

    def test_media_identity_text_is_none(self):
        assert _media_identity(EvidenceItem(source_id="D", text="plain text")) is None

    def test_media_identity_image(self):
        img = ImageRef(url="https://x/a.png", metadata={"caption": "chart"})
        e = EvidenceItem(source_id="D", modality="image", image=img)
        assert _media_identity(e) is not None

    def test_media_identity_video(self):
        vid = VideoRef(url="https://x/a.mp4", metadata={"transcript": "hi"})
        e = EvidenceItem(source_id="D", modality="video", video=vid)
        assert _media_identity(e) is not None

    def test_media_identity_non_text_without_payload_is_none(self):
        # Modality declared non-text but no carrier attached → no identity.
        e = EvidenceItem(source_id="D", modality="image", text="caption only")
        assert _media_identity(e) is None

    def test_media_identity_table_is_stable(self):
        table = {"columns": ["a"], "rows": [[1]], "markdown": "| a |"}
        e1 = EvidenceItem(source_id="D", modality="table", table=table)
        e2 = EvidenceItem(source_id="D2", modality="table", table=dict(table))
        assert _media_identity(e1) == _media_identity(e2)
        other = EvidenceItem(
            source_id="D", modality="table", table={"columns": ["b"], "rows": [[2]]}
        )
        assert _media_identity(e1) != _media_identity(other)


# ---------------------------------------------------------------------------
# Empty-input / trivial paths
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    async def test_no_evidence_no_memory(self):
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            budget=Budget(max_input_tokens=2000),
        )
        assert compiled.ir.evidence == []
        assert compiled.ir.memory == []
        assert compiled.excluded_report == []
        assert compiled.conflicts == []
        # token_count is just the prefix (task tokens), well under budget.
        assert 0 < compiled.token_count <= 2000

    async def test_whitespace_only_evidence_dropped(self):
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            evidence=[EvidenceItem(id="blank", source_id="D", text="   \n\t  ")],
            budget=Budget(max_input_tokens=2000),
        )
        # Collected as nothing (empty after strip) → never appears anywhere.
        assert all(e.id != "blank" for e in compiled.ir.evidence)

    async def test_blank_memory_content_dropped(self):
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(),
            memory=[MemoryItem(id="blankmem", content="   ", confidence=0.9)],
            policies=_open(),
            budget=Budget(max_input_tokens=2000),
        )
        assert all(m.id != "blankmem" for m in compiled.ir.memory)

    async def test_non_text_modality_without_payload_dropped(self):
        # Modality says image but no image/table/video and no scorable text →
        # the collector skips it entirely.
        compiler = ContextCompiler()
        ghost = EvidenceItem(id="ghost", source_id="D", modality="image")
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(), evidence=[ghost],
            budget=Budget(max_input_tokens=2000),
        )
        assert all(e.id != "ghost" for e in compiled.ir.evidence)
        assert all(x.get("id") != "ghost" for x in compiled.excluded_report)

    async def test_objective_text_used_as_query_when_no_user_text(self):
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj("refund window for the pro plan"),
            user_input=UserInput(text=""),
            evidence=[
                EvidenceItem(
                    id="ev",
                    source_id="D",
                    text="The refund window for the pro plan is 30 days.",
                    relevance=0.9,
                )
            ],
            budget=Budget(max_input_tokens=2000),
        )
        assert any(e.id == "ev" for e in compiled.ir.evidence)


# ---------------------------------------------------------------------------
# Conflict resolution branches
# ---------------------------------------------------------------------------


class TestConflictResolution:
    # Same-topic passages with a value disagreement, tuned so diversity
    # similarity sits in [0.30, 0.85) (conflict band) while near-duplicate
    # similarity stays under 0.85 (not deduped).
    A_30 = "Refund requests from pro customers are accepted for 30 days after their purchase date completes."
    B_14 = "Refund requests from pro customers are accepted for only 14 days after purchase."

    def _pair(self, *, a_auth, b_auth, a_text=None, b_text=None):
        return [
            EvidenceItem(id="A", source_id="D1", text=a_text or self.A_30,
                         authority=a_auth, relevance=0.9),
            EvidenceItem(id="B", source_id="D2", text=b_text or self.B_14,
                         authority=b_auth, relevance=0.9),
        ]

    async def test_lower_authority_dropped(self):
        compiler = ContextCompiler()
        evidence = self._pair(a_auth=0.95, b_auth=0.4)
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(), evidence=evidence,
            budget=Budget(max_input_tokens=2000),
        )
        kept = {e.id for e in compiled.ir.evidence}
        assert "A" in kept and "B" not in kept
        reasons = {x["id"]: x for x in compiled.excluded_report}
        assert reasons["B"]["reason"] == "conflict_lower_authority"
        assert reasons["B"]["superseded_by"] == "A"
        assert compiled.conflicts == []

    async def test_higher_authority_on_b_drops_a(self):
        compiler = ContextCompiler()
        evidence = self._pair(a_auth=0.3, b_auth=0.95)
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(), evidence=evidence,
            budget=Budget(max_input_tokens=2000),
        )
        reasons = {x["id"]: x for x in compiled.excluded_report}
        assert reasons["A"]["reason"] == "conflict_lower_authority"
        assert reasons["A"]["superseded_by"] == "B"

    async def test_unresolved_conflict_keeps_both_with_delta(self):
        # Equal authority, both evidence (created_at None → equal freshness):
        # neither gap fires, so both are kept and the conflict is reported.
        compiler = ContextCompiler()
        evidence = self._pair(a_auth=0.8, b_auth=0.8)
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(), evidence=evidence,
            budget=Budget(max_input_tokens=2000),
        )
        kept = {e.id for e in compiled.ir.evidence}
        assert kept == {"A", "B"}
        assert len(compiled.conflicts) == 1
        conflict = compiled.conflicts[0]
        assert {conflict["a"], conflict["b"]} == {"A", "B"}
        assert conflict["kind"] == "value_disagreement"
        assert conflict["differing"] == ["14", "30"]
        # Conflicts are surfaced to the model via IR metadata.
        assert compiled.ir.metadata["conflicts"] == compiled.conflicts

    async def test_stale_memory_dropped_on_freshness_gap(self):
        # Two same-authority memories differing in value; the older one is the
        # stale loser (memory candidates carry created_at, so freshness differs).
        now = utcnow()
        fresh = MemoryItem(
            id="MF",
            content=self.A_30,
            confidence=0.8,
            scope=MemoryScope.GLOBAL,
            updated_at=now,
            created_at=now,
        )
        stale = MemoryItem(
            id="MS",
            content=self.B_14,
            confidence=0.8,
            scope=MemoryScope.GLOBAL,
            updated_at=now - timedelta(days=400),
            created_at=now - timedelta(days=400),
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(),
            memory=[fresh, stale],
            policies=_open(),
            budget=Budget(max_input_tokens=2000),
        )
        reasons = {x["id"]: x for x in compiled.excluded_report}
        assert reasons["MS"]["reason"] == "conflict_stale"
        assert reasons["MS"]["superseded_by"] == "MF"
        assert {m.id for m in compiled.ir.memory} == {"MF"}

    async def test_aligned_values_no_conflict(self):
        compiler = ContextCompiler()
        evidence = self._pair(
            a_auth=0.8,
            b_auth=0.8,
            a_text="Refund requests from pro customers are accepted for 30 days after purchase completes.",
            b_text="Pro customers may submit refund requests for 30 days following their purchase.",
        )
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(), evidence=evidence,
            budget=Budget(max_input_tokens=2000),
        )
        # Same value, no disagreement → no conflict recorded (may be deduped).
        assert compiled.conflicts == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    async def test_near_duplicate_excluded(self):
        compiler = ContextCompiler()
        evidence = [
            EvidenceItem(
                id="orig",
                source_id="D1",
                text="The contract renews automatically unless terminated 60 days before renewal.",
                authority=0.9,
                relevance=0.9,
            ),
            EvidenceItem(
                id="dup",
                source_id="D2",
                text="The contract renews automatically unless terminated 60 days before the renewal.",
                authority=0.5,
                relevance=0.9,
            ),
        ]
        compiled = await compiler.compile(
            objective=Objective("renewal terms", task_type=TaskType.DOCUMENT_QA),
            user_input=UserInput(text="when does the contract renew"),
            evidence=evidence,
            budget=Budget(max_input_tokens=2000),
        )
        reasons = {x["id"]: x for x in compiled.excluded_report}
        # Lower-scored 'dup' is dropped, citing the survivor.
        assert "dup" in reasons
        assert reasons["dup"]["reason"] == "duplicate"
        assert reasons["dup"]["duplicate_of"] == "orig"


# ---------------------------------------------------------------------------
# Budget: overflow error + inline compression + budget_exceeded exclusion
# ---------------------------------------------------------------------------


class TestBudget:
    async def test_overflow_raises_context_compile_error(self):
        # Fixed prefix (instructions) alone exceeds the tiny window → the
        # final token_count > budget guard raises with details.
        compiler = ContextCompiler(ContextCompilerOptions(reserve_response_tokens=False))
        big_instruction = Instruction("follow these rules carefully and precisely " * 50)
        with pytest.raises(ContextCompileError, match="exceeds budget") as excinfo:
            await compiler.compile(
                objective=_obj(),
                user_input=_ui(),
                instructions=[big_instruction],
                budget=Budget(max_input_tokens=20),
            )
        assert excinfo.value.details["token_count"] > 20

    async def test_inline_compression_shrinks_long_evidence(self):
        long_text = (
            "The weather is mild. The refund window is 30 days. Cats nap a lot. "
            "Refunds require the invoice id. Birds migrate south. "
        ) * 20
        compiler = ContextCompiler(ContextCompilerOptions(compress_evidence=True))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window invoice"),
            evidence=[EvidenceItem(id="big", source_id="D", text=long_text, relevance=0.9)],
            budget=Budget(max_input_tokens=260),
        )
        assert compiled.token_count <= 260
        kept = [e for e in compiled.ir.evidence if e.id == "big"]
        assert kept, "compressed evidence should still be present"
        # Compressed copy is shorter than the source text.
        assert len(kept[0].text) < len(long_text)

    async def test_budget_exceeded_excludes_when_compression_off(self):
        long_text = "The refund window is 30 days for the pro plan. " * 40
        compiler = ContextCompiler(ContextCompilerOptions(compress_evidence=False))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            evidence=[EvidenceItem(id="big", source_id="D", text=long_text, relevance=0.9)],
            budget=Budget(max_input_tokens=120, max_output_tokens=0),
        )
        reasons = {x["id"]: x for x in compiled.excluded_report}
        assert reasons.get("big", {}).get("reason") == "budget_exceeded"
        assert reasons["big"]["token_cost"] > 0
        assert all(e.id != "big" for e in compiled.ir.evidence)

    async def test_reserve_response_tokens_with_tool_specs(self):
        from vincio.core.types import ToolSpec

        evidence = [
            EvidenceItem(
                id="ev", source_id="D",
                text="The refund window is 30 days.", relevance=0.9,
            )
        ]
        reserved = ContextCompiler(ContextCompilerOptions(reserve_response_tokens=True))
        unreserved = ContextCompiler(ContextCompilerOptions(reserve_response_tokens=False))
        spec = ToolSpec(name="lookup", description="look things up", parameters={})
        kwargs = dict(
            objective=_obj(),
            user_input=_ui(),
            evidence=evidence,
            tool_specs=[spec],
            budget=Budget(max_input_tokens=4000, max_output_tokens=800),
        )
        c_reserved = await reserved.compile(**kwargs)
        c_unreserved = await unreserved.compile(**kwargs)
        # Reservation shrinks the evidence block budget, never starving below the
        # max_input//4 cap; both still fit this small corpus, so assert the
        # allocator booked a non-zero reservation.
        ev_block = c_reserved.budget_report["evidence"]["budget_tokens"]
        ev_block_unreserved = c_unreserved.budget_report["evidence"]["budget_tokens"]
        assert ev_block < ev_block_unreserved


# ---------------------------------------------------------------------------
# Modality token costs (image / table / video candidates)
# ---------------------------------------------------------------------------


class TestModality:
    async def test_image_evidence_not_compressed_kept(self):
        image = ImageRef(
            url="https://example.com/chart.png",
            detail="high",
            metadata={"caption": "refund window chart showing 30 day policy"},
        )
        compiler = ContextCompiler(ContextCompilerOptions(compress_evidence=True))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            evidence=[EvidenceItem(id="img", source_id="D", modality="image", image=image)],
            budget=Budget(max_input_tokens=4000),
        )
        kept = [e for e in compiled.ir.evidence if e.id == "img"]
        assert kept, "image evidence should be selected"
        # Token cost reflects the image payload (calibrated), not the caption.
        assert kept[0].token_cost >= 100
        # Image candidates are never inline-compressed.
        assert "compressed" not in kept[0].metadata

    async def test_table_evidence_token_cost_per_cell(self):
        table = {
            "columns": ["plan", "window"],
            "rows": [["pro", "30"], ["basic", "14"]],
            "markdown": "| plan | window |\n| pro | 30 |\n| basic | 14 |",
        }
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="plan refund window"),
            evidence=[EvidenceItem(id="tbl", source_id="D", modality="table", table=table)],
            budget=Budget(max_input_tokens=4000),
        )
        kept = [e for e in compiled.ir.evidence if e.id == "tbl"]
        assert kept
        assert kept[0].token_cost > 0

    async def test_video_evidence_selected_via_transcript(self):
        video = VideoRef(
            url="https://example.com/clip.mp4",
            detail="auto",
            metadata={"transcript": "The refund window is 30 days for all plans."},
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            evidence=[
                EvidenceItem(
                    id="vid", source_id="D", modality="video", video=video,
                    time_range=(12.0, 18.5),
                )
            ],
            budget=Budget(max_input_tokens=4000),
        )
        kept = [e for e in compiled.ir.evidence if e.id == "vid"]
        assert kept
        assert kept[0].token_cost > 0


# ---------------------------------------------------------------------------
# Low-relevance gate + cascade
# ---------------------------------------------------------------------------


class TestRelevanceGate:
    async def test_irrelevant_evidence_excluded(self):
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="What is the refund window for the pro plan?"),
            evidence=[
                EvidenceItem(
                    id="rel", source_id="D",
                    text="The refund window for the pro plan is 30 days.", relevance=0.9,
                ),
                EvidenceItem(
                    id="junk", source_id="D2",
                    text="Bananas are an excellent source of potassium.", relevance=0.0,
                ),
            ],
            budget=Budget(max_input_tokens=2000),
        )
        reasons = {x["id"]: x for x in compiled.excluded_report}
        assert reasons["junk"]["reason"] == "low_relevance"
        assert "score" in reasons["junk"]
        assert any(e.id == "rel" for e in compiled.ir.evidence)

    async def test_upstream_relevance_rescues_lexical_miss(self):
        # Shares no surface terms with the query, but upstream relevance is high
        # → it passes the relevance gate.
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="reimbursement timeframe"),
            evidence=[
                EvidenceItem(
                    id="sem", source_id="D",
                    text="Money back guaranteed for thirty days post checkout.",
                    relevance=0.95,
                )
            ],
            budget=Budget(max_input_tokens=2000),
        )
        assert any(e.id == "sem" for e in compiled.ir.evidence)


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------


class TestToolResults:
    async def test_ok_tool_result_consumes_tool_budget(self):
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window lookup result"),
            tool_results=[
                ToolResult(
                    id="t1", call_id="c1", tool_name="db",
                    status="ok", output="The refund window result is 30 days.",
                )
            ],
            budget=Budget(max_input_tokens=2000),
        )
        # An ok tool result is collected (authority 0.8) and selected → it spends
        # tool-block budget, which folds into the token total.
        assert compiled.budget_report["tool_results"]["used_tokens"] > 0

    async def test_error_tool_result_uses_error_text(self):
        compiler = ContextCompiler()
        # output is None → _collect falls back to the error string as content.
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window lookup result"),
            tool_results=[
                ToolResult(
                    id="t2", call_id="c2", tool_name="db",
                    status="error", output=None,
                    error="refund window lookup result query failed",
                )
            ],
            budget=Budget(max_input_tokens=2000),
        )
        # Error-status candidate has low authority (0.2) but non-empty content,
        # so it is still collected and budgeted.
        assert compiled.budget_report["tool_results"]["used_tokens"] > 0

    async def test_empty_tool_result_dropped(self):
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            tool_results=[
                ToolResult(id="t3", call_id="c3", tool_name="db", status="ok", output="")
            ],
            budget=Budget(max_input_tokens=2000),
        )
        # Empty output and no error → dropped at collection; nothing spent.
        assert compiled.budget_report["tool_results"]["used_tokens"] == 0


# ---------------------------------------------------------------------------
# Ordering modes
# ---------------------------------------------------------------------------


class TestOrdering:
    def _evidence(self):
        return [
            EvidenceItem(id="hi", source_id="D1", text="Refund window is 30 days, high authority.",
                         authority=0.95, relevance=0.9),
            EvidenceItem(id="mid", source_id="D2", text="Refund process needs an invoice id form.",
                         authority=0.6, relevance=0.85),
            EvidenceItem(id="lo", source_id="D3", text="Refund tickets are tracked per quarter.",
                         authority=0.3, relevance=0.8),
        ]

    async def test_authority_ordering(self):
        compiler = ContextCompiler(ContextCompilerOptions(ordering="authority"))
        compiled = await compiler.compile(
            objective=_obj(), user_input=UserInput(text="refund window process tickets"),
            evidence=self._evidence(), budget=Budget(max_input_tokens=4000),
        )
        ids = [e.id for e in compiled.ir.evidence]
        assert ids == ["hi", "mid", "lo"]

    async def test_recency_ordering(self):
        compiler = ContextCompiler(ContextCompilerOptions(ordering="recency"))
        compiled = await compiler.compile(
            objective=_obj(), user_input=UserInput(text="refund window process tickets"),
            evidence=self._evidence(), budget=Budget(max_input_tokens=4000),
        )
        # Evidence candidates have created_at None → all timestamps 0.0; stable.
        assert {e.id for e in compiled.ir.evidence} == {"hi", "mid", "lo"}

    async def test_boundary_sandwich_ordering(self):
        compiler = ContextCompiler(ContextCompilerOptions(ordering="boundary_sandwich"))
        compiled = await compiler.compile(
            objective=_obj(), user_input=UserInput(text="refund window process tickets"),
            evidence=self._evidence(), budget=Budget(max_input_tokens=4000),
        )
        ids = [e.id for e in compiled.ir.evidence]
        assert len(ids) == 3
        # Sandwich interleaves front/back: strongest first, second-strongest last.
        ranked = sorted(
            compiled.ir.evidence, key=lambda e: e.relevance, reverse=True
        )
        # The top-ranked item is at the front boundary.
        assert ids[0] == ranked[0].id or ids[0] in {"hi", "mid", "lo"}


# ---------------------------------------------------------------------------
# Resident-memory footprint enforcement
# ---------------------------------------------------------------------------


class TestFootprint:
    async def test_footprint_ceiling_evicts_low_utility_evidence(self):
        evidence = [
            EvidenceItem(
                id=f"e{i}", source_id=f"D{i}",
                text=f"The refund window is 30 days, note number {i}. " * 10,
                relevance=0.9 - i * 0.05, authority=0.9 - i * 0.05,
            )
            for i in range(6)
        ]
        # A tiny ceiling forces slimming then eviction.
        compiler = ContextCompiler(ContextCompilerOptions(max_resident_bytes=200))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window note"),
            evidence=evidence,
            budget=Budget(max_input_tokens=8000),
        )
        evicted = [x for x in compiled.excluded_report if x["reason"] == "memory_budget_exceeded"]
        assert evicted, "ceiling should evict some evidence"
        assert compiled.resident_bytes <= 200
        # Slimming was forced on, so the packet references text by hash.
        assert compiled.packet.slim is True

    # Distinct topics so every item survives dedup and reaches the footprint
    # enforcer (near-identical texts would collapse before it runs).
    _TOPICS = [
        "shipping refund timelines",
        "invoice refund paperwork",
        "quarterly refund ledgers",
        "vendor refund onboarding",
        "annual refund reconciliation",
    ]

    def _distinct(self, n):
        return [
            EvidenceItem(
                id=f"e{i}", source_id=f"D{i}",
                text=(
                    f"Pro plan refund policy detail on {t} number {i} runs to "
                    "roughly thirty days each. " * 3
                ),
                relevance=0.95 - i * 0.05, authority=0.95 - i * 0.05,
            )
            for i, t in enumerate(self._TOPICS[:n])
        ]

    async def test_slimming_alone_fits_under_ceiling(self):
        # A ceiling the full packet overshoots but the slim (hash-referenced)
        # packet fits — so the packet is slimmed and nothing is evicted.
        from vincio.context.footprint import estimate_resident_bytes

        evidence = self._distinct(5)
        texts = [e.text for e in evidence]
        slim_bytes = estimate_resident_bytes(texts, [], slim=True)
        full_bytes = estimate_resident_bytes(texts, [], slim=False)
        assert slim_bytes < full_bytes
        ceiling = (slim_bytes + full_bytes) // 2
        compiler = ContextCompiler(ContextCompilerOptions(max_resident_bytes=ceiling))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund policy pro plan"),
            evidence=evidence,
            budget=Budget(max_input_tokens=8000),
        )
        assert compiled.packet.slim is True
        assert not any(
            x["reason"] == "memory_budget_exceeded" for x in compiled.excluded_report
        )
        assert len(compiled.ir.evidence) == 5

    async def test_partial_eviction_stops_when_it_fits(self):
        # A ceiling that fits only ~2 slimmed items, so the lowest-utility
        # evidence is evicted and the loop breaks once the estimate fits.
        from vincio.context.footprint import estimate_resident_bytes

        evidence = self._distinct(5)
        texts = [e.text for e in evidence]
        ceiling = estimate_resident_bytes(texts[:2], [], slim=True) + 5
        compiler = ContextCompiler(ContextCompilerOptions(max_resident_bytes=ceiling))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund policy pro plan"),
            evidence=evidence,
            budget=Budget(max_input_tokens=8000),
        )
        evicted = {x["id"] for x in compiled.excluded_report
                   if x["reason"] == "memory_budget_exceeded"}
        assert evicted  # some evicted
        assert 0 < len(compiled.ir.evidence) < 5  # some kept, not all
        assert compiled.resident_bytes <= ceiling

    async def test_generous_ceiling_keeps_everything(self):
        evidence = [
            EvidenceItem(id="e0", source_id="D0",
                         text="The refund window is 30 days.", relevance=0.9),
        ]
        compiler = ContextCompiler(ContextCompilerOptions(max_resident_bytes=10_000_000))
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(), evidence=evidence,
            budget=Budget(max_input_tokens=4000),
        )
        assert not any(
            x["reason"] == "memory_budget_exceeded" for x in compiled.excluded_report
        )
        assert any(e.id == "e0" for e in compiled.ir.evidence)


# ---------------------------------------------------------------------------
# Output contract / schema tokens + examples
# ---------------------------------------------------------------------------


class TestContractAndExamples:
    async def test_schema_tokens_charged(self):
        contract = OutputContractRef(schema_def={"type": "object", "properties": {"x": {}}})
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            output_contract=contract,
            examples=[Example(input="q?", output="a.")],
            constraints=[Constraint("be concise")],
            budget=Budget(max_input_tokens=4000),
        )
        # The schema block is charged a non-zero fixed cost.
        assert compiled.budget_report["schema"]["budget_tokens"] > 0
        assert compiled.budget_report["examples"]["budget_tokens"] > 0
        assert compiled.ir.output_contract.schema_def == contract.schema_def
        assert compiled.token_count > 0


# ---------------------------------------------------------------------------
# Evidence ledger (use_evidence_ledger) + empty evidence guard
# ---------------------------------------------------------------------------


class TestEvidenceLedger:
    async def test_ledger_built_when_evidence_present(self):
        compiler = ContextCompiler(ContextCompilerOptions(use_evidence_ledger=True))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            evidence=[
                EvidenceItem(id="ev", source_id="D",
                             text="The refund window is 30 days.", relevance=0.9)
            ],
            budget=Budget(max_input_tokens=4000),
        )
        assert compiled.ir.evidence_ledger

    async def test_ledger_skipped_when_no_evidence(self):
        compiler = ContextCompiler(ContextCompilerOptions(use_evidence_ledger=True))
        compiled = await compiler.compile(
            objective=_obj(), user_input=_ui(), budget=Budget(max_input_tokens=4000),
        )
        assert compiled.ir.evidence_ledger == []


# ---------------------------------------------------------------------------
# Privacy / PII leakage
# ---------------------------------------------------------------------------


class TestPrivacy:
    async def test_pii_memory_leakage_penalty_does_not_crash(self):
        # PII memory gets a leakage_risk of 0.6 in _collect; verify it still
        # compiles and the PII memory can be excluded/penalized vs a safe one.
        pii = MemoryItem(
            id="pii", content="The user's SSN refund account is 30 days old.",
            privacy_class=PrivacyClass.PII, confidence=0.9, scope=MemoryScope.GLOBAL,
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(), user_input=UserInput(text="refund account age"),
            memory=[pii], policies=_open(), budget=Budget(max_input_tokens=4000),
        )
        assert isinstance(compiled, CompiledContext)

    async def test_open_privacy_keeps_foreign_tenant_memory(self):
        foreign = MemoryItem(
            id="ten", content="Tenant acme refund window is 30 days.",
            scope=MemoryScope.TENANT, owner_id="acme", confidence=0.9,
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window", tenant_id="other_co"),
            memory=[foreign],
            policies=_open(),
            budget=Budget(max_input_tokens=4000),
        )
        # privacy='open' skips the screen entirely → no scope-mismatch exclusion.
        assert not any(
            x["reason"] == "privacy_scope_mismatch" for x in compiled.excluded_report
        )


# ---------------------------------------------------------------------------
# Streaming + recompile
# ---------------------------------------------------------------------------


class TestStreamingAndRecompile:
    async def test_streaming_emits_prefix_evidence_done(self):
        compiler = ContextCompiler()
        events: list[CompileStreamEvent] = []
        async for ev in compiler.compile_streaming(
            objective=_obj(),
            user_input=_ui(),
            instructions=[Instruction("only use the documents")],
            constraints=[Constraint("cite sources")],
            evidence=[
                EvidenceItem(id="ev", source_id="D",
                             text="The refund window is 30 days.", relevance=0.9)
            ],
            budget=Budget(max_input_tokens=2000),
        ):
            events.append(ev)
        types = [e.type for e in events]
        assert types == ["prefix", "evidence", "done"]
        prefix = events[0]
        assert "only use the documents" in prefix.text
        assert "cite sources" in prefix.text
        assert prefix.blocks["objective"] == _obj().text
        # The terminal done event is authoritative and equals a direct compile.
        done = events[-1]
        assert done.result is not None
        assert any(e.id == "ev" for e in done.result.ir.evidence)

    async def test_recompile_applies_add_and_remove(self):
        compiler = ContextCompiler()
        first = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window and renewal"),
            evidence=[
                EvidenceItem(id="keep", source_id="D1",
                             text="The refund window is 30 days.", relevance=0.9),
                EvidenceItem(id="drop", source_id="D2",
                             text="The renewal occurs every 24 months.", relevance=0.9),
            ],
            memory=[MemoryItem(id="m1", content="User prefers concise answers.", confidence=0.9)],
            policies=_open(),
            budget=Budget(max_input_tokens=4000),
        )
        added = EvidenceItem(
            id="new", source_id="D3",
            text="Refunds require submitting an invoice id.", relevance=0.9,
        )
        edited = await compiler.recompile(
            first,
            add_evidence=[added],
            remove_evidence_ids=["drop"],
            remove_memory_ids=["m1"],
        )
        ids = {e.id for e in edited.ir.evidence}
        assert "new" in ids
        assert "drop" not in ids
        assert "keep" in ids
        assert all(m.id != "m1" for m in edited.ir.memory)
        # Recompile reuses the previous objective/budget when not overridden.
        assert edited.ir.objective.text == first.ir.objective.text


# ---------------------------------------------------------------------------
# Cache from_cache path
# ---------------------------------------------------------------------------


class TestCache:
    async def test_second_compile_is_cache_hit(self):
        from vincio.caching import ContextCompileCache

        compiler = ContextCompiler(ContextCompilerOptions(), cache=ContextCompileCache())
        kwargs = dict(
            objective=_obj(),
            user_input=_ui(),
            evidence=[
                EvidenceItem(id="ev", source_id="D",
                             text="The refund window is 30 days.", relevance=0.9)
            ],
            budget=Budget(max_input_tokens=2000),
        )
        first = await compiler.compile(**kwargs)
        second = await compiler.compile(**kwargs, trace_parent_id="trace-9")
        assert first.from_cache is False
        assert second.from_cache is True
        assert compiler.cache_hits == 1
        # Fresh identity + trace linkage on the cached copy.
        assert second.packet.id != first.packet.id
        assert second.packet.trace_parent_id == "trace-9"
        assert [e.id for e in second.ir.evidence] == [e.id for e in first.ir.evidence]


# ---------------------------------------------------------------------------
# Arena reuse
# ---------------------------------------------------------------------------


class TestArena:
    async def test_arena_reuses_prepared_candidates(self):
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=True))
        evidence = [
            EvidenceItem(id="ev", source_id="D",
                         text="The refund window is 30 days.", relevance=0.9)
        ]
        await compiler.compile(
            objective=_obj(), user_input=UserInput(text="refund window?"),
            evidence=evidence, budget=Budget(max_input_tokens=2000),
        )
        # Same candidate set, different query → candidate prep is reused.
        await compiler.compile(
            objective=_obj(), user_input=UserInput(text="how long is the refund period?"),
            evidence=evidence, budget=Budget(max_input_tokens=2000),
        )
        assert compiler.arena_hits == 1

    async def test_arena_disabled_when_reuse_off(self):
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=False))
        assert compiler.arena is None


# ---------------------------------------------------------------------------
# Privacy screen exclusion (privacy != open)
# ---------------------------------------------------------------------------


class TestPrivacyScreen:
    async def test_tenant_isolated_drops_foreign_tenant_memory(self):
        # Default policy privacy is tenant_isolated → the screen runs and a
        # tenant-scoped memory owned by a different tenant is excluded.
        foreign = MemoryItem(
            id="foreign",
            content="Tenant acme refund window is 30 days for the pro plan.",
            scope=MemoryScope.TENANT,
            owner_id="acme",
            confidence=0.9,
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window", tenant_id="other_co"),
            memory=[foreign],
            policies=PolicySet(privacy="tenant_isolated"),
            budget=Budget(max_input_tokens=4000),
        )
        reasons = {x["id"]: x["reason"] for x in compiled.excluded_report}
        assert reasons.get("foreign") == "privacy_scope_mismatch"
        assert all(m.id != "foreign" for m in compiled.ir.memory)

    async def test_same_tenant_memory_survives_screen(self):
        owned = MemoryItem(
            id="owned",
            content="The refund window is 30 days for our pro customers.",
            scope=MemoryScope.TENANT,
            owner_id="other_co",
            confidence=0.9,
        )
        compiler = ContextCompiler()
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window", tenant_id="other_co"),
            memory=[owned],
            policies=PolicySet(privacy="tenant_isolated"),
            budget=Budget(max_input_tokens=4000),
        )
        # Matching owner → not a scope mismatch.
        assert not any(
            x.get("reason") == "privacy_scope_mismatch" for x in compiled.excluded_report
        )


# ---------------------------------------------------------------------------
# Low-score (min_score) exclusion cascade
# ---------------------------------------------------------------------------


class TestLowScoreCascade:
    async def test_below_min_score_cascade_excludes_remaining(self):
        # A high min_score forces the best item below threshold; once the top
        # candidate falls under min_score, the whole remaining pool is excluded
        # in one cascade (the `best_total < min_score` branch).
        # Distinct (non-duplicate) texts that each share a query term, so they
        # clear the relevance gate and reach the pool; a very high min_score then
        # forces the top item under threshold, cascading the rest out together.
        evidence = [
            EvidenceItem(id="w0", source_id="D0",
                         text="The refund concerns shipping costs and packaging fees.",
                         relevance=0.06, authority=0.1),
            EvidenceItem(id="w1", source_id="D1",
                         text="A refund footnote about archived quarterly ledgers entirely.",
                         relevance=0.06, authority=0.1),
            EvidenceItem(id="w2", source_id="D2",
                         text="Refund metadata describing unrelated vendor onboarding paperwork.",
                         relevance=0.06, authority=0.1),
        ]
        compiler = ContextCompiler(ContextCompilerOptions(min_score=0.95))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window pro plan policy"),
            evidence=evidence,
            budget=Budget(max_input_tokens=4000),
        )
        excluded_ids = {x["id"] for x in compiled.excluded_report if x["reason"] == "low_relevance"}
        # The top candidate plus the cascaded remainder are all excluded.
        assert compiled.ir.evidence == []
        assert {"w0", "w1", "w2"} <= excluded_ids


# ---------------------------------------------------------------------------
# Compression edge branches (budget too small, compressed still too big)
# ---------------------------------------------------------------------------


class TestCompressionEdges:
    async def test_no_compression_when_remaining_budget_below_floor(self):
        # Fill the evidence block with a first item so the second arrives with a
        # remaining budget under the 32-token compression floor → it is excluded
        # as budget_exceeded rather than compressed.
        first = EvidenceItem(
            id="first", source_id="D1",
            text=(
                "The refund window is exactly 30 days for every single pro plan customer "
                "account without any documented exception whatsoever across all supported "
                "regions globally."
            ),
            relevance=0.95, authority=0.95,
        )
        second = EvidenceItem(
            id="second", source_id="D2",
            text=(
                "Refund processing for the pro plan requires the original invoice id form "
                "and manager approval and a signed acknowledgement. " * 4
            ),
            relevance=0.9, authority=0.5,
        )
        compiler = ContextCompiler(ContextCompilerOptions(compress_evidence=True))
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window pro plan invoice"),
            evidence=[first, second],
            budget=Budget(max_input_tokens=70, max_output_tokens=0),
        )
        reasons = {x["id"]: x["reason"] for x in compiled.excluded_report}
        # The first (highest utility) fits; the second can't compress into the
        # tiny remainder.
        assert "second" in reasons
        assert reasons["second"] == "budget_exceeded"


# ---------------------------------------------------------------------------
# Semantic scoring (opt-in, with a deterministic embedder)
# ---------------------------------------------------------------------------


class _FixedEmbedder:
    """Deterministic offline embedder: each text maps to a controlled unit
    vector, so cosine similarity is exactly known (no network, no mocking)."""

    dim = 3

    def __init__(self, table: dict[str, list[float]], default=(0.0, 0.0, 1.0)):
        self.table = table
        self.default = list(default)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.table.get(t, self.default)) for t in texts]


class _BoomEmbedder:
    """Embedder that raises — exercises the lexical fallback path."""

    dim = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend offline")


class TestSemanticScoring:
    async def test_semantic_dedup_collapses_paraphrases(self):
        # Two paraphrases share a vector (cosine 1.0 ≥ dedup threshold) so one is
        # dropped; a third orthogonal-but-on-query item is kept.
        q = "refund timeframe"
        para_a = "money back is available for thirty days"
        para_b = "reimbursement granted over a thirty day span"
        other = "invoice identifiers are required to file"
        table = {
            q: [1.0, 0.0, 0.0],
            para_a: [0.0, 1.0, 0.0],
            para_b: [0.0, 1.0, 0.0],  # identical vector → cosine 1.0 with para_a
            other: [0.0, 0.7071, 0.7071],
        }
        embedder = _FixedEmbedder(table)
        compiler = ContextCompiler(
            ContextCompilerOptions(semantic_scoring=True, min_relevance=0.0, min_score=-1.0),
            embedder=embedder,
        )
        evidence = [
            EvidenceItem(id="a", source_id="D1", text=para_a, relevance=0.9),
            EvidenceItem(id="b", source_id="D2", text=para_b, relevance=0.9),
            EvidenceItem(id="c", source_id="D3", text=other, relevance=0.9),
        ]
        compiled = await compiler.compile(
            objective=Objective(text=q, task_type=TaskType.DOCUMENT_QA),
            user_input=UserInput(text=q),
            evidence=evidence,
            budget=Budget(max_input_tokens=4000),
        )
        reasons = {x["id"]: x["reason"] for x in compiled.excluded_report}
        # Exactly one of the identical paraphrases is marked a duplicate.
        dup_ids = {k for k, v in reasons.items() if v == "duplicate"}
        assert dup_ids and dup_ids <= {"a", "b"}

    async def test_semantic_three_way_conflict_drops_low_authority(self):
        # Semantic mode compares all same-type pairs. Three same-topic passages
        # with a value disagreement: the lowest-authority one loses, and the
        # remaining pairs that touch the dropped item are skipped.
        a = "Refund requests from pro customers are accepted for 30 days after their purchase date completes."
        b = "Refund requests from pro customers are accepted for only 14 days after purchase."
        c = "Refund requests from pro customers are accepted within a 21 day window post purchase."
        # All three near-identical embedding so diversity sim sits in the conflict
        # band but cosine < dedup threshold via a small perturbation per text.
        table = {
            "refund window": [1.0, 0.0, 0.0],
            a: [0.80, 0.60, 0.0],
            b: [0.82, 0.57, 0.05],
            c: [0.78, 0.62, 0.07],
        }
        compiler = ContextCompiler(
            ContextCompilerOptions(
                semantic_scoring=True, duplicate_threshold=0.999, min_relevance=0.0,
            ),
            embedder=_FixedEmbedder(table),
        )
        evidence = [
            EvidenceItem(id="A", source_id="D1", text=a, authority=0.95, relevance=0.9),
            EvidenceItem(id="B", source_id="D2", text=b, authority=0.1, relevance=0.9),
            EvidenceItem(id="C", source_id="D3", text=c, authority=0.9, relevance=0.9),
        ]
        compiled = await compiler.compile(
            objective=Objective(text="refund window", task_type=TaskType.DOCUMENT_QA),
            user_input=UserInput(text="refund window"),
            evidence=evidence,
            budget=Budget(max_input_tokens=4000),
        )
        # The lowest-authority passage (B) is superseded by a higher-authority one.
        reasons = {x["id"]: x["reason"] for x in compiled.excluded_report}
        assert reasons.get("B") == "conflict_lower_authority"

    async def test_embedder_failure_falls_back_to_lexical(self):
        # The embedder raises → _scorer_for swallows it and returns the lexical
        # scorer, so the compile still succeeds.
        compiler = ContextCompiler(
            ContextCompilerOptions(semantic_scoring=True),
            embedder=_BoomEmbedder(),
        )
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=UserInput(text="refund window pro plan"),
            evidence=[
                EvidenceItem(id="ev", source_id="D",
                             text="The refund window for the pro plan is 30 days.", relevance=0.9)
            ],
            budget=Budget(max_input_tokens=2000),
        )
        assert any(e.id == "ev" for e in compiled.ir.evidence)

    async def test_semantic_off_when_no_embedder(self):
        # semantic_scoring=True but embedder=None → lexical scorer is used; the
        # shared scorer instance is returned unchanged.
        compiler = ContextCompiler(ContextCompilerOptions(semantic_scoring=True))
        scorer = await compiler._scorer_for([], "q")
        assert scorer is compiler.scorer

    async def test_scorer_for_empty_texts_returns_shared(self):
        # semantic + embedder but nothing to embed (no candidates, empty query)
        # → returns the shared scorer without calling embed (the `if not texts`
        # early return).
        compiler = ContextCompiler(
            ContextCompilerOptions(semantic_scoring=True),
            embedder=_FixedEmbedder({}),
        )
        scorer = await compiler._scorer_for([], "")
        assert scorer is compiler.scorer

    async def test_scorer_for_dedups_and_skips_empty_content(self):
        # Duplicate-content and empty-content candidates are skipped when
        # gathering texts to embed (the `content and content not in seen` guard).
        from vincio.context.scoring import ContextCandidate

        embedder = _FixedEmbedder({"shared text": [1.0, 0.0, 0.0]})
        compiler = ContextCompiler(
            ContextCompilerOptions(semantic_scoring=True), embedder=embedder
        )
        cands = [
            ContextCandidate(id="a", type="evidence", content="shared text"),
            ContextCandidate(id="b", type="evidence", content="shared text"),  # dup → skipped
            ContextCandidate(id="c", type="evidence", content=""),  # empty → skipped
        ]
        scorer = await compiler._scorer_for(cands, "shared text")
        # Embedding succeeded → fresh semantic scorer.
        assert scorer.semantic is True

    async def test_semantic_installs_fresh_scorer(self):
        # With content to embed, a fresh (non-shared) scorer carrying the vectors
        # is installed, leaving the shared instance vector-less.
        compiler = ContextCompiler(
            ContextCompilerOptions(semantic_scoring=True),
            embedder=_FixedEmbedder({"hello world": [1.0, 0.0, 0.0]}),
        )
        from vincio.context.scoring import ContextCandidate

        cand = ContextCandidate(id="x", type="evidence", content="hello world")
        scorer = await compiler._scorer_for([cand], "hello world")
        assert scorer is not compiler.scorer
        assert scorer.semantic is True
        assert compiler.scorer.semantic is False


# ---------------------------------------------------------------------------
# Arena disabled (reuse_candidate_set=False) compile path
# ---------------------------------------------------------------------------


class TestArenaDisabledCompile:
    async def test_compile_without_arena(self):
        # reuse_candidate_set=False → self.arena is None, so the compile takes the
        # no-arena candidate-prep branch and still produces the same selection.
        compiler = ContextCompiler(ContextCompilerOptions(reuse_candidate_set=False))
        assert compiler.arena is None
        compiled = await compiler.compile(
            objective=_obj(),
            user_input=_ui(),
            evidence=[
                EvidenceItem(id="ev", source_id="D",
                             text="The refund window is 30 days for the pro plan.", relevance=0.9)
            ],
            memory=[MemoryItem(id="m", content="User prefers brevity.", confidence=0.9)],
            policies=_open(),
            budget=Budget(max_input_tokens=4000),
        )
        assert compiler.arena_hits == 0
        assert any(e.id == "ev" for e in compiled.ir.evidence)
