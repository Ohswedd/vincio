"""Packet compile receipt: the compact, text-light manifest of *why* a context
packet was compiled.

Mirrors the acceptance shape from the feature request: build a packet from a
current authoritative source plus an older contradictory one, assert the receipt
records the current source as included and the older one as excluded/superseded,
that it carries stable hashes/ids, scoring, budget and privacy summaries and a
pointer back to the trace, that it exports without raw prompt/evidence text, and
that a recompile of identical inputs yields the same receipt hash while a changed
source yields an explicit divergence.
"""

import json
from datetime import UTC, datetime

import pytest

from vincio import ContextApp
from vincio.context import CompileReceipt, ContextCompiler, ContextCompilerOptions
from vincio.context.receipt import RenderInfo
from vincio.core.types import (
    Budget,
    EvidenceItem,
    Instruction,
    MemoryItem,
    Objective,
    PolicySet,
    TaskType,
    UserInput,
)
from vincio.providers.mock import MockProvider

_NOW = datetime(2026, 7, 1, tzinfo=UTC)
_OLD = datetime(2019, 1, 1, tzinfo=UTC)

# The raw prompt/evidence strings fed into the contradiction compile, kept as a
# single source of truth so the text-leak regression can guard *exactly* the text
# that entered the compiler rather than a hand-maintained denylist that could
# drift from the fixture.
_USER_QUESTION = "What is the refund window?"
_INSTRUCTION = "Answer only from the provided sources"
_EVIDENCE_CURRENT = "Refunds are allowed within 30 days of purchase."
_EVIDENCE_OLD = "Refunds are allowed within 14 days of purchase."
_EVIDENCE_IRRELEVANT = "Bananas are rich in potassium."

# Every raw string a receipt of ``_compile_contradiction`` must NEVER carry.
_CONTRADICTION_RAW_TEXTS = [
    _USER_QUESTION,
    _INSTRUCTION,
    _EVIDENCE_CURRENT,
    _EVIDENCE_OLD,
    _EVIDENCE_IRRELEVANT,
]


def _iter_export_strings(obj):
    """Yield every string anywhere in a JSON-safe export — dict keys, dict values,
    and list elements, recursively.

    The text-leak regression walks these instead of eyeballing a few known fields,
    so a *new* receipt field that ever carried free-form prompt/evidence text is
    caught structurally rather than slipping through a fixed denylist."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _iter_export_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_export_strings(value)


def assert_export_text_light(export, raw_texts):
    """Assert the trust boundary: no raw prompt/evidence string appears anywhere in
    an exported receipt — neither as a whole leaf nor as a substring of one — and
    the structural ``omitted_raw_text`` guarantee is set.

    ``export`` is the dict from :meth:`CompileReceipt.to_export`; ``raw_texts`` are
    the exact prompt/evidence strings fed into the compile."""
    leaves = list(_iter_export_strings(export))
    for raw in raw_texts:
        assert raw, "empty raw text is not a meaningful guard"
        for leaf in leaves:
            assert raw not in leaf, (
                f"raw text leaked into a receipt export field: {raw!r} in {leaf!r}"
            )
    # Belt-and-suspenders: the serialized form carries none of it either.
    blob = json.dumps(export)
    for raw in raw_texts:
        assert raw not in blob, f"raw text leaked into the serialized receipt: {raw!r}"
    assert export["privacy"]["omitted_raw_text"] is True


async def _compile_contradiction(**opts):
    """Current authoritative source (30 days) vs an older, lower-authority,
    contradictory source (14 days), plus an irrelevant item. The compiler keeps
    the authoritative source and supersedes the contradictory one on authority."""
    compiler = ContextCompiler(ContextCompilerOptions(**opts))
    return await compiler.compile(
        objective=Objective("refund window", task_type=TaskType.DOCUMENT_QA),
        user_input=UserInput(text=_USER_QUESTION),
        instructions=[Instruction(_INSTRUCTION)],
        evidence=[
            EvidenceItem(
                id="D1",
                source_id="refunds.md",
                text=_EVIDENCE_CURRENT,
                authority=0.9,
                relevance=0.95,
                created_at=_NOW,
                page=2,
            ),
            EvidenceItem(
                id="D9",
                source_id="refunds_old.md",
                text=_EVIDENCE_OLD,
                authority=0.4,
                relevance=0.9,
                created_at=_OLD,
            ),
            EvidenceItem(
                id="D3",
                source_id="misc.md",
                text=_EVIDENCE_IRRELEVANT,
                authority=0.5,
                relevance=0.01,
            ),
        ],
        budget=Budget(max_input_tokens=2000),
    )


class TestCompileReceiptDecision:
    @pytest.mark.asyncio
    async def test_included_and_superseded_are_recorded(self):
        compiled = await _compile_contradiction()
        receipt = compiled.receipt()

        included = {it.id: it for it in receipt.included}
        excluded = {it.id: it for it in receipt.excluded}

        # The current authoritative source is included; the older contradictory
        # source is excluded and points at the winner; the irrelevant one is out.
        assert "D1" in included
        assert "D9" in excluded
        assert excluded["D9"].reason == "conflict_lower_authority"
        assert excluded["D9"].superseded_by == "D1"
        assert "D3" in excluded and excluded["D3"].reason == "low_relevance"

        # The conflict is summarized with a winner, loser, and the deciding rule.
        resolved = [c for c in receipt.conflicts if c.loser == "D9"]
        assert resolved and resolved[0].winner == "D1"
        assert resolved[0].rule == "higher_authority"

    @pytest.mark.asyncio
    async def test_scores_hashes_budget_and_privacy_summary(self):
        compiled = await _compile_contradiction()
        receipt = compiled.receipt(run_id="run_x", trace_id="trace_x")

        d1 = next(it for it in receipt.included if it.id == "D1")
        # Per-item scoring signals that drove selection.
        assert d1.score is not None and d1.authority == pytest.approx(0.9)
        assert d1.source_ref == "refunds.md:p2"  # citation locator, not text
        assert d1.source_hash and d1.source_hash.startswith("sha256:")
        assert d1.reason == "selected"

        # Stable ids and input fingerprint.
        assert receipt.packet_id == compiled.packet.id
        assert receipt.input_fingerprint == "sha256:" + compiled.packet.spec_hash
        assert receipt.compiler_version.startswith("vincio/")

        # Budget summary.
        assert receipt.budget.max_input_tokens == 2000
        assert receipt.budget.used_tokens == compiled.token_count
        assert "evidence" in receipt.budget.blocks

        # Privacy summary — and a pointer back to the trace/run.
        assert receipt.privacy.omitted_raw_text is True
        assert receipt.privacy.privacy_scope == "tenant_isolated"
        assert receipt.trace_id == "trace_x" and receipt.run_id == "run_x"

    @pytest.mark.asyncio
    async def test_export_carries_no_raw_text(self):
        compiled = await _compile_contradiction()
        assert_export_text_light(compiled.receipt().to_export(), _CONTRADICTION_RAW_TEXTS)

    @pytest.mark.asyncio
    async def test_unresolved_conflict_does_not_leak_disputed_values(self):
        # Same authority and freshness, contradictory values -> the conflict is
        # unresolved and both are kept. The receipt must report *that* they
        # disagreed and on how many units, never the disputed values themselves.
        compiler = ContextCompiler(ContextCompilerOptions())
        compiled = await compiler.compile(
            objective=Objective("refund window", task_type=TaskType.DOCUMENT_QA),
            user_input=UserInput(text="What is the refund window?"),
            evidence=[
                EvidenceItem(id="A", source_id="a.md", text="Refunds are allowed within 30 days of purchase.", authority=0.6, relevance=0.9),
                EvidenceItem(id="B", source_id="b.md", text="Refunds are allowed within 14 days of purchase.", authority=0.6, relevance=0.9),
            ],
            budget=Budget(max_input_tokens=2000),
        )
        receipt = compiled.receipt()
        unresolved = [c for c in receipt.conflicts if c.rule == "unresolved_both_included"]
        assert unresolved and unresolved[0].differing_count >= 1
        # The disputed value tokens must not appear anywhere in the export.
        blob = json.dumps(receipt.to_export())
        for value in ('"14"', '"30"', "14 days", "30 days"):
            assert value not in blob

    @pytest.mark.asyncio
    async def test_privacy_scope_exclusion_is_summarized(self):
        compiler = ContextCompiler(ContextCompilerOptions())
        from vincio.core.types import MemoryScope

        compiled = await compiler.compile(
            objective=Objective("tenant spend"),
            user_input=UserInput(text="what does the tenant pay", tenant_id="other_co"),
            memory=[
                MemoryItem(
                    id="M_foreign",
                    content="Tenant Acme pays 50k annually (internal-only note)",
                    scope=MemoryScope.TENANT,
                    owner_id="acme",
                    confidence=0.9,
                )
            ],
            budget=Budget(max_input_tokens=2000),
        )
        receipt = compiled.receipt()
        assert receipt.privacy.scope_excluded_count == 1
        excluded = {it.id: it for it in receipt.excluded}
        assert excluded["M_foreign"].reason == "privacy_scope_mismatch"
        assert excluded["M_foreign"].kind == "memory"
        # The scope-excluded memory's content is never echoed into the receipt.
        assert_export_text_light(
            receipt.to_export(), ["Tenant Acme pays 50k annually (internal-only note)"]
        )


class TestCompileReceiptDeterminism:
    @pytest.mark.asyncio
    async def test_identical_inputs_same_hash_and_no_divergence(self):
        a = (await _compile_contradiction()).receipt()
        b = (await _compile_contradiction()).receipt()
        # Per-run ids differ, but the compile decision (and its hash) is identical.
        assert a.packet_id != b.packet_id
        assert a.receipt_hash == b.receipt_hash
        assert a.diverges_from(b) is None

    @pytest.mark.asyncio
    async def test_changed_source_produces_explicit_divergence(self):
        baseline = (await _compile_contradiction()).receipt()

        # Same query, but the authoritative source now says 45 days.
        compiler = ContextCompiler(ContextCompilerOptions())
        changed = (
            await compiler.compile(
                objective=Objective("refund window", task_type=TaskType.DOCUMENT_QA),
                user_input=UserInput(text="What is the refund window?"),
                instructions=[Instruction("Answer only from the provided sources")],
                evidence=[
                    EvidenceItem(
                        id="D1",
                        source_id="refunds.md",
                        text="Refunds are allowed within 45 days of purchase.",
                        authority=0.9,
                        relevance=0.95,
                        created_at=_NOW,
                        page=2,
                    ),
                ],
                budget=Budget(max_input_tokens=2000),
            )
        ).receipt()

        assert changed.receipt_hash != baseline.receipt_hash
        divergence = changed.diverges_from(baseline)
        assert divergence is not None
        assert divergence["receipt_hash"]["current"] == changed.receipt_hash
        # D1's content hash moved, so its score/fingerprint changed.
        stamped = changed.with_divergence(baseline)
        assert stamped.divergence is not None

    @pytest.mark.asyncio
    async def test_swapped_source_locator_is_detected(self):
        # Same evidence id and text, but a different source document/page: the
        # provenance changed, so the receipt hash must change and the divergence
        # must name the source_ref move (even on the compiler-only render=None path).
        async def _one(source_id: str, page: int):
            compiler = ContextCompiler(ContextCompilerOptions())
            compiled = await compiler.compile(
                objective=Objective("refund window", task_type=TaskType.DOCUMENT_QA),
                user_input=UserInput(text="What is the refund window?"),
                evidence=[
                    EvidenceItem(
                        id="D1",
                        source_id=source_id,
                        text="Refunds are allowed within 30 days of purchase.",
                        authority=0.9,
                        relevance=0.95,
                        page=page,
                    )
                ],
                budget=Budget(max_input_tokens=2000),
            )
            return compiled.receipt()

        baseline = await _one("refunds.md", 2)
        swapped = await _one("other_source.md", 2)
        assert swapped.receipt_hash != baseline.receipt_hash
        divergence = swapped.diverges_from(baseline)
        assert divergence is not None
        changed = {c["id"]: c["changed"] for c in divergence["content_changes"]}
        assert "D1" in changed and "source_ref" in changed["D1"]

    @pytest.mark.asyncio
    async def test_receipt_verifies_from_its_own_bytes(self):
        receipt = (await _compile_contradiction()).receipt(
            render=RenderInfo(provider="mock", model="m", context_ir_hash="a", rendered_packet_hash="b")
        )
        assert receipt.verify() is True
        restored = CompileReceipt.model_validate(receipt.to_export())
        assert restored.receipt_hash == receipt.receipt_hash
        assert restored.verify() is True


class TestCompileReceiptTrustBoundary:
    """The regression fixture from the 7.3 review (issue #140).

    A reviewer attaches a receipt to a PR or an incident precisely because it lets
    them reason about the compile — including *the exact bytes that rendered* — with
    no access to the underlying evidence text. The boundary they lean on has two
    halves that must hold *together*: a changed render identity (the
    ``rendered_packet_hash`` and the other render-identity fields) is surfaced as an
    **explicit** divergence, while ``to_export()`` provably carries **no** raw prompt
    or evidence text. This fixture nails both, so neither can regress silently."""

    # Each render-identity field paired with a changed value. Any one of these
    # moving must (a) change the receipt hash, (b) surface as ``render_changed`` in
    # the divergence, and (c) leave the *selection* decision untouched — the render
    # identity moved alone.
    _RENDER_FIELDS = [
        ("rendered_packet_hash", "sha256:pkt_rerendered"),
        ("context_ir_hash", "sha256:ir_v2"),
        ("prompt_spec_hash", "sha256:spec_v2"),
        ("provider", "anthropic"),
        ("model", "other-model"),
    ]

    @staticmethod
    def _base_render():
        return RenderInfo(
            provider="mock",
            model="m1",
            context_ir_hash="sha256:ir_v1",
            rendered_packet_hash="sha256:pkt_v1",
            prompt_spec_hash="sha256:spec_v1",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field,new_value", _RENDER_FIELDS)
    async def test_render_identity_change_diverges_while_export_stays_text_light(
        self, field, new_value
    ):
        # Same compile decision, rendered two different ways: only the render
        # identity differs between the baseline and the re-render.
        compiled = await _compile_contradiction()
        base_render = self._base_render()
        rerendered_render = base_render.model_copy(update={field: new_value})

        baseline = compiled.receipt(run_id="run_a", trace_id="trace_a", render=base_render)
        rerendered = compiled.receipt(
            run_id="run_b", trace_id="trace_b", render=rerendered_render
        )

        # A different render identity is a different compile-decision hash...
        assert rerendered.receipt_hash != baseline.receipt_hash

        # ...surfaced as an *explicit* divergence flagging the render change, with
        # the selection decision itself provably untouched (render moved alone).
        divergence = rerendered.diverges_from(baseline)
        assert divergence is not None
        assert divergence["render_changed"] is True
        assert divergence["input_fingerprint_changed"] is False
        assert divergence["included_added"] == [] and divergence["included_removed"] == []
        assert divergence["excluded_added"] == [] and divergence["excluded_removed"] == []
        assert divergence["score_changes"] == [] and divergence["content_changes"] == []
        assert divergence["used_tokens_delta"] == 0

        # The divergence rides along on the stamped receipt for the reviewer.
        stamped = rerendered.with_divergence(baseline)
        assert stamped.divergence is not None
        assert stamped.divergence["render_changed"] is True

        # The other half of the boundary: no export — baseline, re-render, or the
        # divergence-stamped copy — carries any raw prompt or evidence text, and
        # each still verifies from its own bytes.
        for receipt in (baseline, rerendered, stamped):
            assert_export_text_light(receipt.to_export(), _CONTRADICTION_RAW_TEXTS)
            assert receipt.verify() is True

    @pytest.mark.asyncio
    async def test_export_is_text_light_over_the_whole_export_tree(self):
        # The text-light half on its own, exhaustively: walk *every* string in the
        # export (with a render identity present so the render block is covered),
        # not a fixed denylist, so a future receipt field that carried text fails.
        compiled = await _compile_contradiction()
        receipt = compiled.receipt(
            run_id="run_x", trace_id="trace_x", render=self._base_render()
        )
        assert_export_text_light(receipt.to_export(), _CONTRADICTION_RAW_TEXTS)

    def test_text_light_guard_has_teeth(self):
        # A guard that can never fail proves nothing. This asserts the text-leak
        # guard *catches* a real leak — a raw evidence string smuggled into a nested
        # export field — so the whole fixture above is meaningful, not vacuous.
        clean = {"privacy": {"omitted_raw_text": True}, "included": [{"reason": "selected"}]}
        assert_export_text_light(clean, _CONTRADICTION_RAW_TEXTS)  # a clean export passes

        leaked = {
            "privacy": {"omitted_raw_text": True},
            "included": [{"reason": "selected", "note": _EVIDENCE_CURRENT}],
        }
        with pytest.raises(AssertionError):
            assert_export_text_light(leaked, _CONTRADICTION_RAW_TEXTS)


class TestCompileReceiptRuntimeLinkage:
    @pytest.mark.asyncio
    async def test_run_links_receipt_on_result_and_trace(self):
        app = ContextApp(name="receipt", provider=MockProvider(default_text="Within 30 days."))
        app.pending_evidence = [
            EvidenceItem(
                id="doc1",
                source_id="policy.md",
                text="Customers may request a refund within 30 days of purchase.",
                authority=0.9,
                relevance=0.95,
                page=2,
            )
        ]
        result = await app.arun("What is the refund window?")

        export = result.metadata.get("compile_receipt")
        assert export is not None
        receipt = CompileReceipt.model_validate(export)
        assert receipt.verify() is True
        # Pointers back to the run and trace.
        assert receipt.run_id == result.run_id
        assert receipt.trace_id == result.trace_id
        assert receipt.packet_id == result.context_packet_id
        # Render identity captured at prompt render.
        assert receipt.render is not None and receipt.render.provider == "mock"
        assert receipt.render.rendered_packet_hash

        # The receipt is linked from the trace's render span.
        trace = app.tracer.exporter.load(result.trace_id)
        render_spans = [s for s in trace.spans if s.type == "prompt_render"]
        assert render_spans
        attrs = render_spans[0].attributes
        assert attrs.get("receipt_hash") == receipt.receipt_hash
        assert attrs["compile_receipt"]["packet_id"] == receipt.packet_id

    @pytest.mark.asyncio
    async def test_pii_redaction_count_surfaces_in_privacy_summary(self):
        app = ContextApp(
            name="receipt-pii",
            provider=MockProvider(default_text="ok"),
            policies=PolicySet(redact_pii_in_context=True),
        )
        result = await app.arun("Please email me at alice@example.com about my order.")
        export = result.metadata.get("compile_receipt")
        assert export is not None
        assert export["privacy"]["redacted_count"] >= 1
        assert export["privacy"]["redact_pii_in_context"] is True


class TestCompileReceiptRobustness:
    def test_non_standard_budget_report_degrades_not_crashes(self):
        from vincio.context import ContextPacket
        from vincio.core.types import Objective

        # A direct caller with an out-of-contract (but type-valid) budget_report
        # must get a valid receipt, not an AttributeError/ValueError.
        packet = ContextPacket(
            objective=Objective("q"),
            budget_report={"total": 42, "evidence": {"used_tokens": 5, "note": "high"}},
        )
        receipt = packet.compile_receipt()
        assert receipt.verify() is True
        # Scalars are dropped; numeric leaves survive; non-numeric leaves dropped.
        assert receipt.budget.blocks["total"] == {}
        assert receipt.budget.blocks["evidence"] == {"used_tokens": 5.0}


class TestCompileReceiptCLI:
    def test_trace_receipt_command(self, tmp_path, capsys):
        from vincio import VincioConfig
        from vincio.cli.main import main

        traces_dir = str(tmp_path / "traces")
        cfg = VincioConfig()
        cfg.observability.exporter = "jsonl"
        cfg.observability.traces_dir = traces_dir
        app = ContextApp(
            name="cli-receipt", provider=MockProvider(default_text="Within 30 days."), config=cfg
        )
        app.pending_evidence = [
            EvidenceItem(
                id="D1",
                source_id="refunds.md",
                text="Refunds are allowed within 30 days of purchase.",
                authority=0.9,
                relevance=0.95,
                page=2,
            ),
            EvidenceItem(
                id="D9",
                source_id="refunds_old.md",
                text="Refunds are allowed within 14 days of purchase.",
                authority=0.4,
                relevance=0.9,
            ),
        ]
        result = app.run("What is the refund window?")
        capsys.readouterr()  # drain

        # Human summary.
        code = main(["trace", "receipt", result.trace_id, "--traces-dir", traces_dir])
        out = capsys.readouterr().out
        assert code == 0
        assert "compile receipt" in out
        assert result.run_id in out
        assert "conflict_lower_authority" in out or "D9" in out
        assert "Refunds are allowed" not in out  # text-light

        # JSON form round-trips to a verifying receipt.
        code = main(["trace", "receipt", result.trace_id, "--json", "--traces-dir", traces_dir])
        payload = capsys.readouterr().out
        assert code == 0
        receipt = CompileReceipt.model_validate(json.loads(payload))
        assert receipt.verify() is True
