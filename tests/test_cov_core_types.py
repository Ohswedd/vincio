"""Real-behavior coverage for vincio.core.types.

Targets the logic-bearing surface of the core data contracts: positional
constructors, validators, computed properties, the budget/usage arithmetic,
the modality-aware evidence helpers, the bi-temporal/ACL memory logic, and the
model-lifecycle date math. No mocks: every model is constructed for real and
every assertion checks a specific computed outcome, value, or raised error.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from vincio.core.types import (
    AudioRef,
    Budget,
    BudgetUsage,
    Chunk,
    Constraint,
    ContentPart,
    EvidenceItem,
    ImageRef,
    Instruction,
    MemoryItem,
    MemoryScope,
    MemoryType,
    Message,
    ModelCapabilities,
    ModelProfile,
    ModelRequest,
    Objective,
    PolicySet,
    PrivacyClass,
    TaskType,
    TokenUsage,
    ToolCallRequest,
    ToolSpec,
    TrustLevel,
    VideoRef,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_trust_level_allowed_to_instruct_model_per_member():
    # SYSTEM/DEVELOPER/USER may instruct; the three untrusted tags may not.
    assert TrustLevel.SYSTEM.allowed_to_instruct_model is True
    assert TrustLevel.DEVELOPER.allowed_to_instruct_model is True
    assert TrustLevel.USER.allowed_to_instruct_model is True
    assert TrustLevel.UNTRUSTED_DOCUMENT.allowed_to_instruct_model is False
    assert TrustLevel.UNTRUSTED_TOOL.allowed_to_instruct_model is False
    assert TrustLevel.UNTRUSTED_EXTERNAL.allowed_to_instruct_model is False


def test_str_enum_values_are_plain_strings():
    # StrEnum members compare equal to their string value (used as dict keys).
    assert TaskType.CODING == "coding"
    assert PrivacyClass.PII == "pii"
    assert MemoryScope.TEAM == "team"
    assert MemoryType.RELATIONSHIP == "relationship"


# ---------------------------------------------------------------------------
# Positional constructors (custom __init__)
# ---------------------------------------------------------------------------


def test_objective_positional_text_and_defaults():
    obj = Objective("Review contracts")
    assert obj.text == "Review contracts"
    assert obj.task_type is TaskType.GENERAL
    assert obj.id.startswith("obj_")
    assert obj.metadata == {}


def test_objective_keyword_only_construction():
    # text supplied purely via kwargs (positional left None -> setdefault skipped).
    obj = Objective(text="kwarg only", task_type=TaskType.CODING)
    assert obj.text == "kwarg only"
    assert obj.task_type is TaskType.CODING


def test_objective_missing_text_raises():
    with pytest.raises(ValidationError):
        Objective()


def test_instruction_positional_and_defaults():
    ins = Instruction("Be concise")
    assert ins.text == "Be concise"
    assert ins.priority == 100
    assert ins.category == "rule"
    assert ins.source is None


def test_instruction_rejects_invalid_category():
    with pytest.raises(ValidationError):
        Instruction("x", category="not-a-category")


def test_constraint_positional_and_default_hard():
    c = Constraint("No PII")
    assert c.text == "No PII"
    assert c.hard is True
    soft = Constraint("Prefer brevity", hard=False)
    assert soft.hard is False


# ---------------------------------------------------------------------------
# Budget.scaled
# ---------------------------------------------------------------------------


def test_budget_scaled_proportional_values():
    b = Budget(max_input_tokens=1000, max_output_tokens=400, max_latency_ms=10_000, max_cost_usd=2.0)
    half = b.scaled(0.5)
    assert half.max_input_tokens == 500
    assert half.max_output_tokens == 200
    assert half.max_latency_ms == 5000
    assert half.max_cost_usd == 1.0
    # steps/tool_calls/retries are NOT scaled.
    assert half.max_steps == b.max_steps
    assert half.max_tool_calls == b.max_tool_calls
    assert half.max_retries == b.max_retries


def test_budget_scaled_floors_token_dims_at_one():
    b = Budget(max_input_tokens=1, max_output_tokens=1, max_latency_ms=1)
    tiny = b.scaled(0.0)
    # max(1, int(... * 0)) clamps each integer dimension to at least 1.
    assert tiny.max_input_tokens == 1
    assert tiny.max_output_tokens == 1
    assert tiny.max_latency_ms == 1
    # cost is a float and is NOT floored.
    assert tiny.max_cost_usd == 0.0


# ---------------------------------------------------------------------------
# BudgetUsage.add / .exceeds
# ---------------------------------------------------------------------------


def test_budget_usage_add_accumulates_every_dimension():
    a = BudgetUsage(input_tokens=10, output_tokens=5, latency_ms=100, cost_usd=0.5,
                    steps=1, tool_calls=2, retries=1)
    b = BudgetUsage(input_tokens=3, output_tokens=7, latency_ms=20, cost_usd=0.25,
                    steps=2, tool_calls=4, retries=0)
    a.add(b)
    assert a.input_tokens == 13
    assert a.output_tokens == 12
    assert a.latency_ms == 120
    assert a.cost_usd == 0.75
    assert a.steps == 3
    assert a.tool_calls == 6
    assert a.retries == 1


def test_budget_usage_exceeds_empty_when_within_budget():
    budget = Budget()
    assert BudgetUsage().exceeds(budget) == []


def test_budget_usage_exceeds_reports_each_breached_dimension():
    budget = Budget(max_input_tokens=100, max_output_tokens=10, max_latency_ms=50,
                    max_cost_usd=1.0, max_steps=2, max_tool_calls=3)
    # steps=1 -> output ceiling stays at 10, so output_tokens=11 breaches.
    usage = BudgetUsage(input_tokens=101, output_tokens=11, latency_ms=51,
                        cost_usd=1.5, steps=3, tool_calls=4)
    breaches = usage.exceeds(budget)
    assert set(breaches) == {"input_tokens", "latency_ms",
                             "cost_usd", "steps", "tool_calls"}


def test_budget_usage_output_tokens_scale_by_steps():
    # The output budget is multiplied by max(1, steps): with 3 steps, the
    # effective per-output ceiling is 10 * 3 = 30, so 25 is within budget.
    budget = Budget(max_output_tokens=10)
    usage = BudgetUsage(output_tokens=25, steps=3)
    assert "output_tokens" not in usage.exceeds(budget)
    # but 31 trips it (3 * 10 = 30).
    usage2 = BudgetUsage(output_tokens=31, steps=3)
    assert "output_tokens" in usage2.exceeds(budget)


def test_budget_usage_output_tokens_zero_steps_uses_factor_one():
    # steps == 0 -> max(1, 0 or 1) == 1, so the raw output ceiling applies.
    budget = Budget(max_output_tokens=10)
    assert "output_tokens" in BudgetUsage(output_tokens=11, steps=0).exceeds(budget)


# ---------------------------------------------------------------------------
# EvidenceItem.citation_ref
# ---------------------------------------------------------------------------


def test_citation_ref_time_range_wins_over_page():
    # time_range takes priority over page and renders compact seconds.
    ev = EvidenceItem(source_id="D1", page=4, time_range=(12.0, 18.5))
    assert ev.citation_ref == "D1:t12-18.5"


def test_citation_ref_page_when_no_time_range():
    ev = EvidenceItem(source_id="D7", page=4)
    assert ev.citation_ref == "D7:p4"


def test_citation_ref_falls_back_to_id():
    ev = EvidenceItem(source_id="D1")
    assert ev.citation_ref == ev.id
    assert ev.id.startswith("ev_")


def test_citation_ref_seconds_formatting_strips_trailing_zeros_and_handles_zero():
    # _fmt_seconds: 30.00 -> "30", 0.00 -> "0" (the `or "0"` branch).
    ev = EvidenceItem(source_id="V", time_range=(0.0, 30.0))
    assert ev.citation_ref == "V:t0-30"


# ---------------------------------------------------------------------------
# EvidenceItem.scorable_text  (one branch per modality)
# ---------------------------------------------------------------------------


def test_scorable_text_prefers_explicit_text():
    ev = EvidenceItem(source_id="D1", modality="image", text="explicit",
                      image=ImageRef(metadata={"caption": "ignored"}))
    assert ev.scorable_text == "explicit"


def test_scorable_text_table_uses_markdown_then_caption():
    md = EvidenceItem(source_id="D1", modality="table", table={"markdown": "| a |"})
    assert md.scorable_text == "| a |"
    cap = EvidenceItem(source_id="D1", modality="table", table={"caption": "Sales by region"})
    assert cap.scorable_text == "Sales by region"


def test_scorable_text_image_uses_caption_then_alt():
    cap = EvidenceItem(source_id="D1", modality="image",
                       image=ImageRef(metadata={"caption": "a cat"}))
    assert cap.scorable_text == "a cat"
    alt = EvidenceItem(source_id="D1", modality="image",
                       image=ImageRef(metadata={"alt": "alt text"}))
    assert alt.scorable_text == "alt text"


def test_scorable_text_video_uses_transcript_then_caption():
    tr = EvidenceItem(source_id="D1", modality="video",
                      video=VideoRef(metadata={"transcript": "hello world"}))
    assert tr.scorable_text == "hello world"
    cap = EvidenceItem(source_id="D1", modality="video",
                       video=VideoRef(metadata={"caption": "intro clip"}))
    assert cap.scorable_text == "intro clip"


def test_scorable_text_empty_when_no_carrier():
    # image modality but no image payload -> "" (final fallthrough).
    assert EvidenceItem(source_id="D1", modality="image").scorable_text == ""
    assert EvidenceItem(source_id="D1", modality="table").scorable_text == ""
    assert EvidenceItem(source_id="D1", modality="video").scorable_text == ""


# ---------------------------------------------------------------------------
# EvidenceItem.estimated_token_cost  (calibrated tables)
# ---------------------------------------------------------------------------


def test_estimated_token_cost_image_by_detail():
    assert EvidenceItem(source_id="D", modality="image",
                        image=ImageRef(detail="low")).estimated_token_cost() == 85
    assert EvidenceItem(source_id="D", modality="image",
                        image=ImageRef(detail="high")).estimated_token_cost() == 765
    assert EvidenceItem(source_id="D", modality="image",
                        image=ImageRef(detail="auto")).estimated_token_cost() == 512


def test_estimated_token_cost_video_by_detail():
    assert EvidenceItem(source_id="D", modality="video",
                        video=VideoRef(detail="low")).estimated_token_cost() == 256
    assert EvidenceItem(source_id="D", modality="video",
                        video=VideoRef(detail="high")).estimated_token_cost() == 2048
    assert EvidenceItem(source_id="D", modality="video",
                        video=VideoRef(detail="auto")).estimated_token_cost() == 1024


def test_estimated_token_cost_table_per_cell():
    # 2 rows x 2 cells = 4 cells + 2 columns = 6 units * 3 tokens/cell = 18.
    ev = EvidenceItem(source_id="D", modality="table",
                      table={"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]})
    assert ev.estimated_token_cost() == 18


def test_estimated_token_cost_table_floored_by_explicit_token_cost():
    # token_cost=100 dominates the tiny computed estimate (max(...)).
    ev = EvidenceItem(source_id="D", modality="table", token_cost=100,
                      table={"columns": ["a"], "rows": [[1]]})
    assert ev.estimated_token_cost() == 100


def test_estimated_token_cost_text_returns_token_cost():
    ev = EvidenceItem(source_id="D", modality="text", token_cost=42)
    assert ev.estimated_token_cost() == 42


def test_estimated_token_cost_image_modality_without_payload_uses_token_cost():
    # modality says image but no image -> falls through to token_cost.
    ev = EvidenceItem(source_id="D", modality="image", token_cost=7)
    assert ev.estimated_token_cost() == 7


# ---------------------------------------------------------------------------
# MemoryItem: confidence clamp, valid_at, readable_by
# ---------------------------------------------------------------------------


def test_memory_confidence_clamped_into_unit_interval():
    assert MemoryItem(content="x", confidence=2.5).confidence == 1.0
    assert MemoryItem(content="x", confidence=-0.3).confidence == 0.0
    assert MemoryItem(content="x", confidence=0.42).confidence == 0.42


def test_memory_valid_at_open_ended_from_creation():
    created = datetime(2020, 1, 1, tzinfo=UTC)
    m = MemoryItem(content="x", created_at=created)
    assert m.valid_at(datetime(2020, 1, 1, tzinfo=UTC)) is True   # boundary inclusive at start
    assert m.valid_at(datetime(2025, 1, 1, tzinfo=UTC)) is True
    assert m.valid_at(datetime(2019, 12, 31, tzinfo=UTC)) is False  # before creation


def test_memory_valid_at_closed_interval_end_is_exclusive():
    m = MemoryItem(
        content="x",
        valid_from=datetime(2021, 1, 1, tzinfo=UTC),
        valid_to=datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert m.valid_at(datetime(2021, 6, 1, tzinfo=UTC)) is True
    # valid_to is exclusive: exactly at valid_to is NOT valid.
    assert m.valid_at(datetime(2022, 1, 1, tzinfo=UTC)) is False
    assert m.valid_at(datetime(2020, 12, 31, tzinfo=UTC)) is False


def test_memory_valid_at_treats_naive_datetimes_as_utc():
    # A naive moment must not raise against an aware interval.
    m = MemoryItem(content="x", valid_from=datetime(2021, 1, 1, tzinfo=UTC))
    assert m.valid_at(datetime(2021, 6, 1)) is True  # naive -> UTC


def test_memory_readable_by_open_acl_admits_everyone_including_none():
    m = MemoryItem(content="x")  # empty acl
    assert m.readable_by("anyone") is True
    assert m.readable_by(None) is True


def test_memory_readable_by_populated_acl_admits_only_listed():
    m = MemoryItem(content="x", acl=["alice", "bob"])
    assert m.readable_by("alice") is True
    assert m.readable_by("carol") is False
    assert m.readable_by(None) is False  # None never admitted to a closed ACL


# ---------------------------------------------------------------------------
# ToolSpec.is_cacheable
# ---------------------------------------------------------------------------


def test_tool_spec_is_cacheable_default_by_side_effect():
    assert ToolSpec(name="r", description="", side_effects="read").is_cacheable is True
    assert ToolSpec(name="n", description="", side_effects="none").is_cacheable is True
    assert ToolSpec(name="w", description="", side_effects="write").is_cacheable is False
    assert ToolSpec(name="e", description="", side_effects="external").is_cacheable is False


def test_tool_spec_explicit_cacheable_overrides_side_effect_default():
    # explicit True on a write tool, explicit False on a read tool.
    assert ToolSpec(name="w", description="", side_effects="write", cacheable=True).is_cacheable is True
    assert ToolSpec(name="r", description="", side_effects="read", cacheable=False).is_cacheable is False


# ---------------------------------------------------------------------------
# PolicySet.set / .get  (known field vs custom)
# ---------------------------------------------------------------------------


def test_policy_set_known_field_roundtrip():
    p = PolicySet()
    p.set("require_citations", True)
    assert p.require_citations is True
    assert p.get("require_citations") is True
    # known fields never land in custom.
    assert p.custom == {}


def test_policy_set_unknown_field_routes_to_custom():
    p = PolicySet()
    p.set("max_widgets", 9)
    assert p.custom == {"max_widgets": 9}
    assert p.get("max_widgets") == 9


def test_policy_set_get_unknown_returns_default():
    p = PolicySet()
    assert p.get("nope", default="fallback") == "fallback"
    assert p.get("nope") is None


# ---------------------------------------------------------------------------
# TokenUsage.total_tokens / .add
# ---------------------------------------------------------------------------


def test_token_usage_total_excludes_cached_and_reasoning():
    u = TokenUsage(input_tokens=100, output_tokens=20, cached_input_tokens=50, reasoning_tokens=30)
    assert u.total_tokens == 120


def test_token_usage_add_accumulates_all_four_counters():
    a = TokenUsage(input_tokens=1, output_tokens=2, cached_input_tokens=3, reasoning_tokens=4)
    a.add(TokenUsage(input_tokens=10, output_tokens=20, cached_input_tokens=30, reasoning_tokens=40))
    assert (a.input_tokens, a.output_tokens, a.cached_input_tokens, a.reasoning_tokens) == (11, 22, 33, 44)


# ---------------------------------------------------------------------------
# Message.text  (str vs structured content parts)
# ---------------------------------------------------------------------------


def test_message_text_with_string_content():
    assert Message(role="user", content="hello").text == "hello"


def test_message_text_joins_only_text_parts():
    msg = Message(
        role="user",
        content=[
            ContentPart(type="text", text="first"),
            ContentPart(type="image", image=ImageRef(path="/x.png")),
            ContentPart(type="text", text="second"),
        ],
    )
    # non-text parts are skipped; text parts joined by newline.
    assert msg.text == "first\nsecond"


def test_message_text_part_with_none_text_becomes_empty_string():
    msg = Message(role="assistant", content=[ContentPart(type="text", text=None)])
    assert msg.text == ""


# ---------------------------------------------------------------------------
# Chunk.citation_ref
# ---------------------------------------------------------------------------


def test_chunk_citation_ref_uses_document_and_index():
    chunk = Chunk(document_id="doc_42", text="body", index=3)
    assert chunk.citation_ref == "doc_42:C3"


# ---------------------------------------------------------------------------
# ModelRequest.hash  (content-addressed, deterministic, sensitive to content)
# ---------------------------------------------------------------------------


def test_model_request_hash_is_deterministic_and_content_sensitive():
    req = ModelRequest(model="gpt", messages=[Message(role="user", content="hi")])
    same = ModelRequest(model="gpt", messages=[Message(role="user", content="hi")])
    other = ModelRequest(model="gpt", messages=[Message(role="user", content="bye")])
    # Same content -> identical stable hash; different content -> different hash.
    assert req.hash == same.hash
    assert req.hash != other.hash
    assert isinstance(req.hash, str) and len(req.hash) == 16


# ---------------------------------------------------------------------------
# ModelProfile.lifecycle  (date math + parse failures)
# ---------------------------------------------------------------------------


def _profile(**kw) -> ModelProfile:
    base = dict(name="m", provider="p", model="m-1")
    base.update(kw)
    return ModelProfile(**base)


def test_lifecycle_ga_when_no_dates():
    assert _profile().lifecycle(as_of=date(2026, 1, 1)) == "ga"


def test_lifecycle_retired_takes_priority_over_deprecated():
    prof = _profile(deprecation_date="2024-01-01", retirement_date="2025-01-01")
    # as_of past both -> retired wins.
    assert prof.lifecycle(as_of=date(2026, 1, 1)) == "retired"


def test_lifecycle_deprecated_before_retirement_date():
    prof = _profile(deprecation_date="2024-01-01", retirement_date="2025-01-01")
    # past deprecation but before retirement.
    assert prof.lifecycle(as_of=date(2024, 6, 1)) == "deprecated"


def test_lifecycle_ga_before_deprecation():
    prof = _profile(deprecation_date="2024-01-01")
    assert prof.lifecycle(as_of=date(2023, 1, 1)) == "ga"


def test_lifecycle_boundary_dates_are_inclusive():
    # today >= retired / >= deprecated triggers the state exactly on the date.
    assert _profile(retirement_date="2025-01-01").lifecycle(as_of=date(2025, 1, 1)) == "retired"
    assert _profile(deprecation_date="2025-01-01").lifecycle(as_of=date(2025, 1, 1)) == "deprecated"


def test_lifecycle_unparseable_date_is_ignored():
    # A garbage retirement date parses to None and is skipped (ga).
    prof = _profile(retirement_date="not-a-date")
    assert prof.lifecycle(as_of=date(2026, 1, 1)) == "ga"


def test_lifecycle_parses_full_iso_timestamp_prefix():
    # value[:10] slice lets an ISO datetime string parse to its date.
    prof = _profile(retirement_date="2025-01-01T00:00:00Z")
    assert prof.lifecycle(as_of=date(2025, 6, 1)) == "retired"


def test_lifecycle_default_as_of_is_today_and_ga_for_future_retirement():
    # No as_of -> uses utcnow().date(); a far-future retirement stays ga.
    prof = _profile(retirement_date="2999-01-01")
    assert prof.lifecycle() == "ga"


# ---------------------------------------------------------------------------
# Defaults / capability factory lists are independent (no shared mutable default)
# ---------------------------------------------------------------------------


def test_model_capabilities_modality_lists_default_and_isolated():
    a = ModelCapabilities()
    b = ModelCapabilities()
    assert a.input_modalities == ["text"]
    assert a.output_modalities == ["text"]
    a.input_modalities.append("image")
    # the default_factory must give each instance its own list.
    assert b.input_modalities == ["text"]


def test_imageref_defaults():
    img = ImageRef()
    assert img.media_type == "image/png"
    assert img.detail == "auto"
    assert img.path is None and img.url is None


def test_audioref_and_videoref_defaults():
    assert AudioRef().media_type == "audio/wav"
    v = VideoRef()
    assert v.media_type == "video/mp4"
    assert v.detail == "auto"
    assert v.duration_seconds is None and v.fps is None


def test_tool_call_request_generates_prefixed_id():
    req = ToolCallRequest(name="search")
    assert req.name == "search"
    assert req.id.startswith("tcr_")
    assert req.arguments == {}
