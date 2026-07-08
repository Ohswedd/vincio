"""Universal reasoning: adaptive depth, non-native passes, web and correction."""

from __future__ import annotations

import json
import warnings

import httpx
import pytest
from pydantic import BaseModel

from vincio import ContextApp, UniversalReasoningEngine, UniversalReasoningPolicy
from vincio.core.config import VincioConfig
from vincio.core.errors import WebPolicyError
from vincio.core.types import Budget, FileRef, ModelCapabilities, RunConfig, UserInput
from vincio.providers import MockProvider
from vincio.stability import VincioExperimentalWarning
from vincio.web import SearchResult, StaticSearchBackend

warnings.simplefilter("ignore", VincioExperimentalWarning)


def _app(tmp_path, provider: MockProvider) -> ContextApp:
    config = VincioConfig()
    config.observability.exporter = "memory"
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(name="reason", provider=provider, model="mock-1", config=config)


def _route_payload(
    *,
    language: str,
    depth: str,
    task_kinds: list[str],
    live: bool = False,
    web_preference: str = "auto",
    tools: list[str] | None = None,
    confidence: float = 0.95,
) -> str:
    signal_by_kind = {
        "mathematical": "calculation",
        "logical": "logic",
        "multi_step": "multi_step",
        "tool_dependent": "tool_request",
    }
    signals = [signal_by_kind[kind] for kind in task_kinds if kind in signal_by_kind]
    if live:
        signals.append("current_external_fact")
    if web_preference == "forbidden":
        signals.append("web_prohibited")
    return json.dumps(
        {
            "language": language,
            "primary_task": "general",
            "depth": depth,
            "difficulty": {"direct": 0.1, "standard": 0.5, "deep": 0.8}[depth],
            "task_kinds": task_kinds,
            "needs_live_external_information": live,
            "web_preference": web_preference,
            "tool_names": tools or [],
            "confidence": confidence,
            "signals": signals,
        }
    )


def test_assessment_keeps_simple_work_direct_and_routes_hard_work(tmp_path):
    engine = UniversalReasoningEngine(_app(tmp_path, MockProvider(default_text="ok")))

    simple = engine.assess("Rewrite this title in uppercase")
    hard = engine.assess(
        "Prove whether the assumptions are logically consistent; derive the equation and detect contradictions."
    )

    assert simple.depth == "direct" and not simple.multiple_passes
    assert hard.depth == "deep" and hard.multiple_passes
    assert {"logical", "mathematical"}.issubset(hard.task_kinds)

    math = engine.assess(
        "A service handles 240 requests/minute. Traffic rises 25%, then two workers split it."
    )
    plan = engine.plan(
        "A service handles 240 requests/minute. Traffic rises 25%, then two workers split it.",
        math,
    )
    assert plan.verified_facts["expected_numeric_answer"] == "150"


def test_non_reasoning_model_gets_bounded_multi_pass_reasoning(tmp_path):
    provider = MockProvider(default_text="A concise answer.", reasoning=False)
    app = _app(tmp_path, provider)

    outcome = app.reason(
        "Compare the trade-offs, identify the root cause, and derive a logically consistent recommendation."
    )

    assert len(outcome.passes) == 3
    # Deep multi-step work on a non-native model spends one extra bounded
    # internal planning call before the candidate passes.
    assert provider.call_count == 4
    assert outcome.result.metadata["universal_reasoning"]["model_calls"] == 4
    assert outcome.assessment.native_reasoning is False
    assert outcome.result.metadata["universal_reasoning"]["strategy"] == "logic_check"
    assert all(not hasattr(item, "raw_text") for item in outcome.passes)


def test_native_reasoning_is_used_inside_same_engine(tmp_path):
    provider = MockProvider(default_text="Checked answer.", reasoning=True)
    app = _app(tmp_path, provider)

    outcome = app.reason("Prove this logical implication and check for a counterexample.")

    assert outcome.assessment.native_reasoning
    assert provider.requests
    assert all(request.reasoning_effort == "high" for request in provider.requests)
    assert outcome.result.usage.reasoning_tokens > 0


def test_refuted_arithmetic_triggers_bounded_correction(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "bounded answer correction" in prompt:
            return "2 + 2 = 4."
        return "2 + 2 = 5."

    app = _app(tmp_path, MockProvider(responder=responder))
    outcome = app.reason(
        "Prove logically whether 2 + 2 = 5, calculate the equality, and detect the contradiction."
    )

    assert outcome.corrected
    assert outcome.result.raw_text == "2 + 2 = 4."
    assert [item.verification for item in outcome.passes] == [
        "refuted",
        "refuted",
        "refuted",
        "verified",
    ]
    assert len(outcome.passes) == 4


def test_installed_engine_adapts_normal_run(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="answer"))
    app.use_reasoning_engine()

    result = app.run(
        "Prove the logical consistency of these constraints and explain the trade-offs."
    )

    assert "universal_reasoning" in result.metadata
    assert app.reasoning_engine.last_result.result.run_id == result.run_id
    assert any(entry.action == "universal_reasoning" for entry in app.audit.entries)


def test_live_fact_check_uses_governed_web_evidence(tmp_path):
    url = "https://example.org/release"
    backend = StaticSearchBackend(
        default=[
            SearchResult(
                rank=1,
                title="Release notes",
                url=url,
                snippet="Version 4.2 shipped today.",
                source="static",
            )
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=(
                "<html><title>Release notes</title><body><main>"
                "Version 4.2 shipped today with deterministic verification and security fixes."
                "</main></body></html>"
            ),
        )

    provider = MockProvider(default_text=f"Version 4.2 shipped. Source: {url}")
    app = _app(tmp_path, provider)
    app.use_web_search(
        backend=backend,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    outcome = app.reason("Search for the latest release version and fact-check it with sources.")

    assert outcome.assessment.needs_search
    assert outcome.web_evidence and outcome.web_verified
    assert outcome.web_evidence[0].metadata["reasoning_live_verification"]
    assert app.web_browser.report().verify(app.web_browser.snapshots)
    assert all(
        not {tool.name for tool in request.tools}.intersection({"web_search", "web_read"})
        for request in provider.requests
    )
    assert all(
        "You can call `web_search" not in "\n".join(message.text for message in request.messages)
        for request in provider.requests
    )


def test_policy_caps_candidates_and_disables_web(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="answer"))
    policy = UniversalReasoningPolicy(max_passes=1, max_parallel_candidates=4, web="off")
    engine = UniversalReasoningEngine(app, policy)
    assessment = engine.assess("Find the latest result, compare it, and prove it logically.")
    plan = engine.plan("Find the latest result, compare it, and prove it logically.", assessment)

    assert not assessment.needs_search
    assert plan.candidate_passes == 1


def test_tool_dependent_reasoning_never_duplicates_tool_passes(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="done"))

    def create_ticket(title: str) -> str:
        return title

    app.add_tool(create_ticket)
    engine = UniversalReasoningEngine(app)
    assessment = engine.assess("Use create_ticket to create the incident, then explain the result.")
    plan = engine.plan(
        "Use create_ticket to create the incident, then explain the result.", assessment
    )

    assert assessment.needs_tools
    assert plan.strategy == "tool_plan"
    assert plan.candidate_passes == 1
    assert not plan.allow_correction


def test_outer_step_budget_caps_total_reasoning_passes(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="answer"))
    outcome = app.reason(
        "Prove this logical implication and detect every contradiction.",
        config=RunConfig(budget=Budget(max_steps=1)),
    )

    assert len(outcome.passes) == 1


def test_required_live_verification_fails_closed_without_browser(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="unsupported current claim"))
    engine = UniversalReasoningEngine(app, {"web": "required"})

    with pytest.raises(WebPolicyError):
        engine.run("Who is the current CEO?")


def test_unrequested_printed_scratch_work_is_corrected_to_answer_only(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "bounded answer correction" in prompt:
            return "17 * 23 = 391."
        return "## Step 1: multiply privately.\n17 * 23 = 391."

    app = _app(tmp_path, MockProvider(responder=responder))
    outcome = app.reason("Calculate 17 * 23 and verify the equality.")

    assert outcome.corrected
    assert outcome.result.raw_text == "17 * 23 = 391."


def test_task_refuted_answer_uses_only_deterministically_verified_fallback(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="17 * 23 = 369."))
    outcome = app.reason("Calculate 17 * 23 and verify the equality.")

    assert outcome.deterministic_fallback and not outcome.refused
    assert "391" in outcome.result.raw_text and "369" not in outcome.result.raw_text


def test_routing_avoids_ambiguous_freshness_words(tmp_path):
    engine = UniversalReasoningEngine(_app(tmp_path, MockProvider(default_text="ok")))

    for prompt in (
        "Rewrite the current paragraph in uppercase.",
        "Rewrite the latest paragraph in uppercase.",
        "Explain semantic version control.",
        "Score this essay from 1 to 10.",
    ):
        assessment = engine.assess(prompt)
        assert assessment.depth == "direct"
        assert assessment.search_decision == "not_needed"

    current = engine.assess("Who is the CEO of OpenAI?")
    assert current.needs_live_verification
    assert current.search_decision == "search"
    assert current.search_reasons == ["unstable_fact"]


def test_assessment_includes_supplied_modalities_and_reasoning_families(tmp_path):
    engine = UniversalReasoningEngine(_app(tmp_path, MockProvider(default_text="ok")))
    supplied = UserInput(
        text="Summarize the differences.",
        files=[FileRef(path="a.md"), FileRef(path="b.md")],
    )

    assessment = engine.assess(supplied)
    decision = engine.assess(
        "Choose the best option while keeping cost below $10 and latency below 100 ms."
    )
    spatial = engine.assess("Find the shortest path between these map coordinates.")

    assert assessment.input_modalities == ["text", "file"]
    assert "multi_source" in assessment.task_kinds
    assert assessment.needs_reasoning
    assert {"decision_analysis", "constraint_satisfaction"}.issubset(decision.task_kinds)
    assert "spatial_reasoning" in spatial.task_kinds


def test_high_stakes_requests_require_fresh_evidence(tmp_path):
    engine = UniversalReasoningEngine(_app(tmp_path, MockProvider(default_text="ok")))

    assessment = engine.assess("Could these symptoms indicate a drug interaction?")

    assert assessment.needs_search
    assert "high_stakes" in assessment.search_reasons


def test_user_web_opt_out_is_obeyed_and_required_policy_reports_conflict(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="The current CEO is Example Person."))
    engine = UniversalReasoningEngine(app)

    assessment = engine.assess("Who is the current CEO? Do not browse the web.")
    assert assessment.needs_live_verification and not assessment.needs_search
    assert assessment.search_decision == "user_declined"

    outcome = engine.run("Who is the current CEO? Do not browse the web.")
    assert outcome.refused
    assert outcome.result.status.value == "failed"

    required = UniversalReasoningEngine(app, {"web": "required"})
    with pytest.raises(WebPolicyError, match="user declined"):
        required.run("Who is the current CEO? Do not browse the web.")


def test_unavailable_auto_web_refuses_an_unsupported_live_claim(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="The latest release is 99.0."))

    outcome = app.reason("What is the latest stable release?")

    assert outcome.refused
    assert outcome.answer_verification == "refuted"
    assert outcome.result.raw_text == ""


def test_requested_url_is_read_directly_once_without_redundant_search(tmp_path):
    url = "https://example.org/report"
    backend = StaticSearchBackend()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><main>The report contains the audited result.</main></html>",
        )

    app = _app(tmp_path, MockProvider(default_text=f"Audited result. Source: {url}"))
    app.use_web_search(
        backend=backend,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    outcome = app.reason(f"Summarize {url}")

    assert outcome.plan.source_urls == [url]
    assert outcome.plan.search_queries == []
    assert backend.queries == []
    assert app.web_browser.fetches_used == 1
    assert len(app.web_browser.reads) == 1


def test_tool_routing_requires_bounded_name_or_description_match(tmp_path):
    app = _app(tmp_path, MockProvider(default_text="done"))

    def create_ticket(title: str) -> str:
        """Create an incident ticket from its title."""
        return title

    app.add_tool(create_ticket)
    engine = UniversalReasoningEngine(app)

    unrelated = engine.assess("Run through the history of incident management.")
    explicit = engine.assess("Use create_ticket to create an incident ticket.")

    assert not unrelated.needs_tools and unrelated.matched_tools == []
    assert explicit.needs_tools and explicit.matched_tools == ["create_ticket"]


def test_task_facts_are_conservative_for_ambiguous_math_and_logic(tmp_path):
    engine = UniversalReasoningEngine(_app(tmp_path, MockProvider(default_text="ok")))

    math_text = "Calculate 2 + 2 and 3 + 3."
    math_plan = engine.plan(math_text, engine.assess(math_text))
    assert "expected_numeric_answer" not in math_plan.verified_facts

    consistent = "All cats are mammals, and no dogs are cats. Are these premises consistent?"
    consistent_plan = engine.plan(consistent, engine.assess(consistent))
    assert "expected_consistency" not in consistent_plan.verified_facts

    contradiction = (
        "All red keys open door A. No brass key opens door A. "
        "Key K is red and brass. Are the premises mutually consistent?"
    )
    contradiction_plan = engine.plan(contradiction, engine.assess(contradiction))
    assert contradiction_plan.verified_facts["expected_consistency"] == "inconsistent"
    assert "red key/brass key" in contradiction_plan.verified_facts["contradiction_summary"]


def test_deterministic_fallback_never_breaks_a_structured_contract(tmp_path):
    class Answer(BaseModel):
        value: int

    config = VincioConfig()
    config.observability.exporter = "memory"
    config.storage.metadata = f"sqlite:///{tmp_path}/structured.db"
    config.security.audit_dir = str(tmp_path / "structured-audit")
    app = ContextApp(
        name="reason-structured",
        provider=MockProvider(default_text='{"value": 369}'),
        model="mock-1",
        output_schema=Answer,
        config=config,
    )

    outcome = app.reason("Calculate 17 * 23 and verify the equality.")

    assert outcome.refused and not outcome.deterministic_fallback
    assert outcome.result.output is None


def test_explicit_yes_cannot_pass_an_inconsistent_logic_verdict(tmp_path):
    app = _app(
        tmp_path,
        MockProvider(
            default_text=(
                "Yes. The premises are inconsistent because the red and brass key "
                "creates a contradiction."
            )
        ),
    )
    prompt = (
        "All red keys open door A. No brass key opens door A. Key K is red and brass. "
        "Are the premises mutually consistent? Answer yes or no and identify the contradiction."
    )

    outcome = app.reason(prompt)

    assert outcome.deterministic_fallback
    assert outcome.result.raw_text.startswith("No.")
    assert "contradiction" in outcome.result.raw_text.lower()
    assert all(item.verification == "refuted" for item in outcome.passes)


def test_standard_reasoning_reserves_two_not_four_latency_slots(tmp_path, monkeypatch):
    app = _app(tmp_path, MockProvider(default_text="17 * 23 = 391."))
    observed: list[int] = []
    original = app._runtime.execute

    async def capture(user_input, config):
        observed.append((config.budget or app.budget).max_latency_ms)
        return await original(user_input, config)

    monkeypatch.setattr(app._runtime, "execute", capture)
    outcome = app.reason("Calculate 17 * 23 and verify the equality.")

    assert outcome.assessment.depth == "standard"
    assert observed == [app.budget.max_latency_ms // 2]


def test_final_reasoning_leak_uses_only_a_task_proven_fallback(tmp_path):
    app = _app(
        tmp_path,
        MockProvider(default_text="## Step 1: multiply privately.\n17 * 23 = 391."),
    )

    outcome = app.reason("Calculate 17 * 23 and verify the equality.")

    assert outcome.deterministic_fallback
    assert "Step" not in outcome.result.raw_text
    assert outcome.result.raw_text.endswith("391.")


def test_model_native_semantic_router_handles_non_english_direct_task(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            assert request.output_schema is not None
            assert request.output_schema["additionalProperties"] is False
            assert set(request.output_schema["required"]) == set(
                request.output_schema["properties"]
            )
            return _route_payload(language="es", depth="direct", task_kinds=["summarization"])
        return "Resumen breve en español."

    provider = MockProvider(responder=responder)
    app = _app(tmp_path, provider)

    outcome = app.reason("Resume brevemente este párrafo sin buscar en internet.")

    assert outcome.assessment.semantic_routing_used
    assert outcome.assessment.semantic_routing_succeeded
    assert outcome.assessment.detected_language == "es"
    assert outcome.assessment.depth == "direct"
    assert provider.call_count == 2  # compact route + the normal governed answer
    assert outcome.result.raw_text == "Resumen breve en español."
    receipt = outcome.result.metadata["universal_reasoning"]
    assert receipt["semantic_routing_tokens"] > 0
    assert receipt["semantic_routing_trace_id"]
    route_audit = next(
        entry for entry in app.audit.entries if entry.action == "reasoning_semantic_route"
    )
    assert "Resume brevemente" not in json.dumps(route_audit.details)


def test_semantic_router_supports_unlisted_latin_language_via_model(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            return _route_payload(
                language="sw",
                depth="standard",
                task_kinds=["decision_analysis"],
            )
        return "Chaguo la pili lina uwiano bora."

    app = _app(tmp_path, MockProvider(responder=responder))
    text = "Linganisha chaguo hizi na uchague lenye uwiano bora."

    preview = UniversalReasoningEngine(app).assess(text)
    outcome = app.reason(text)

    assert preview.detected_language != "sw"  # finite offline profiles misidentify it
    assert outcome.assessment.detected_language == "sw"
    assert outcome.assessment.semantic_routing_succeeded
    assert "decision_analysis" in outcome.assessment.task_kinds


def test_multilingual_web_prohibition_is_enforced_without_locale_regex(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            return _route_payload(
                language="ja",
                depth="standard",
                task_kinds=["factual_verification"],
                live=True,
                web_preference="forbidden",
            )
        return "[UNVERIFIED] ライブ情報を確認できません。"

    provider = MockProvider(responder=responder)
    app = _app(tmp_path, provider)
    outcome = app.reason("ウェブを使わずに、現在のCEOを教えてください。")

    assert outcome.assessment.search_decision == "user_declined"
    assert not outcome.assessment.needs_search
    assert outcome.answer_verification == "verified"
    assert outcome.result.status.value == "succeeded"
    assert provider.call_count == 2


def test_multilingual_semantic_tool_choice_is_allow_list_bounded(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            return _route_payload(
                language="ar",
                depth="standard",
                task_kinds=["tool_dependent"],
                tools=["create_ticket", "unregistered_admin_tool"],
            )
        return "تم إنشاء الطلب."

    app = _app(tmp_path, MockProvider(responder=responder))

    def create_ticket(title: str) -> str:
        return title

    app.add_tool(create_ticket)
    outcome = app.reason("استخدم أداة إنشاء التذكرة للحادث الجديد.")

    assert outcome.assessment.matched_tools == ["create_ticket"]
    assert outcome.plan.strategy == "tool_plan"
    assert outcome.plan.candidate_passes == 1
    assert not outcome.plan.allow_correction


def test_low_confidence_semantic_route_fails_conservatively(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            return _route_payload(
                language="und",
                depth="direct",
                task_kinds=[],
                confidence=0.2,
            )
        return "Jibu la tahadhari."

    app = _app(tmp_path, MockProvider(responder=responder))
    outcome = app.reason("Fafanua uhusiano huu kwa makini.")

    assert outcome.assessment.semantic_routing_used
    assert not outcome.assessment.semantic_routing_succeeded
    assert outcome.assessment.depth == "standard"
    assert "semantic_unclassified" in outcome.assessment.task_kinds


def test_one_step_budget_skips_semantic_probe_and_preserves_answer_slot(tmp_path):
    provider = MockProvider(default_text="Risposta.")
    app = _app(tmp_path, provider)

    outcome = app.reason(
        "Spiega questo.",
        config=RunConfig(budget=Budget(max_steps=1)),
    )

    assert not outcome.assessment.semantic_routing_used
    assert provider.call_count == 1
    assert len(outcome.passes) == 1


def test_multilingual_semantic_route_preserves_app_output_contract(tmp_path):
    class LocalizedAnswer(BaseModel):
        text: str

    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            return _route_payload(language="fr", depth="direct", task_kinds=["extraction"])
        return '{"text":"Paris"}'

    config = VincioConfig()
    config.observability.exporter = "memory"
    config.storage.metadata = f"sqlite:///{tmp_path}/multilingual-structured.db"
    config.security.audit_dir = str(tmp_path / "multilingual-structured-audit")
    app = ContextApp(
        name="multilingual-structured",
        provider=MockProvider(responder=responder),
        model="mock-1",
        output_schema=LocalizedAnswer,
        config=config,
    )

    outcome = app.reason("Extrais uniquement le nom de la ville : Paris.")

    assert isinstance(outcome.result.output, LocalizedAnswer)
    assert outcome.result.output.text == "Paris"
    assert outcome.assessment.semantic_routing_succeeded


def test_semantic_router_degrades_to_json_text_without_structured_output(tmp_path):
    class TextOnlyMock(MockProvider):
        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(structured_output=False, reasoning=False)

    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            assert request.output_schema is None
            return (
                "```json\n"
                + _route_payload(language="de", depth="direct", task_kinds=["summarization"])
                + "\n```"
            )
        return "Kurze Zusammenfassung."

    app = _app(tmp_path, TextOnlyMock(responder=responder))
    outcome = app.reason("Fasse diesen Absatz kurz zusammen.")

    assert outcome.assessment.semantic_routing_succeeded
    assert outcome.assessment.detected_language == "de"
    assert outcome.result.raw_text == "Kurze Zusammenfassung."


def test_multilingual_semantic_search_and_unicode_evidence_verification(tmp_path):
    url = "https://example.org/ja-release"
    backend = StaticSearchBackend(
        default=[
            SearchResult(
                rank=1,
                title="公式リリース",
                url=url,
                snippet="Pythonの最新安定版は3.14です",
                source="static",
            )
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><main>公式情報ではPythonの最新安定版は3.14です。</main></html>",
        )

    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "semantic request router" in prompt:
            return _route_payload(
                language="ja",
                depth="standard",
                task_kinds=["factual_verification"],
                live=True,
            )
        return f"Pythonの最新安定版は3.14です。 Source: {url}"

    app = _app(tmp_path, MockProvider(responder=responder))
    app.use_web_search(
        backend=backend,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    outcome = app.reason("Pythonの最新安定版をウェブで確認してください。")

    assert outcome.assessment.semantic_routing_succeeded
    assert outcome.assessment.needs_search
    assert backend.queries and "Python" in backend.queries[0]
    assert outcome.web_verified
    assert outcome.answer_verification == "verified"
    assert outcome.result.status.value == "succeeded"


def _plan_payload(
    steps: list[dict] | None = None,
    *,
    assumptions: list[str] | None = None,
    evidence_queries: list[str] | None = None,
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "steps": steps
            or [
                {"goal": "List the material trade-offs", "kind": "analyze", "depends_on": [], "check": "none"},
                {"goal": "Compare the options against the constraints", "kind": "compare", "depends_on": [0], "check": "constraint"},
                {"goal": "Recommend one option with its rationale", "kind": "decide", "depends_on": [1], "check": "none"},
            ],
            "assumptions": assumptions or [],
            "evidence_queries": evidence_queries or [],
            "confidence": confidence,
        }
    )


def test_plan_mode_produces_typed_steps_and_is_fully_accounted(tmp_path):
    def responder(request):
        prompt = "\n".join(message.text for message in request.messages)
        if "internal task planner" in prompt:
            return _plan_payload(assumptions=["Latency matters more than cost"])
        return "A concise, consistent recommendation."

    provider = MockProvider(responder=responder, reasoning=False)
    app = _app(tmp_path, provider)
    outcome = app.reason(
        "Compare the trade-offs, identify the root cause, and derive a logically consistent recommendation."
    )

    from vincio import PlannedStep

    receipt = outcome.result.metadata["universal_reasoning"]
    assert outcome.plan.plan_mode_used
    assert all(isinstance(step, PlannedStep) for step in outcome.plan.steps)
    assert [step.kind for step in outcome.plan.steps] == ["analyze", "compare", "decide"]
    assert outcome.plan.steps[1].depends_on == [0]
    assert outcome.plan.subproblems == [step.goal for step in outcome.plan.steps]
    assert receipt["plan_mode_used"] and receipt["plan_steps"] == 3
    assert receipt["plan_tokens"] > 0 and receipt["plan_trace_id"]
    assert receipt["model_calls"] == len(outcome.passes) + 1
    candidate_prompts = [
        "\n".join(message.text for message in request.messages)
        for request in provider.requests
        if "internal task planner" not in "\n".join(message.text for message in request.messages)
    ]
    assert candidate_prompts
    assert all("Step 2 (compare; after step 1; check: constraint)" in p for p in candidate_prompts)
    assert all("assumption" in p.lower() for p in candidate_prompts)


def test_plan_mode_skips_simple_work_and_respects_off_policy(tmp_path):
    prompts_seen: list[str] = []

    def responder(request):
        prompts_seen.append("\n".join(message.text for message in request.messages))
        return "ANSWER"

    app = _app(tmp_path, MockProvider(responder=responder, reasoning=False))
    simple = app.reason("Rewrite this title in uppercase")
    assert not simple.plan.plan_mode_used
    assert simple.result.metadata["universal_reasoning"]["plan_steps"] == 0

    engine = UniversalReasoningEngine(app, UniversalReasoningPolicy(plan_mode="off"))
    deep = engine.run(
        "Compare the trade-offs, identify the root cause, and derive a logically consistent recommendation."
    )
    assert not deep.plan.plan_mode_used
    assert all("internal task planner" not in prompt for prompt in prompts_seen)


def test_plan_mode_is_bounded_and_cannot_open_the_web(tmp_path):
    from vincio.agents.universal_reasoning import _InternalPlanDecision

    app = _app(tmp_path, MockProvider(default_text="answer"))
    engine = UniversalReasoningEngine(app)
    request = "Compare the trade-offs, identify the root cause, and recommend a fix."
    assessment = engine.assess(request)
    plan = engine.plan(request, assessment)
    decision = _InternalPlanDecision.model_validate(
        {
            "steps": [
                {"goal": f"Do bounded step {i}", "kind": "analyze", "depends_on": [i - 1] if i else [99], "check": "none"}
                for i in range(9)
            ],
            "assumptions": [],
            "evidence_queries": ["latest market prices", "current exchange rate"],
            "confidence": 0.9,
        }
    )
    merged = engine._merge_plan(plan, assessment, decision)

    assert merged.plan_mode_used
    assert len(merged.steps) == engine.policy.plan_max_steps
    assert merged.steps[0].depends_on == []
    # The deterministic policy declined search, so plan queries never open the web.
    assert not assessment.needs_search
    assert all("market prices" not in query for query in merged.search_queries)

    low = engine.plan(request, assessment)
    low_decision = decision.model_copy(update={"confidence": 0.2})
    assert not engine._merge_plan(low, assessment, low_decision).plan_mode_used


def test_fabricated_source_is_refuted_and_withheld(tmp_path):
    url = "https://example.org/release"
    backend = StaticSearchBackend(
        default=[
            SearchResult(rank=1, title="Release notes", url=url, snippet="Version 4.2 shipped.", source="static")
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><title>Release notes</title><body><main>Version 4.2 shipped today.</main></body></html>",
        )

    provider = MockProvider(
        default_text="Version 9.9 shipped. According to nytimes.com, see https://fake.example.net/story."
    )
    app = _app(tmp_path, provider)
    app.use_web_search(
        backend=backend,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    outcome = app.reason("Search for the latest release version and fact-check it with sources.")

    assert outcome.refused
    assert outcome.result.raw_text == ""
    fabricated = outcome.result.metadata["universal_reasoning"]["fabricated_sources"]
    assert "nytimes.com" in fabricated and "https://fake.example.net/story" in fabricated


def test_genuine_sources_and_request_urls_are_never_flagged(tmp_path):
    from vincio.agents.universal_reasoning import _fabricated_sources
    from vincio.core.types import EvidenceItem, TrustLevel

    evidence = [
        EvidenceItem(
            id="web:abc",
            source_id="https://docs.python.org/3/whatsnew/",
            source_type="web",
            text="evidence",
            trust_level=TrustLevel.UNTRUSTED_TOOL,
            metadata={"url": "https://docs.python.org/3/whatsnew/"},
        )
    ]
    honest = "According to python.org, 3.14 is current. See https://docs.python.org/3/whatsnew/."
    assert _fabricated_sources(honest, evidence, "") == []
    assert _fabricated_sources("See https://internal.wiki/page.", [], "Summarize https://internal.wiki/page") == []
    assert _fabricated_sources("The answer is 42, according to my analysis.", [], "") == []
    assert _fabricated_sources("According to fabricated-news.com it doubled.", evidence, "") == [
        "fabricated-news.com"
    ]


def test_plan_validation_is_tolerant_of_wordy_small_models(tmp_path):
    from vincio.agents.universal_reasoning import _InternalPlanDecision

    decision = _InternalPlanDecision.model_validate(
        {
            "steps": [
                {
                    "goal": "Step goal " * 60,  # far past any bound: clipped, not rejected
                    "kind": "Deep-Analysis",  # unknown kind: falls back to default
                    "depends_on": [0, "x", 1.0],
                    "check": "CONSTRAINT ",  # normalized
                }
            ]
            + [{"goal": f"Extra step {i}", "kind": "analyze"} for i in range(15)],
            "assumptions": ["  ", "One real assumption"] + [f"extra {i}" for i in range(9)],
            "evidence_queries": 7,  # junk type: dropped, not rejected
            "confidence": "0.9",
        }
    )

    assert len(decision.steps) == 12
    assert decision.steps[0].kind == "analyze"
    assert decision.steps[0].check == "constraint"
    assert decision.steps[0].depends_on == [0, 1]
    assert len(decision.steps[0].goal) <= 400
    assert decision.assumptions[0] == "One real assumption" and len(decision.assumptions) <= 4
    assert decision.evidence_queries == []
    assert decision.confidence == 0.9

    app = _app(tmp_path, MockProvider(default_text="answer"))
    engine = UniversalReasoningEngine(app)
    request = "Compare the trade-offs, identify the root cause, and recommend a fix."
    assessment = engine.assess(request)
    merged = engine._merge_plan(engine.plan(request, assessment), assessment, decision)
    assert merged.plan_mode_used
    assert len(merged.steps) == engine.policy.plan_max_steps
    assert all(len(step.goal) <= 160 for step in merged.steps)


def test_concurrent_web_reads_preserve_rank_order(tmp_path):
    urls = [f"https://site{i}.example.org/page" for i in range(3)]
    backend = StaticSearchBackend(
        default=[
            SearchResult(rank=i + 1, title=f"Result {i}", url=url, snippet="release", source="static")
            for i, url in enumerate(urls)
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=f"<html><title>{request.url.host}</title><body><main>"
            f"Version 4.2 release notes hosted on {request.url.host}.</main></body></html>",
        )

    app = _app(tmp_path, MockProvider(default_text="Version 4.2 shipped. Source: " + urls[0]))
    app.use_web_search(
        backend=backend,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    outcome = app.reason("Search for the latest release version and fact-check it with sources.")

    hosts = [item.metadata["url"] for item in outcome.web_evidence]
    assert hosts == urls[: len(hosts)] and len(hosts) == 3


def test_transient_provider_failure_is_salvaged_with_one_spaced_retry(tmp_path):
    from vincio.core.errors import ProviderResponseError

    calls = {"n": 0}

    class FlakyProvider(MockProvider):
        async def generate(self, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ProviderResponseError("no choices in response", provider="mock")
            return await super().generate(request)

    app = _app(tmp_path, FlakyProvider(default_text="17 * 23 = 391."))
    engine = UniversalReasoningEngine(app, UniversalReasoningPolicy(salvage_backoff_ms=0))
    outcome = engine.run("Calculate 17 * 23 and verify the equality.")

    assert outcome.result.status.value == "succeeded"
    assert outcome.result.raw_text == "17 * 23 = 391."
    assert [item.kind for item in outcome.passes] == ["candidate", "salvage"]
    assert outcome.result.metadata["universal_reasoning"]["salvaged"]
    assert outcome.answer_verification == "verified"


def test_salvage_respects_policy_off_and_pass_ceiling(tmp_path):
    from vincio.core.errors import ProviderResponseError

    class DeadProvider(MockProvider):
        async def generate(self, request):
            raise ProviderResponseError("no choices in response", provider="mock")

    app = _app(tmp_path, DeadProvider())
    engine = UniversalReasoningEngine(
        app, UniversalReasoningPolicy(salvage_transient_failures=False)
    )
    outcome = engine.run("Calculate 17 * 23 and verify the equality.")
    assert outcome.result.status.value == "failed"
    assert all(item.kind == "candidate" for item in outcome.passes)

    persistent = UniversalReasoningEngine(
        app, UniversalReasoningPolicy(salvage_backoff_ms=0)
    ).run("Calculate 17 * 23 and verify the equality.")
    assert persistent.result.status.value == "failed"
    assert [item.kind for item in persistent.passes] == ["candidate", "salvage"]
    assert not persistent.result.metadata["universal_reasoning"]["corrected"]
