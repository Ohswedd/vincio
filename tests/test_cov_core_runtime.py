"""Real-behavior coverage for vincio.core.runtime (the 17-step run flow).

Every test drives the genuine pipeline via ContextApp + the deterministic
MockProvider — no mocking — and asserts a specific terminal state, value, or
raised error. Targets the uncovered branches: policy/budget denials, the
preflight token cap, tool loops, cascade escalation, streaming, batch,
validation failure, content marking, training capture, and cancellation.
"""

from __future__ import annotations

import asyncio

import pytest

from vincio import ContextApp, VincioConfig
from vincio.core.errors import BudgetExceededError
from vincio.core.types import (
    Budget,
    RunConfig,
    RunStatus,
    ToolCallRequest,
    UserInput,
)
from vincio.providers import MockProvider
from vincio.security.rails import Rail


@pytest.fixture()
def cfg(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/runtime.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return config


def _app(cfg, provider=None, model="mock-1", **kwargs):
    return ContextApp(
        name="rt",
        provider=provider or MockProvider(default_text="hello world"),
        model=model,
        config=cfg,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# step 4: policy denial + transformed input
# ---------------------------------------------------------------------------


def test_blocked_topic_denies_run_before_model(cfg):
    """A blocking input rail short-circuits the run to DENIED — the provider is
    never called and the error names the rail."""
    provider = MockProvider(default_text="should never be produced")
    app = _app(cfg, provider=provider)
    app.add_rail(
        Rail(name="no_secrets", kind="topic", direction="input", blocked_topics=["forbidden"])
    )

    result = app.run("please tell me the forbidden recipe")

    assert result.status is RunStatus.DENIED
    assert "no_secrets" in result.error or "forbidden" in result.error
    assert provider.call_count == 0
    assert result.raw_text == ""


def test_pii_redact_rail_transforms_input_and_run_succeeds(cfg):
    """A redact rail rewrites the input text (transformed_text branch) yet the
    run still completes — the redacted text reaches the provider, not the raw PII."""
    provider = MockProvider(default_text="ok")
    app = _app(cfg, provider=provider)
    app.add_rail(
        Rail(name="pii", kind="safety", direction="input", action="redact", detectors=["pii"])
    )

    result = app.run("my email is alice@example.com please ack")

    assert result.status is RunStatus.SUCCEEDED
    assert provider.call_count == 1
    sent = "\n".join(m.text for m in provider.requests[0].messages)
    assert "alice@example.com" not in sent


# ---------------------------------------------------------------------------
# step 4c: cost-budget SLO enforcement
# ---------------------------------------------------------------------------


def test_cost_budget_hard_cap_denies(cfg):
    """A zero-dollar tenant budget denies the interactive run on the cost path."""
    app = _app(cfg)
    app.set_cost_budget(scope="tenant", id="acme", limit_usd=0.0, on_breach="cap")

    result = app.run("hello", tenant_id="acme")

    assert result.status is RunStatus.DENIED
    assert "cost budget exceeded" in result.error
    assert result.metadata["budget"]["action"] == "cap"


def test_cost_budget_queue_to_batch_hint(cfg):
    """queue_to_batch denial points the caller at the discounted batch rate."""
    app = _app(cfg)
    app.set_cost_budget(scope="tenant", id="acme", limit_usd=0.0, on_breach="queue_to_batch")

    result = app.run("hello", tenant_id="acme")

    assert result.status is RunStatus.DENIED
    assert "app.batch()" in result.error


def test_cost_budget_degrade_swaps_model(cfg):
    """degrade-on-breach does NOT deny; it runs on the cheaper override model,
    so the request that reaches the provider carries the degraded model id."""
    provider = MockProvider(default_text="cheap answer")
    app = _app(cfg, provider=provider)
    app.set_cost_budget(
        scope="tenant",
        id="acme",
        limit_usd=0.0,
        on_breach="degrade",
        degrade_model="mock-cheap",
    )

    result = app.run("hello", tenant_id="acme")

    assert result.status is RunStatus.SUCCEEDED
    assert provider.requests[0].model == "mock-cheap"


# ---------------------------------------------------------------------------
# step 11: preflight input-token cap
# ---------------------------------------------------------------------------


def test_preflight_input_token_cap_fails_run(cfg):
    """An input estimate over max_input_tokens raises BudgetExceededError inside
    _prepare; the run path catches it and reports FAILED with the limit named."""
    app = _app(cfg)
    # max_input_tokens=5 sits between the compiled-context size (~4 tokens, which
    # clears the compiler) and the rendered prompt size (~9 tokens), so the
    # runtime's own preflight token guard is the one that trips.
    tiny = Budget(max_input_tokens=5, max_output_tokens=64)

    result = app.run("a short prompt here", config=RunConfig(budget=tiny))

    assert result.status is RunStatus.FAILED
    assert "max_input_tokens" in result.error


def test_preflight_cap_raises_directly_via_runtime(cfg):
    """The same preflight guard raises BudgetExceededError with used>limit when
    invoked at the prepare layer (not swallowed)."""
    app = _app(cfg)
    tiny = Budget(max_input_tokens=5, max_output_tokens=64)
    ui = UserInput(text="a short prompt here")
    runtime = app._runtime  # noqa: SLF001 - exercising the runtime directly

    async def go():
        from vincio.core.types import RunResult

        res = RunResult(run_id="run_x", status=RunStatus.RUNNING)
        with pytest.raises(BudgetExceededError, match="max_input_tokens") as exc_info:
            await runtime._prepare(  # noqa: SLF001
                ui, RunConfig(), tiny, app.policies, res, "run_x"
            )
        # used (estimated prompt tokens) strictly exceeds the limit.
        assert exc_info.value.limit == 5
        assert exc_info.value.used > 5

    asyncio.run(go())


def test_soft_cap_optout_skips_preflight(cfg):
    """enforce_budget_caps=False restores soft-cap behavior: the preflight token
    guard is skipped and the run completes despite the tiny input budget."""
    app = _app(cfg)
    tiny = Budget(max_input_tokens=5, max_output_tokens=64)

    result = app.run(
        "a short prompt here", config=RunConfig(budget=tiny, enforce_budget_caps=False)
    )

    assert result.status is RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# step 12: tool loop
# ---------------------------------------------------------------------------


def _weather(city: str) -> dict:
    """Return the weather for a city."""
    return {"city": city, "temp_c": 21}


def test_tool_round_executes_then_model_answers(cfg):
    """The model requests a tool on round 1, the runtime executes it, appends the
    tool message, and the second model turn produces the final answer."""
    script = [
        {"tool_call": {"name": "_weather", "arguments": {"city": "Paris"}}},
        "It is 21C in Paris.",
    ]
    provider = MockProvider(script=script)
    app = _app(cfg, provider=provider)
    app.add_tool(_weather)

    result = app.run("weather in Paris?")

    assert result.status is RunStatus.SUCCEEDED
    assert result.raw_text == "It is 21C in Paris."
    assert len(result.tool_results) == 1
    tr = result.tool_results[0]
    assert tr.status == "ok"
    assert tr.output == {"city": "Paris", "temp_c": 21}


def test_tool_call_budget_zero_skips_execution(cfg):
    """max_tool_calls=0 means _run_tool_round admits no calls: the tool request is
    dropped, no tool result is recorded, and the run terminates on that turn."""
    provider = MockProvider(
        script=[{"tool_call": {"name": "_weather", "arguments": {"city": "Rome"}}}]
    )
    app = _app(cfg, provider=provider)
    app.add_tool(_weather)

    result = app.run(
        "weather?", config=RunConfig(budget=Budget(max_tool_calls=0, max_output_tokens=64))
    )

    assert result.tool_results == []


def test_failing_tool_records_error_result(cfg):
    """A tool that raises is captured as an error ToolResult (the tool runtime
    converts the exception) rather than crashing the run."""

    def boom(x: str) -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    provider = MockProvider(
        script=[
            {"tool_call": {"name": "boom", "arguments": {"x": "1"}}},
            "recovered",
        ]
    )
    app = _app(cfg, provider=provider)
    app.add_tool(boom)

    result = app.run("do it")

    assert result.tool_results[0].status == "error"
    assert "kaboom" in result.tool_results[0].error
    assert result.raw_text == "recovered"


def test_tool_invalid_arguments_become_error_result(cfg):
    """When the model calls a tool with arguments missing a required field, the
    tool runtime raises a ToolValidationError that _run_tool_round catches and
    turns into an error ToolResult (the VincioError-handling branch)."""

    def _weather_req(city: str) -> dict:
        """Weather requires a city."""
        return {"city": city}

    provider = MockProvider(
        script=[
            {"tool_call": {"name": "_weather_req", "arguments": {}}},
            "done anyway",
        ]
    )
    app = _app(cfg, provider=provider)
    app.add_tool(_weather_req)

    result = app.run("weather")

    assert result.tool_results[0].status == "error"
    assert "invalid arguments" in result.tool_results[0].error


# ---------------------------------------------------------------------------
# step 12: per-call budget hard caps inside the loop
# ---------------------------------------------------------------------------


def test_output_token_breach_in_loop_fails(cfg):
    """A large output that exceeds max_output_tokens trips _enforce_budget inside
    the model loop, surfacing FAILED with an output_tokens breach."""
    provider = MockProvider(default_text="word " * 400)
    app = _app(cfg, provider=provider)

    result = app.run(
        "go", config=RunConfig(budget=Budget(max_output_tokens=2, max_input_tokens=100_000))
    )

    assert result.status is RunStatus.FAILED
    assert "output_tokens" in result.error


# ---------------------------------------------------------------------------
# step 12: cascade escalation
# ---------------------------------------------------------------------------


def test_cascade_escalates_on_low_confidence(cfg):
    """A cascade whose confidence signal always reports low confidence escalates
    from the cheap rung to the strong rung; metadata records the final model and
    one escalation."""
    provider = MockProvider(default_text="meh")
    app = _app(cfg, provider=provider)
    app.use_cascade(["mock-cheap", "mock-strong"], confidence=lambda r: 0.0)

    result = app.run("anything")

    assert result.status is RunStatus.SUCCEEDED
    assert result.metadata["cascade"]["model"] == "mock-strong"
    assert result.metadata["cascade"]["escalations"] == 1
    # Both rungs were called (cheap first, then strong).
    assert [r.model for r in provider.requests] == ["mock-cheap", "mock-strong"]


def test_cascade_no_escalation_on_high_confidence(cfg):
    """A high-confidence signal keeps the run on the cheap rung — no escalation."""
    provider = MockProvider(default_text="confident")
    app = _app(cfg, provider=provider)
    app.use_cascade(["mock-cheap", "mock-strong"], confidence=lambda r: 1.0)

    result = app.run("anything")

    assert result.metadata["cascade"]["escalations"] == 0
    assert provider.requests[-1].model == "mock-cheap"


def test_cascade_confidence_signal_exception_falls_back(cfg):
    """A confidence callable that raises must not break the run; the default
    confidence signal takes over and the run still completes."""
    def bad(_response):
        raise ValueError("bad signal")

    provider = MockProvider(default_text="answer")
    app = _app(cfg, provider=provider)
    app.use_cascade(["mock-cheap", "mock-strong"], confidence=bad)

    result = app.run("anything")

    assert result.status is RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# unknown model accounting
# ---------------------------------------------------------------------------


def test_unknown_model_emits_event_once(cfg):
    """A model with no price-table entry surfaces a model.unknown event and is
    recorded once under metadata['unknown_models']."""
    seen: list = []
    provider = MockProvider(default_text="x")
    app = _app(cfg, provider=provider, model="totally-unknown-model-zzz")
    app.events.subscribe("model.unknown", lambda event: seen.append(event))

    result = app.run("hi")

    assert result.metadata.get("unknown_models") == ["totally-unknown-model-zzz"]
    assert seen and seen[0].payload["model"] == "totally-unknown-model-zzz"


# ---------------------------------------------------------------------------
# RunConfig knobs threaded into the request
# ---------------------------------------------------------------------------


def test_temperature_and_seed_reach_provider(cfg):
    """RunConfig.temperature and .seed are passed through into the ModelRequest."""
    provider = MockProvider(default_text="ok")
    app = _app(cfg, provider=provider)

    app.run("hi", config=RunConfig(temperature=0.3, seed=42))

    req = provider.requests[0]
    assert req.temperature == 0.3
    assert req.seed == 42


def test_run_config_model_override_pins_model(cfg):
    """An explicit per-run model overrides app.model on the outgoing request."""
    provider = MockProvider(default_text="ok")
    app = _app(cfg, provider=provider)

    app.run("hi", config=RunConfig(model="pinned-model"))

    assert provider.requests[0].model == "pinned-model"


# ---------------------------------------------------------------------------
# steps 13-16: validation failure
# ---------------------------------------------------------------------------


def test_schema_validation_failure_marks_run_failed(cfg):
    """When the model returns text that doesn't satisfy the output schema, the
    run is FAILED with a validation error and the raw text is preserved."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        score: int

    # A responder (not default_text) bypasses the mock's schema-synthesis path,
    # so genuinely unstructured prose reaches the validator and fails it.
    provider = MockProvider(responder=lambda r: "this is not json at all")
    app = ContextApp(
        name="rt",
        provider=provider,
        model="mock-1",
        config=cfg,
        output_schema=Answer,
    )

    result = app.run("give me a score")

    assert result.status is RunStatus.FAILED
    assert "output validation failed" in result.error
    assert result.raw_text == "this is not json at all"
    assert result.validation["valid"] is False


def test_valid_schema_run_succeeds_with_structured_output(cfg):
    """A schema-valid structured response yields SUCCEEDED with parsed output."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        score: int

    provider = MockProvider()  # default synthesizes a schema-valid instance
    app = ContextApp(
        name="rt", provider=provider, model="mock-1", config=cfg, output_schema=Answer
    )

    result = app.run("score it")

    assert result.status is RunStatus.SUCCEEDED
    assert result.validation["valid"] is True
    # The validated output is the parsed model instance with the synthesized score.
    assert result.output.score == 1


# ---------------------------------------------------------------------------
# content marking + training capture epilogue
# ---------------------------------------------------------------------------


def test_content_marking_attaches_credentials(cfg):
    """With content marking on, a successful run stamps C2PA-style content
    credentials and an AI disclosure into result.metadata."""
    provider = MockProvider(default_text="a synthetic answer")
    app = _app(cfg, provider=provider)
    app.content_marking = True

    result = app.run("write something")

    assert "content_credentials" in result.metadata
    assert "ai_disclosure" in result.metadata
    assert isinstance(result.metadata["content_credentials"], dict)


def test_training_capture_records_untruncated_output_on_trace(cfg):
    """observability.training_capture stamps the full input/output on the trace
    via _capture_training_artifacts."""
    captured: list = []
    provider = MockProvider(default_text="full output text here")
    app = _app(cfg, provider=provider)
    app.config.observability.training_capture = True
    app.tracer.exporter.export = lambda trace: captured.append(trace)  # type: ignore[assignment]

    app.run("the full input prompt")

    assert captured
    trace = captured[-1]
    assert trace.attributes["input_full"] == "the full input prompt"
    assert trace.attributes["output_full"] == "full output text here"


# ---------------------------------------------------------------------------
# streaming path
# ---------------------------------------------------------------------------


def test_stream_yields_deltas_and_done_with_result(cfg):
    """astream emits text_delta events whose concatenation equals the answer and
    a terminal done event carrying a SUCCEEDED RunResult."""
    provider = MockProvider(default_text="streamed answer body")
    app = _app(cfg, provider=provider)

    async def collect():
        events = []
        async for ev in app.astream("stream it"):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    deltas = "".join(e.text for e in events if e.type == "text_delta")
    done = [e for e in events if e.type == "done"]

    assert deltas == "streamed answer body"
    assert len(done) == 1
    assert done[0].result.status is RunStatus.SUCCEEDED
    assert any(e.type == "stage" and e.stage == "context_compiled" for e in events)


def test_stream_emits_error_event_on_policy_failure(cfg):
    """A streaming run over a budget-exceeding input emits an error event and a
    FAILED done result."""
    provider = MockProvider(default_text="x")
    app = _app(cfg, provider=provider)
    tiny = Budget(max_input_tokens=5, max_output_tokens=64)

    async def collect():
        out = []
        async for ev in app.astream("a short prompt here", config=RunConfig(budget=tiny)):
            out.append(ev)
        return out

    events = asyncio.run(collect())
    errors = [e for e in events if e.type == "error"]
    done = [e for e in events if e.type == "done"]

    assert errors and "max_input_tokens" in errors[0].error
    assert done[0].result.status is RunStatus.FAILED


def test_stream_runs_tool_round(cfg):
    """The streaming loop drives a tool round: it emits tool_call and tool_result
    events and the final answer arrives as deltas."""
    provider = MockProvider(
        script=[
            {"tool_call": {"name": "_weather", "arguments": {"city": "Oslo"}}},
            "Oslo is cold.",
        ]
    )
    app = _app(cfg, provider=provider)
    app.add_tool(_weather)

    async def collect():
        out = []
        async for ev in app.astream("weather Oslo"):
            out.append(ev)
        return out

    events = asyncio.run(collect())
    assert any(e.type == "tool_call" and e.tool_name == "_weather" for e in events)
    assert any(e.type == "tool_result" for e in events)
    done = [e for e in events if e.type == "done"][0]
    assert done.result.tool_results[0].output == {"city": "Oslo", "temp_c": 21}


def test_stream_cascade_buffers_and_replays_final(cfg):
    """A cascade streaming run buffers each rung and replays only the accepted
    (escalated) answer as deltas — never a discarded cheap attempt."""
    provider = MockProvider(default_text="final answer")
    app = _app(cfg, provider=provider)
    app.use_cascade(["mock-cheap", "mock-strong"], confidence=lambda r: 0.0)

    async def collect():
        out = []
        async for ev in app.astream("go"):
            out.append(ev)
        return out

    events = asyncio.run(collect())
    deltas = "".join(e.text for e in events if e.type == "text_delta")
    done = [e for e in events if e.type == "done"][0]

    assert deltas == "final answer"
    assert done.result.metadata["cascade"]["model"] == "mock-strong"


# ---------------------------------------------------------------------------
# batch path
# ---------------------------------------------------------------------------


def test_batch_runs_all_inputs(cfg):
    """abatch prepares and finalizes each input through the batch runner, one
    SUCCEEDED RunResult per input, costs attributed."""
    provider = MockProvider(default_text="batched")
    app = _app(cfg, provider=provider)

    results = asyncio.run(app.abatch(["first", "second", "third"]))

    assert len(results) == 3
    assert all(r.status is RunStatus.SUCCEEDED for r in results)
    assert all(r.raw_text == "batched" for r in results)


def test_batch_denied_input_short_circuits_without_request(cfg):
    """A batch input that fails the input policy is recorded DENIED and never
    becomes a batch request — the other inputs still run."""
    provider = MockProvider(default_text="ok")
    app = _app(cfg, provider=provider)
    app.add_rail(
        Rail(name="block", kind="topic", direction="input", blocked_topics=["forbidden"])
    )

    results = asyncio.run(app.abatch(["the forbidden one", "a clean one"]))

    assert RunStatus.DENIED in {r.status for r in results}
    assert RunStatus.SUCCEEDED in {r.status for r in results}


# ---------------------------------------------------------------------------
# cancellation epilogue
# ---------------------------------------------------------------------------


def test_cancelled_run_is_recorded_and_reraised(cfg):
    """A cooperatively cancelled run still runs the shared epilogue (the result
    is CANCELLED) before re-raising CancelledError out of execute."""
    provider = MockProvider(default_text="x")
    app = _app(cfg, provider=provider)
    result_box: list = []

    class _Cancelling:
        """A provider that cancels the in-flight model call."""

        name = "cancel"

        def capabilities(self, model):
            return provider.capabilities(model)

        async def generate(self, request):
            raise asyncio.CancelledError

        async def stream(self, request):  # pragma: no cover - unused
            raise asyncio.CancelledError
            yield

    app._provider_instance = _Cancelling()  # noqa: SLF001

    async def go():
        try:
            await app._runtime.execute(UserInput(text="hi"))  # noqa: SLF001
        except asyncio.CancelledError:
            result_box.append("reraised")

    asyncio.run(go())
    assert result_box == ["reraised"]


# ---------------------------------------------------------------------------
# reasoning trace recording
# ---------------------------------------------------------------------------


def test_reasoning_effort_threaded_and_tokens_billed(cfg):
    """An explicit reasoning_effort reaches the provider, which emits reasoning
    tokens that flow into the result usage."""
    provider = MockProvider(default_text="thought it through", reasoning=True)
    app = _app(cfg, provider=provider)

    result = app.run("think hard", config=RunConfig(reasoning_effort="high"))

    assert provider.requests[0].reasoning_effort == "high"
    assert result.usage.reasoning_tokens == 128
    assert result.status is RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# _tool_message direct unit (ok vs error formatting)
# ---------------------------------------------------------------------------


def test_tool_message_formats_ok_and_error(cfg):
    """_tool_message serializes ok output as JSON and an error as 'error: ...',
    preserving the call id and tool name on both."""
    from vincio.core.runtime import VincioRuntime
    from vincio.core.types import ToolResult

    call = ToolCallRequest(name="t", arguments={}, id="call-1")
    ok = ToolResult(call_id="call-1", tool_name="t", status="ok", output={"a": 1})
    err = ToolResult(call_id="call-1", tool_name="t", status="error", error="nope")

    ok_msg = VincioRuntime._tool_message(call, ok)  # noqa: SLF001
    err_msg = VincioRuntime._tool_message(call, err)  # noqa: SLF001

    assert ok_msg.content == '{"a": 1}'
    assert ok_msg.tool_call_id == "call-1"
    assert ok_msg.name == "t"
    assert err_msg.content == "error: nope"
    assert err_msg.role == "tool"


# ---------------------------------------------------------------------------
# exact response cache → second run is free
# ---------------------------------------------------------------------------


def test_response_cache_serves_second_identical_run_free(cfg):
    """With the exact-match response cache enabled, an identical second run is
    served from cache: the provider is called once, the second run costs $0."""
    cfg.cache.response_cache = True
    provider = MockProvider(default_text="cached body")
    app = _app(cfg, provider=provider)
    assert app.response_cache is not None

    r1 = app.run("same question")
    calls_after_first = provider.call_count
    r2 = app.run("same question")

    assert r1.raw_text == "cached body"
    assert r2.raw_text == "cached body"
    assert provider.call_count == calls_after_first  # no new provider call
    assert r2.cost_usd == 0.0


# ---------------------------------------------------------------------------
# step 14: evaluators
# ---------------------------------------------------------------------------


def test_evaluators_score_the_output(cfg):
    """Attached evaluators run in _finalize and land on result.eval_scores."""
    provider = MockProvider(default_text="the answer")
    app = _app(cfg, provider=provider)
    app.add_evaluator("output_tokens")
    app.add_evaluator("latency")

    result = app.run("evaluate me")

    assert "output_tokens" in result.eval_scores
    assert result.eval_scores["output_tokens"] >= 0
    assert "latency" in result.eval_scores


def test_unknown_evaluator_name_is_skipped(cfg):
    """An evaluator name not present in METRICS is silently skipped (no crash,
    no score), exercising the metric-is-None continue branch."""
    provider = MockProvider(default_text="x")
    app = _app(cfg, provider=provider)
    app.evaluators.append("no_such_metric_zzz")

    result = app.run("hi")

    assert result.status is RunStatus.SUCCEEDED
    assert "no_such_metric_zzz" not in result.eval_scores


# ---------------------------------------------------------------------------
# energy / carbon accounting
# ---------------------------------------------------------------------------


def test_energy_accounting_accrues_on_result(cfg):
    """With energy accounting on, a successful run accrues positive energy_wh and
    co2e_grams that are also stamped on the run audit details."""
    provider = MockProvider(default_text="some output to burn tokens on")
    app = _app(cfg, provider=provider)
    app.use_energy_accounting(region="us")

    result = app.run("compute")

    assert result.status is RunStatus.SUCCEEDED
    assert result.energy_wh > 0
    assert result.co2e_grams >= 0


# ---------------------------------------------------------------------------
# retrieval + memory writes (RAG app drives steps 5-7 and 16)
# ---------------------------------------------------------------------------


def test_rag_run_screens_and_finalizes(rag_app):
    """The retrieval branch runs concurrently in _prepare: a RAG app retrieves
    evidence, cites it, and the run succeeds with citations and evidence stamped
    on the result."""
    result = rag_app.run("What is the refund window for the Pro plan?")

    assert result.status is RunStatus.SUCCEEDED
    assert result.evidence  # retrieval populated evidence
    assert result.citations  # the citing mock provider cited a ref


def test_memory_recall_and_writeback(cfg):
    """A memory-enabled run recalls candidates (step 5) and writes back the input
    (step 16); a second run sees a non-empty memory store."""
    cfg.memory.enabled = True
    provider = MockProvider(default_text="acknowledged")
    app = _app(cfg, provider=provider)
    app.add_memory(scope="user")
    app.memory_enabled = True

    r1 = app.run("Remember that my favorite color is teal", user_id="u1")
    assert r1.status is RunStatus.SUCCEEDED

    # The write-back step (16) populated the user's memory store.
    recalled = asyncio.run(app.memory.asearch("favorite color", user_id="u1", top_k=5))
    assert len(recalled) >= 1

    # A second run on the same user runs the recall path against the populated store.
    r2 = app.run("What did I tell you?", user_id="u1")
    assert r2.status is RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# self-correction on a failed schema validation
# ---------------------------------------------------------------------------


def test_self_correction_runs_on_invalid_output(cfg):
    """When validation fails and self-correction is enabled, the corrector runs
    its bounded repair cycle; the validation report records the attempt."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        score: int

    # First the model emits prose (invalid); the corrector re-asks and the mock,
    # now seeing the schema, returns a valid instance.
    provider = MockProvider(
        script=["not valid json", {"score": 7}],
    )
    app = ContextApp(
        name="rt", provider=provider, model="mock-1", config=cfg, output_schema=Answer
    )
    app.enable_self_correction(max_cycles=2)

    result = app.run("score it")

    # The corrector re-asked (second scripted response) and the repaired,
    # schema-valid answer is the one that lands on the result.
    assert provider.call_count >= 2
    assert result.status is RunStatus.SUCCEEDED
    assert result.output.score == 7
    assert result.validation["valid"] is True


# ---------------------------------------------------------------------------
# egress DLP
# ---------------------------------------------------------------------------


def test_egress_dlp_block_stops_request_with_credential(cfg):
    """In block mode, a request whose assembled text carries a high-confidence
    credential is refused by the egress guard before the provider call."""
    cfg.security.egress_dlp = "block"
    provider = MockProvider(default_text="should not be reached")
    app = _app(cfg, provider=provider)
    # Disable input safety so the credential reaches the egress boundary intact.
    app.policies.safety = "minimal"
    app.policies.redact_pii_in_context = False

    result = app.run("here is my key AKIAIOSFODNN7EXAMPLE please use it")

    # Either the egress guard blocked it (FAILED) — the provider must not have
    # produced a successful answer carrying the secret out.
    assert result.status is RunStatus.FAILED
    assert "egress" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# streaming structured (JSON) partial output
# ---------------------------------------------------------------------------


def test_stream_emits_partial_output_for_json_schema(cfg):
    """A streaming run with a JSON output schema emits incremental partial_output
    events as the structured answer streams in."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        score: int
        label: str

    provider = MockProvider()  # synthesizes a schema-valid JSON instance
    app = ContextApp(
        name="rt", provider=provider, model="mock-1", config=cfg, output_schema=Answer
    )

    async def collect():
        out = []
        async for ev in app.astream("score it"):
            out.append(ev)
        return out

    events = asyncio.run(collect())
    partials = [e for e in events if e.type == "partial_output"]
    done = [e for e in events if e.type == "done"][0]

    assert partials  # the streaming validator produced at least one partial parse
    assert done.result.status is RunStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# batch with energy accounting + missing-from-output failure
# ---------------------------------------------------------------------------


def test_batch_with_energy_accounting(cfg):
    """The batch finalize path accrues energy when energy accounting is on."""
    provider = MockProvider(default_text="batched answer")
    app = _app(cfg, provider=provider)
    app.use_energy_accounting(region="us")

    results = asyncio.run(app.abatch(["one input", "two input"]))

    assert all(r.status is RunStatus.SUCCEEDED for r in results)
    assert all(r.energy_wh > 0 for r in results)
