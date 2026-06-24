"""Real-behavior coverage for vincio.providers.shadow.

Exercises ShadowProvider (primary-returned, candidate dual-dispatched, divergence
and cost diffing, blocking/background dispatch, candidate-failure capture) and
CanaryRouter (deterministic percentage routing, online scoring, auto-rollback,
prompt rollback) through the real API with deterministic MockProviders only.
"""

from __future__ import annotations

from vincio.core.types import (
    Message,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from vincio.providers import MockProvider
from vincio.providers.shadow import (
    CanaryRouter,
    CanaryState,
    ShadowObservation,
    ShadowProvider,
    _last_user_text,
)


def _req(text: str = "hello", model: str = "primary-1") -> ModelRequest:
    return ModelRequest(model=model, messages=[Message(role="user", content=text)])


class _BoomProvider(MockProvider):
    """A provider whose generate always raises (to drive the candidate-error path)."""

    async def generate(self, request: ModelRequest) -> ModelResponse:  # type: ignore[override]
        raise RuntimeError("candidate exploded")


class _NoDoneProvider(MockProvider):
    """Streams text deltas but never emits a terminal 'done' event."""

    async def stream(self, request: ModelRequest):  # type: ignore[override]
        from vincio.core.types import ModelEvent

        yield ModelEvent(type="text_delta", text="partial")


class _RecordingEvents:
    """A real event sink that records (topic, payload) pairs."""

    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    def emit(self, topic: str, payload: dict) -> None:
        self.emitted.append((topic, payload))


class _RecordingRegistry:
    def __init__(self, *, fail: bool = False) -> None:
        self.rolled_back: list[str] = []
        self._fail = fail

    def rollback(self, name: str) -> None:
        if self._fail:
            raise RuntimeError("registry locked")
        self.rolled_back.append(name)


# --------------------------------------------------------------------------- #
# _last_user_text
# --------------------------------------------------------------------------- #


def test_last_user_text_prefers_latest_user_message():
    req = ModelRequest(
        model="m",
        messages=[
            Message(role="user", content="first"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="second"),
        ],
    )
    assert _last_user_text(req) == "second"


def test_last_user_text_falls_back_to_last_message_when_no_user():
    # No user/developer role -> loop falls through (45->44) and returns last text (47).
    req = ModelRequest(
        model="m",
        messages=[
            Message(role="system", content="sys"),
            Message(role="assistant", content="only-assistant"),
        ],
    )
    assert _last_user_text(req) == "only-assistant"


def test_last_user_text_empty_messages_returns_empty():
    req = ModelRequest.model_construct(model="m", messages=[])
    assert _last_user_text(req) == ""


# --------------------------------------------------------------------------- #
# ShadowProvider.generate — blocking path, divergence detection
# --------------------------------------------------------------------------- #


async def test_shadow_returns_primary_and_records_identical_pair():
    primary = MockProvider(default_text="same answer")
    candidate = MockProvider(default_text="same answer")
    sp = ShadowProvider(primary, candidate, block=True)

    resp = await sp.generate(_req("q"))

    # User always gets the PRIMARY response.
    assert resp.text == "same answer"
    assert len(sp.observations) == 1
    obs = sp.observations[0]
    assert obs.primary_text == "same answer"
    assert obs.candidate_text == "same answer"
    # Identical text short-circuits to a perfect similarity (the == branch).
    assert obs.output_similarity == 1.0
    assert obs.candidate_error is None


async def test_shadow_detects_divergence_with_partial_similarity():
    primary = MockProvider(default_text="the quick brown fox")
    candidate = MockProvider(default_text="the quick brown cat")
    sp = ShadowProvider(primary, candidate, block=True)

    await sp.generate(_req())
    obs = sp.observations[0]

    assert obs.candidate_text == "the quick brown cat"
    # Not identical -> SequenceMatcher ratio in (0, 1).
    assert obs.output_similarity is not None
    assert 0.0 < obs.output_similarity < 1.0


async def test_shadow_candidate_failure_is_captured_not_raised():
    primary = MockProvider(default_text="ok")
    candidate = _BoomProvider()
    sp = ShadowProvider(primary, candidate, block=True)

    resp = await sp.generate(_req())

    # The user is unaffected by the candidate blowing up.
    assert resp.text == "ok"
    obs = sp.observations[0]
    assert obs.candidate_text is None
    assert obs.candidate_error == "RuntimeError: candidate exploded"


async def test_shadow_recorder_and_events_fire_on_dispatch():
    recorded: list[ShadowObservation] = []
    events = _RecordingEvents()
    sp = ShadowProvider(
        MockProvider(default_text="a"),
        MockProvider(default_text="a"),
        block=True,
        recorder=recorded.append,
        events=events,
    )

    await sp.generate(_req())

    assert len(recorded) == 1
    assert recorded[0].candidate_text == "a"
    assert len(events.emitted) == 1
    topic, payload = events.emitted[0]
    assert topic == "model.shadow"
    assert payload["candidate_text"] == "a"


async def test_shadow_candidate_model_override_changes_request_model():
    primary = MockProvider(default_text="P")
    candidate = MockProvider(default_text="C")
    sp = ShadowProvider(primary, candidate, candidate_model="cand-9", block=True)

    await sp.generate(_req(model="primary-1"))

    # The candidate saw a copy of the request retargeted to candidate_model.
    assert candidate.requests[-1].model == "cand-9"
    obs = sp.observations[0]
    assert obs.primary_model == "primary-1"
    assert obs.candidate_model == "cand-9"


# --------------------------------------------------------------------------- #
# ShadowProvider — background (non-blocking) dispatch + drain + aclose
# --------------------------------------------------------------------------- #


async def test_shadow_background_dispatch_completes_after_drain():
    primary = MockProvider(default_text="prim")
    candidate = MockProvider(default_text="cand")
    sp = ShadowProvider(primary, candidate, block=False)

    resp = await sp.generate(_req())
    assert resp.text == "prim"

    await sp.drain()  # awaits the in-flight background task (189-190)

    obs = sp.observations[0]
    assert obs.candidate_text == "cand"


async def test_shadow_drain_noop_when_nothing_pending():
    sp = ShadowProvider(MockProvider(), MockProvider(), block=True)
    # block=True keeps nothing in the background pool; drain must be a clean no-op.
    await sp.drain()
    assert sp._pending == set()


async def test_shadow_aclose_drains_and_closes_both():
    primary = MockProvider(default_text="p")
    candidate = MockProvider(default_text="c")
    sp = ShadowProvider(primary, candidate, block=False)
    await sp.generate(_req())

    await sp.aclose()  # drain + primary.aclose + candidate.aclose (215-217)

    assert sp.observations[0].candidate_text == "c"


# --------------------------------------------------------------------------- #
# ShadowProvider.stream
# --------------------------------------------------------------------------- #


async def test_shadow_stream_records_after_done_blocking():
    primary = MockProvider(default_text="streamed primary")
    candidate = MockProvider(default_text="streamed candidate")
    sp = ShadowProvider(primary, candidate, block=True)

    chunks = [ev.text for ev in [e async for e in sp.stream(_req())] if ev.type == "text_delta"]
    # Concatenated deltas reproduce the primary text exactly.
    assert "".join(chunks) == "streamed primary"

    obs = sp.observations[0]
    assert obs.primary_text == "streamed primary"
    assert obs.candidate_text == "streamed candidate"
    assert obs.output_similarity is not None and obs.output_similarity < 1.0


async def test_shadow_stream_background_then_drain():
    sp = ShadowProvider(
        MockProvider(default_text="abc"),
        MockProvider(default_text="abc"),
        block=False,
    )
    async for _ in sp.stream(_req()):
        pass
    await sp.drain()
    assert sp.observations[0].candidate_text == "abc"
    assert sp.observations[0].output_similarity == 1.0


# --------------------------------------------------------------------------- #
# ShadowProvider._cost / diff
# --------------------------------------------------------------------------- #


async def test_shadow_uses_explicit_cost_usd_when_present():
    primary = MockProvider(
        responder=lambda r: ModelResponse(text="p", cost_usd=0.5)
    )
    candidate = MockProvider(
        responder=lambda r: ModelResponse(text="c", cost_usd=0.25)
    )
    sp = ShadowProvider(primary, candidate, block=True)

    await sp.generate(_req())
    obs = sp.observations[0]
    assert obs.primary_cost_usd == 0.5
    assert obs.candidate_cost_usd == 0.25


def test_shadow_cost_swallows_pricing_errors():
    class _BadPriceTable:
        def cost(self, model, usage):
            raise ValueError("no price")

    sp = ShadowProvider(MockProvider(), MockProvider(), price_table=_BadPriceTable())
    # cost_usd unset -> falls into price table -> raises -> swallowed to 0.0 (112-113).
    resp = ModelResponse(text="x", model="unknown", usage=TokenUsage(input_tokens=3))
    assert sp._cost(resp) == 0.0


def test_shadow_diff_aggregates_pairs_errors_and_costs():
    sp = ShadowProvider(MockProvider(), MockProvider())
    sp.observations.append(
        ShadowObservation(
            primary_text="a",
            candidate_text="a",
            output_similarity=1.0,
            primary_cost_usd=0.10,
            candidate_cost_usd=0.04,
        )
    )
    sp.observations.append(
        ShadowObservation(
            primary_text="b",
            candidate_text="z",
            output_similarity=0.5,
            primary_cost_usd=0.10,
            candidate_cost_usd=0.06,
        )
    )
    sp.observations.append(
        ShadowObservation(primary_text="c", candidate_error="Timeout", primary_cost_usd=0.10)
    )

    d = sp.diff()
    assert d["observations"] == 3
    assert d["paired"] == 2
    assert d["candidate_errors"] == 1
    assert d["candidate_error_rate"] == round(1 / 3, 4)
    assert d["mean_output_similarity"] == 0.75
    assert d["primary_cost_usd"] == 0.30
    assert d["candidate_cost_usd"] == 0.10
    assert d["cost_ratio"] == round(0.10 / 0.30, 4)


def test_shadow_diff_empty_uses_safe_defaults():
    d = ShadowProvider(MockProvider(), MockProvider()).diff()
    assert d["observations"] == 0
    assert d["candidate_error_rate"] == 0.0
    assert d["mean_output_similarity"] is None
    assert d["cost_ratio"] is None


def test_shadow_observations_bounded_by_max():
    sp = ShadowProvider(MockProvider(), MockProvider(), max_observations=2)
    for i in range(5):
        sp.observations.append(ShadowObservation(primary_text=str(i)))
    # deque(maxlen) keeps only the most recent two.
    assert [o.primary_text for o in sp.observations] == ["3", "4"]


def test_shadow_max_observations_floor_is_one():
    sp = ShadowProvider(MockProvider(), MockProvider(), max_observations=0)
    assert sp.observations.maxlen == 1


async def test_shadow_capabilities_delegates_to_primary():
    primary = MockProvider()
    sp = ShadowProvider(primary, MockProvider())
    caps = sp.capabilities("mock-1")
    assert caps.tool_calling is True
    assert caps == primary.capabilities("mock-1")


# --------------------------------------------------------------------------- #
# CanaryRouter — routing, scoring, rollback
# --------------------------------------------------------------------------- #


def test_canary_percent_clamped_into_range():
    assert CanaryRouter(MockProvider(), MockProvider(), percent=250).percent == 100.0
    assert CanaryRouter(MockProvider(), MockProvider(), percent=-7).percent == 0.0


async def test_canary_all_primary_at_zero_percent():
    primary = MockProvider(default_text="P")
    candidate = MockProvider(default_text="C")
    router = CanaryRouter(primary, candidate, percent=0.0)

    for _ in range(10):
        resp = await router.generate(_req())
        assert resp.text == "P"

    assert candidate.call_count == 0
    assert router.state().candidate_n == 0


async def test_canary_routes_deterministic_share_to_candidate():
    primary = MockProvider(default_text="P")
    candidate = MockProvider(default_text="C")
    router = CanaryRouter(primary, candidate, percent=50.0)

    texts = [(await router.generate(_req())).text for _ in range(10)]
    # 50% accumulator routes every 2nd call to the candidate -> exactly 5.
    assert texts.count("C") == 5
    assert texts.count("P") == 5
    assert router.calls == 10


async def test_canary_candidate_model_override():
    primary = MockProvider(default_text="P")
    candidate = MockProvider(default_text="C")
    router = CanaryRouter(primary, candidate, percent=100.0, candidate_model="cand-x")

    await router.generate(_req(model="orig"))
    assert candidate.requests[-1].model == "cand-x"


async def test_canary_auto_rollback_on_regression():
    events = _RecordingEvents()
    rolled: list[CanaryState] = []
    registry = _RecordingRegistry()
    # Candidate scores 0 (empty text -> response_confidence 0), primary scores 1.
    primary = MockProvider(default_text="solid answer")
    candidate = MockProvider(
        responder=lambda r: ModelResponse(text="", finish_reason="error")
    )
    router = CanaryRouter(
        primary,
        candidate,
        percent=50.0,
        min_samples=3,
        regression_threshold=0.05,
        on_rollback=rolled.append,
        prompt_registry=registry,
        prompt_name="qa-prompt",
        events=events,
    )

    for _ in range(20):
        await router.generate(_req())

    state = router.state()
    assert state.rolled_back is True
    assert router.percent == 0.0
    assert "candidate mean" in state.rollback_reason
    # Prompt registry rolled back to prior head.
    assert registry.rolled_back == ["qa-prompt"]
    # on_rollback + event both fired exactly once.
    assert len(rolled) == 1
    assert rolled[0].rolled_back is True
    assert [t for t, _ in events.emitted] == ["canary.rollback"]


async def test_canary_no_rollback_when_candidate_matches_primary():
    primary = MockProvider(default_text="answer")
    candidate = MockProvider(default_text="answer")
    router = CanaryRouter(primary, candidate, percent=50.0, min_samples=3)

    for _ in range(20):
        await router.generate(_req())

    assert router.rolled_back is False
    st = router.state()
    assert st.primary_mean == 1.0
    assert st.candidate_mean == 1.0


async def test_canary_waits_for_min_samples_before_rollback():
    primary = MockProvider(default_text="good")
    candidate = MockProvider(responder=lambda r: ModelResponse(text="", finish_reason="error"))
    router = CanaryRouter(primary, candidate, percent=50.0, min_samples=10)

    # Only 4 calls: candidate arm has < min_samples, so no rollback yet (314-315).
    for _ in range(4):
        await router.generate(_req())
    assert router.rolled_back is False


def test_canary_set_percent_ignored_after_rollback():
    router = CanaryRouter(MockProvider(), MockProvider(), percent=20.0)
    router.set_percent(40.0)
    assert router.percent == 40.0

    router.rolled_back = True
    router.set_percent(80.0)  # ignored (285-286 guard)
    assert router.percent == 40.0


def test_canary_set_percent_clamps():
    router = CanaryRouter(MockProvider(), MockProvider())
    router.set_percent(900.0)
    assert router.percent == 100.0


async def test_canary_custom_score_fn_used():
    primary = MockProvider(default_text="x")
    candidate = MockProvider(default_text="x")
    router = CanaryRouter(
        primary, candidate, percent=50.0, min_samples=2, score_fn=lambda r: 0.42
    )
    for _ in range(8):
        await router.generate(_req())
    st = router.state()
    assert st.primary_mean == 0.42
    assert st.candidate_mean == 0.42


async def test_canary_score_fn_exception_scores_zero():
    def boom(_resp):
        raise ValueError("bad signal")

    primary = MockProvider(default_text="x")
    candidate = MockProvider(default_text="x")
    router = CanaryRouter(primary, candidate, percent=0.0, min_samples=2, score_fn=boom)
    for _ in range(3):
        await router.generate(_req())
    # All primary, score_fn raises -> 0.0 each (290-293).
    assert router.state().primary_mean == 0.0


async def test_canary_rollback_swallows_registry_error():
    primary = MockProvider(default_text="good")
    candidate = MockProvider(responder=lambda r: ModelResponse(text="", finish_reason="error"))
    registry = _RecordingRegistry(fail=True)
    router = CanaryRouter(
        primary,
        candidate,
        percent=50.0,
        min_samples=3,
        prompt_registry=registry,
        prompt_name="p",
    )
    for _ in range(20):
        await router.generate(_req())
    # Registry raised but rollback still completed (328-329 except branch).
    assert router.rolled_back is True
    assert registry.rolled_back == []


async def test_canary_stream_routes_and_observes():
    primary = MockProvider(default_text="primary stream")
    candidate = MockProvider(default_text="candidate stream")
    router = CanaryRouter(primary, candidate, percent=100.0, min_samples=2)

    out = []
    async for ev in router.stream(_req()):
        if ev.type == "text_delta":
            out.append(ev.text)
    # 100% -> candidate served; concatenated deltas reproduce candidate text.
    assert "".join(out) == "candidate stream"
    st = router.state()
    assert st.calls == 1
    assert st.candidate_n == 1
    assert st.primary_n == 0


async def test_canary_stream_primary_arm():
    primary = MockProvider(default_text="from primary")
    candidate = MockProvider(default_text="from candidate")
    router = CanaryRouter(primary, candidate, percent=0.0)

    out = []
    async for ev in router.stream(_req()):
        if ev.type == "text_delta":
            out.append(ev.text)
    assert "".join(out) == "from primary"
    assert router.state().primary_n == 1
    assert router.state().candidate_n == 0


async def test_canary_capabilities_and_aclose_delegate():
    primary = MockProvider()
    candidate = MockProvider()
    router = CanaryRouter(primary, candidate)
    assert router.capabilities("m") == primary.capabilities("m")
    # aclose must not raise and closes both arms (388-389).
    await router.aclose()


# --------------------------------------------------------------------------- #
# Remaining branch edges: no terminal 'done', rollback without prompt registry,
# block=True dispatch helper.
# --------------------------------------------------------------------------- #


async def test_shadow_stream_without_done_records_nothing():
    # Primary stream never emits 'done' -> final stays None -> 179->exit.
    sp = ShadowProvider(_NoDoneProvider(), MockProvider(default_text="c"), block=True)
    chunks = [ev.text async for ev in sp.stream(_req()) if ev.type == "text_delta"]
    assert chunks == ["partial"]
    assert len(sp.observations) == 0


async def test_canary_stream_without_done_does_not_observe():
    # Provider never emits 'done' -> final None -> 369->exit, no score recorded.
    router = CanaryRouter(_NoDoneProvider(), MockProvider(), percent=0.0, min_samples=2)
    chunks = [ev.text async for ev in router.stream(_req()) if ev.type == "text_delta"]
    assert chunks == ["partial"]
    st = router.state()
    assert st.calls == 1
    assert st.primary_n == 0
    assert st.candidate_n == 0


async def test_canary_rollback_without_prompt_registry():
    # No prompt_registry -> the registry branch is skipped (325->330) but the
    # rollback itself still completes.
    primary = MockProvider(default_text="good")
    candidate = MockProvider(responder=lambda r: ModelResponse(text="", finish_reason="error"))
    router = CanaryRouter(primary, candidate, percent=50.0, min_samples=3)
    for _ in range(20):
        await router.generate(_req())
    assert router.rolled_back is True
    assert router.rollback_reason != ""


async def test_shadow_dispatch_candidate_block_uses_ensure_future():
    # Directly exercise the block=True arm of _dispatch_candidate (155): it
    # registers an awaitable future that drain() then completes.
    candidate = MockProvider(default_text="cand")
    sp = ShadowProvider(MockProvider(default_text="prim"), candidate, block=True)
    obs = ShadowObservation(primary_text="prim")
    sp.observations.append(obs)
    sp._dispatch_candidate(obs, _req(), "prim")
    assert len(sp._pending) == 1
    await sp.drain()
    assert obs.candidate_text == "cand"
