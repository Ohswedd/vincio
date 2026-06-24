"""Coverage tests for vincio.observability.record_replay.

Targets the uncovered edges of the record-replay debugger: the recording and
replay *providers* (streaming reconstruction, capabilities/embed fallbacks),
the tool/retrieval replay shims (hit / miss / fallback), branch re-execution
that diverges, file/store portability, and the small pure helpers. Everything
is deterministic and offline; model interaction uses MockProvider or tiny real
ModelProvider subclasses — no unittest.mock.
"""

from __future__ import annotations

import math

import pytest

from vincio import ContextApp, VincioConfig
from vincio.core.errors import ReplayDivergenceError
from vincio.core.types import (
    Message,
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolCallRequest,
    ToolResult,
    UserInput,
)
from vincio.observability.record_replay import (
    BranchEdit,
    Divergence,
    RecordedEdge,
    Recorder,
    Recording,
    Replayer,
    ReplayProvider,
    _deterministic_embed,
    _EdgeSink,
    _input_text,
    _RecordingProvider,
    _ReplayBook,
    _retrieval_key,
    _tool_key,
    content_hash,
)
from vincio.providers.mock import MockProvider


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


def _app(text: str = "RECORDED ANSWER") -> ContextApp:
    return ContextApp(config=_config(), provider=MockProvider(default_text=text))


# ---------------------------------------------------------------------------
# Pure helpers / data-model methods (110, 320, 358, 566, 573)
# ---------------------------------------------------------------------------


def test_retrieval_key_depends_on_query_and_params():
    base = _retrieval_key("refunds?", {"top_k": 5})
    assert _retrieval_key("refunds?", {"top_k": 5}) == base  # stable
    assert _retrieval_key("other?", {"top_k": 5}) != base  # query matters
    assert _retrieval_key("refunds?", {"top_k": 9}) != base  # params matter


def test_tool_key_is_argument_sensitive():
    a = _tool_key("lookup", {"q": "x"})
    assert _tool_key("lookup", {"q": "x"}) == a
    assert _tool_key("lookup", {"q": "y"}) != a
    assert _tool_key("other", {"q": "x"}) != a


def test_input_text_unwraps_userinput_and_passes_str():
    assert _input_text("plain string") == "plain string"
    assert _input_text(UserInput(text="wrapped")) == "wrapped"
    # A UserInput with no text falls back to the empty string, not None.
    assert _input_text(UserInput(text=None)) == ""


def test_replay_result_summary_counts_divergences():
    from vincio.observability.record_replay import ReplayResult

    rr = ReplayResult(
        recording_id="rec_1",
        faithful=False,
        output_identical=False,
        recorded_output="a",
        replayed_output="b",
        replayed_status="succeeded",
        served_from_recording=3,
        divergences=[Divergence(kind="tool_call", key="k", detail="d")],
    )
    assert rr.summary() == {
        "recording_id": "rec_1",
        "faithful": False,
        "output_identical": False,
        "served_from_recording": 3,
        "divergences": 1,
    }


def test_branch_result_summary_reports_counters():
    from vincio.observability.record_replay import BranchResult

    br = BranchResult(
        recording_id="rec_2",
        branched_input="q",
        output="o",
        status="succeeded",
        served_from_recording=2,
        reexecuted=1,
        edits_applied=4,
    )
    assert br.summary() == {
        "recording_id": "rec_2",
        "served_from_recording": 2,
        "reexecuted": 1,
        "edits_applied": 4,
    }


# ---------------------------------------------------------------------------
# _RecordingProvider streaming + passthrough (412-440)
# ---------------------------------------------------------------------------


async def test_recording_provider_stream_captures_terminal_response():
    """Streaming through the recorder accumulates deltas and records the terminal
    response (the MockProvider emits a `done` event carrying it)."""
    sink = _EdgeSink()
    rec = _RecordingProvider(MockProvider(default_text="STREAMED ANSWER HERE"), sink)
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])

    deltas = [
        e.text async for e in rec.stream(request) if e.type == "text_delta" and e.text
    ]
    assert "".join(deltas) == "STREAMED ANSWER HERE"
    # exactly one model_call edge captured, valued with the terminal response.
    edges = [e for e in sink.edges if e.kind == "model_call"]
    assert len(edges) == 1
    assert edges[0].value["text"] == "STREAMED ANSWER HERE"


class _NoDoneProvider(MockProvider):
    """A provider whose stream emits deltas + a tool call but never a `done`
    event, forcing the recorder to reconstruct the terminal response itself."""

    async def stream(self, request: ModelRequest):
        yield ModelEvent(type="text_delta", text="ab")
        yield ModelEvent(type="text_delta", text="cd")
        yield ModelEvent(
            type="tool_call_delta",
            tool_call=ToolCallRequest(name="lookup", arguments={"q": "x"}),
        )


async def test_recording_provider_stream_reconstructs_when_no_done_event():
    sink = _EdgeSink()
    rec = _RecordingProvider(_NoDoneProvider(default_text="ignored"), sink)
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])

    events = [e async for e in rec.stream(request)]
    assert [e.type for e in events] == ["text_delta", "text_delta", "tool_call_delta"]
    edges = [e for e in sink.edges if e.kind == "model_call"]
    assert len(edges) == 1
    # text was reconstructed by concatenating the deltas, tool call preserved.
    assert edges[0].value["text"] == "abcd"
    assert edges[0].value["tool_calls"][0]["name"] == "lookup"


async def test_recording_provider_records_capabilities_and_delegates_rest():
    sink = _EdgeSink()
    rec = _RecordingProvider(MockProvider(default_text="x"), sink)

    caps = rec.capabilities("mock-1")
    assert caps.tool_calling is True
    assert [e.kind for e in sink.edges] == ["capabilities"]
    assert sink.edges[0].key == "mock-1"

    # embed / list_models / aclose delegate to the inner provider without recording.
    vectors = await rec.embed(["hello world"])
    assert len(vectors) == 1 and len(vectors[0]) == 64
    assert await rec.list_models() == await MockProvider(default_text="x").list_models()
    assert await rec.aclose() is None


# ---------------------------------------------------------------------------
# ReplayProvider streaming, fallback, capabilities, embed (667-713)
# ---------------------------------------------------------------------------


async def test_replay_provider_stream_emits_tool_calls_usage_done():
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])
    response = ModelResponse(
        model="mock-1",
        text="answer with a tool",
        tool_calls=[ToolCallRequest(name="lookup", arguments={"q": "x"})],
    )
    recording = Recording(
        edges=[RecordedEdge.of("model_call", 0, request.hash, response.model_dump(mode="json"))]
    )
    provider = ReplayProvider(_ReplayBook(recording), [], {}, on_miss="error")

    events = [e async for e in provider.stream(request)]
    types = [e.type for e in events]
    assert types[-2:] == ["usage", "done"]
    assert any(e.type == "tool_call_delta" and e.tool_call.name == "lookup" for e in events)
    text = "".join(e.text for e in events if e.type == "text_delta")
    assert text == "answer with a tool"


async def test_replay_provider_stream_falls_back_on_miss():
    """A streamed miss in fallback mode delegates the whole stream to the
    fallback provider instead of raising."""
    book = _ReplayBook(Recording())  # empty: every lookup misses
    divergences: list[Divergence] = []
    fallback = MockProvider(default_text="FALLBACK STREAM")
    provider = ReplayProvider(book, divergences, {}, on_miss="fallback", fallback=fallback)
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])

    deltas = [
        e.text async for e in provider.stream(request) if e.type == "text_delta" and e.text
    ]
    assert "".join(deltas) == "FALLBACK STREAM"
    assert divergences and divergences[0].kind == "model_call"


async def test_replay_provider_stream_reraises_in_strict_mode():
    book = _ReplayBook(Recording())
    provider = ReplayProvider(book, [], {}, on_miss="error")
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])
    with pytest.raises(ReplayDivergenceError, match="no recorded model response"):
        [e async for e in provider.stream(request)]


async def test_replay_provider_generate_fallback_increments_reexecuted():
    book = _ReplayBook(Recording())
    divergences: list[Divergence] = []
    counters: dict[str, int] = {}
    fallback = MockProvider(default_text="LIVE FALLBACK")
    provider = ReplayProvider(book, divergences, counters, on_miss="fallback", fallback=fallback)
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])

    resp = await provider.generate(request)
    assert resp.text == "LIVE FALLBACK"
    assert counters["reexecuted"] == 1
    assert divergences[0].key == request.hash


async def test_replay_provider_capabilities_recorded_then_fallback_then_default():
    request_caps = ModelCapabilities(tool_calling=True, vision=True)
    recording = Recording(
        edges=[RecordedEdge.of("capabilities", 0, "mock-1", request_caps.model_dump(mode="json"))]
    )
    # recorded model -> served verbatim.
    served = ReplayProvider(_ReplayBook(recording), [], {}, on_miss="error")
    caps = served.capabilities("mock-1")
    assert caps.vision is True and caps.tool_calling is True

    # unknown model with a fallback -> fallback's capabilities.
    with_fb = ReplayProvider(
        _ReplayBook(recording), [], {}, on_miss="fallback", fallback=MockProvider(default_text="x")
    )
    assert with_fb.capabilities("unknown-model").tool_calling is True

    # unknown model, no fallback -> empty default capabilities.
    bare = ReplayProvider(_ReplayBook(Recording()), [], {}, on_miss="error")
    assert bare.capabilities("anything") == ModelCapabilities()


async def test_replay_provider_embed_uses_fallback_when_present():
    fallback = MockProvider(default_text="x")
    provider = ReplayProvider(
        _ReplayBook(Recording()), [], {}, on_miss="fallback", fallback=fallback
    )
    out = await provider.embed(["hello"])
    assert out == await fallback.embed(["hello"])


async def test_replay_provider_embed_deterministic_without_fallback():
    provider = ReplayProvider(_ReplayBook(Recording()), [], {}, on_miss="error")
    out = await provider.embed(["hello world", ""])
    assert len(out) == 2 and all(len(v) == 64 for v in out)
    # the empty string yields a zero vector (no tokens to hash).
    assert out[1] == [0.0] * 64
    # nonzero vector is L2-normalized.
    assert math.isclose(math.sqrt(sum(v * v for v in out[0])), 1.0, rel_tol=1e-9)


async def test_replay_provider_list_models_and_aclose():
    provider = ReplayProvider(_ReplayBook(Recording()), [], {}, on_miss="error")
    assert await provider.list_models() == []
    assert await provider.aclose() is None


def test_deterministic_embed_is_stable_and_normalized():
    a = _deterministic_embed("refund policy window")
    assert a == _deterministic_embed("refund policy window")
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-9)
    # empty text -> the all-zero vector (norm defended to 1.0, not div-by-zero).
    assert _deterministic_embed("") == [0.0] * 64


# ---------------------------------------------------------------------------
# Replayer tool/retrieval shims via _instruments (751-791)
# ---------------------------------------------------------------------------


def _book_and_shims(recording: Recording, *, on_miss):
    book = _ReplayBook(recording)
    divergences: list[Divergence] = []
    counters: dict[str, int] = {}
    replayer = Replayer.__new__(Replayer)
    _pf, tool_shim, retrieval_shim = replayer._instruments(
        book, divergences, counters, on_miss=on_miss, fallback=None
    )
    return book, divergences, counters, tool_shim, retrieval_shim


async def test_replay_tool_shim_serves_hit_then_diverges_on_miss():
    call = ToolCall(tool_name="lookup", arguments={"q": "policy"})
    recorded = ToolResult(call_id="c1", tool_name="lookup", output="RECORDED OUTPUT")
    recording = Recording(
        edges=[
            RecordedEdge.of(
                "tool_call",
                0,
                _tool_key("lookup", {"q": "policy"}),
                recorded.model_dump(mode="json"),
            )
        ]
    )
    _book, divergences, counters, tool_shim, _rs = _book_and_shims(recording, on_miss="error")

    async def _orig(c, *a, **k):  # noqa: ARG001 - never reached on a hit
        raise AssertionError("original execute must not run on a hit")

    execute = tool_shim(_orig)
    served = await execute(call)
    assert served.output == "RECORDED OUTPUT"
    assert counters["served"] == 1

    # second call to the same identity: queue exhausted -> divergence, strict raise.
    with pytest.raises(ReplayDivergenceError, match="no recorded output for tool 'lookup'"):
        await execute(call)
    assert divergences and divergences[0].kind == "tool_call"


async def test_replay_tool_shim_miss_falls_back_to_original():
    call = ToolCall(tool_name="lookup", arguments={"q": "x"})
    _book, divergences, counters, tool_shim, _rs = _book_and_shims(Recording(), on_miss="fallback")

    async def _orig(c, *a, **k):  # noqa: ARG001
        return ToolResult(call_id="c", tool_name=c.tool_name, output="LIVE RERUN")

    execute = tool_shim(_orig)
    result = await execute(call)
    assert result.output == "LIVE RERUN"
    assert counters["reexecuted"] == 1
    assert divergences[0].key == _tool_key("lookup", {"q": "x"})


async def test_replay_retrieval_shim_serves_hit_then_diverges_on_miss():
    from vincio.retrieval.engine import RetrievalResult

    recorded = RetrievalResult(evidence=[], metadata={"served": True})
    key = _retrieval_key("refunds?", {"top_k": 3})
    recording = Recording(
        edges=[RecordedEdge.of("retrieval", 0, key, recorded.model_dump(mode="json"))]
    )
    _book, divergences, counters, _ts, retrieval_shim = _book_and_shims(recording, on_miss="error")

    async def _orig(q, **k):  # noqa: ARG001
        raise AssertionError("original retrieve must not run on a hit")

    retrieve = retrieval_shim(_orig)
    served = await retrieve("refunds?", top_k=3)
    assert served.metadata == {"served": True}
    assert counters["served"] == 1

    with pytest.raises(ReplayDivergenceError, match="no recorded retrieval for this query"):
        await retrieve("refunds?", top_k=3)
    assert divergences and divergences[0].kind == "retrieval"


async def test_replay_retrieval_shim_miss_falls_back_to_original():
    from vincio.retrieval.engine import RetrievalResult

    _book, divergences, counters, _ts, retrieval_shim = _book_and_shims(
        Recording(), on_miss="fallback"
    )

    async def _orig(q, **k):  # noqa: ARG001
        return RetrievalResult(evidence=[], metadata={"live": q})

    retrieve = retrieval_shim(_orig)
    out = await retrieve("new query", top_k=1)
    assert out.metadata == {"live": "new query"}
    assert counters["reexecuted"] == 1
    assert divergences[0].key == _retrieval_key("new query", {"top_k": 1})


# ---------------------------------------------------------------------------
# Recorder tool/retrieval capture shims (533-542) + seed (514)
# ---------------------------------------------------------------------------


async def test_recorder_captures_tool_edge_and_seed_from_config():
    """A recorded run that calls a tool produces a tool_call edge keyed by
    name+args, and the seed edge reflects the run config's seed."""
    script = [
        {"tool_call": {"name": "lookup", "arguments": {"q": "policy"}}},
        "FINAL",
    ]
    app = ContextApp(config=_config(), provider=MockProvider(script=list(script)))

    @app.tool_registry.register(name="lookup")
    def lookup(q: str) -> str:
        return "TOOL OUT"

    app.enabled_tools.append("lookup")

    from vincio.core.types import RunConfig

    run_config = RunConfig(seed=4242)
    _result, recording = await Recorder(app).record("refund policy?", config=run_config)

    assert len(recording.tool_calls) == 1
    assert recording.tool_calls[0].value["output"] == "TOOL OUT"
    seed_edge = recording.edges_of("seed")[0]
    assert seed_edge.value == 4242


# ---------------------------------------------------------------------------
# Replay error-catch path + branch divergence (824-828, branch reexec)
# ---------------------------------------------------------------------------


async def test_replay_nonstrict_serves_recording_without_divergence():
    """Non-strict replay of a clean recording is still faithful: no edge misses,
    so the fallback path is never taken and the recorded output is reproduced."""
    _result, recording = await Recorder(_app("RECORDED ANSWER")).record("q?")
    out = await Replayer(_app("LIVE WRONG")).replay(recording, strict=False)
    assert out.replayed_output == "RECORDED ANSWER"
    assert out.output_identical
    assert not out.divergences


async def test_branch_new_input_reexecutes_and_reports_trajectory_empty():
    """Branching with a fresh input re-executes from the start against the
    fallback; nothing is served from the recording."""
    _result, recording = await Recorder(_app("ORIG")).record("original?")
    branch = await Replayer(_app()).branch(
        recording,
        input="totally different question",
        fallback=MockProvider(default_text="REBRANCHED"),
    )
    assert branch.branched_input == "totally different question"
    assert branch.output == "REBRANCHED"
    assert branch.served_from_recording == 0
    assert branch.reexecuted >= 1
    assert branch.edits_applied == 0


# ---------------------------------------------------------------------------
# Recording.render_text + content_hash + RecordedEdge integrity
# ---------------------------------------------------------------------------


def test_recorded_edge_hash_matches_detects_tamper():
    edge = RecordedEdge.of("model_call", 0, "k", {"text": "ok"})
    assert edge.hash_matches() is True
    edge.value = {"text": "tampered"}
    assert edge.hash_matches() is False


def test_content_hash_is_16_hex_and_stable():
    h = content_hash("hello")
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)
    assert content_hash("hello") == h
    assert content_hash("world") != h


# ---------------------------------------------------------------------------
# Recording inspection surface + verification (193, 198-205, 222-250)
# ---------------------------------------------------------------------------


async def test_fidelity_report_and_verify_on_clean_recording():
    _result, recording = await Recorder(_app("CLEAN")).record("q?")
    report = recording.fidelity_report()
    assert report["ok"] is True
    assert report["digest_ok"] is True
    assert report["corrupt_edges"] == []
    assert report["expected_digest"] == report["actual_digest"]
    assert report["edges"] == len(recording.edges)
    assert recording.verify() is True


async def test_fidelity_report_flags_digest_mismatch():
    _result, recording = await Recorder(_app("CLEAN")).record("q?")
    # Tamper with the stored digest only: edges still hash-match, but digest_ok flips.
    recording.fidelity_digest = "0" * 16
    report = recording.fidelity_report()
    assert report["digest_ok"] is False
    assert report["corrupt_edges"] == []  # payloads themselves untouched
    assert report["ok"] is False
    assert recording.verify() is False


def test_render_text_and_kind_accessors():
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])
    response = ModelResponse(model="mock-1", text="ans")
    from vincio.retrieval.engine import RetrievalResult

    recording = Recording(
        recording_id="rec_show",
        app_name="demo",
        status="succeeded",
        input="the input question",
        output_text="the output answer",
        fidelity_digest="deadbeef",
        edges=[
            RecordedEdge.of("model_call", 0, request.hash, response.model_dump(mode="json")),
            RecordedEdge.of(
                "retrieval",
                1,
                _retrieval_key("q", {}),
                RetrievalResult(evidence=[]).model_dump(mode="json"),
            ),
            RecordedEdge.of("clock", 2, "run", "2026-01-01T00:00:00+00:00"),
        ],
    )
    assert len(recording.model_calls) == 1
    assert len(recording.retrievals) == 1
    assert recording.tool_calls == []
    # steps() with no trace -> empty list.
    assert recording.steps() == []

    text = recording.render_text()
    assert "recording rec_show" in text
    assert "app=demo" in text
    assert "status=succeeded" in text
    assert "model=1 tool=0 retrieval=1" in text
    assert "digest: deadbeef" in text
    # only replay-edges (model_call/retrieval) get an indented per-edge line; the
    # clock edge is summarized but not listed.
    assert "model_call" in text and "retrieval" in text
    assert "clock" not in text.split("digest:")[1]


async def test_steps_returns_span_tree_when_trace_present():
    _result, recording = await Recorder(_app("with-trace")).record("q?")
    steps = recording.steps()
    assert isinstance(steps, list) and steps  # trace was captured
    assert all("type" in node for node in steps)


# ---------------------------------------------------------------------------
# Portability: put/from_store/save/load + record(store=) (260, 265-282, 287, 566)
# ---------------------------------------------------------------------------


async def test_record_writes_to_store_when_given():
    from vincio.context.evidence_store import InMemoryEvidenceStore

    store = InMemoryEvidenceStore()
    _result, recording = await Recorder(_app("STORED")).record("q?", store=store)
    # record(store=) put the recording; we can load it straight back by address.
    address = recording.put(store)
    loaded = Recording.from_store(store, address)
    assert loaded.fidelity_digest == recording.fidelity_digest
    assert loaded.output_text == "STORED"


async def test_save_creates_parent_dirs_and_load_roundtrips(tmp_path):
    _result, recording = await Recorder(_app("FILED")).record("q?")
    nested = tmp_path / "deep" / "nested" / "rec.json"
    written = recording.save(nested)
    assert written.exists()
    loaded = Recording.load(written)
    assert loaded.recording_id == recording.recording_id
    assert loaded.output_text == "FILED"
    assert loaded.verify()


# ---------------------------------------------------------------------------
# Branch edits applied to matching edges (876-880)
# ---------------------------------------------------------------------------


async def test_branch_edit_to_model_edge_changes_served_response():
    """Editing the recorded model response and replaying with no input change
    serves the *edited* answer (the edge identity is unchanged, so it is still a
    hit, but its payload was rewritten)."""
    _result, recording = await Recorder(_app("ORIGINAL ANSWER")).record("q?")
    model_key = recording.model_calls[0].key
    edited = ModelResponse(model="mock-1", text="EDITED ANSWER").model_dump(mode="json")

    branch = await Replayer(_app("LIVE WRONG")).branch(
        recording,
        edits=[BranchEdit(kind="model_call", key=model_key, value=edited)],
    )
    assert branch.edits_applied == 1
    assert branch.output == "EDITED ANSWER"
    assert branch.served_from_recording >= 1


async def test_branch_edit_with_nonmatching_key_applies_nothing():
    _result, recording = await Recorder(_app("ANSWER")).record("q?")
    branch = await Replayer(_app("ANSWER")).branch(
        recording,
        edits=[BranchEdit(kind="tool_call", key="no-such-key", value={"x": 1})],
    )
    assert branch.edits_applied == 0
    # nothing edited, no input change -> the recorded answer is reproduced.
    assert branch.output == "ANSWER"


async def test_from_store_missing_address_raises_with_address_detail():
    from vincio.context.evidence_store import InMemoryEvidenceStore

    store = InMemoryEvidenceStore()
    with pytest.raises(ReplayDivergenceError, match="no recording at content address"):
        Recording.from_store(store, "0000000000000000")


async def test_save_to_bare_filename_in_cwd(tmp_cwd):
    """A bare relative filename has no real parent dir, so the mkdir branch is
    skipped and the file is written into the cwd."""
    _result, recording = await Recorder(_app("BARE")).record("q?")
    written = recording.save("bare_rec.json")
    assert written.name == "bare_rec.json"
    assert (tmp_cwd / "bare_rec.json").exists()
    assert Recording.load(tmp_cwd / "bare_rec.json").output_text == "BARE"


# ---------------------------------------------------------------------------
# Recorder retrieval capture + _instrument retrieval branch (467, 476, 533-542)
# ---------------------------------------------------------------------------


class _ToolRuntime:
    async def execute(self, call, *args, **kwargs):  # noqa: ARG002 - placeholder
        return ToolResult(call_id="c", tool_name=call.tool_name, output="x")


class _Retrieval:
    """A real retrieval object whose retrieve() returns a deterministic result;
    its `retrieve` attribute is swapped by the record/replay shims."""

    async def retrieve(self, query, **kwargs):  # noqa: ARG002
        from vincio.retrieval.engine import RetrievalResult

        return RetrievalResult(evidence=[], metadata={"q": query})


class _Tracer:
    def __init__(self):
        from vincio.observability.spans import Trace

        self._trace = Trace(id="trace-fake")
        self.app_name = "fake-app"

        class _Exporter:
            def export(self_inner, trace):  # noqa: ANN001, N805
                self_inner.last = trace

        self.exporter = _Exporter()

    def trace(self):
        return self._trace


class _FakeApp:
    """A minimal ContextApp-shaped object that, on arun, performs one retrieval —
    enough to drive the recorder's retrieval capture shim end to end."""

    app_name = "fake-app"

    def __init__(self, provider):
        self._provider = provider
        self.tool_runtime = _ToolRuntime()
        self.retrieval = _Retrieval()
        self.tracer = _Tracer()
        self.response_cache = object()  # nulled out by _instrument, restored after

    def resolve_provider(self, run_config=None):  # noqa: ARG002
        return self._provider

    async def arun(self, user_input, **kwargs):  # noqa: ARG002
        from vincio.core.types import RunResult, RunStatus

        # one retrieval read -> the shim captures a retrieval edge.
        await self.retrieval.retrieve(user_input, top_k=2)
        # publish the trace through the (now-wrapped) exporter so the recording
        # picks up a trace keyed by the result's trace id.
        self.tracer.exporter.export(self.tracer.trace())
        return RunResult(
            status=RunStatus.SUCCEEDED,
            raw_text="FAKE OUTPUT",
            trace_id=self.tracer.trace().id,
        )


async def test_recorder_captures_retrieval_edge_and_restores_app():
    provider = MockProvider(default_text="x")
    app = _FakeApp(provider)
    original_retrieve = app.retrieval.retrieve
    original_cache = app.response_cache

    _result, recording = await Recorder(app).record("find refunds")

    # the retrieval edge was captured with the query+params identity.
    rets = recording.retrievals
    assert len(rets) == 1
    assert rets[0].key == _retrieval_key("find refunds", {"top_k": 2})
    assert rets[0].value["metadata"] == {"q": "find refunds"}
    assert recording.output_text == "FAKE OUTPUT"
    # _instrument restored the swapped retrieve and the response cache on exit.
    assert app.retrieval.retrieve == original_retrieve
    assert app.response_cache is original_cache
