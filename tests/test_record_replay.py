"""Tests for the causal record-replay debugger (vincio.observability.record_replay).

All deterministic and offline: runs are driven by :class:`MockProvider`, recorded
edge-by-edge, then replayed byte-for-byte. The headline guarantees are
byte-faithful replay (the recording, not the live provider, serves the answer)
and divergence detection (changed code is caught against the recording).
"""

from __future__ import annotations

import pytest

from vincio import ContextApp, VincioConfig
from vincio.context.evidence_store import InMemoryEvidenceStore
from vincio.core.errors import ReplayDivergenceError
from vincio.observability import (
    BranchEdit,
    Recorder,
    Recording,
    Replayer,
)
from vincio.observability.record_replay import ReplayProvider, _ReplayBook
from vincio.providers.mock import MockProvider


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


def _app(text: str = "RECORDED ANSWER") -> ContextApp:
    return ContextApp(config=_config(), provider=MockProvider(default_text=text))


def _tool_app(final: str = "FINAL ANSWER") -> ContextApp:
    """An app whose model calls a tool once, then answers from the tool output."""
    script = [
        {"tool_call": {"name": "lookup", "arguments": {"q": "policy"}}},
        final,
    ]
    app = ContextApp(config=_config(), provider=MockProvider(script=list(script)))

    @app.tool_registry.register(name="lookup")
    def lookup(q: str) -> str:
        return "LIVE TOOL OUTPUT"

    app.enabled_tools.append("lookup")
    return app


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


async def test_record_captures_edges_and_digest():
    app = _app()
    result, recording = await Recorder(app).record("What is the refund policy?")

    assert result.status.value == "succeeded"
    assert recording.output_text == "RECORDED ANSWER"
    assert len(recording.model_calls) == 1
    # capabilities + clock + seed + at least one model edge are all captured.
    assert recording.edges_of("capabilities")
    assert recording.edges_of("clock") and recording.edges_of("seed")
    assert recording.fidelity_digest
    assert recording.verify()
    assert recording.trace is not None and recording.trace.id == result.trace_id


async def test_recording_inspection_surface():
    app = _tool_app()
    _, recording = await Recorder(app).record("refund policy?")

    assert len(recording.model_calls) == 2
    assert len(recording.tool_calls) == 1
    steps = recording.steps()
    assert steps and any(s["type"] == "model_call" for s in _flatten(steps))
    text = recording.render_text()
    assert "recording" in text and "model_call" in text


def _flatten(tree):
    for node in tree:
        yield node
        yield from _flatten(node.get("children", []))


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------


async def test_replay_is_byte_faithful_from_recording_not_live():
    """Replay must serve the recorded answer even when the live provider would
    return something different — proving the recording, not the provider, drives
    the run."""
    _, recording = await Recorder(_app("RECORDED ANSWER")).record("question?")

    # A fresh app whose live provider answers "WRONG LIVE": faithful replay must
    # still yield "RECORDED ANSWER".
    replay_app = _app("WRONG LIVE")
    result = await Replayer(replay_app).replay(recording)

    assert result.faithful
    assert result.output_identical
    assert result.replayed_output == "RECORDED ANSWER"
    assert result.served_from_recording >= 1
    assert not result.divergences


async def test_record_replay_is_faithful_with_response_cache_enabled():
    """The recording — not a warm response cache — is the source of truth: even
    with the response cache on, replay serves recorded edges and stays faithful."""
    config = _config()
    config.cache.response_cache = True
    app = ContextApp(config=config, provider=MockProvider(default_text="CACHED-OK"))
    _, recording = await Recorder(app).record("question?")

    replay_config = _config()
    replay_config.cache.response_cache = True
    replay_app = ContextApp(config=replay_config, provider=MockProvider(default_text="WRONG"))
    result = await Replayer(replay_app).replay(recording)
    assert result.faithful
    assert result.replayed_output == "CACHED-OK"


async def test_replay_detects_divergence_when_code_changes():
    """Changing the prompt changes the model request hash, so the recorded edge
    no longer matches — a divergence the debugger reports."""
    _, recording = await Recorder(_app()).record("question?")

    diverged = _app("WRONG LIVE")
    diverged.configure(objective="A completely different objective that rewrites the prompt")
    result = await Replayer(diverged).replay(recording)

    assert not result.faithful
    assert result.divergences
    assert result.divergences[0].kind == "model_call"


async def test_replay_with_tools_is_faithful():
    _, recording = await Recorder(_tool_app("FINAL ANSWER")).record("refund policy?")

    # Live tool returns something else and the script would too; faithful replay
    # serves recorded model + tool edges, reproducing the recorded final answer.
    replay_app = _tool_app("DIFFERENT")
    result = await Replayer(replay_app).replay(recording)

    assert result.faithful
    assert result.replayed_output == "FINAL ANSWER"
    # both model calls and the tool call are served from the recording.
    assert result.served_from_recording >= 3


# ---------------------------------------------------------------------------
# Branch-and-edit
# ---------------------------------------------------------------------------


async def test_branch_edit_tool_reexecutes_only_suffix():
    """Editing a recorded tool output keeps the unchanged prefix (the
    decide-to-call-tool model step) served from the recording and re-executes
    only the affected suffix against the fallback."""
    _, recording = await Recorder(_tool_app("FINAL ANSWER")).record("refund policy?")
    tool_key = recording.tool_calls[0].key

    branch = await Replayer(_tool_app()).branch(
        recording,
        edits=[
            BranchEdit(
                kind="tool_call",
                key=tool_key,
                value={
                    "call_id": "x",
                    "tool_name": "lookup",
                    "status": "ok",
                    "output": "EDITED TOOL OUTPUT",
                },
            )
        ],
        fallback=MockProvider(default_text="BRANCHED ANSWER from edited tool"),
    )

    assert branch.edits_applied == 1
    assert branch.output == "BRANCHED ANSWER from edited tool"
    # the first model call + the (edited) tool call are served from the recording;
    # the second model call re-executes against the fallback.
    assert branch.served_from_recording >= 2
    assert branch.reexecuted >= 1


async def test_branch_changed_input_reexecutes_from_start():
    _, recording = await Recorder(_app("RECORDED ANSWER")).record("original question?")

    branch = await Replayer(_app()).branch(
        recording,
        input="a completely different question",
        fallback=MockProvider(default_text="NEW ANSWER"),
    )

    assert branch.served_from_recording == 0
    assert branch.reexecuted >= 1
    assert branch.output == "NEW ANSWER"


async def test_branch_does_not_mutate_original_recording():
    _, recording = await Recorder(_tool_app()).record("q?")
    digest_before = recording.fidelity_digest
    tool_key = recording.tool_calls[0].key
    await Replayer(_tool_app()).branch(
        recording,
        edits=[BranchEdit(kind="tool_call", key=tool_key, value={"call_id": "x", "tool_name": "lookup", "status": "ok", "output": "Z"})],
        fallback=MockProvider(default_text="B"),
    )
    assert recording.fidelity_digest == digest_before
    assert recording.tool_calls[0].value["output"] != "Z"


# ---------------------------------------------------------------------------
# Portability and verification
# ---------------------------------------------------------------------------


async def test_content_addressed_store_roundtrip():
    _, recording = await Recorder(_app()).record("question?")
    store = InMemoryEvidenceStore()
    address = recording.put(store)

    loaded = Recording.from_store(store, address)
    assert loaded.fidelity_digest == recording.fidelity_digest
    assert loaded.verify()
    # A round-tripped recording still replays faithfully.
    result = await Replayer(_app("WRONG")).replay(loaded)
    assert result.faithful


async def test_from_store_missing_address_raises():
    store = InMemoryEvidenceStore()
    with pytest.raises(ReplayDivergenceError):
        Recording.from_store(store, "deadbeefdeadbeef")


async def test_save_and_load_file_roundtrip(tmp_path):
    _, recording = await Recorder(_app()).record("question?")
    path = recording.save(tmp_path / "rec.json")
    loaded = Recording.load(path)
    assert loaded.fidelity_digest == recording.fidelity_digest
    assert loaded.verify()


async def test_tampered_recording_fails_verification():
    _, recording = await Recorder(_app()).record("question?")
    # Corrupt a payload without updating its content address.
    recording.model_calls[0].value = {"text": "tampered"}
    report = recording.fidelity_report()
    assert not report["ok"]
    assert report["corrupt_edges"]
    assert not recording.verify()


# ---------------------------------------------------------------------------
# ReplayProvider unit behavior
# ---------------------------------------------------------------------------


async def test_replay_provider_streaming_reconstructs_response():
    from vincio.core.types import Message, ModelRequest, ModelResponse
    from vincio.observability.record_replay import RecordedEdge

    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])
    response = ModelResponse(model="mock-1", text="a recorded answer streamed back in deltas")
    recording = Recording(
        edges=[RecordedEdge.of("model_call", 0, request.hash, response.model_dump(mode="json"))]
    )
    provider = ReplayProvider(_ReplayBook(recording), [], {}, on_miss="error")

    chunks = [event.text async for event in provider.stream(request) if event.type == "text_delta"]
    assert "".join(chunks) == response.text


async def test_replay_provider_miss_raises_in_strict_mode():
    from vincio.core.types import Message, ModelRequest

    book = _ReplayBook(Recording())  # empty recording -> every lookup misses
    divergences: list = []
    provider = ReplayProvider(book, divergences, {}, on_miss="error")
    request = ModelRequest(model="mock-1", messages=[Message(role="user", content="hi")])
    with pytest.raises(ReplayDivergenceError):
        await provider.generate(request)
    assert divergences and divergences[0].kind == "model_call"
