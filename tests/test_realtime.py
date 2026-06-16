"""1.5 voice/realtime module: session lifecycle, VAD, interruption, in-session
tool dispatch through the permissioned runtime, wire-event translation, and
missing-dependency handling. All deterministic and offline."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from vincio import ContextApp
from vincio.core.errors import VincioError
from vincio.realtime import (
    GeminiLiveBackend,
    InProcessRealtimeBackend,
    OpenAIRealtimeBackend,
    RealtimeConfig,
    RealtimeEvent,
    RealtimeSession,
    RealtimeToolCall,
    VADConfig,
    connect_realtime,
)


async def _drain(session: RealtimeSession, *, stop_on: str, timeout: float = 2.0) -> list[str]:
    types: list[str] = []

    async def consume() -> None:
        async for event in session.events():
            types.append(event.type)
            if event.type == stop_on:
                break

    await asyncio.wait_for(consume(), timeout=timeout)
    return types


@pytest.mark.asyncio
async def test_session_basic_turn_lifecycle():
    session = RealtimeSession()
    async with session:
        await session.send_text("hello")
        await session.commit()
        types = await _drain(session, stop_on="turn.end")
    assert types == [
        "session.started",
        "input.transcript",
        "turn.start",
        "response.text",
        "response.audio",
        "response.done",
        "turn.end",
    ]


@pytest.mark.asyncio
async def test_send_before_connect_raises():
    session = RealtimeSession()
    with pytest.raises(VincioError):
        await session.send_text("hi")


@pytest.mark.asyncio
async def test_vad_turn_detection_from_audio_energy():
    session = RealtimeSession(config=RealtimeConfig(vad=VADConfig(threshold=0.5)))
    async with session:
        await session.send_audio(bytes([200] * 32))  # speech
        await session.send_audio(bytes([1] * 32))  # silence -> ends the turn
        types = await _drain(session, stop_on="turn.end")
    assert "vad.speech_start" in types
    assert "vad.speech_stop" in types
    assert "turn.start" in types and "turn.end" in types


@pytest.mark.asyncio
async def test_vad_disabled_emits_no_turn():
    backend = InProcessRealtimeBackend()
    session = RealtimeSession(backend, config=RealtimeConfig(vad=VADConfig(enabled=False)))
    async with session:
        await session.send_audio(bytes([200] * 32))
        # No VAD => no events queued beyond session.started.
        event = await asyncio.wait_for(backend._queue.get(), timeout=1.0)
        assert event.type == "session.started"
        assert backend._queue.empty()


@pytest.mark.asyncio
async def test_interruption_stops_response_stream():
    def script(text, config):
        return [RealtimeEvent(type="response.text", text=f"c{i}") for i in range(6)]

    session = RealtimeSession(InProcessRealtimeBackend(script=script))
    types: list[str] = []
    async with session:
        await session.send_text("go")
        await session.commit()

        async def consume() -> None:
            async for event in session.events():
                types.append(event.type)
                if event.type == "response.text" and types.count("response.text") == 1:
                    await session.interrupt()
                if event.type == "turn.end":
                    break

        await asyncio.wait_for(consume(), timeout=2.0)
    assert "interrupted" in types
    assert types.count("response.text") < 6  # barge-in truncated the response


@pytest.mark.asyncio
async def test_in_session_tool_dispatch():
    def script(text, config):
        return [
            RealtimeEvent(
                type="tool_call",
                tool_call=RealtimeToolCall(call_id="c1", name="lookup", arguments={"q": "paris"}),
            )
        ]

    seen_args = {}

    async def dispatch(name: str, arguments: dict) -> dict:
        seen_args["name"] = name
        seen_args["arguments"] = arguments
        return {"temp_c": 17}

    session = RealtimeSession(InProcessRealtimeBackend(script=script), tool_dispatcher=dispatch)
    results = []
    async with session:
        await session.send_text("weather?")
        await session.commit()

        async def consume() -> None:
            async for event in session.events():
                if event.type == "tool_result":
                    results.append(event.data)
                if event.type == "response.done":
                    break

        await asyncio.wait_for(consume(), timeout=2.0)
    assert seen_args == {"name": "lookup", "arguments": {"q": "paris"}}
    assert results[0]["result"] == {"temp_c": 17}


@pytest.mark.asyncio
async def test_tool_dispatch_error_surfaces_event():
    def script(text, config):
        return [RealtimeEvent(type="tool_call", tool_call=RealtimeToolCall(call_id="c1", name="boom"))]

    async def dispatch(name: str, arguments: dict) -> dict:
        raise RuntimeError("tool failed")

    session = RealtimeSession(InProcessRealtimeBackend(script=script), tool_dispatcher=dispatch)
    errors = []
    async with session:
        await session.send_text("x")
        await session.commit()

        async def consume() -> None:
            async for event in session.events():
                if event.type == "error":
                    errors.append(event.text)
                if event.type == "response.done":
                    break

        await asyncio.wait_for(consume(), timeout=2.0)
    assert errors and "tool failed" in errors[0]


@pytest.mark.asyncio
async def test_app_realtime_session_routes_tools_through_runtime():
    """app.realtime_session dispatches tool calls through the permissioned,
    audited tool runtime — the same path as a native tool call."""
    app = ContextApp(name="rt")

    def get_weather(city: str) -> dict:
        return {"city": city, "temp_c": 12}

    app.add_tool(get_weather, permission="read_only")

    def script(text, config):
        return [
            RealtimeEvent(
                type="tool_call",
                tool_call=RealtimeToolCall(
                    call_id="c1", name="get_weather", arguments={"city": "Paris"}
                ),
            )
        ]

    session = app.realtime_session(backend="inprocess", script=script)
    results = []
    async with session:
        await session.send_text("weather in Paris?")
        await session.commit()

        async def consume() -> None:
            async for event in session.events():
                if event.type == "tool_result":
                    results.append(event.data["result"])
                if event.type == "response.done":
                    break

        await asyncio.wait_for(consume(), timeout=2.0)
    assert results[0] == {"city": "Paris", "temp_c": 12}


@pytest.mark.asyncio
async def test_app_realtime_session_honors_approval_gate():
    """An approval-required (write) tool must NOT auto-execute on the realtime
    path — it hits the same approval gate as the text path. With no approval
    callback wired, the runtime denies and the call surfaces as an error event,
    never a successful tool_result."""
    app = ContextApp(name="rt-approval")

    executed = {"called": False}

    def delete_account(user_id: str) -> dict:
        executed["called"] = True
        return {"deleted": user_id}

    app.add_tool(delete_account, approval_required=True, side_effects="write")

    def script(text, config):
        return [
            RealtimeEvent(
                type="tool_call",
                tool_call=RealtimeToolCall(
                    call_id="c1", name="delete_account", arguments={"user_id": "u1"}
                ),
            )
        ]

    session = app.realtime_session(backend="inprocess", script=script)
    types: list[str] = []
    async with session:
        await session.send_text("delete my account")
        await session.commit()

        async def consume() -> None:
            async for event in session.events():
                types.append(event.type)
                if event.type in ("response.done", "turn.end"):
                    break

        await asyncio.wait_for(consume(), timeout=2.0)
    assert "error" in types  # approval gate blocked it
    assert executed["called"] is False  # the write tool never ran


def test_connect_realtime_unknown_backend():
    with pytest.raises(VincioError):
        connect_realtime("not-a-backend")


# -- hosted backends (offline: translation + missing-dependency) ----------------


def test_openai_translate_text_audio_tool_done():
    translate = OpenAIRealtimeBackend._translate
    assert translate({"type": "response.text.delta", "delta": "hi"}).text == "hi"
    audio = translate({"type": "response.audio.delta", "delta": base64.b64encode(b"ab").decode()})
    assert audio.type == "response.audio" and audio.audio == b"ab"
    tool = translate(
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c9",
            "name": "search",
            "arguments": json.dumps({"q": "x"}),
        }
    )
    assert tool.tool_call.name == "search" and tool.tool_call.arguments == {"q": "x"}
    assert translate({"type": "response.done"}).type == "response.done"


def test_gemini_translate_setup_and_turn():
    translate = GeminiLiveBackend._translate
    assert translate({"setupComplete": {}}).type == "session.started"
    text = translate({"serverContent": {"modelTurn": {"parts": [{"text": "yo"}]}}})
    assert text.type == "response.text" and text.text == "yo"
    done = translate({"serverContent": {"turnComplete": True}})
    assert done.type == "response.done"
    tool = translate({"toolCall": {"functionCalls": [{"id": "1", "name": "f", "args": {"a": 1}}]}})
    assert tool.tool_call.name == "f" and tool.tool_call.arguments == {"a": 1}


@pytest.mark.parametrize("backend_cls", [OpenAIRealtimeBackend, GeminiLiveBackend])
def test_hosted_backend_missing_websockets_is_helpful(backend_cls, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "websockets":
            raise ImportError("no websockets")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    backend = backend_cls(api_key="k")
    with pytest.raises(VincioError, match="realtime"):
        asyncio.run(backend.connect(RealtimeConfig()))
