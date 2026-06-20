"""End-to-end voice agent: a realtime session wired to deep research, the
memory OS, and the deterministic rails — exercised offline on the in-process
backend."""

from __future__ import annotations

import pytest

from vincio import ContextApp, VoiceAgent
from vincio.core.types import Document
from vincio.providers import MockProvider
from vincio.realtime import RealtimeEvent, RealtimeToolCall


@pytest.fixture()
def voice_app():
    app = ContextApp(
        name="voice",
        provider=MockProvider(responder=lambda r: "The refund window is 30 days. [E1]"),
        model="mock-1",
    )
    app.add_source("kb", documents=[Document(id="d1", title="Policy",
                                             text="The refund window is 30 days.")])
    return app


async def _drain(agent, *, stop="turn.end"):
    events = []
    async for event in agent.events():
        events.append(event)
        if event.type == stop:
            break
    return events


def test_voice_agent_registers_research_and_memory_tools(voice_app):
    agent = voice_app.voice_agent()
    assert isinstance(agent, VoiceAgent)
    assert "research" in voice_app.tool_registry
    # The memory OS tools are wired through the permissioned runtime.
    for name in ("memory_append", "memory_search", "memory_replace", "memory_archive"):
        assert name in voice_app.enabled_tools


async def test_voice_agent_answers_from_research(voice_app):
    def script(text, config):
        return [
            RealtimeEvent(type="tool_call",
                          tool_call=RealtimeToolCall(call_id="c1", name="research",
                                                     arguments={"question": text})),
            RealtimeEvent(type="response.text", text="The refund window is 30 days."),
            RealtimeEvent(type="response.done"),
        ]

    agent = voice_app.voice_agent(script=script)
    async with agent:
        await agent.send_text("what is the refund window?")
        await agent.commit()
        events = await _drain(agent)
    tool_results = [e for e in events if e.type == "tool_result"]
    assert tool_results, "the research tool should run through the permissioned dispatch"
    result = tool_results[0].data["result"]
    assert "answer" in result and result["answer"]
    assert result["citations"]


async def test_voice_output_rail_redacts_pii(voice_app):
    voice_app.add_rail(name="pii_out", kind="safety", direction="output",
                       detectors=["pii"], action="redact")

    def script(text, config):
        return [
            RealtimeEvent(type="response.text",
                          text="Your SSN 123-45-6789 is on file."),
            RealtimeEvent(type="response.done"),
        ]

    agent = voice_app.voice_agent(research=False, memory_os=False, script=script)
    async with agent:
        await agent.send_text("read my ssn")
        await agent.commit()
        events = await _drain(agent)
    spoken = [e for e in events if e.type == "response.text"][0]
    assert "123-45-6789" not in (spoken.text or "")
    assert "pii_out" in spoken.data.get("redacted", [])


async def test_voice_input_rail_blocks_off_topic(voice_app):
    voice_app.add_rail(name="no_legal", kind="topic", direction="input",
                       blocked_topics=["lawsuit"], action="block")

    def script(text, config):
        return [RealtimeEvent(type="response.text", text="ok"),
                RealtimeEvent(type="response.done")]

    agent = voice_app.voice_agent(research=False, memory_os=False, script=script)
    async with agent:
        await agent.send_text("should I file a lawsuit?")
        await agent.commit()
        events = await _drain(agent)
    transcript = [e for e in events if e.type == "input.transcript"][0]
    assert transcript.data.get("blocked") == ["no_legal"]
    assert "lawsuit" not in (transcript.transcript or "")


async def test_voice_rails_can_be_disabled(voice_app):
    voice_app.add_rail(name="pii_out", kind="safety", direction="output",
                       detectors=["pii"], action="redact")

    def script(text, config):
        return [RealtimeEvent(type="response.text", text="SSN 123-45-6789"),
                RealtimeEvent(type="response.done")]

    agent = voice_app.voice_agent(research=False, memory_os=False, rails=False, script=script)
    async with agent:
        await agent.send_text("x")
        await agent.commit()
        events = await _drain(agent)
    spoken = [e for e in events if e.type == "response.text"][0]
    assert "123-45-6789" in (spoken.text or "")  # rails off: passed through verbatim
