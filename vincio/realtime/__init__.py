"""Voice / realtime sessions (optional module).

A separate, opt-in module for stateful bidirectional voice/realtime sessions
(OpenAI Realtime, Gemini Live): a WebSocket session with voice-activity
detection, interruption (barge-in), and in-session tool calls that route
through the **same permissioned, sandboxed, audited tool runtime** as every
other Vincio tool. This is explicitly scoped as a stateful bidirectional
module, *not* core context engineering.

Install the extra: ``pip install "vincio[realtime]"`` (adds ``websockets`` for
the hosted backends). The dependency-free :class:`InProcessRealtimeBackend` is
the default and offline-test path.

    from vincio.realtime import RealtimeSession, RealtimeConfig

    session = RealtimeSession(config=RealtimeConfig(model="gpt-realtime"))
    async with session:
        await session.send_text("What's the weather in Paris?")
        await session.commit()
        async for event in session.events():
            ...
"""

from __future__ import annotations

from .backends import (
    GeminiLiveBackend,
    InProcessRealtimeBackend,
    OpenAIRealtimeBackend,
)
from .session import (
    RealtimeBackend,
    RealtimeConfig,
    RealtimeEvent,
    RealtimeSession,
    RealtimeToolCall,
    VADConfig,
    connect_realtime,
)
from .voice_agent import VoiceAgent

__all__ = [
    "RealtimeBackend",
    "RealtimeConfig",
    "RealtimeEvent",
    "RealtimeSession",
    "RealtimeToolCall",
    "VADConfig",
    "connect_realtime",
    "VoiceAgent",
    "InProcessRealtimeBackend",
    "OpenAIRealtimeBackend",
    "GeminiLiveBackend",
]
