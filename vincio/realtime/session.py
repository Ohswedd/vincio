"""Realtime session: a provider-neutral, transport-abstracted bidirectional
voice/realtime session over a pluggable :class:`RealtimeBackend`.

The session owns the protocol-agnostic concerns — config, the event stream,
interruption, and in-session tool dispatch through Vincio's permissioned tool
runtime — while a backend owns the wire (in-process, OpenAI Realtime, or
Gemini Live).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..core.errors import VincioError

__all__ = [
    "VADConfig",
    "RealtimeConfig",
    "RealtimeToolCall",
    "RealtimeEvent",
    "RealtimeBackend",
    "RealtimeSession",
    "connect_realtime",
]

ToolDispatcher = Callable[[str, dict[str, Any]], Awaitable[Any]]


class VADConfig(BaseModel):
    """Server-side voice-activity-detection settings."""

    enabled: bool = True
    threshold: float = 0.5  # speech-energy threshold in [0, 1]
    silence_ms: int = 500  # trailing silence that ends a turn
    prefix_padding_ms: int = 300


class RealtimeConfig(BaseModel):
    """Session configuration shared across backends."""

    model: str = "gpt-realtime"
    voice: str = "alloy"
    modalities: list[str] = Field(default_factory=lambda: ["text", "audio"])
    instructions: str = ""
    temperature: float = 0.8
    vad: VADConfig = Field(default_factory=VADConfig)
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"


class RealtimeToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RealtimeEvent(BaseModel):
    """A normalized event from a realtime backend.

    ``type`` is a dotted name: ``session.started``, ``input.transcript``,
    ``vad.speech_start`` / ``vad.speech_stop``, ``turn.start`` / ``turn.end``,
    ``response.text`` / ``response.audio`` / ``response.done``, ``tool_call``,
    ``tool_result``, ``interrupted``, ``error``.
    """

    type: str
    text: str | None = None
    transcript: str | None = None
    audio: bytes | None = None
    tool_call: RealtimeToolCall | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class RealtimeBackend(Protocol):  # pragma: no cover - structural
    async def connect(self, config: RealtimeConfig) -> None: ...
    async def send_text(self, text: str) -> None: ...
    async def send_audio(self, chunk: bytes) -> None: ...
    async def commit(self) -> None: ...
    async def interrupt(self) -> None: ...
    async def send_tool_result(self, call_id: str, result: Any) -> None: ...
    def events(self) -> AsyncIterator[RealtimeEvent]: ...
    async def close(self) -> None: ...


class RealtimeSession:
    """A bidirectional realtime session.

    Drives a :class:`RealtimeBackend` and layers on the cross-cutting
    concerns: interruption (barge-in) and **in-session tool calls dispatched
    through the permissioned runtime** (pass ``tool_dispatcher``). Use it as an
    async context manager, send text/audio, and iterate :meth:`events`.
    """

    def __init__(
        self,
        backend: RealtimeBackend | None = None,
        *,
        config: RealtimeConfig | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
    ) -> None:
        from .backends import InProcessRealtimeBackend

        self.config = config or RealtimeConfig()
        self.backend: RealtimeBackend = backend or InProcessRealtimeBackend()
        self._tool_dispatcher = tool_dispatcher
        self._connected = False

    async def connect(self) -> RealtimeSession:
        if not self._connected:
            await self.backend.connect(self.config)
            self._connected = True
        return self

    async def __aenter__(self) -> RealtimeSession:
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _require_connected(self) -> None:
        if not self._connected:
            raise VincioError("realtime session is not connected; call connect() first")

    async def send_text(self, text: str) -> None:
        self._require_connected()
        await self.backend.send_text(text)

    async def send_audio(self, chunk: bytes) -> None:
        self._require_connected()
        await self.backend.send_audio(chunk)

    async def commit(self) -> None:
        """Commit the pending input (end the user's turn)."""
        self._require_connected()
        await self.backend.commit()

    async def interrupt(self) -> None:
        """Barge-in: cancel the in-flight model response."""
        self._require_connected()
        await self.backend.interrupt()

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield backend events, dispatching tool calls inline.

        A ``tool_call`` event is surfaced, then (when a ``tool_dispatcher`` is
        configured) executed through the permissioned runtime and its result
        is sent back to the model; a ``tool_result`` event follows. Backends
        without tool support simply never emit ``tool_call``."""
        async for event in self.backend.events():
            if event.type == "tool_call" and event.tool_call is not None:
                yield event
                if self._tool_dispatcher is not None:
                    call = event.tool_call
                    try:
                        result = await self._tool_dispatcher(call.name, call.arguments)
                        await self.backend.send_tool_result(call.call_id, result)
                        yield RealtimeEvent(
                            type="tool_result",
                            tool_call=call,
                            data={"call_id": call.call_id, "result": result},
                        )
                    except Exception as exc:  # noqa: BLE001 - surfaced as an event
                        await self.backend.send_tool_result(call.call_id, {"error": str(exc)})
                        yield RealtimeEvent(type="error", text=str(exc), tool_call=call)
            else:
                yield event

    async def close(self) -> None:
        if self._connected:
            await self.backend.close()
            self._connected = False


def connect_realtime(
    backend: str = "inprocess",
    *,
    config: RealtimeConfig | None = None,
    tool_dispatcher: ToolDispatcher | None = None,
    **backend_kwargs: Any,
) -> RealtimeSession:
    """Build a :class:`RealtimeSession` over a named backend.

    - ``inprocess`` → deterministic offline backend (default)
    - ``openai`` → OpenAI Realtime (WebSocket; ``vincio[realtime]``)
    - ``gemini`` → Gemini Live (WebSocket; ``vincio[realtime]``)
    """
    from .backends import GeminiLiveBackend, InProcessRealtimeBackend, OpenAIRealtimeBackend

    backends: dict[str, type] = {
        "inprocess": InProcessRealtimeBackend,
        "openai": OpenAIRealtimeBackend,
        "gemini": GeminiLiveBackend,
    }
    if backend not in backends:
        raise VincioError(f"unknown realtime backend {backend!r}; known: {list(backends)}")
    instance = backends[backend](**backend_kwargs)
    return RealtimeSession(instance, config=config, tool_dispatcher=tool_dispatcher)
