"""Realtime backends: a deterministic in-process backend (offline default and
test path) and lazy WebSocket backends for OpenAI Realtime and Gemini Live.

The hosted backends import ``websockets`` lazily and raise a clear error when
``vincio[realtime]`` is not installed; they are stateful and network-bound, so
the in-process backend is what the offline test suite exercises.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..core.errors import ProviderAuthError, VincioError
from .session import RealtimeConfig, RealtimeEvent, RealtimeToolCall, VADConfig

__all__ = ["InProcessRealtimeBackend", "OpenAIRealtimeBackend", "GeminiLiveBackend"]

Script = Callable[[str, RealtimeConfig], list[RealtimeEvent]]


def _default_script(text: str, config: RealtimeConfig) -> list[RealtimeEvent]:
    return [
        RealtimeEvent(type="response.text", text=f"You said: {text}"),
        RealtimeEvent(type="response.audio", audio=b"\x00\x01"),
        RealtimeEvent(type="response.done"),
    ]


class InProcessRealtimeBackend:
    """Deterministic, dependency-free realtime backend.

    Models a full turn lifecycle — input transcript, server VAD turn detection
    from audio energy, a scriptable model response, interruption, and tool
    round-trips — entirely in process, so realtime flows are reproducible and
    testable offline. Pass ``script`` to drive the model's response (default:
    echo the user text)."""

    def __init__(self, *, script: Script | None = None) -> None:
        self._script = script or _default_script
        self._queue: asyncio.Queue[RealtimeEvent | None] = asyncio.Queue()
        self._config: RealtimeConfig | None = None
        self._pending_text: list[str] = []
        self._speech_active = False
        self._interrupted = False
        self._turn_task: asyncio.Task[None] | None = None

    async def connect(self, config: RealtimeConfig) -> None:
        self._config = config
        await self._emit(RealtimeEvent(type="session.started", data={"model": config.model}))

    async def send_text(self, text: str) -> None:
        self._pending_text.append(text)
        await self._emit(RealtimeEvent(type="input.transcript", transcript=text))

    async def send_audio(self, chunk: bytes) -> None:
        vad = self._config.vad if self._config else VADConfig()
        if not vad.enabled or not chunk:
            return
        energy = sum(chunk) / len(chunk) / 255.0
        if energy >= vad.threshold and not self._speech_active:
            self._speech_active = True
            await self._emit(RealtimeEvent(type="vad.speech_start"))
        elif energy < vad.threshold and self._speech_active:
            self._speech_active = False
            await self._emit(RealtimeEvent(type="vad.speech_stop"))
            await self.commit()  # trailing silence ends the user's turn

    async def commit(self) -> None:
        self._interrupted = False
        text = " ".join(self._pending_text) or "<audio>"
        self._pending_text = []
        self._turn_task = asyncio.ensure_future(self._run_turn(text))

    async def _run_turn(self, text: str) -> None:
        await self._emit(RealtimeEvent(type="turn.start"))
        for event in self._script(text, self._config or RealtimeConfig()):
            await asyncio.sleep(0)  # cooperative point so interrupt() can land
            if self._interrupted:
                await self._emit(RealtimeEvent(type="interrupted"))
                break
            await self._emit(event)
        await self._emit(RealtimeEvent(type="turn.end"))

    async def interrupt(self) -> None:
        self._interrupted = True

    async def send_tool_result(self, call_id: str, result: Any) -> None:
        await self._emit(
            RealtimeEvent(type="response.text", text=f"tool {call_id} -> {result}")
        )
        await self._emit(RealtimeEvent(type="response.done"))

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass
        await self._queue.put(None)

    async def _emit(self, event: RealtimeEvent) -> None:
        await self._queue.put(event)


class _WebSocketBackend:
    """Shared scaffolding for WebSocket realtime backends."""

    extra = "realtime"

    def __init__(self, *, api_key: str | None = None, url: str | None = None) -> None:
        self.api_key = api_key
        self.url = url or self.default_url
        self._ws: Any = None
        self._config: RealtimeConfig | None = None

    default_url = ""

    def _require_websockets(self) -> Any:
        try:
            import websockets
        except ImportError as exc:
            raise VincioError(
                f'Realtime support requires websockets: pip install "vincio[{self.extra}]"'
            ) from exc
        if not self.api_key:
            raise ProviderAuthError("missing API key for realtime backend", provider=self.extra)
        return websockets

    def _connect_kwargs(self) -> dict[str, Any]:  # pragma: no cover - overridden
        return {}

    async def send_audio(self, chunk: bytes) -> None:  # pragma: no cover - network
        import base64

        await self._send(self._audio_event(base64.b64encode(chunk).decode("ascii")))

    async def _send(self, payload: dict[str, Any]) -> None:  # pragma: no cover - network
        if self._ws is None:
            raise VincioError("realtime backend is not connected")
        await self._ws.send(json.dumps(payload))

    async def close(self) -> None:  # pragma: no cover - network
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # Subclass protocol hooks
    def _audio_event(self, b64: str) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


class OpenAIRealtimeBackend(_WebSocketBackend):
    """OpenAI Realtime API over WebSocket (``wss://api.openai.com/v1/realtime``).

    Requires ``vincio[realtime]`` and an OpenAI API key. Translates the
    Realtime server events into :class:`RealtimeEvent`s."""

    default_url = "wss://api.openai.com/v1/realtime"

    def _connect_kwargs(self) -> dict[str, Any]:  # pragma: no cover - network
        model = self._config.model if self._config else "gpt-realtime"
        return {
            "additional_headers": {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
            "uri": f"{self.url}?model={model}",
        }

    async def connect(self, config: RealtimeConfig) -> None:
        self._config = config
        websockets = self._require_websockets()
        kwargs = self._connect_kwargs()
        uri = kwargs.pop("uri", self.url)
        self._ws = await websockets.connect(uri, **kwargs)  # pragma: no cover - network
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "modalities": config.modalities,
                    "instructions": config.instructions,
                    "voice": config.voice,
                    "turn_detection": {"type": "server_vad"} if config.vad.enabled else None,
                },
            }
        )

    def _audio_event(self, b64: str) -> dict[str, Any]:  # pragma: no cover - network
        return {"type": "input_audio_buffer.append", "audio": b64}

    async def send_text(self, text: str) -> None:  # pragma: no cover - network
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def commit(self) -> None:  # pragma: no cover - network
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

    async def interrupt(self) -> None:  # pragma: no cover - network
        await self._send({"type": "response.cancel"})

    async def send_tool_result(self, call_id: str, result: Any) -> None:  # pragma: no cover - network
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        await self._send({"type": "response.create"})

    async def events(self) -> AsyncIterator[RealtimeEvent]:  # pragma: no cover - network
        if self._ws is None:
            raise VincioError("realtime backend is not connected")
        async for raw in self._ws:
            yield self._translate(json.loads(raw))

    @staticmethod
    def _translate(message: dict[str, Any]) -> RealtimeEvent:  # pragma: no cover - network
        kind = message.get("type", "")
        if kind == "response.audio_transcript.delta" or kind == "response.text.delta":
            return RealtimeEvent(type="response.text", text=message.get("delta", ""))
        if kind == "response.audio.delta":
            import base64

            return RealtimeEvent(type="response.audio", audio=base64.b64decode(message.get("delta", "")))
        if kind == "response.function_call_arguments.done":
            return RealtimeEvent(
                type="tool_call",
                tool_call=RealtimeToolCall(
                    call_id=message.get("call_id", ""),
                    name=message.get("name", ""),
                    arguments=json.loads(message.get("arguments") or "{}"),
                ),
            )
        if kind == "response.done":
            return RealtimeEvent(type="response.done")
        if kind == "error":
            return RealtimeEvent(type="error", text=str(message.get("error")))
        return RealtimeEvent(type=kind, data=message)


class GeminiLiveBackend(_WebSocketBackend):
    """Gemini Live API over WebSocket.

    Requires ``vincio[realtime]`` and a Google API key. Translates Live server
    messages into :class:`RealtimeEvent`s."""

    default_url = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )

    async def connect(self, config: RealtimeConfig) -> None:
        self._config = config
        websockets = self._require_websockets()
        self._ws = await websockets.connect(f"{self.url}?key={self.api_key}")  # pragma: no cover - network
        await self._send(
            {
                "setup": {
                    "model": f"models/{config.model}",
                    "generation_config": {"response_modalities": config.modalities},
                    "system_instruction": {"parts": [{"text": config.instructions}]},
                }
            }
        )

    def _audio_event(self, b64: str) -> dict[str, Any]:  # pragma: no cover - network
        return {
            "realtime_input": {
                "media_chunks": [{"mime_type": "audio/pcm", "data": b64}]
            }
        }

    async def send_text(self, text: str) -> None:  # pragma: no cover - network
        await self._send(
            {
                "client_content": {
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turn_complete": False,
                }
            }
        )

    async def commit(self) -> None:  # pragma: no cover - network
        await self._send({"client_content": {"turns": [], "turn_complete": True}})

    async def interrupt(self) -> None:  # pragma: no cover - network
        # Gemini Live interrupts implicitly on new client input; send an empty
        # activity to signal barge-in.
        await self._send({"realtime_input": {"media_chunks": []}})

    async def send_tool_result(self, call_id: str, result: Any) -> None:  # pragma: no cover - network
        await self._send(
            {
                "tool_response": {
                    "function_responses": [
                        {"id": call_id, "response": {"result": result}}
                    ]
                }
            }
        )

    async def events(self) -> AsyncIterator[RealtimeEvent]:  # pragma: no cover - network
        if self._ws is None:
            raise VincioError("realtime backend is not connected")
        async for raw in self._ws:
            yield self._translate(json.loads(raw))

    @staticmethod
    def _translate(message: dict[str, Any]) -> RealtimeEvent:  # pragma: no cover - network
        if "setupComplete" in message:
            return RealtimeEvent(type="session.started")
        server = message.get("serverContent") or {}
        model_turn = server.get("modelTurn") or {}
        for part in model_turn.get("parts", []):
            if "text" in part:
                return RealtimeEvent(type="response.text", text=part["text"])
            if "inlineData" in part:
                import base64

                return RealtimeEvent(
                    type="response.audio",
                    audio=base64.b64decode(part["inlineData"].get("data", "")),
                )
        if server.get("turnComplete"):
            return RealtimeEvent(type="response.done")
        if "toolCall" in message:
            calls = message["toolCall"].get("functionCalls", [])
            if calls:
                call = calls[0]
                return RealtimeEvent(
                    type="tool_call",
                    tool_call=RealtimeToolCall(
                        call_id=call.get("id", ""),
                        name=call.get("name", ""),
                        arguments=call.get("args") or {},
                    ),
                )
        return RealtimeEvent(type="message", data=message)
