"""End-to-end voice agent: a realtime session wired to the full stack.

:class:`~vincio.realtime.RealtimeSession` owns the wire — audio in, audio out,
VAD, interruption, in-session tool dispatch through the permissioned runtime. A
spoken assistant needs more than the wire: it should be able to *look things up*
(grounded, cited, budget-bounded), *remember* across the conversation, and be
*guarded* on both the spoken-in and spoken-out boundary exactly like the text
path. :class:`VoiceAgent` is that wiring, assembled from parts that already exist:

- **Deep research** — registers :meth:`ContextApp.research` as an in-session
  ``research`` tool, so a spoken question runs the cited search → read → verify →
  synthesize loop and answers from sources, not from the model's memory.
- **Memory OS** — enables the self-editing memory tools
  (:meth:`ContextApp.enable_memory_os`), so the agent can recall and update its
  own memory mid-conversation, on the same audited, permissioned path.
- **Rails** — runs the app's deterministic input/output rails over every spoken
  transcript and every spoken reply, redacting or blocking before audio is
  produced, recorded on the audit chain.

Tool calls (including ``research`` and the memory ops) route through the app's
permissioned, sandboxed, budgeted, audited tool runtime — a voice turn cannot do
anything a text turn could not. The dependency-free in-process backend is the
default, so the whole flow runs and tests offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .session import RealtimeConfig, RealtimeEvent, RealtimeSession

if TYPE_CHECKING:
    from ..core.app import ContextApp

__all__ = ["VoiceAgent"]


class VoiceAgent:
    """A grounded, remembering, guarded voice session over a :class:`ContextApp`.

    Construct via :meth:`ContextApp.voice_agent`. Use it as an async context
    manager, drive it with :meth:`send_text` / :meth:`send_audio` / :meth:`commit`,
    and iterate :meth:`events` — the events stream is rail-screened.
    """

    def __init__(
        self,
        app: ContextApp,
        *,
        backend: str = "inprocess",
        config: RealtimeConfig | None = None,
        research: bool = True,
        memory_os: bool = True,
        rails: bool = True,
        owner_id: str = "voice",
        research_tool: str = "research",
        **backend_kwargs: Any,
    ) -> None:
        self.app = app
        self.rails_enabled = rails
        self.research_tool = research_tool

        if research and research_tool not in app.tool_registry:
            app.add_tool(self._research_callable(), name=research_tool, side_effects="read",
                         description="Answer a question from the app's sources with citations.")
        if research and research_tool not in app.enabled_tools:
            app.enabled_tools.append(research_tool)
        if memory_os:
            app.enable_memory_os(owner_id=owner_id)

        instructions = (config.instructions if config else "") or (
            "You are a spoken assistant. Use the research tool to answer factual "
            "questions from the knowledge base, and remember what the user tells you."
        )
        if config is None:
            config = RealtimeConfig(instructions=instructions)
        self.session: RealtimeSession = app.realtime_session(
            backend=backend, config=config, **backend_kwargs
        )

    def _research_callable(self) -> Any:
        app = self.app

        def research(question: str) -> dict[str, Any]:
            """Look up an answer from the app's sources, with citations."""
            report = app.research(question)
            return {
                "answer": report.answer,
                "citations": [s.id for s in report.sources],
                "citation_coverage": report.metrics.get("citation_coverage", 0.0),
            }

        return research

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> VoiceAgent:
        await self.session.connect()
        return self

    async def __aenter__(self) -> VoiceAgent:
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self.session.close()

    # -- I/O -----------------------------------------------------------------

    async def send_text(self, text: str) -> None:
        await self.session.send_text(text)

    async def send_audio(self, chunk: bytes) -> None:
        await self.session.send_audio(chunk)

    async def commit(self) -> None:
        await self.session.commit()

    async def interrupt(self) -> None:
        await self.session.interrupt()

    # -- guarded event stream ------------------------------------------------

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield session events with input/output rails applied.

        A spoken transcript is screened on the way in and a spoken reply on the
        way out: a ``redact`` rail rewrites the text in place, a ``block`` rail
        replaces it with a refusal and records a blocking decision on the audit
        chain. Tool-call / tool-result events (research, memory ops) pass through
        from the session's permissioned dispatch unchanged.
        """
        async for event in self.session.events():
            if not self.rails_enabled:
                yield event
                continue
            if event.type == "input.transcript" and event.transcript:
                yield self._screen(event, event.transcript, "input", field="transcript")
            elif event.type == "response.text" and event.text:
                yield self._screen(event, event.text, "output", field="text")
            else:
                yield event

    def _screen(
        self, event: RealtimeEvent, text: str, direction: str, *, field: str
    ) -> RealtimeEvent:
        check = self.app.rail_engine.check(text, direction=direction)
        if check.allowed and not check.violations:
            return event
        names = [v.rail for v in check.violations]
        self.app.audit.record(
            "voice_rail",
            decision="deny" if not check.allowed else "redact",
            resource="voice",
            details={"direction": direction, "rails": names},
        )
        self.app.events.emit(
            "voice.rail", {"direction": direction, "allowed": check.allowed, "rails": names}
        )
        if not check.allowed:
            safe = "I can't help with that request."
            return event.model_copy(update={field: safe, "data": {**event.data, "blocked": names}})
        if check.transformed_text is not None:
            return event.model_copy(
                update={field: check.transformed_text, "data": {**event.data, "redacted": names}}
            )
        return event
