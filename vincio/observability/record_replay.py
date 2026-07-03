"""Causal record-replay: byte-faithful, deterministic replay of a whole run.

The eval-replay runner re-runs a recorded *input* through a target app and diffs
the result; durable-graph time-travel re-executes from a checkpoint. Neither
reproduces a past run *byte-for-byte*. This module does: it records every
non-deterministic edge of a run — model responses, tool outputs, retrieval hits,
the capabilities a request was negotiated against, and the clock/seed — keyed to
the run's trace, then serves each edge back in order so the run replays exactly,
turning "step, inspect, and branch a past run" into a first-class tool instead of
a bespoke script.

The design rests on one observation: a Vincio run is deterministic *except* at
its edges — every place it reads the outside world. Capture those edges by a
stable identity (a model call by its :attr:`ModelRequest.hash`, a tool call by
its name + canonical arguments, a retrieval by its query + params) and replay is
just "serve the recorded edge for this identity". Because the recorded bytes are
returned verbatim, replay is byte-faithful; because the identity is the same hash
the live code computes, a *changed* edge is a cache miss — and a miss is exactly a
**divergence**: live code no longer matches the recording, reported with the edge
that drifted.

Three surfaces:

* :class:`Recorder` — the capture layer. It instruments an app's provider, tool
  runtime, and retrieval for one run and returns a portable, content-addressed
  :class:`Recording`.
* :class:`Replayer` — the deterministic replay runtime. ``replay`` re-executes a
  recording against an app, serving every edge from the recording and reporting
  any divergence; ``branch`` forks a recording, edits an edge or the input, and
  re-executes only the affected suffix (the unchanged prefix is still served from
  the recording) so a fix is validated against the exact failing run.
* :class:`Recording` — the artifact. It is self-contained and JSON-serializable,
  carries a ``fidelity_digest`` so it is verifiable, and writes to / loads from a
  content-addressed :class:`~vincio.context.evidence_store.EvidenceStore` so it is
  portable across processes.

Everything here is additive and opt-in; the recorder is never in the live path
unless you wrap a run with it.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import ReplayDivergenceError
from ..core.types import (
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    ModelResponse,
    RunResult,
    ToolCall,
    ToolResult,
    UserInput,
)
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from ..providers.base import ModelProvider
from ..stability import deprecated
from .spans import Trace
from .traces import trace_diff

if TYPE_CHECKING:  # pragma: no cover
    from ..context.evidence_store import EvidenceStore
    from ..retrieval.engine import RetrievalResult

__all__ = [
    "EdgeKind",
    "RecordedEdge",
    "Recording",
    "Recorder",
    "ReplayProvider",
    "Replayer",
    "Divergence",
    "ReplayResult",
    "BranchEdit",
    "BranchResult",
]

EdgeKind = Literal["model_call", "tool_call", "retrieval", "capabilities", "clock", "seed"]

# Edge kinds whose identity is a genuine non-deterministic read of the world: a
# miss on one of these signals that live code diverged from the recording.
_REPLAY_EDGES: tuple[EdgeKind, ...] = ("model_call", "tool_call", "retrieval")


def content_hash(content: str) -> str:
    """Stable 16-hex content address (matches the evidence store's addressing)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _tool_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Stable identity of a tool call: its name + canonical arguments."""
    return stable_hash({"tool": tool_name, "arguments": arguments})


def _retrieval_key(query: str, params: dict[str, Any]) -> str:
    """Stable identity of a retrieval: its query + the params that shape it."""
    return stable_hash({"query": query, "params": to_jsonable(params)})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class RecordedEdge(BaseModel):
    """One captured non-deterministic edge of a run.

    ``key`` is the edge's stable identity (the model request hash, the tool
    name+args hash, the retrieval query+params hash, or the model name for a
    capabilities edge); ``value`` is the recorded payload (a JSON-serialized
    :class:`ModelResponse` / :class:`ToolResult` / :class:`RetrievalResult` /
    :class:`ModelCapabilities`); ``value_hash`` content-addresses that payload so
    the recording is verifiable; ``seq`` preserves capture order so repeated
    identical calls replay in the order they were made.
    """

    kind: EdgeKind
    seq: int
    key: str
    value: Any = None
    value_hash: str = ""
    span_id: str | None = None

    @classmethod
    def of(cls, kind: EdgeKind, seq: int, key: str, value: Any, *, span_id: str | None = None) -> RecordedEdge:
        jsonable = to_jsonable(value)
        return cls(
            kind=kind,
            seq=seq,
            key=key,
            value=jsonable,
            value_hash=stable_hash(jsonable),
            span_id=span_id,
        )

    def hash_matches(self) -> bool:
        """Whether ``value_hash`` is the content address of ``value`` (integrity)."""
        return self.value_hash == stable_hash(self.value)


class Recording(BaseModel):
    """A portable, verifiable recording of a whole run.

    Self-contained (every edge payload is inlined) and JSON-serializable, so it
    can be saved to a file or written to a content-addressed
    :class:`~vincio.context.evidence_store.EvidenceStore` with :meth:`put` and
    loaded back with :meth:`from_store`. :meth:`verify` recomputes the
    ``fidelity_digest`` and every edge's content address, so a tampered or
    truncated recording is detected before it is trusted for replay.
    """

    recording_id: str = Field(default_factory=lambda: new_id("rec"))
    app_name: str = ""
    run_id: str = ""
    trace_id: str = ""
    input: str = ""
    created_at: str = ""
    status: str = ""
    output_text: str = ""
    edges: list[RecordedEdge] = Field(default_factory=list)
    trace: Trace | None = None
    fidelity_digest: str = ""

    # -- digest / verification -------------------------------------------------

    def digest(self) -> str:
        """The replay-fidelity anchor: a hash over the ordered edge identities,
        their content addresses, and the recorded output."""
        return stable_hash(
            {
                "edges": [[e.kind, e.seq, e.key, e.value_hash] for e in self.edges],
                "output": self.output_text,
                "status": self.status,
            }
        )

    @deprecated(since="7.5", removed_in="8.0", alternative="digest()")
    def compute_digest(self) -> str:
        """Deprecated name for :meth:`digest`."""
        return self.digest()

    def verify(self) -> bool:
        """True when every edge's payload matches its content address and the
        recording's ``fidelity_digest`` matches the recomputed digest."""
        return self.fidelity_report()["ok"]

    def fidelity_report(self) -> dict[str, Any]:
        """A structured integrity check: the recomputed digest, whether it
        matches, and any edges whose payload no longer matches its hash."""
        recomputed = self.digest()
        corrupt = [
            {"seq": e.seq, "kind": e.kind, "key": e.key}
            for e in self.edges
            if not e.hash_matches()
        ]
        digest_ok = recomputed == self.fidelity_digest
        return {
            "ok": digest_ok and not corrupt,
            "digest_ok": digest_ok,
            "expected_digest": self.fidelity_digest,
            "actual_digest": recomputed,
            "corrupt_edges": corrupt,
            "edges": len(self.edges),
        }

    # -- inspection surface ----------------------------------------------------

    def edges_of(self, kind: EdgeKind) -> list[RecordedEdge]:
        """The recorded edges of a kind, in capture order."""
        return [e for e in self.edges if e.kind == kind]

    @property
    def model_calls(self) -> list[RecordedEdge]:
        return self.edges_of("model_call")

    @property
    def tool_calls(self) -> list[RecordedEdge]:
        return self.edges_of("tool_call")

    @property
    def retrievals(self) -> list[RecordedEdge]:
        return self.edges_of("retrieval")

    def steps(self) -> list[dict[str, Any]]:
        """The recorded run's span tree (the step/inspect surface)."""
        return self.trace.span_tree() if self.trace is not None else []

    def render_text(self) -> str:
        """A readable, deterministic summary for the CLI / inspection."""
        lines = [
            f"recording {self.recording_id}  app={self.app_name or '-'}  status={self.status}",
            f"  input:  {self.input[:80]!r}",
            f"  output: {self.output_text[:80]!r}",
            f"  edges:  {len(self.edges)} "
            f"(model={len(self.model_calls)} tool={len(self.tool_calls)} "
            f"retrieval={len(self.retrievals)})",
            f"  digest: {self.fidelity_digest}",
        ]
        for edge in self.edges:
            if edge.kind in _REPLAY_EDGES:
                lines.append(f"    [{edge.seq:>3}] {edge.kind:<11} {edge.key}")
        return "\n".join(lines)

    # -- content-addressed store / file portability ----------------------------

    def put(self, store: EvidenceStore) -> str:
        """Write the recording to a content-addressed store; return its address.

        The whole recording is serialized and stored under its content hash, so
        the returned address both locates and verifies the bytes.
        """
        return store.put(json.dumps(to_jsonable(self.model_dump(mode="json")), ensure_ascii=False))

    @classmethod
    def from_store(cls, store: EvidenceStore, address: str) -> Recording:
        """Load a recording previously written with :meth:`put`."""
        blob = store.get(address)
        if blob is None:
            raise ReplayDivergenceError(
                f"no recording at content address {address!r}",
                details={"address": address},
            )
        return cls.model_validate(json.loads(blob))

    def save(self, path: str | Path) -> Path:
        """Write the recording to a JSON file (portable across machines)."""
        target = Path(path)
        if target.parent and str(target.parent):
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(to_jsonable(self.model_dump(mode="json")), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        """Load a recording from a JSON file written by :meth:`save`."""
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class Divergence(BaseModel):
    """A point where replayed code no longer matched the recording.

    ``kind`` / ``key`` identify the edge the live code asked for that was not in
    the recording (a model call with a different request, a tool called with
    different arguments, a retrieval with a different query); ``detail`` explains
    it. A divergence is how the debugger pinpoints *where* a run drifted.
    """

    kind: EdgeKind
    key: str
    detail: str = ""
    span_id: str | None = None


class ReplayResult(BaseModel):
    """The outcome of replaying a recording against an app."""

    recording_id: str
    faithful: bool
    output_identical: bool
    recorded_output: str
    replayed_output: str
    replayed_status: str
    replayed_trace_id: str = ""
    served_from_recording: int = 0
    divergences: list[Divergence] = Field(default_factory=list)
    trajectory: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "faithful": self.faithful,
            "output_identical": self.output_identical,
            "served_from_recording": self.served_from_recording,
            "divergences": len(self.divergences),
        }


class BranchEdit(BaseModel):
    """An edit applied to a recording before a branch re-execution.

    Replaces the recorded payload served for an edge identity — e.g. make a
    recorded tool call return a different output, or pin a different model
    response — so you can ask "what would this run have done if this step had
    returned X". The ``key`` comes from the recording (``recording.tool_calls``,
    ``recording.model_calls``, ...).
    """

    kind: Literal["model_call", "tool_call", "retrieval"]
    key: str
    value: Any


class BranchResult(BaseModel):
    """The outcome of forking a recording, editing it, and re-executing."""

    recording_id: str
    branched_input: str
    output: str
    status: str
    served_from_recording: int = 0
    reexecuted: int = 0
    edits_applied: int = 0
    replayed_trace_id: str = ""
    trajectory: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "served_from_recording": self.served_from_recording,
            "reexecuted": self.reexecuted,
            "edits_applied": self.edits_applied,
        }


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class _EdgeSink:
    """Collects recorded edges in capture order during a recorded run."""

    def __init__(self) -> None:
        self.edges: list[RecordedEdge] = []
        self._seq = 0

    def add(self, kind: EdgeKind, key: str, value: Any, *, span_id: str | None = None) -> None:
        self.edges.append(RecordedEdge.of(kind, self._seq, key, value, span_id=span_id))
        self._seq += 1


class _CaptureExporter:
    """Wraps an exporter to capture exported traces by id, still forwarding them
    so recording never loses observability data."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.captured: dict[str, Trace] = {}

    def export(self, trace: Trace) -> None:
        self.captured[trace.id] = trace
        self._inner.export(trace)


class _RecordingProvider(ModelProvider):
    """Wraps a provider to record every model response and the capabilities each
    request was negotiated against, delegating the call to the real provider."""

    name = "recording"

    def __init__(self, inner: ModelProvider, sink: _EdgeSink) -> None:
        self.inner = inner
        self._sink = sink

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await self.inner.generate(request)
        self._sink.add("model_call", request.hash, response.model_dump(mode="json"))
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        terminal: ModelResponse | None = None
        accumulated: list[str] = []
        tool_calls: list[Any] = []
        async for event in self.inner.stream(request):
            if event.type == "text_delta" and event.text:
                accumulated.append(event.text)
            elif event.type == "tool_call_delta" and event.tool_call is not None:
                tool_calls.append(event.tool_call)
            elif event.type == "done" and event.response is not None:
                terminal = event.response
            yield event
        if terminal is None:
            terminal = ModelResponse(model=request.model, text="".join(accumulated))
            terminal.tool_calls = list(tool_calls)
        self._sink.add("model_call", request.hash, terminal.model_dump(mode="json"))

    def capabilities(self, model: str) -> ModelCapabilities:
        caps = self.inner.capabilities(model)
        self._sink.add("capabilities", model, caps.model_dump(mode="json"))
        return caps

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return await self.inner.embed(texts, model)

    async def list_models(self) -> list[Any]:
        return await self.inner.list_models()

    async def aclose(self) -> None:
        return None


@contextmanager
def _instrument(app: Any, *, provider_factory: Any, tool_shim: Any, retrieval_shim: Any) -> Iterator[_CaptureExporter]:
    """Install capture/replay shims on an app for the duration of a run.

    Shadows ``resolve_provider`` (so cascade/failover providers are wrapped too),
    ``tool_runtime.execute``, ``retrieval.retrieve``, and the tracer's exporter,
    restoring every one of them on exit — the run path is otherwise untouched.
    """
    original_resolve = app.resolve_provider
    original_execute = app.tool_runtime.execute
    original_retrieve = app.retrieval.retrieve if app.retrieval is not None else None
    original_exporter = app.tracer.exporter
    # Bypass the response cache for the duration so every model call flows through
    # the recording/replay provider — the recording, not a warm cache, is the
    # single source of truth (a cache hit would otherwise skip capture/replay).
    original_cache = getattr(app, "response_cache", None)
    capture = _CaptureExporter(original_exporter)

    def resolve(run_config: Any = None) -> ModelProvider:
        return provider_factory(original_resolve(run_config))

    app.resolve_provider = resolve
    app.tool_runtime.execute = tool_shim(original_execute)
    if original_retrieve is not None:
        app.retrieval.retrieve = retrieval_shim(original_retrieve)
    app.tracer.exporter = capture
    app.response_cache = None
    try:
        yield capture
    finally:
        app.resolve_provider = original_resolve
        app.tool_runtime.execute = original_execute
        if original_retrieve is not None:
            app.retrieval.retrieve = original_retrieve
        app.tracer.exporter = original_exporter
        app.response_cache = original_cache


class Recorder:
    """Records a run's non-deterministic edges into a portable :class:`Recording`.

    Instruments the app's provider, tool runtime, and retrieval for one run; the
    run executes normally against the *real* provider/tools/retrieval while every
    edge is captured. The returned recording replays byte-for-byte through
    :class:`Replayer`.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def record(
        self,
        user_input: str | UserInput,
        *,
        store: EvidenceStore | None = None,
        **run_kwargs: Any,
    ) -> tuple[RunResult, Recording]:
        """Run *user_input* through the app, capturing a :class:`Recording`.

        When ``store`` is given, the recording is also written to it (content
        addressed) so it is durable and portable. Extra keyword arguments are
        forwarded to ``app.arun`` (``tenant_id``, ``user_id``, ``config``, ...).
        """
        app = self.app
        sink = _EdgeSink()
        # Record the clock/seed anchor so the recording is complete and auditable
        # (replay determinism comes from the served edges, not the wall clock).
        sink.add("clock", "run", utcnow().isoformat())
        seed = None
        config = run_kwargs.get("config")
        if config is not None:
            seed = getattr(config, "seed", None)
        sink.add("seed", "run", seed)

        def provider_factory(inner: ModelProvider) -> ModelProvider:
            return _RecordingProvider(inner, sink)

        def tool_shim(original_execute: Any) -> Any:
            async def execute(call: ToolCall, *args: Any, **kwargs: Any) -> ToolResult:
                result = await original_execute(call, *args, **kwargs)
                sink.add(
                    "tool_call",
                    _tool_key(call.tool_name, call.arguments),
                    result.model_dump(mode="json"),
                )
                return result

            return execute

        def retrieval_shim(original_retrieve: Any) -> Any:
            async def retrieve(query: str, **kwargs: Any) -> RetrievalResult:
                result = await original_retrieve(query, **kwargs)
                sink.add(
                    "retrieval",
                    _retrieval_key(query, kwargs),
                    result.model_dump(mode="json"),
                )
                return result

            return retrieve

        with _instrument(
            app,
            provider_factory=provider_factory,
            tool_shim=tool_shim,
            retrieval_shim=retrieval_shim,
        ) as capture:
            result = await app.arun(user_input, **run_kwargs)

        trace = capture.captured.get(result.trace_id)
        recording = Recording(
            app_name=getattr(app, "app_name", "") or getattr(app.tracer, "app_name", ""),
            run_id=result.run_id,
            trace_id=result.trace_id,
            input=_input_text(user_input),
            created_at=utcnow().isoformat(),
            status=result.status.value,
            output_text=result.raw_text or "",
            edges=sink.edges,
            trace=trace,
        )
        recording.fidelity_digest = recording.digest()
        if store is not None:
            recording.put(store)
        return result, recording


def _input_text(user_input: str | UserInput) -> str:
    if isinstance(user_input, str):
        return user_input
    return user_input.text or ""


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class _ReplayBook:
    """Indexes a recording's edges into per-(kind,key) FIFO queues so repeated
    identical calls replay in capture order."""

    def __init__(self, recording: Recording) -> None:
        self._queues: dict[tuple[str, str], deque[Any]] = {}
        # Capabilities are deterministic config, not a divergence signal: keep the
        # last seen value per model so an extra lookup never reports a divergence.
        self._caps: dict[str, Any] = {}
        for edge in recording.edges:
            if edge.kind == "capabilities":
                self._caps[edge.key] = edge.value
                continue
            if edge.kind in _REPLAY_EDGES:
                self._queues.setdefault((edge.kind, edge.key), deque()).append(edge.value)

    def pop(self, kind: str, key: str) -> tuple[bool, Any]:
        """Pop the next recorded value for an edge identity.

        Returns ``(hit, value)``; ``hit`` is False when the recording has no
        (more) values for this identity — a divergence.
        """
        queue = self._queues.get((kind, key))
        if not queue:
            return False, None
        return True, queue.popleft()

    def capabilities(self, model: str) -> Any:
        return self._caps.get(model)


class ReplayProvider(ModelProvider):
    """Serves recorded model responses by request identity.

    On a hit the recorded :class:`ModelResponse` is returned verbatim (so replay
    is byte-faithful). On a miss the request the live code produced is not in the
    recording: a :class:`Divergence` is recorded and, in strict mode, a
    :class:`~vincio.core.errors.ReplayDivergenceError` is raised; in fallback
    mode (branch-and-edit) the call is delegated to a fallback provider so the
    affected suffix re-executes live while the unchanged prefix is still served
    from the recording.
    """

    name = "replay"

    def __init__(
        self,
        book: _ReplayBook,
        divergences: list[Divergence],
        counters: dict[str, int],
        *,
        on_miss: Literal["error", "fallback"] = "error",
        fallback: ModelProvider | None = None,
    ) -> None:
        self._book = book
        self._divergences = divergences
        self._counters = counters
        self._on_miss = on_miss
        self._fallback = fallback

    def _miss(self, request: ModelRequest) -> Divergence:
        div = Divergence(
            kind="model_call",
            key=request.hash,
            detail=f"no recorded model response for request on model {request.model!r}",
        )
        self._divergences.append(div)
        return div

    async def generate(self, request: ModelRequest) -> ModelResponse:
        hit, value = self._book.pop("model_call", request.hash)
        if not hit:
            self._miss(request)
            if self._on_miss == "fallback" and self._fallback is not None:
                self._counters["reexecuted"] = self._counters.get("reexecuted", 0) + 1
                return await self._fallback.generate(request)
            raise ReplayDivergenceError(
                f"replay diverged: no recorded model response for request hash {request.hash}",
                details={"kind": "model_call", "key": request.hash, "model": request.model},
            )
        self._counters["served"] = self._counters.get("served", 0) + 1
        return ModelResponse.model_validate(value)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        try:
            response = await self.generate(request)
        except ReplayDivergenceError:
            if self._on_miss == "fallback" and self._fallback is not None:
                async for event in self._fallback.stream(request):
                    yield event
                return
            raise
        chunk = 16
        for start in range(0, len(response.text), chunk):
            yield ModelEvent(type="text_delta", text=response.text[start : start + chunk])
        for tool_call in response.tool_calls:
            yield ModelEvent(type="tool_call_delta", tool_call=tool_call)
        yield ModelEvent(type="usage", usage=response.usage)
        yield ModelEvent(type="done", response=response)

    def capabilities(self, model: str) -> ModelCapabilities:
        recorded = self._book.capabilities(model)
        if recorded is not None:
            return ModelCapabilities.model_validate(recorded)
        if self._fallback is not None:
            return self._fallback.capabilities(model)
        return ModelCapabilities()

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        if self._fallback is not None:
            return await self._fallback.embed(texts, model)
        # Deterministic, dependency-free fallback so an incidental embed (e.g. a
        # speculative prefetch) never errors during replay.
        return [_deterministic_embed(text) for text in texts]

    async def list_models(self) -> list[Any]:
        return []

    async def aclose(self) -> None:
        return None


def _deterministic_embed(text: str, dim: int = 64) -> list[float]:
    import hashlib
    import math

    vector = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class Replayer:
    """Deterministic replay runtime over a :class:`~vincio.core.app.ContextApp`.

    Re-executes a recording against an app, serving every edge from the
    recording. The app must be configured like the one that produced the
    recording (same instructions, tools, contract, retrieval) — the recording
    supplies only the *edges*, the app supplies the *code*; that is exactly what
    makes a divergence meaningful when the code has changed.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    def _instruments(
        self,
        book: _ReplayBook,
        divergences: list[Divergence],
        counters: dict[str, int],
        *,
        on_miss: Literal["error", "fallback"],
        fallback: ModelProvider | None,
    ) -> tuple[Any, Any, Any]:
        def provider_factory(inner: ModelProvider) -> ModelProvider:
            return ReplayProvider(
                book, divergences, counters,
                on_miss=on_miss,
                fallback=fallback if fallback is not None else (inner if on_miss == "fallback" else None),
            )

        def tool_shim(original_execute: Any) -> Any:
            async def execute(call: ToolCall, *args: Any, **kwargs: Any) -> ToolResult:
                hit, value = book.pop("tool_call", _tool_key(call.tool_name, call.arguments))
                if hit:
                    counters["served"] = counters.get("served", 0) + 1
                    return ToolResult.model_validate(value)
                divergences.append(
                    Divergence(
                        kind="tool_call",
                        key=_tool_key(call.tool_name, call.arguments),
                        detail=f"no recorded output for tool {call.tool_name!r}",
                    )
                )
                if on_miss == "fallback":
                    counters["reexecuted"] = counters.get("reexecuted", 0) + 1
                    return await original_execute(call, *args, **kwargs)
                raise ReplayDivergenceError(
                    f"replay diverged: no recorded output for tool {call.tool_name!r}",
                    details={"kind": "tool_call", "tool": call.tool_name},
                )

            return execute

        def retrieval_shim(original_retrieve: Any) -> Any:
            async def retrieve(query: str, **kwargs: Any) -> RetrievalResult:
                from ..retrieval.engine import RetrievalResult

                hit, value = book.pop("retrieval", _retrieval_key(query, kwargs))
                if hit:
                    counters["served"] = counters.get("served", 0) + 1
                    return RetrievalResult.model_validate(value)
                divergences.append(
                    Divergence(
                        kind="retrieval",
                        key=_retrieval_key(query, kwargs),
                        detail="no recorded retrieval for this query",
                    )
                )
                if on_miss == "fallback":
                    counters["reexecuted"] = counters.get("reexecuted", 0) + 1
                    return await original_retrieve(query, **kwargs)
                raise ReplayDivergenceError(
                    "replay diverged: no recorded retrieval for this query",
                    details={"kind": "retrieval", "query": query[:120]},
                )

            return retrieve

        return provider_factory, tool_shim, retrieval_shim

    async def replay(
        self,
        recording: Recording,
        *,
        strict: bool = True,
    ) -> ReplayResult:
        """Replay *recording* against the app and report any divergence.

        With ``strict`` (the default) the first edge the live code asks for that
        is not in the recording stops the run and is reported as a divergence —
        the byte-faithful, regression-detecting mode. The result's ``faithful``
        is True only when no edge diverged and the replayed output is byte-
        identical to the recording.
        """
        app = self.app
        book = _ReplayBook(recording)
        divergences: list[Divergence] = []
        counters: dict[str, int] = {}
        provider_factory, tool_shim, retrieval_shim = self._instruments(
            book, divergences, counters, on_miss="error" if strict else "fallback", fallback=None
        )
        with _instrument(
            app,
            provider_factory=provider_factory,
            tool_shim=tool_shim,
            retrieval_shim=retrieval_shim,
        ) as capture:
            try:
                result = await app.arun(recording.input)
            except ReplayDivergenceError:
                # Defensive: the runtime catches VincioError and marks the run
                # failed, so this rarely propagates; either way the divergence is
                # already recorded in the shared list.
                result = None  # type: ignore[assignment]

        replayed_output = result.raw_text if result is not None else ""
        replayed_status = result.status.value if result is not None else "failed"
        replayed_trace = capture.captured.get(result.trace_id) if result is not None else None
        output_identical = replayed_output == recording.output_text
        trajectory = (
            trace_diff(recording.trace, replayed_trace)
            if recording.trace is not None and replayed_trace is not None
            else {}
        )
        return ReplayResult(
            recording_id=recording.recording_id,
            faithful=not divergences and output_identical,
            output_identical=output_identical,
            recorded_output=recording.output_text,
            replayed_output=replayed_output,
            replayed_status=replayed_status,
            replayed_trace_id=result.trace_id if result is not None else "",
            served_from_recording=counters.get("served", 0),
            divergences=divergences,
            trajectory=trajectory,
        )

    async def branch(
        self,
        recording: Recording,
        *,
        input: str | None = None,
        edits: list[BranchEdit] | None = None,
        fallback: ModelProvider | None = None,
    ) -> BranchResult:
        """Fork *recording*, optionally edit an edge or the input, and re-execute.

        The unchanged prefix is still served from the recording; the affected
        suffix — every edge whose identity changed because of an edit or a new
        input — re-executes against ``fallback`` (or the app's own provider /
        tools / retrieval when no fallback is given). This validates a fix
        against the exact recorded run without re-paying for the unaffected
        prefix.
        """
        app = self.app
        edits = edits or []
        # A branch works on a copy so the original recording is never mutated, and
        # edits replace the served payload for their edge identity.
        branched = recording.model_copy(deep=True)
        applied = 0
        for edit in edits:
            for edge in branched.edges:
                if edge.kind == edit.kind and edge.key == edit.key:
                    edge.value = to_jsonable(edit.value)
                    edge.value_hash = stable_hash(edge.value)
                    applied += 1
        book = _ReplayBook(branched)
        divergences: list[Divergence] = []
        counters: dict[str, int] = {}
        provider_factory, tool_shim, retrieval_shim = self._instruments(
            book, divergences, counters, on_miss="fallback", fallback=fallback
        )
        branched_input = recording.input if input is None else input
        with _instrument(
            app,
            provider_factory=provider_factory,
            tool_shim=tool_shim,
            retrieval_shim=retrieval_shim,
        ) as capture:
            result = await app.arun(branched_input)

        replayed_trace = capture.captured.get(result.trace_id)
        trajectory = (
            trace_diff(recording.trace, replayed_trace)
            if recording.trace is not None and replayed_trace is not None
            else {}
        )
        return BranchResult(
            recording_id=recording.recording_id,
            branched_input=branched_input,
            output=result.raw_text or "",
            status=result.status.value,
            served_from_recording=counters.get("served", 0),
            reexecuted=counters.get("reexecuted", 0),
            edits_applied=applied,
            replayed_trace_id=result.trace_id,
            trajectory=trajectory,
        )
