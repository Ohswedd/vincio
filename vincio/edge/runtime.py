"""The edge in-process runtime — compile → score → rail → pack at the edge.

:class:`EdgeRuntime` is a thin, synchronous, dependency-free boundary around the
*same* context compiler, rail engine, and prompt compiler the server runs. It
takes an :class:`EdgeRequest` (a task, instructions, constraints, evidence, and
memory) and returns an :class:`EdgeResult` (a bounded, slim context packet, the
rendered model-ready prompt, the rail outcome, and the measured footprint and
latency) without a model call, a network hop, the filesystem, or an event loop
the caller has to own.

Everything runs in-process and offline, so the identical context-engineering
core that runs on a server runs in a browser (Pyodide/WASM) or an edge worker —
held inside an :class:`~vincio.edge.profile.EdgeProfile`'s resident-memory and
token bounds. It is parity by construction: the runtime *delegates* to
:class:`~vincio.context.compiler.ContextCompiler` and
:class:`~vincio.security.rails.RailEngine`, never re-implementing them, so a
capability can never silently diverge between server and edge
(:func:`~vincio.edge.parity.verify_edge_parity` proves it).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from ..context.compiler import ContextCompiler
from ..context.packet import ContextPacket
from ..core.errors import EdgeError
from ..core.types import (
    Budget,
    Constraint,
    EvidenceItem,
    Instruction,
    MemoryItem,
    Objective,
    PolicySet,
    TaskType,
    UserInput,
)
from ..prompts.compiler import PromptCompiler
from ..prompts.templates import PromptSpec
from ..security.rails import Rail, RailCheck, RailEngine, RailViolation
from .profile import EdgeProfile

__all__ = ["EdgeRequest", "EdgeResult", "EdgeRuntime"]


def _run_sync(coro: Any) -> Any:
    """Drive a coroutine to completion from sync code, inside or outside a loop.

    A standalone copy (the edge core deliberately depends on nothing outside the
    compile/score/rail/pack path), so :meth:`EdgeRuntime.run` works in a plain
    script, in a notebook, and under a WASM host's event loop alike.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class EdgeRequest(BaseModel):
    """A self-contained context-engineering request for the edge runtime.

    Reuses the platform's own typed inputs — :class:`~vincio.core.types.EvidenceItem`,
    :class:`~vincio.core.types.MemoryItem`, and :class:`~vincio.security.rails.Rail`
    — so an edge call is the same shape as a server run, never a parallel schema.
    """

    task: str = ""
    objective: str | None = None
    task_type: TaskType = TaskType.GENERAL
    instructions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    memory: list[MemoryItem] = Field(default_factory=list)
    rails: list[Rail] = Field(default_factory=list)
    tenant_id: str | None = None
    # Optional budget override; defaults to the profile's token window.
    budget: Budget | None = None


class EdgeResult(BaseModel):
    """The outcome of one edge compile.

    Carries the bounded, slim :class:`~vincio.context.packet.ContextPacket`, the
    rendered model-ready ``prompt`` (and ``system_text``), the deterministic
    resident-byte footprint, the measured ``latency_ms``, the merged input/output
    rail outcome, and ``within_profile`` — whether the packet held inside the
    profile's resident-memory and token bounds.
    """

    packet: ContextPacket
    prompt: str = ""
    system_text: str = ""
    token_count: int = 0
    prompt_tokens: int = 0
    resident_bytes: int = 0
    latency_ms: float = 0.0
    within_profile: bool = True
    allowed: bool = True
    rail_check: RailCheck = Field(default_factory=RailCheck)
    excluded: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    profile: str = "edge"


class EdgeRuntime:
    """A bounded, in-process context-engineering runtime for the edge.

    Construct it with an :class:`~vincio.edge.profile.EdgeProfile` (defaulting to
    the edge-worker profile) and optional baseline rails; then call
    :meth:`run` (sync) or :meth:`arun` (async) with an :class:`EdgeRequest` or a
    plain task string. The runtime holds no provider, store, or tracer — it is
    the dependency-free compile/score/rail/pack core and nothing else, which is
    exactly what compiles for a browser or edge worker.
    """

    def __init__(
        self,
        profile: EdgeProfile | None = None,
        *,
        rails: list[Rail] | None = None,
    ) -> None:
        self.profile = profile or EdgeProfile.default()
        # The *same* compiler the server runs, under the profile's bounded options.
        self.compiler = ContextCompiler(self.profile.to_compiler_options())
        self.rail_engine = RailEngine(list(rails or []))
        self.prompt_compiler = PromptCompiler()
        self._spec = PromptSpec(name="edge")

    @staticmethod
    def _coerce(request: EdgeRequest | str) -> EdgeRequest:
        if isinstance(request, str):
            return EdgeRequest(task=request)
        return request

    def _rails_for(self, request: EdgeRequest) -> RailEngine:
        """Combine the runtime's baseline rails with any per-request rails,
        sharing the deterministic detector instances so a per-request safety rail
        runs the same engine the server does."""
        if not request.rails:
            return self.rail_engine
        engine = RailEngine(
            list(self.rail_engine.rails) + list(request.rails),
            pii_detector=self.rail_engine.pii,
            secret_scanner=self.rail_engine.secrets,
            injection_detector=self.rail_engine.injection,
        )
        engine._predicates = self.rail_engine._predicates
        return engine

    async def arun(
        self, request: EdgeRequest | str, *, strict: bool = False
    ) -> EdgeResult:
        """Compile, rail, and pack one request, fully in-process and offline.

        With ``strict=True`` a packet that cannot be held inside the profile's
        resident-memory and token bounds raises :class:`~vincio.core.errors.EdgeError`
        instead of being reported as ``within_profile=False``; the default
        reports the bound rather than raising.
        """
        req = self._coerce(request)
        if not (req.task or req.objective):
            raise EdgeError(
                "an edge request needs a task or an objective",
                hint="set EdgeRequest.task (the user message) or .objective",
            )
        started = time.perf_counter()

        rails = self._rails_for(req)
        in_check = rails.check(req.task or "", direction="input")
        task_text = in_check.transformed_text or req.task or ""

        objective = Objective(
            text=req.objective or task_text or "edge context",
            task_type=req.task_type,
        )
        user_input = UserInput(text=task_text, tenant_id=req.tenant_id)
        budget = req.budget or Budget(
            max_input_tokens=self.profile.max_input_tokens,
            max_output_tokens=self.profile.max_output_tokens,
        )

        compiled = await self.compiler.compile(
            objective=objective,
            user_input=user_input,
            instructions=[Instruction(text=t) for t in req.instructions],
            constraints=[Constraint(text=t) for t in req.constraints],
            evidence=list(req.evidence),
            memory=list(req.memory),
            budget=budget,
            policies=PolicySet(),
        )

        rendered = self.prompt_compiler.compile(
            self._spec,
            user_task=task_text,
            memory_items=compiled.ir.memory_as_items(),
            evidence_items=compiled.ir.evidence_as_items(),
        )

        # Output rails over the assembled context catch a secret or PII that
        # leaked from retrieved evidence into the rendered prompt — the edge
        # runtime refuses to emit a context it would refuse on the server.
        out_check = rails.check(rendered.user_text, direction="output")
        violations: list[RailViolation] = list(in_check.violations) + list(out_check.violations)
        rail_check = RailCheck(
            allowed=in_check.allowed and out_check.allowed,
            violations=violations,
            transformed_text=in_check.transformed_text,
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        within_profile = (
            compiled.resident_bytes <= self.profile.max_resident_bytes
            and compiled.token_count <= self.profile.max_input_tokens
        )
        if strict and not within_profile:
            raise EdgeError(
                f"compiled context exceeds the {self.profile.name!r} edge profile "
                f"({compiled.resident_bytes}B / {compiled.token_count} tok vs caps "
                f"{self.profile.max_resident_bytes}B / {self.profile.max_input_tokens} tok)",
                hint="raise the profile's max_resident_bytes / max_input_tokens, "
                "or trim the request's evidence",
            )

        return EdgeResult(
            packet=compiled.packet,
            prompt=rendered.user_text,
            system_text=rendered.system_text,
            token_count=compiled.token_count,
            prompt_tokens=rendered.token_count,
            resident_bytes=compiled.resident_bytes,
            latency_ms=latency_ms,
            within_profile=within_profile,
            allowed=rail_check.allowed,
            rail_check=rail_check,
            excluded=list(compiled.excluded_report),
            conflicts=list(compiled.conflicts),
            profile=self.profile.name,
        )

    def run(self, request: EdgeRequest | str, *, strict: bool = False) -> EdgeResult:
        """Synchronous :meth:`arun` — usable from a plain script or a WASM host."""
        return _run_sync(self.arun(request, strict=strict))
