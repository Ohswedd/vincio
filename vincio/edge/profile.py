"""Bounded edge profiles and runtime-environment detection.

An :class:`EdgeProfile` is the constrained-target analogue of the per-app
resident-memory budget the server run already holds: a declarative cap on the
compiled packet's resident footprint, the input-token window, the evidence
count, and a latency budget. It lowers directly to the
:class:`~vincio.context.compiler.ContextCompilerOptions` the *same* compiler
reads on the server, so an edge build is bounded by the identical mechanism — a
profile, not a fork.

:func:`edge_environment` reports whether the process is running on a WASM target
(Pyodide / CPython-on-Emscripten, or a WASI runtime) so a caller can pick a
profile without guessing; the core itself runs identically either way.
"""

from __future__ import annotations

import sys
from typing import Literal

from pydantic import BaseModel, Field

from ..context.compiler import ContextCompilerOptions

__all__ = [
    "EdgeProfile",
    "EdgeEnvironment",
    "edge_environment",
    "is_wasm_runtime",
]


EdgeOrdering = Literal["relevance", "authority", "recency", "boundary_sandwich"]


class EdgeProfile(BaseModel):
    """A bounded resident-memory and latency profile for a constrained target.

    The fields are hard caps the runtime holds the compiled packet under:

    - ``max_resident_bytes`` — the packet's estimated resident footprint
      (:mod:`vincio.context.footprint`); the compiler slims the packet and evicts
      the lowest-utility evidence until the estimate fits, exactly as the
      server's per-app ceiling does.
    - ``max_input_tokens`` — the token window the constrained target can spend on
      one compiled context.
    - ``max_evidence_items`` / ``max_memory_items`` — selection caps that bound
      the candidate set a browser or edge worker assembles.
    - ``max_latency_ms`` — the latency budget the edge-scaling SLO is held
      against; the runtime measures and reports the compile latency against it.

    The defaults target an edge worker; :meth:`browser` is tighter and
    :meth:`server_like` is looser for parity testing against an unconstrained
    compile.
    """

    name: str = "edge"
    max_resident_bytes: int = Field(default=1_048_576, ge=1)  # 1 MiB
    max_input_tokens: int = Field(default=16_384, ge=64)
    max_evidence_items: int = Field(default=16, ge=1)
    max_memory_items: int = Field(default=8, ge=1)
    max_output_tokens: int = Field(default=1_024, ge=1)
    max_latency_ms: float = Field(default=100.0, gt=0.0)
    ordering: EdgeOrdering = "relevance"
    # Edge packets always slim (zero-copy: evidence text is referenced by hash,
    # not duplicated inline) and compress text evidence to fit the window.
    slim_packets: bool = True
    compress_evidence: bool = True

    def to_compiler_options(self) -> ContextCompilerOptions:
        """Lower this profile to the options the *same* context compiler reads.

        This is the parity contract in one method: an edge compile is the server
        compiler under a bounded option set, never a re-implementation. The
        resident ceiling, slim packets, evidence/memory caps, ordering, and
        compression all map onto the existing
        :class:`~vincio.context.compiler.ContextCompilerOptions` fields.
        """
        return ContextCompilerOptions(
            max_resident_bytes=self.max_resident_bytes,
            max_evidence_items=self.max_evidence_items,
            max_memory_items=self.max_memory_items,
            ordering=self.ordering,
            slim_packets=self.slim_packets,
            compress_evidence=self.compress_evidence,
        )

    @classmethod
    def browser(cls) -> EdgeProfile:
        """A tight profile for an in-browser (Pyodide/WASM) target."""
        return cls(
            name="browser",
            max_resident_bytes=262_144,  # 256 KiB
            max_input_tokens=4_096,
            max_evidence_items=8,
            max_memory_items=4,
            max_output_tokens=512,
            max_latency_ms=50.0,
        )

    @classmethod
    def worker(cls) -> EdgeProfile:
        """The default profile for an edge worker (e.g. a WASI runtime)."""
        return cls()

    @classmethod
    def server_like(cls) -> EdgeProfile:
        """A loose profile that mirrors an unconstrained server compile.

        Useful for parity testing: the edge runtime under this profile and a
        direct server-side compile should produce a byte-identical packet, since
        only the option set differs, never the code.
        """
        return cls(
            name="server_like",
            max_resident_bytes=8_388_608,  # 8 MiB
            max_input_tokens=100_000,
            max_evidence_items=24,
            max_memory_items=8,
            max_output_tokens=4_096,
            max_latency_ms=1_000.0,
        )

    @classmethod
    def default(cls) -> EdgeProfile:
        """The default bounded edge profile (an edge worker)."""
        return cls.worker()


class EdgeEnvironment(BaseModel):
    """A report of the runtime the edge core is executing on.

    ``is_wasm`` is true under a browser (Pyodide / CPython-on-Emscripten) or a
    WASI runtime; ``runtime`` names the detected host. The dependency-free core
    runs identically on every target — this only helps a caller pick a profile
    and decide whether thread- or filesystem-dependent extras are available.
    """

    platform: str
    runtime: Literal["pyodide", "emscripten", "wasi", "cpython", "unknown"]
    is_wasm: bool
    has_threads: bool
    has_filesystem: bool


def is_wasm_runtime() -> bool:
    """True when running on a WASM target (Emscripten/Pyodide or WASI)."""
    return sys.platform in ("emscripten", "wasi")


def edge_environment() -> EdgeEnvironment:
    """Detect the current runtime and report its edge-relevant capabilities.

    Detection reads ``sys.platform`` (``emscripten`` for Pyodide/browser builds,
    ``wasi`` for a WASI runtime) — never executing anything — so it is safe to
    call at import time on any host.
    """
    platform = sys.platform
    is_wasm = is_wasm_runtime()
    runtime: Literal["pyodide", "emscripten", "wasi", "cpython", "unknown"]
    if platform == "emscripten":
        # Pyodide exposes its loader module; otherwise it's a bare Emscripten build.
        runtime = "pyodide" if "pyodide" in sys.modules else "emscripten"
    elif platform == "wasi":
        runtime = "wasi"
    elif platform in ("linux", "darwin", "win32"):
        runtime = "cpython"
    else:
        runtime = "unknown"
    # Threads and a real filesystem are absent on a typical single-threaded WASM
    # build; the core never depends on either, but extras might.
    has_threads = not is_wasm
    has_filesystem = not is_wasm
    return EdgeEnvironment(
        platform=platform,
        runtime=runtime,
        is_wasm=is_wasm,
        has_threads=has_threads,
        has_filesystem=has_filesystem,
    )
