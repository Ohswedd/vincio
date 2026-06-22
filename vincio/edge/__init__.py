"""Edge / WASM in-process runtime — the same context engineering at the edge.

Vincio's dependency-free core (the prompt and context compilers, the vectorized
scorer with its pure-Python fallback, the deterministic rails, and the
offline-first evidence path) has no native dependencies on the default path.
This subpackage compiles that core for constrained and browser/WASM targets, so
the same compile → score → rail → pack pipeline runs at the edge and in the
browser — not only on a server, and not as a fork.

- :class:`EdgeRuntime` — a thin, synchronous, in-process boundary that turns an
  :class:`EdgeRequest` into an :class:`EdgeResult` (a bounded, slim packet plus
  the rendered prompt and rail outcome) with no model call, network, filesystem,
  or caller-owned event loop.
- :class:`EdgeProfile` — a bounded resident-memory and latency profile for a
  constrained target, held by the same mechanism as the server's resident-memory
  budget.
- :func:`verify_edge_parity` / :func:`edge_manifest` — the mechanical "parity,
  not a fork" guarantees: a byte-identical packet versus a server compile, and a
  static certificate that the core imports nothing native.
- :func:`edge_environment` — detect a WASM (Pyodide/WASI) host.
"""

from __future__ import annotations

from .parity import (
    EDGE_CORE_MODULES,
    EdgeManifest,
    EdgeParityReport,
    edge_manifest,
    verify_edge_parity,
)
from .profile import (
    EdgeEnvironment,
    EdgeProfile,
    edge_environment,
    is_wasm_runtime,
)
from .runtime import EdgeRequest, EdgeResult, EdgeRuntime

__all__ = [
    "EdgeRuntime",
    "EdgeRequest",
    "EdgeResult",
    "EdgeProfile",
    "EdgeEnvironment",
    "edge_environment",
    "is_wasm_runtime",
    "EdgeManifest",
    "EdgeParityReport",
    "edge_manifest",
    "verify_edge_parity",
    "EDGE_CORE_MODULES",
]
