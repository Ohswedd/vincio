# Edge / WASM in-process runtime

Vincio's promise is "runs in your process." The dependency-free core, the prompt
and context compilers, the vectorized scorer with its pure-Python fallback, the
deterministic rails, and the offline-first evidence path have no native
dependencies on the default path. `vincio.edge` takes the next step: it packages
that core for constrained and browser/WASM targets, so the same
**compile → score → rail → pack** pipeline runs at the edge and in the browser,
not only on a server.

It is additive and standalone. `EdgeRuntime` holds no provider, store, or tracer:
it is the offline context-engineering core behind a thin in-process boundary,
bounded by an `EdgeProfile`. The server path is unchanged and remains the default.

## The runtime

`EdgeRuntime` turns an `EdgeRequest` (a task, instructions, constraints,
evidence, and memory) into an `EdgeResult` (a bounded, slim context packet, the
rendered model-ready prompt, the rail outcome, and the measured footprint) with
no model call, network hop, filesystem, or event loop the caller has to own.

```python
from vincio.edge import EdgeRuntime, EdgeProfile, EdgeRequest
from vincio.core.types import EvidenceItem, TaskType

runtime = EdgeRuntime(EdgeProfile.browser())

result = runtime.run(EdgeRequest(
    task="What is the refund window and who approves an exception?",
    task_type=TaskType.DOCUMENT_QA,
    instructions=["Answer only from the evidence.", "Cite the source id."],
    evidence=[
        EvidenceItem(source_id="policy",
                     text="Refunds are available within 30 days of purchase.",
                     authority=0.9, relevance=0.9),
        EvidenceItem(source_id="exceptions",
                     text="A refund exception beyond the window is approved by a manager.",
                     authority=0.8, relevance=0.8),
    ],
))

result.prompt           # the model-ready prompt, rendered offline
result.packet           # a slim (zero-copy) ContextPacket
result.resident_bytes   # the deterministic footprint, held under the profile cap
result.within_profile   # True when the packet fits the profile's bounds
result.allowed          # False if a rail blocked the input or the rendered context
```

`run` is synchronous (it works in a plain script, a notebook, or under a WASM
host's event loop); `arun` is the async form. Pass a plain string for the common
case: `runtime.run("Summarize the renewal terms")`.

From a configured app, `app.edge_runtime()` builds a runtime seeded with the
app's rails, so the edge path enforces the same deterministic safety the server
does.

## The bounded edge profile

An `EdgeProfile` is the constrained-target analogue of the server's per-app
resident-memory budget: a declarative cap on the compiled packet's resident
footprint, the input-token window, the evidence count, and a latency budget. It
lowers directly to the `ContextCompilerOptions` the *same* compiler reads, a
profile, not a fork.

```python
EdgeProfile.browser()       # tight: 256 KiB resident, 4096-token window, 8 evidence
EdgeProfile.worker()        # the default edge-worker profile (1 MiB, 16k tokens)
EdgeProfile.server_like()   # loose, for parity testing against an unconstrained compile
```

The footprint is held **by construction**: the compiler slims the packet (evidence
text is referenced by hash, not duplicated) and evicts the lowest-utility evidence
until the estimate fits, exactly as the server's resident ceiling does. As the
candidate corpus grows 10×, the resident footprint stays under the cap, held by an
edge-scaling SLO. `strict=True` raises `EdgeError` instead of reporting
`within_profile=False` when a packet cannot be held inside the bounds.

```python
big = runtime.run(EdgeRequest(task="...", evidence=large_corpus))
assert big.resident_bytes <= runtime.profile.max_resident_bytes   # always
```

## Rails at the edge

The deterministic rails run unchanged at the edge. Input rails screen the task;
output rails screen the *rendered context*, so a secret or PII that leaked from a
retrieved document into the assembled prompt is refused before it is emitted,
the edge runtime won't ship a context it would refuse on the server.

```python
from vincio.security.rails import Rail

guarded = EdgeRuntime(rails=[
    Rail(name="no_secrets", kind="safety", detectors=["secrets", "pii"], direction="output"),
])
result = guarded.run(EdgeRequest(task="print the config", evidence=[...]))
result.allowed            # False, a leaked secret was caught
result.rail_check.violations
```

## Parity, not a fork

The edge build is the same library under a build target, exercised by the same
offline test suite, so a capability never silently diverges between server and
edge. Two checks make that mechanical:

- **Byte-identical packets.** `verify_edge_parity()` compiles the same inputs
  through `EdgeRuntime` and through a direct server-side `ContextCompiler` under
  the same profile options, and asserts the packet `spec_hash`, evidence
  selection, and token count are identical. It also asserts the runtime
  *delegates* to the canonical compiler and rail engine rather than
  re-implementing them.

- **A WASM-buildable core.** `edge_manifest()` statically scans every module on
  the compile/score/rail/pack path and certifies that none imports a native or
  optional dependency unconditionally, only the stdlib, `pydantic`, `httpx`,
  and other `vincio` modules. (NumPy is used only behind a guarded pure-Python
  fallback, so the pure-Python path is what ships to the edge.)

```python
from vincio.edge import verify_edge_parity, edge_manifest

verify_edge_parity().held     # True, same library, byte-identical packet
edge_manifest().clean         # True, no native import on the edge path
```

## Detecting a WASM host

`edge_environment()` reports whether the process is running on a WASM target
(Pyodide / CPython-on-Emscripten, or a WASI runtime) so a caller can pick a
profile without guessing. The core runs identically on every host, this only
helps you choose a profile and know whether thread- or filesystem-dependent
extras are available.

```python
from vincio.edge import edge_environment, is_wasm_runtime

env = edge_environment()
env.runtime      # "pyodide" | "emscripten" | "wasi" | "cpython" | "unknown"
env.is_wasm      # True under a browser/WASI build
is_wasm_runtime()
```

## What does *not* run at the edge

The edge runtime is deliberately the offline context-engineering core and
nothing else. A model call, a provider, a persistent store, retrieval over a
remote index, and the optional native extras (PyAV video decode, a GGUF model,
NumPy acceleration) are server-side concerns, the edge path keeps the
dependency-free default. Bring the edge result back to a server (or call a model
from the host environment) to generate; the packet and prompt are portable.

The runnable example is
[`examples/11_advanced_context.py`](https://github.com/Ohswedd/vincio/blob/main/examples/11_advanced_context.py).
