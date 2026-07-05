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

### How it works: compile → score → rail → pack

`run` executes the *same* four stages the server pipeline does, with the model,
network, store, and event loop stripped away:

```
EdgeRequest ─▶ compile (chunk + assemble) ─▶ score (relevance/authority + MMR)
            ─▶ rail (input + rendered-context screen) ─▶ pack (budget + slim)
            ─▶ EdgeResult (prompt · packet · footprint · rail outcome)
```

There is no separate edge compiler: the `EdgeProfile` lowers to the
`ContextCompilerOptions` the canonical `ContextCompiler` already reads, so the
edge path *delegates* to the server's compiler rather than re-implementing it —
which is exactly what `verify_edge_parity()` asserts byte-for-byte.

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

## Best practice & gotchas

- **Pick the profile by host, not by guess.** `edge_environment()` reports the
  runtime; use `EdgeProfile.browser()` under Pyodide/Emscripten (tight),
  `.worker()` for an edge worker, and `.server_like()` only for parity testing.
- **`within_profile=False` is a soft report; `strict=True` is a hard stop.** By
  default an over-budget packet is slimmed and evicted until it fits, and
  `within_profile` tells you whether it did; pass `strict=True` to raise
  `EdgeError` instead when a packet cannot be held inside the bounds.
- **Output rails screen the *rendered context*, not just the task.** A secret
  that leaked from a retrieved document into the assembled prompt is only caught
  if a rail runs with `direction="output"` — otherwise the leak ships.
- **The footprint is held by construction.** Evidence text is referenced by hash,
  not duplicated, and the lowest-utility evidence is evicted until the estimate
  fits, so `result.resident_bytes <= profile.max_resident_bytes` holds even as the
  candidate corpus grows.
- **No provider, store, or tracer lives here.** `EdgeRuntime` is the offline
  context core only; bring the portable packet/prompt back to a server (or call a
  model from the host) to generate. NumPy is used only behind a pure-Python
  fallback, so the shipped edge path stays dependency-free.

## What does *not* run at the edge

The edge runtime is deliberately the offline context-engineering core and
nothing else. A model call, a provider, a persistent store, retrieval over a
remote index, and the optional native extras (PyAV video decode, a GGUF model,
NumPy acceleration) are server-side concerns, the edge path keeps the
dependency-free default. Bring the edge result back to a server (or call a model
from the host environment) to generate; the packet and prompt are portable.

The runnable example is
[`11_advanced_context.py`](../../examples/11_advanced_context.py).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 11_advanced_context.py](../../examples/11_advanced_context.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#serving)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
