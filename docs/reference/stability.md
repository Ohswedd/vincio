# Reference: API stability & deprecation policy

From **1.0**, Vincio follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
on its public API. This page is the contract: what's covered, what isn't, and
exactly how things get removed.

## What is "public API"

The public surface is:

- Every symbol re-exported from the top-level `vincio` package — i.e.
  `vincio.__all__`, also returned by `vincio.stability.public_api()`.
- The documented entry points of each subpackage listed in the
  [API reference](api.md).
- The `vincio` CLI commands documented in the [CLI reference](cli.md).
- The config schema (`vincio config schema`) field names and semantics.

Everything else is internal and may change at any time without notice:

- Any name beginning with an underscore (`_helper`, `vincio.core._runtime`).
- Modules and attributes not listed in the API reference.
- Symbols marked [`@experimental`](#experimental-apis).
- The exact wording of log lines, trace span internals, and error messages
  (error *types* and `.code` values are stable; message strings are not).

## Versioning guarantees

Within a major version (`1.x.y`):

| Bump | Example | Guarantee |
|---|---|---|
| **PATCH** | `1.0.0 → 1.0.1` | Bug fixes only. No public behavior changes. |
| **MINOR** | `1.0.0 → 1.1.0` | Additive only: new symbols, new **optional** parameters with defaults. Existing code keeps working. |
| **MAJOR** | `1.x → 2.0.0` | May remove or change public API — but only after the deprecation contract below. |

`vincio.API_VERSION` (`"1.0"`) is the contract version SemVer is applied
against; it changes only on a major bump.

## Deprecation contract

The policy is mechanical, not just documented:

1. A public symbol is **never removed in a minor or patch release.** It is first
   marked deprecated in a **minor** release.
2. While deprecated, calling it emits a `VincioDeprecationWarning` naming the
   version it was deprecated in, the version scheduled for removal, and the
   replacement.
3. Removal happens no earlier than the **next major** release.

```python
from vincio import deprecated, experimental, stability_of

@deprecated(since="1.2", removed_in="2.0", alternative="new_api")
def old_api(...): ...

@experimental(since="1.0", note="signature may change")
def fresh_api(...): ...

stability_of(old_api)
# {'level': StabilityLevel.DEPRECATED, 'since': '1.2',
#  'removed_in': '2.0', 'alternative': 'new_api', 'note': None}
```

`VincioDeprecationWarning` subclasses the built-in `DeprecationWarning`, so it's
silent by default but visible under `python -W` and in test runs. To make
deprecations hard errors in your CI:

```python
import warnings
from vincio import VincioDeprecationWarning

warnings.simplefilter("error", VincioDeprecationWarning)
```

### Renamed symbols

A renamed function keeps its old name working via a forwarding alias for one
major cycle:

```python
from vincio.stability import deprecated_alias

new_name = _impl
old_name = deprecated_alias(new_name, old_name="old_name",
                            since="1.2", removed_in="2.0")
```

## Experimental APIs

Symbols marked `@experimental` are public and usable but carry **no stability
guarantee** — they may change or be removed in any release, including a minor.
They emit a one-time `VincioExperimentalWarning` per process so their status is
visible without being noisy. Use them, but pin your Vincio version if you
depend on their exact shape.

The **1.1 protocol & interoperability surfaces are experimental** while the
underlying standards settle: `vincio.mcp` (MCP client/server), `vincio.a2a`
(agent-to-agent), `vincio.skills` (Agent Skills), and the `ContextApp` methods
`add_mcp_server` / `serve_mcp` / `add_skill` / `serve_a2a` (which emit the
warning). The unified reasoning controls (`RunConfig.reasoning_effort` /
`thinking_budget_tokens`) and the `OpenAIResponsesProvider` are additive and
do not change existing behavior.

The **1.2 continuous-quality entry points are experimental** while their shape
settles: the `ContextApp` methods `add_online_evaluator`, `experiment`, and
`add_metric_rail` (which emit the warning). The rest of 1.2 is stable, additive
API: the new trajectory/conversational metrics in `METRICS`, the `Trajectory`
model and `RunOutput.from_*` constructors, `Simulator` / `DriftMonitor` /
`AnnotationQueue` / `Experiment` / `metric_guardrail`, `dataset_from_traces`'s
`group_by_session` parameter, and the new `vincio eval drift|annotate` commands.

None of this removes or repurposes any 1.0 public symbol — upgrading across the
1.x line never breaks working code.

## Breaking windows: 2.0 and 3.0

`API_VERSION` (returned by `vincio.stability.API_VERSION`) tracks the frozen
public-API contract and bumps only on a deliberate breaking window. There have
been two: **2.0** (the structural refactor — facades, async-first stores, the
multimodal-native packet, enterprise endpoints) and **3.0** (the unified
self-improvement contract, provable erasure with consent modeling, and the
async-canonical core). Each shipped through the mechanical deprecation runway
above, and nothing breaks *outside* a window.

### Collapsed at 3.0

3.0 collapses the separately-wired self-improvement convenience methods into the
unified `app.self_improvement(...)` contract. The superseded methods are **kept
working** through the entire 3.x line and removed no earlier than 4.0:

| Deprecated (`since=3.0`, `removed_in=4.0`) | Replacement |
| --- | --- |
| `app.continuous_improvement(...)` | `app.self_improvement(policy)` — the policy's `online` arm |
| `app.experiment_proposer(...)` | `app.self_improvement(policy)` — the policy's `propose` arm |

The underlying organs (`ContinuousImprovementController`, `ExperimentProposer`,
`ImprovementLoop`) remain public; only the app-level convenience entry points are
deprecated in favour of the one contract. The 3.0 additions —
`SelfImprovementPolicy` / `SelfImprovementController` / `app.deploy`, provable
erasure (`ErasureProof`), the `ConsentLedger`, and the bi-temporal `MemoryItem`
fields (`valid_from` / `valid_to` / `acl` / `purpose`) — are all `@experimental`
while their shape settles; the new `MemoryItem` fields default backward-compatibly.

## Supported versions

See [SECURITY.md](https://github.com/Ohswedd/vincio/blob/main/SECURITY.md) for
which release lines receive security fixes.
