# Security Policy

## Supported versions

Vincio follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Security fixes land on the latest minor of the current major. Older majors are
not maintained — upgrade to the latest release.

| Version | Supported |
| ------- | --------- |
| 3.x     | ✅ |
| < 3.0   | ❌ (upgrade to the latest 3.x release) |

## Threat model

Vincio is a library you run on your own infrastructure. What it defends against
(prompt injection, secret/PII leakage, cross-tenant access, audit tampering,
runaway tool execution) and what it explicitly does not (kernel-level sandbox
escape, a compromised host/provider) is documented in detail in the
[threat model](docs/security/threat-model.md). Security, permissions, and
validation are enforced deterministically in code — never gated on model output.

A principle runs through every section below: Vincio adds capability **in your
process**, not new external services. Observability, evaluation, distribution, the
agent fabric, and the benchmark suite all run on your own infrastructure, on the
same hash-chained audit chain, never as a hosted control plane that widens your
exposure surface. Online-eval scores, drift baselines, and human annotations are
written to your own metadata store; the user simulator runs against your app, and
judge calibration uses your own labelled data — none of it is mirrored to an
external service.

### Untrusted input & prompt injection

All retrieved and ingested content is treated as untrusted. The prompt-injection
detector runs a normalization + decode pre-pass — NFKC fold, zero-width strip,
homoglyph/leetspeak fold, and recursive (depth- and size-bounded) base64/hex/rot13
decode — before its signals, so an obfuscated attack is scored against its decoded
form. **Authority/provenance RAG-poisoning detection** scores retrieved evidence,
and the PII / injection / secret detectors accept a pluggable `DetectorBackend`
where an ML model *merges with*, never replaces, the deterministic rules.

Untrusted content enters through one provenance-stamped pipeline regardless of
source: MCP resources, OCR'd pages, transcripts, figure crops, and extracted form
fields all arrive as injection-scanned evidence (with the extractor recorded for
honesty), exactly like any loaded file. Optional cloud Document-AI / transcription
backends are lazy and opt-in.

**Containment that holds when detection misses.** Detection is necessary but not
sufficient — an attacker needs only one missed instruction. The control plane is
therefore separated from the data plane. Provenance is promoted to a typed
`TrustLabel` (`trusted` / `untrusted` / `quarantined`) that propagates through
`TaintedValue` derivations and `ContextPacket.materialize()`, so a value derived
from untrusted data is tainted end-to-end. A `DualPlaneExecutor` runs a
privileged planner that never sees untrusted bytes — only typed, schema-validated
extractions of them — and gates every side-effecting tool call on an unforgeable
`CapabilityToken` minted by a `CapabilityBroker` from the *user's* request
(HMAC-signed, principal- and argument-scoped, TTL-bounded). An argument carrying
an untrusted taint cannot reach a write/external tool without a capability or an
explicit human approval; the attempt is refused (`ContainmentError`) and recorded.
A `ContainmentMonitor` lets `verify_containment` prove the invariant
`untrusted ⇒ no unapproved capability` held across a whole run, and the VincioBench
`containment` family holds the escalation rate at **0** on an adversarial corpus.
The guarantee rests on capability-key secrecy, not on detecting the attack;
capability-scoped tools are also available at the permission layer via
`ToolPermissionChecker(broker=...)`.

### Formal verification of governance invariants

The controls above are *enforced* at runtime and recorded on the audit chain. A
`GovernanceVerifier` (`app.verify_governance()`) adds the rung beside that: a
**machine-checkable proof that the governance invariants hold across their whole
bounded, typed state space, ahead of any run**. Four invariants — injection
containment (`untrusted ⇒ no unapproved capability`), in-jurisdiction residency, the
budget hard cap, and the erasure-proof content binding — are stated as formal
specifications over the pipeline's typed state and checked exhaustively by a
deterministic in-process verifier. The verifier binds to the *same* decision
functions the runtime uses (the containment gate is `requires_authority`; the erasure
binding is `verify_erasure_proof`), so a holding verdict is a proof about the shipped
machinery, not a re-implementation. A violation returns a concrete, delta-minimized
**counterexample** — the input, the labels, the capability gap — so a governance
regression is debuggable, not merely flagged; the residency invariant reflects the
app's own `deny_on_unknown` posture, so a fail-open configuration is caught. The
content-hashed verdict is reproducible and lands on the hash-chained audit log as a
`governance_verification` decision, computed in-process with **no external prover
service**. The VincioBench `verification` family holds a property-holds, a
counterexample-on-violation, and an auditable-offline SLO. See the
[verification guide](docs/guides/governance-verification.md).

### Secret & PII leakage, and egress control

Deterministic PII / secret detection and redaction runs in-process, with
non-English locale packs (France/Germany/Spain/India/Singapore/Brazil/UK)
extending the built-in patterns. An output `redact` rail masks detected
identifiers in the deliverable — including the string fields of a **structured**
output, with the schema and field types preserved — so a typed result (e.g. a
clinical or KYC assessment from a vertical pack) never ships an identifier the
rail caught.

A **mandatory egress DLP scan** (`PolicyEngine.scan_egress`, mode
`security.egress_dlp`: `off` / `warn` / `block`) inspects the *fully-assembled*
provider request — system prompt, every message, and tool schemas — at both the
non-streaming and streaming dispatch points, independent of how earlier checks
were wired. It is the always-on last line of defense: a call site that bypassed
every other check still passes through it, and in `block` mode an outbound
credential or sensitive identifier raises `EgressBlockedError`, recorded as a deny
on the audit chain.

Telemetry is data-minimizing by default. A `ContentCapturePolicy` gates content
at the *export boundary* — the OpenTelemetry exporter and the tool runtime — so
structural telemetry (model, tokens, cost, latency, scores) exports while raw
prompt/completion text is dropped unless you explicitly opt in, and when you do it
is PII-redacted and truncated first.

A causal **`Recording`** (`Recorder` / `Replayer` in `vincio.observability`) is a
deliberate exception: to replay a run byte-for-byte it captures the *full* model
responses, tool outputs, and retrieval hits — not the truncated previews trace
spans keep — so a recording is a complete record of a run's inputs and outputs.
Treat it like any export of model inputs/outputs: it may contain business data and
secrets, so store it where you would store the underlying data, not in a shared
artifact bucket. Recording is never on the live path unless you wrap a run with
the recorder, recordings are content-addressed and carry a verifiable fidelity
digest (`recording.verify()`) so tampering is detectable, and replaying one runs
entirely in-process against the recorded edges (no new egress).

The **learned semantic cache** (`LearnedSemanticCache`, `app.use_semantic_cache`)
serves a *near-miss* — a recent answer to a semantically-equivalent, not
byte-identical, request — so it carries two risks the design contains explicitly.
**Correctness:** a near-miss is served only at or above an acceptance threshold
*calibrated from labelled traces* to clear a precision target, and when the target
is unreachable the threshold falls back to `1.0` (off) rather than guess — so the
cache never serves below the bar. Every accepted hit is auditable (`cache.audit()`)
and reversible (`cache.revoke(key)`), and a cache whose calibration has drifted is
caught by `SemanticCacheGate` — the same eval-replay no-regression check that gates
a model swap — before it ships. **Isolation:** entries are partitioned by
`policy_scope` (model + stable prompt head) and output schema, so a cached answer
is never served across a tenant or policy boundary; a policy / schema / scope
change clears the cache through the same `InvalidationManager` that clears the
exact-match caches. The cache holds the response payload and the query embedding
only, bounded under the resident-memory budget, and is consulted on the live path
only when you opt in.

### Multi-tenant isolation & access control

Access is governed by RBAC scopes and ABAC rules through one `AccessController`.
Tenant/ACL scope is pushed **into the retrieval engine**: `app.tenant_filter`
returns a structured `FilterSpec` the vector store applies server-side (Qdrant
native filter, pgvector `jsonb` `WHERE`, and the other supported backends), so
other tenants' rows are never fetched to the client and dropped — closing the
fetch-to-filter exfiltration gap. Isolation can **fail closed**:
`AccessController(require_explicit_tenant=True)` stops treating an untagged
resource as globally readable.

When you expose a server, the boundary is enforced too: an MCP/A2A server you run
validates bearer tokens (OAuth 2.1 resource server) and applies the policy engine
and audit log on every inbound call.

### Tool & code execution

MCP tools (consumed from a server), Agent-Skill bundled scripts, computer-use
actions, and provider-native hosted tools all run through the **same
permissioned, sandboxed, audited, budgeted runtime** as any local tool. The
default sandbox is OS-process isolation with `setrlimit` CPU/memory/fd limits.

That default is explicitly **not a security boundary** for adversarial code: the
pluggable `IsolationBackend` flags the subprocess backend as `real=False`, and
`require_real_isolation` refuses to run code-executing or computer-use workloads
on it — those must run behind a container / microVM / gVisor / WASM backend.
Computer-use and `code_interpreter`/`computer_use` hosted tools register as
approval-gated, `external`-side-effecting tools on the same RBAC + audit + budget
path, so a hosted capability is governed exactly like a local one. Write actions
are idempotent and approval-gated.

### Runaway execution & cost

The advertised `Budget` is a hard cap: cost / token / step overruns raise
`BudgetExceededError` (and `meter_media_cost` raises it *before* an over-budget
image/audio generation commits), recorded on the same audit chain. An unknown
model warns rather than silently billing $0. Agents and crews are bounded and
guaranteed to terminate; budget breaches surface as policy violations on the audit
path. A declared per-app resident-memory ceiling (`performance.memory_budget_mb`)
bounds the compiled context packet against memory exhaustion: when the selected
context would exceed it, the compiler slims the packet and evicts the
lowest-utility evidence to fit, recording each eviction in the excluded report
and surfacing the footprint as `RunResult.memory_bytes`.

### Self-improvement safety

Self-modification is **gated and reversible**. The online improvement controller
acts only on a sustained, debounced drift signal, spends a bounded global eval
budget, and takes one gated, audited action — a re-eval, a significance-gated
re-optimization, or a registry rollback to the last known-good prompt — with a
held-out, growing golden regression suite blocking any promotion that regresses a
prior fix, so an autonomous promotion can never silently undo earlier work.
Guarded online bandits carry a **safety floor**: they never explore on safety- or
high-risk-tagged traffic, track per-arm regret, and auto-freeze / roll back to the
safe arm on regression. A trained student model is promoted only past the
significance swap gate, so the distillation flywheel cannot silently ship a
regression. The self-editing memory OS exposes memory mutation only as
permissioned, audited tools over the guarded write pipeline.

**On-device adaptation** keeps the same discipline without a network round-trip.
The `LocalLoRATrainer` fits a LoRA-class adapter **in your process** from the
flywheel's grounded data — no training traffic leaves the machine, so an
air-gapped or edge deployment can improve on its own traffic without exporting
it. A new adapter version is promoted only when the `ContinualAdaptation` loop's
no-regression gate (the on-device analogue of the swap gate) confirms the adapted
model is at-least-as-good as its base on a held-out set; the adapter is **bounded**
(it reshapes only in-distribution requests, leaving unseen traffic to the base
model), every version is **content-addressed and versioned** in the
`AdapterRegistry`, and a regressing one is refused and rolled back — applied or
unloaded live via `app.use_local_adapter`, every decision on the audit chain.

**Federated / cross-org self-improvement** extends that discipline *across* trust
boundaries without weakening it. The only thing that leaves a member is a
`Contribution`: the `d × d` weighted **scatter** of its local prompt-embedding
subspace — a second-moment sufficient statistic from which no individual prompt or
response is recoverable — never the raw traffic. Three layers bound what a member
exposes, set by `PrivacyConfig`: **clipping** caps each contribution's Frobenius
norm, bounding one member's sensitivity (its maximum influence on the merged result)
and braking a poisoned outlier; an optional **differential-privacy** Gaussian
mechanism (`dp_epsilon`/`dp_delta`) makes the merged scatter `(ε, δ)`-private with
respect to any single example; and **secure-aggregation** masks make an individual
update indistinguishable from noise on the wire — the pairwise masks cancel only
when summed across the exact participant set, so the `SecureAggregator` recovers the
fleet geometry without ever observing one member's update. A round below the
`min_contributors` k-anonymity floor, or one mixing base models, embedding
dimensions, or disallowed residency regions, is **refused**. A contribution is gated
behind the consent ledger's TRAINING purpose (`require_consent`) and stamped with the
member's residency tag. Adoption is reversible and gated: the adopting member re-fits
its **own** adapter against the shared subspace (keeping its grounded answers local)
and adopts it only when the same no-regression gate confirms it is at-least-as-good
as the base — versioned in the `AdapterRegistry`, rolled back on regression, every
decision on the audit chain. The mask seeds and DP noise are deterministic for
offline testing; a production deployment derives the mask seeds from a key-agreement
protocol rather than a shared seed.

**Cross-fleet reputation** adds a defense against a member that contributes
*consistently harmful* geometry. Equal-weight aggregation lets a repeatedly-regressing
or adversarial member pull the shared consensus as hard as a reliable one; a
`ReputationLedger` (`app.use_reputation_ledger`) earns a per-member reliability score
— a Beta-Bernoulli posterior over how each past contribution fared against the
no-regression gate, accrued only from gate verdicts on the signed audit chain, never
from raw traffic — and the `SecureAggregator` weights a member's contribution by that
score, **discounting an unreliable member without singling it out**. The weight is
folded into the contribution *before* the secure-aggregation masks, so the masks still
cancel exactly; a masked contribution must carry the assigned weight, and the
aggregator refuses to re-weight it after the fact (which would break cancellation).
The discount is **bounded and reversible by construction**: a weight never leaves
`[weight_floor, 1]`, so a bad reputation only ever *lowers* a member's pull — it can
never zero a member out, never raise one past parity, and never bypass the quality
bar, because adoption still clears the same no-regression and canary gates. Reputation
changes only which geometry the fleet converges toward when every candidate already
passes the gate; it is a discount on influence, not a substitute for the gate. Every
update is on the audit chain and reconstructable from it (`ReputationLedger.from_audit`),
so a member's standing is a mechanical, auditable, tamper-evident number.

### Audit integrity & tamper-evidence

Every run lands on an append-only, hash-chained audit log, verifiable offline
(`vincio audit verify`). The chain is **tamper-evident against a privileged
attacker**: with a `ChainSigner` configured (`security.audit_signing_key` for
HMAC, or an `Ed25519Signer` for third-party verifiability) every entry is signed
over its `entry_hash`, so forging history requires the key, not just the public
hash algorithm. Periodic Merkle-root checkpoints (`AuditLog.checkpoint`) let a
root be witnessed externally to pin history irreversibly. Unsigned logs still
verify their chain.

Distributed execution stays inside this boundary: a TTL lease + checkpoint-version
CAS guarantees exactly-once super-step execution across workers (a lost race
raises `CheckpointConflictError`, never a double-write), and Redis-backed shared
rate-limit / idempotency state enforces one coherent limit across a multi-worker
fleet.

### Provenance & generated media

Every generated image, audio, and video asset auto-attaches a media-aware C2PA
manifest bound to the asset's bytes by SHA-256 (a tampered asset fails
`verify_manifest`), embedded in file metadata where the container supports it or as
a `*.c2pa.json` sidecar otherwise, with edits marked, and each generation is metered
against the run `Budget` and recorded as an `image_generate` / `speech_synthesize` /
`video_generate` (or `video_edit`) audit event — so synthetic video is as
tamper-evident as a generated image or a text answer. Manifests can be cryptographically **signed**
(built-in symmetric `HmacSigner`, or your own asymmetric `ContentSigner`); Vincio
assumes no signing authority — the key and any PKI are yours, and the
invisible-watermark hook is a point you supply. Document generation is grounded by
construction: the `DocumentBuilder` consumes an already-validated result against a
`DocumentContract`, repairs formatting only, never invents content, and fails
loudly (`DocumentContractError`) rather than silently padding a deficient
deliverable; the cited-report path verifies per-claim entailment.

### Data rights & governance

Governance evidence is *generated from data Vincio already holds* — model/system
cards, the OWASP LLM / OWASP Agentic / NIST AI RMF / MITRE ATLAS / ISO IEC 42001
coverage matrix, and the AI-BOM are views over the live config, audit chain, eval
reports, and price table. A control reads as `covered` only when backed by
*measured* red-team / eval evidence; a configured-but-unmeasured control is
`partial`. The EU AI Act conformity pack (`app.risk_tier` / `annex_iv` / `fria`)
is recorded as `conformity_doc` audit events, with the risk-tier classification
advisory and the final determination the operator's.

**Data residency** resolves the provider region from a region-pinned endpoint with
jurisdiction-aware matching, then refuses to *send* a request to a disallowed
region (a blocking policy violation on the audit path) — since Vincio cannot
guarantee where a global provider runs a request, the strongest posture is a
region-pinned endpoint plus this client-side egress refusal.

**Erasure is provable, not merely logged.** `app.erase_source` purges a source
from every index, memory, and cache and returns a signed, content-bound
`ErasureProof`: a manifest of exactly which chunk / document / memory /
generated-artifact ids were removed, bound by SHA-256 over the sorted removed-id
set (tampering breaks `verify_erasure_proof`), signed with the app's
`content_signer`, and anchored to the audit chain's Merkle root — so a
right-to-erasure claim is checkable offline against the precise removal. A
`ConsentLedger` binds a data subject to a GDPR `Purpose` and `LawfulBasis`;
`AccessController.check_purpose` consults it and memory recall drops any item whose
purpose lost consent, so purpose limitation is enforced in code. Memory is
bi-temporal (`valid_from` / `valid_to`, as-of recall) with per-memory ACLs and a
`TEAM` scope, so team-shared memory surfaces only to permitted readers and a
corrected fact never silently rewrites history. Every grant, revoke, denied check,
and erasure proof lands on the audit chain.

**A per-subject differential-privacy budget bounds what learning can leak.** The
consent ledger answers *"may we process this, and for what?"*; the privacy
accountant answers *"how much has a subject's data already leaked into what we
learned, and is there budget left?"*. `app.use_privacy_accountant(...)` attaches a
Rényi/moments `PrivacyAccountant` that composes the cumulative `(ε, δ)` a subject's
data spends across **every** memory consolidation and federated contribution into
one running budget — far more tightly than naively summing each step's `ε`. A
release that would push a subject past their `PrivacyBudget` is **refused** (the
privacy analogue of a hard cost cap) or **down-weighted** (clipped harder so its
sensitivity, and therefore its privacy cost, fits the remaining budget). Every
spend and refusal lands on the same hash-chained audit log, and
`app.privacy_report()` rolls up each subject's spent / remaining `ε` next to the
cost report — so the privacy guarantee is a mechanical, auditable number, not a
policy doc. The accountant is opt-in and additive: with none attached, consolidation
and contributions are unaccounted exactly as before.

**A run's energy and carbon are an auditable, in-process number — never an external
call.** Sustainability-reporting regimes are beginning to require disclosure of a
workload's energy and carbon footprint. `app.use_energy_accounting(...)` accrues a
per-run **energy** (watt-hours) and **carbon** (grams CO₂e) estimate on the same
cost-report surface, computed **entirely in-process** from the run's own token
accounting against a built-in, deterministic intensity table (a per-model factor by
tier and a per-region grid factor) — **no external service is consulted**, so enabling
the estimate opens no new egress channel and leaks no run metadata to a third party.
The estimate is mechanical and reproducible: the same run yields the same number. It is
**budgeted like a dollar** — `app.set_energy_budget(...)` sets an energy or carbon
envelope, and a run that would exceed it is **refused** on the same audit path as a hard
cost cap (an `energy_budget` decision on the chain). Both the per-run estimate and every
refusal land on the hash-chained, tamper-evident audit log, so the sustainability figure
an auditor sees is a verifiable number, not a vendor claim. Accounting is off by default
and additive: with it disabled, a run behaves exactly as before.

### Edge / WASM runtime — the same deterministic safety, offline

**The edge runtime carries the deterministic rails to a constrained target without
weakening them.** `vincio.edge.EdgeRuntime` packages the dependency-free
compile → score → rail → pack core for a browser/WASM or edge-worker target behind a
thin in-process boundary. It runs the **same** `RailEngine` the server does, so the
deterministic PII / secret / injection detectors enforce at the edge exactly as in a
server run: input rails screen the task, and **output rails screen the rendered
context**, so a secret or PII that leaked from a retrieved document into the assembled
prompt is refused before the prompt is ever emitted. The runtime holds **no provider,
store, network, or filesystem** — it compiles a context and renders a prompt offline,
opening no egress channel; generation stays a server-side concern. Resource bounds are
enforced by construction: an `EdgeProfile` caps the compiled packet's resident footprint
and token window, held under the cap by the same slimming + eviction the server's
resident-memory budget uses, so a malicious or oversized corpus cannot exhaust a
constrained host. And the edge build is **parity, not a fork**: `verify_edge_parity()`
proves an edge compile is byte-identical to a direct server compile, and
`edge_manifest()` statically certifies the core path imports nothing native — so a
safety control can never silently diverge between the server and the edge.

### Governed discovery & interoperability

Agent and tool discovery is governed by construction: an `AgentDirectory` resolves
an agent or MCP server only through an `AllowListGate` — a fail-closed allow-list
(deny patterns first, then allow; default deny) over the same `AccessController`
the data plane uses — and **every resolution is recorded as an `agent_resolve`
access decision** on the audit chain, so an unlisted agent is unreachable and each
reachable one is accountable. AGNTCY/ACP and MCP-registry discovery normalize
remote records into the same governed directory; discovery never auto-trusts a
result. The **MCP-server marketplace bridge** (`app.add_mcp_from_registry`) runs
the same gate: a server discovered from a registry is resolved through a governed
`AgentDirectory` before any tool is registered, so reachability is an audited
decision and an unlisted server is refused — discovery and connection never
bypass governance. Generative-UI (AG-UI) streaming opens no new data channel — it
is a translation of the run's existing stream, inheriting its provenance, budget
metering, and audit.

The **community pack & skill registry** (`CommunityRegistry`) extends the same
model to opt-in content. Each bundle is **content-bound** (SHA-256 over its
payload) and may be **signed** with a `ChainSigner` (HMAC, or Ed25519 so a
consumer verifies with only the public key); resolution passes the same
`AllowListGate`, verifies the digest and signature, and records an audited
`bundle_resolve` decision — a tampered, unsigned-when-required, or unlisted
bundle is denied rather than loaded. Vincio ships the engine; the signing keys
and any PKI are yours.

**Third-party plugins execute in your process.** The `vincio.plugins` entry-point
system imports and runs code from any installed distribution advertising a
`vincio.<kind>` entry point — treat plugins like any dependency and vet them
before installing. Discovery (`installed_plugins()`) does **not** import target
objects, so listing what is installed is side-effect-free; only `load_plugins()`
(and the lazy auto-load on a registry name miss) imports them, and a plugin that
fails to import is isolated and reported, never breaking the rest. A distribution
that declares an incompatible plugin-API major is reported and **not** loaded.

### Data you export

The distillation flywheel writes a fine-tuning **JSONL file you own** from the
runs/traces you choose; treat it as any export of model inputs/outputs — it may
contain business data. Every exported example is grounding-checked, full trace
capture (`enable_training_capture()`) is opt-in and off by default, and the
faithful `runs=` path reads only the `RunResult`s you pass it. Apply your own
retention and access controls before sending it to a provider's fine-tuning API;
the job itself runs through the provider's own authenticated transport, so the
egress DLP scan and audit apply.

## Supply-chain integrity

Releases are built in CI and published with a **CycloneDX SBOM** and **SLSA
build-provenance attestations** (`actions/attest-build-provenance`), so you can
verify a downloaded wheel/sdist was built from this repository by the release
workflow (`.github/workflows/release.yml`). Verify with:

```bash
gh attestation verify <artifact> --repo Ohswedd/vincio
```

The dependency SBOM is complemented by an **AI-BOM** (`vincio governance aibom`)
recording the base model, embedding/rerank models, and fine-tune/prompt versions
with SHA-256 hashes, for blast-radius assessment when a model or dataset is found
compromised.

You can also confirm the integrity of a persisted audit log offline:

```bash
vincio audit verify .vincio/audit/audit.jsonl
```

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report privately via GitHub's
[private vulnerability reporting](https://github.com/Ohswedd/vincio/security/advisories/new).

Please include a description of the issue, steps to reproduce, and the affected version. We aim to
acknowledge reports within a few days and will keep you updated on remediation progress. Thank you
for helping keep Vincio and its users safe.
