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

An MCP server's mid-call **elicitation** request (a server asking the user for a
structured value) is treated as an untrusted-input boundary, not a trusted one. The
`ElicitationGate` gates it with the same approval and rail machinery a write tool
passes: an approver may deny the request before any value is collected, the
collected value is screened through the input rail engine (a secret, PII, or
injection value is declined — an injection-flagged value is quarantined), and an
accepted value is wrapped as an `untrusted` `TaintedValue`, so it propagates taint
and can never silently authorize a side effect. Server-rendered UI (**MCP Apps**)
surfaced through the AG-UI channel is likewise `untrusted_external` content,
token-metered against the run budget (an oversized render is refused), and audited.

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

### Agent negotiation & contracting

A negotiation between agents (`app.negotiate` / `vincio.negotiation`) inherits the
fabric's discipline rather than opening a new trust surface. It **terminates by
construction**: a `NegotiationBudget` bounds the offer exchanges and an optional
wall-clock deadline returns a partial result, so an adversarial counterparty
cannot stall a bargain into an unbounded loop — the worst case is a clean no-deal.
The negotiated **`Contract`** is **content-bound** (a stable hash over buyer,
seller, terms, rounds, and timestamp) and **signed by both parties** with the same
`ChainSigner` the audit chain uses (HMAC, or Ed25519 so a counterparty verifies
with only the public key); `contract.verify(signer)` recomputes the hash and checks
every signature **offline from the bytes alone**, so a tampered term or a forged
signature is caught without the live parties, and a negotiation outcome and its
signed contract land on the hash-chained audit log (`negotiation` /
`contract_signed`). The contract is **enforced like a budget** — `to_budget()`
lowers the agreed price/SLA into the same hard-cap machinery the runtime already
enforces, and `app.enforce_contract` records a fulfilment or breach — so an agreed
term is a mechanical cap, not a self-asserted promise. Reputation weighting is
**bounded and reversible** (a counterparty's weight never leaves `[floor, 1]`, only
lowering its pull), so a regressing agent is discounted without being singled out
and a high reputation can never bypass the contract's terms. Over the A2A fabric, a
remote party's identity is **pinned to the directory-resolved member id**, never the
self-asserted one on the wire, so a counterparty cannot spoof another member's
reputation; every offer exchange is a bounded, audited A2A task.

### Cross-org workflow choreography

A cross-org saga (`app.choreograph` / `vincio.choreography`) coordinates durable
work across organizations **without opening a shared control plane**: the only
things that cross a trust boundary are the typed `StepRequest` handoff and the
audited result, governed by the same A2A and contract discipline. **Governance is
per-org**: the coordinator records each dispatched handoff on *its* hash-chained
audit log (the `choreography_step` action) while each participant records its own
execution on *its* chain, so no single party holds an authoritative log of another's
steps and each side's record is independently verifiable. The **`SagaJournal`** is
**content-bound** — every step record links to the previous by a stable hash (and,
with a signer, carries a signature over it), so `journal.verify(verifier)` recomputes
the chain **offline from the bytes alone** and pinpoints any record edited, inserted,
or dropped; the journal is checkpointed to the metadata store after every step, so a
restart resumes from durable state rather than re-running or losing committed work.
**Failure is contained, not dangling**: a forward step that fails, raises, or
**breaches its step `Contract`** (the delivered cost/latency/quality checked against
the agreed terms) triggers deterministic compensation of the completed steps in
reverse order — a half-completed cross-org transaction unwinds cleanly rather than
stranding partial state across boundaries, and a compensation that itself fails ends
the saga in an explicit `failed` state with the residue pinpointed, never silently
swallowed. Like negotiation, it **terminates by construction**: a failure compensates
and stops (it never loops), and a remote participant is driven over the same bounded,
audited A2A task surface — so an adversarial or unreachable counterparty yields a
recorded failure-and-unwind, not an unbounded hang.

### Cross-org workflow discovery & dynamic choreography

Resolving a saga step's counterparty at run time (`app.choreograph(..., directory=)`
/ `vincio.choreography.discovery`) **changes who runs a step, never how it is
governed**. A discovered step's candidate set is the agents that advertise the
needed capability in the **same governed `AgentDirectory`** the rest of the fabric
uses, so discovery inherits its discipline by construction: every candidate is
resolved through the directory's fail-closed `AllowListGate` and **each resolution
is recorded as an `agent_resolve` decision** before it can be bound — an unlisted
candidate scores zero and is **never bound**, exactly as it is unreachable to a
direct tool call. The binder considers only candidates that are both allow-listed
**and** reachable (present in the coordinator's participant set), so an advertised
but unbindable org is rejected rather than dispatched to a dead end, and a
capability **no** allowed, reachable candidate advertises is **refused** with a
`ChoreographyError` rather than silently skipped. The binding decision itself —
the chosen org and the full ranked candidate field — lands on the coordinator's
hash-chained chain as a `choreography_bind` entry and is carried on the saga
journal (`result.bindings`), so *who* was chosen and *why* is as auditable as the
handoff that follows. The ranking signals are **bounded and earned, never
self-asserted**: a candidate's pull comes from its `ReputationLedger` standing
(its no-regression / contract-fulfilment track record, bounded to `[floor, 1]`,
discounting a regressor without singling it out) and its prior `SettlementBook`
record (the share of settlements it honoured and how its delivered cost fit the
agreed price) — both accrued from audited outcomes, not from anything a candidate
claims on the wire. Once bound, the resolved org runs under the **same** contract
enforcement, per-org audit, deterministic compensation (unwound at the org it was
actually bound to, recorded on the journal — never a freshly re-resolved one), and
durable-resume guarantees a statically-wired participant does; a restart re-binds
only the steps that had not yet run. There is no hosted matching service or shared
registry of intent — discovery is a library-side resolution over a directory you
operate, governed and audited in your process.

### Agent-to-agent settlement & metering

Closing the books on contracted cross-org work (`app.settle` / `app.settle_saga` /
`vincio.settlement`) produces a **verifiable ledger of what was owed and delivered,
never a payment rail**: no money moves, there is no escrow or clearing service, and a
settlement is a typed record each org holds and verifies itself. A `SettlementRecord`
is **content-bound**: both parties sign one *reconciliation hash* over the economic
facts alone — contract, parties, agreed terms, delivered metrics, balance — which is
deliberately independent of run id, timestamp, and book position, so two
independently-produced records co-sign the **same** hash when the books agree and a
tampered figure changes it. `record.verify(verifier)` recomputes the hash and checks
every signature **offline from the bytes alone**, so a forged signature or an edited
balance is caught without the live parties, and `reconcile(ours, theirs)` ties two
orgs' records out and **pinpoints a disagreement as a dispute** rather than letting one
side's number stand unchallenged. The **`SettlementBook`** is a **hash-chained** ledger
— each record links to the previous by an entry hash — so `book.verify()` recomputes
the whole ledger and catches any record edited, inserted, or dropped (`broken_at`),
and it is checkpointed to the metadata store so a restart resumes from durable state.
**Metering is total-preserving**: a reading's cost/latency/usage totals are exactly the
sum of the accrued events (quality is the weakest link held against a floor), so a
settlement built from it cannot silently under- or over-count delivery. A delivered
**breach is contained, not hidden**: an overrun or shortfall reconciles to an explicit
`status="breached"` record with the breaching dimensions named, and — closing the
reputation loop — debits the seller's standing so an unreliable counterparty is
discounted in the next negotiation, bounded and reversible, never singled out. Like
negotiation and choreography, settlement is **deterministic and offline by
construction**: it asserts nothing it cannot verify from the bytes, and crosses a trust
boundary only as a signed record reconciled on each side's own chain.

### Cross-org settlement netting & multilateral clearing

Netting a fleet's bilateral books (`app.clear_settlements` / `net_settlements` /
`net_books` / `vincio.settlement.netting`) is a **clearing calculation, never a
clearing house**: no money moves, there is no central ledger, and a `NettingSet` is a
content-bound artifact each party can recompute. Netting **reads only the existing
signed, hash-chained records and asserts nothing it cannot recompute**. A source
record whose reconciliation hash no longer recomputes — a tampered economic figure —
is **refused outright** rather than netted, and with a verifier a forged signature is
too, so a clearing is never built on a falsified book. The same bilateral settlement
seen from both parties' books is **deduplicated by its reconciliation hash, not
double-counted**, and when two books carry *different* facts for the same contract the
contract is **pinpointed as a `NettingDispute` and excluded** from the clearing — a
disagreement is named, never silently absorbed (`require_clean()` raises on one). The
`NettingSet` is **content-bound** the way a record is: a netting hash binds the fleet,
the exact source records read (by their hashes), the net positions, and the cleared
obligations, so `netting.verify(verifier)` recomputes it **offline from the bytes
alone** — the hash matches, the net positions balance to zero (every payable is a
receivable), and the cleared transfers reproduce every org's position, so a value
created or lost in clearing is caught. The obligation a settlement contributes is the
agreed price the buyer owes the seller for the scope; a breach is surfaced by the
settlement's own status and the reputation loop, it never silently alters what is
cleared. Because the netting hash excludes local metadata, two clearers reading the
same records co-sign the **same** hash — a cleared balance is a mechanical, verifiable
artifact, never a self-asserted claim.

### Cross-org dispute resolution & arbitration

Arbitration (`app.arbitrate` / `arbitrate` / `book.arbitrate` /
`vincio.settlement.arbitration`) resolves a pinpointed disagreement and is a
**deterministic adjudication over the parties' own signed records, never a hosted
arbitration service or a court of record**: no third party rules by fiat, and a
`Resolution` is a content-bound artifact each party can recompute. Arbitration **reads
only the existing signed records and asserts nothing it cannot recompute**. The
decision rests on verifiable evidence: a reconciliation hash that **both** the buyer
and the seller signed — each on their own record, the two co-signing one figure — is
mutually corroborated and **upheld**; a unilateral claim contradicting the corroborated
figure is **rejected and pinpointed**, not silently overruled; and when neither side's
figure is corroborated the dispute is honestly left **unresolved** rather than decided
without evidence — a security-conscious refusal to fabricate a winner. Unlike netting,
which *refuses* to clear over a tampered book, arbitration is the venue where a bad
claim is adjudicated: a claim whose reconciliation hash no longer recomputes (a tampered
figure), one carrying no signature, or — with a verifier — one with a forged signature
is marked **inadmissible** and pinpointed (`ClaimVerdict.reason`), never silently
dropped and never crashing the resolution. The `Resolution` is **content-bound** the
way a record is: a resolution hash binds the contract, the parties, the outcome, and
every adjudicated claim (by reconciliation hash, corroborating signers, admissibility,
and whether it stands), so `resolution.verify(verifier)` recomputes it **offline from
the bytes alone** and **re-derives the whole decision from the recorded claims** — a
flipped verdict, a swapped winner, or a smuggled-in standing claim is caught even when
the hash was recomputed to match. Because the resolution hash excludes the arbiter and
the local metadata, two arbiters reading the same records co-sign the **same** hash. A
settled dispute also closes the reputation loop on the party whose claim did not stand —
and only in the upheld case, where corroboration *proves* the rejected claim wrong, so
an honest party in an unresolved standoff is never debited.

### Cross-org reputation attestation & portability

Reputation attestation (`app.attest_reputation` / `attest_reputation` / `book.attest` and
`app.import_reputation` / `combine_attestations` / `vincio.settlement.attestation`) makes a
counterparty's earned standing **portable**, and is a **signed, offline-verifiable claim
that combines into an evidence-weighted prior, never a hosted reputation bureau or a
central score**. An attestation **reads only the issuer's own existing signed records and
asserts nothing it cannot recompute**: it counts settlements where the subject was the
seller (a fulfilled delivery a success, a breach a failure) and arbitration dissents,
**skipping a record whose reconciliation hash no longer recomputes** so a tampered own
record cannot inflate a standing, and binds the exact source hashes the evidence came
from. The `ReputationAttestation` is **content-bound** the way a record is: an attestation
hash binds the issuer, the subject, the evidence counts, the prior, and the source hashes,
so `attestation.verify(verifier)` recomputes it **offline from the bytes alone** and
**re-derives the attested reputation from the evidence counts** — a tampered score is caught
even after re-sealing (the score no longer re-derives), and a forged issuer signature is
caught. The issuer is bound into the hash, so an attestation is one issuer's signed claim,
not an issuer-independent recomputation. Combining several issuers' attestations is an
**evidence-weighted pool, never a single self-asserted number**: an issuer that vouches for
itself (`issuer == subject`) is **refused**, an issuer cannot stack its own pull (only its
largest attestation for a subject is counted, the rest pinpointed as superseded), a tampered
or forged attestation is **refused and pinpointed** (`PortableReputation.refused`) rather
than silently dropped, an optional `per_issuer_cap` bounds any one issuer's mass, and the
importer's own prior anchors the pooled posterior so a thin attestation barely moves it. The
imported prior weights a negotiation under the **same bounded `[floor, 1]` rule a local
reputation does** — a regressor is discounted, never zeroed and never singled out, and the
discount is reversible — and a counterparty the importer already knows keeps its own earned
local standing (the `base` ledger), so a portable attestation only ever fills the gap where
there is no local history; it can never *raise* a counterparty's pull past parity or bypass
the quality bar.

Because standing changes, the portable prior is **time-aware and revocable** — it reflects
*current* standing, never a frozen snapshot, and still **reads only the existing signed
artifacts**, asserting nothing it cannot recompute. **Freshness:** an attestation carries an
issuer-declared validity window (`horizon_days`) bound into its signed hash, so against an
as-of clock a stale attestation is **excluded and pinpointed** (`PortableReputation.stale`)
while an older one within its window **decays** out of the pooled prior by an importer
`half_life_days` (its evidence mass halved each half-life, its attested ratio preserved) —
an old attestation eases out rather than anchoring the prior forever. **Revocation:** an
issuer signs a content-bound `AttestationRevocation` (`app.revoke_attestation` /
`book.revoke`) that withdraws or supersedes a prior attestation **by its hash**, and
`revocation.verify(verifier)` recomputes it **offline from the bytes alone** — so the
withdrawn claim is **excluded and pinpointed** (`PortableReputation.revoked`), never silently
honored. A revocation withdraws an attestation only when it both verifies and is issued by
the **same party** whose attestation it names, so a **forged revocation, or one naming
another org's attestation, cannot cancel a claim** — there is no central revocation service
or bulletin board, and freshness and revocation fold into the *same* bounded `[floor, 1]`
weighting a local reputation uses.

### Cross-org reputation gossip & attestation exchange

The attestation exchange (`app.serve_attestations` / `attestation_a2a_server` and
`app.gather_reputation` / `gather_reputation` / `AttestationExchange` in
`vincio.settlement.exchange`) makes portable standing **discoverable** over the A2A fabric,
and is a **bounded pull of signed artifacts from peers you govern, never a hosted reputation
registry or a push-based gossip bus**. It is *pull, never push*: a peer
(`app.serve_attestations`) only ever **answers** a subject query, and only with **its own
signed artifacts** — the current attestation it can issue from its own settlement book plus
the revocations it has signed. An importer **trusts nothing on a peer's word**: every fetched
artifact is **independently verified from the bytes** before it is counted, exactly as a
directly-handed bundle is, so a forged or tampered artifact a peer serves is **refused**, and
a revocation a peer gossips is honored only when it verifies and is issued by the same party
(it still cannot cancel another org's claim). The fan-out is **bounded and governed**: the
exchange visits at most `max_peers` peers, each cleared through the `AgentDirectory`
allow-list (a denied peer **skipped and pinpointed** in `GatheredReputation.visits`, its
resolution audited), deduplicates by content hash, and folds the result into the **same**
`combine_attestations` under the same freshness, revocation, and `[floor, 1]` discipline — so
**gossip changes only where the evidence comes from, never how it is weighed**. Every peer
visited (`reputation_peer`) and every artifact fetched (`reputation_fetch`) lands on the
hash-chained audit log, and the whole exchange runs byte-for-byte the same against
deterministic in-process peers as over the live fabric — there is no central source of truth,
only verifiable artifacts you pull from peers you control.

### Cross-org transitive trust & Sybil-resistant weighting

Issuer-trust weighting (`build_trust_model` / `TrustConfig` / `TrustModel` in
`vincio.settlement.attestation`, opt-in via `combine_attestations(..., trust_config=)` /
`app.import_reputation` / `app.gather_reputation`) is a defense against **Sybil and
volume-stuffing attacks** on a pooled prior, and is a **bounded, transitive weighting
computed in-process from your own ledger, never a central trust authority or a hosted
Sybil-detection service**. Without it, every counted issuer's evidence pools with **equal
pull**, weighted only by *how much* it attests — so a clutch of unknown peers can
out-evidence a few you have lived through, and an adversary can spin up a cluster of
**Sybil** issuers that all vouch the same way to manufacture a standing. The trust kernel
scales each issuer's contributed evidence *mass* by the importer's **own trust in that
issuer** (its successes and failures together, so it changes how much an issuer *pulls*,
never the reputation it attests), under a bounded web-of-trust: an issuer the importer
**knows first-hand** in its `base` `ReputationLedger` is trusted as much as that ledger
weights it (hop 0); trust **composes at most `max_depth` hops** outward — a trusted issuer
that *attests another issuer* (vouches for it as a counterparty) lends it trust derived
from that pooled standing, attenuated by a per-hop `hop_decay` — under a **hard depth
bound**, so a long unverifiable chain cannot manufacture standing and the computation stays
finite, deterministic, and offline. Only **admissible (verified)** attestations vouch, and
an issuer **never bootstraps its own trust** (a self-subject attestation is ignored for
vouching). The kernel is **Sybil-resistant by construction**: trust is lent only *outward
from a trusted root*, so a ring of mutually-vouching unknown issuers is **never reached**
and every member stays at the floor — **pull follows earned trust, not issuer count**, so a
Sybil clutch cannot outvote a few corroborating trusted peers. Every multiplier is bounded
`[trust_floor, 1]` — an unknown issuer is **floored, never zeroed or singled out**, and
recoverable — and is **pinpointed** (`AttestationVerdict.trust`, `SubjectStanding.issuer_trust`,
`IssuerTrust`), never a silent exclusion. The weighting is **strictly opt-in**: with no
`trust` / `trust_config`, the combination pools with equal pull exactly as before, and the
weighted prior still weights a negotiation under the same `[floor, 1]` discipline — it can
never *raise* an issuer's pull past full or bypass the quality bar.

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
