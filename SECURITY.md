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

### Secret & PII leakage, and egress control

Deterministic PII / secret detection and redaction runs in-process, with
non-English locale packs (France/Germany/Spain/India/Singapore/Brazil/UK)
extending the built-in patterns.

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
path.

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

Every generated image and audio asset auto-attaches a media-aware C2PA manifest
bound to the asset's bytes by SHA-256 (a tampered asset fails `verify_manifest`),
embedded in file metadata where the container supports it or as a `*.c2pa.json`
sidecar otherwise, with edits marked, and each generation is metered against the
run `Budget` and recorded as an `image_generate` / `speech_synthesize` audit
event. Manifests can be cryptographically **signed**
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

### Governed discovery & interoperability

Agent and tool discovery is governed by construction: an `AgentDirectory` resolves
an agent or MCP server only through an `AllowListGate` — a fail-closed allow-list
(deny patterns first, then allow; default deny) over the same `AccessController`
the data plane uses — and **every resolution is recorded as an `agent_resolve`
access decision** on the audit chain, so an unlisted agent is unreachable and each
reachable one is accountable. AGNTCY/ACP and MCP-registry discovery normalize
remote records into the same governed directory; discovery never auto-trusts a
result. Generative-UI (AG-UI) streaming opens no new data channel — it is a
translation of the run's existing stream, inheriting its provenance, budget
metering, and audit.

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
