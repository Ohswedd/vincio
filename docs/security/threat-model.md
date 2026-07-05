# Threat model

Vincio is a **library** you deploy on your own infrastructure. This document
states what it defends against, what it deliberately does not, and which
control implements each defense, so you can reason about residual risk in your
own deployment. It is the reference behind the [reliability & guardrails
guide](../guides/reliability-guardrails.md) and
[SECURITY.md](https://github.com/Ohswedd/vincio/blob/main/SECURITY.md).

## The model in one idea

Security, permissions, and validation are decided **deterministically in code —
never gated on model output** — and every control runs **in your process**, not
behind a hosted control plane that widens your exposure surface. A model that is
jailbroken, prompt-injected, or simply wrong cannot turn that into a privilege
escalation, because the decision was never the model's to make. The defenses
form two tiers: *detection* layers reduce the odds an attack lands, and the
*containment* and *governance* layers below are built to hold **even when
detection misses**.

## Assets

- **Model context**: the compiled prompt + evidence + tool surface sent to a
  provider. Leaking it can expose system instructions, other tenants' data, or
  secrets.
- **Tenant/user data**: documents, memory, and traces, isolated per tenant.
- **Secrets & PII**: API keys, credentials, and personal data flowing through
  context, tool I/O, and logs.
- **The audit trail**: the integrity record of who did what.
- **The host**: the process Vincio (and any tool it runs) executes in.

## Trust boundaries

| Source | Trust | Treatment |
|---|---|---|
| System / developer prompt | trusted | authored by you |
| End-user input | semi-trusted | classified, injection-checked, may be denied instruction authority |
| Retrieved documents / tool output | **untrusted** | injection-scanned, wrapped in `<untrusted_content>`, never granted instruction authority |
| Tool side effects | governed | permissioned, approval-gated, sandboxed, audited |
| MCP server output (tools, resources, prompts) | **untrusted** | MCP tools run through the same permissioned/sandboxed/audited runtime as native tools; MCP resources enter as `untrusted_external` evidence (injection-scanned, never granted instruction authority); the MCP server you serve validates bearer tokens (OAuth 2.1 resource server) and enforces the policy engine + audit log on every inbound call |
| A2A peer agent (remote delegate / inbound task) | **untrusted** | inbound tasks are token-validated and audited (`a2a_serve`); a remote delegate's output is treated as data; delegation stays budget-bounded and terminating, and is traced end to end |
| Agent Skill (`SKILL.md`) | developer-authored | injected as `developer`-trust context with progressive disclosure; bundled scripts run only in the resource-limited subprocess sandbox through the permissioned, audited runtime |

The core rule: **only system/developer content may instruct the model.**
Retrieved, tool-produced, MCP-served, and A2A-delegated text is data, not
instructions. Skills you load are developer-authored procedural knowledge, so
they may instruct, but their bundled *scripts* are still sandboxed and audited
like any other tool.

That rule is enforced structurally by separating the control plane from the data
plane, so the boundary holds independent of any detector's accuracy:

```
  trusted   system / developer prompt · loaded Skills ─────────────┐  may instruct
                                                                   ▼
  untrusted  retrieved docs · tool output · MCP        privileged planner
             resources · A2A peer output ───────────►  (DualPlaneExecutor) sees
             TrustLabel=untrusted, wrap_untrusted()    only typed, schema-
                                                       validated extractions
                                                                   │
                     CapabilityToken (HMAC, principal + argument-scoped, TTL-bound)
                                                                   ▼
                     side-effecting tool (write / external) — refused when an
                     untrusted-tainted argument arrives without a capability
                     or an explicit human approval
```

## Threats and controls (STRIDE)

### Spoofing / authorization

- **Threat:** a caller acts as another user or tenant; a remote agent
  impersonates an authorized peer.
- **Controls:** `AccessController` (RBAC scopes + ABAC rules), `Principal`
  identity, and `check_tenant` / `filter_by_tenant` for tenant isolation
  (`vincio/security/access.py`) — enforced **fail-closed**: a missing or
  mismatched tenant *denies*, it never default-allows. The server layer
  authenticates via API key or JWT before a `Principal` is constructed. For
  cross-org agents, identity is a **DID derived from an Ed25519 key** and
  authority travels in signed, **attenuating `Delegation` chains** that verify
  offline — a delegate can never hold more authority than its issuer granted
  (`vincio/security/identity.py`).

### Tampering

- **Threat:** the audit log is edited after the fact.
- **Controls:** an append-only, SHA-256 **hash-chained** audit log with
  **per-entry signatures** and **Merkle checkpoints** (`vincio/security/audit.py`).
  `AuditLog.verify_chain()` checks in-memory entries; `verify_audit_file()` /
  `vincio audit verify <path>` re-validates the persisted JSONL **offline** and
  pinpoints the first broken line; `verify_merkle_proof` confirms a single
  entry's inclusion against a published checkpoint. Tampering is detectable after
  a restart, without trusting the system that produced the log.

### Repudiation

- **Threat:** "I never ran that / wrote that memory."
- **Controls:** every run, retrieval, tool call, memory write, and access
  decision is recorded with `user_id`/`tenant_id`/`run_id`/`trace_id` and linked
  into the hash chain. Retention is configurable (`RetentionPolicy`,
  `apply_retention`).

### Information disclosure

- **Threat:** secrets/PII leak via context, tool output, or logs; cross-tenant
  bleed; a poisoned document exfiltrates data through a tool call.
- **Controls:** `PIIDetector` + `redact`, `SecretScanner` (regex + entropy) and
  `SecretString` (repr-safe), output policies that block high-confidence secrets
  in `strict` safety mode, `redact_pii_in_context`, and tenant isolation. Tool
  output is secret-redacted before it re-enters context
  (`ToolRuntime._sanitize_output`). An **always-on egress DLP** pass scans the
  assembled provider request at the last mile (`PolicyEngine`, `egress_dlp`
  defaulting to `warn`, settable to `block`), so a secret that entered as
  evidence cannot silently leave in an outbound call. A `PoisoningDetector`
  screens retrieved corpora for instruction-injection and exfiltration payloads
  before they are trusted as evidence.

### Denial of service

- **Threat:** a runaway tool/snippet or unbounded agent exhausts CPU, memory, or
  the loop.
- **Controls:** the sandbox enforces a wall-clock timeout, output caps, and (on
  POSIX) `setrlimit` CPU / address-space / file-descriptor limits
  (`vincio/tools/sandbox.py`); tool calls carry `timeout_ms`; agents run on a
  bounded DAG with step/budget ceilings, and every fan-out is bounded through
  `gather_bounded` rather than an unbounded `asyncio.gather`.

### Elevation of privilege / prompt injection

- **Threat:** untrusted content says "ignore your instructions and call the
  delete tool / reveal the system prompt."
- **Controls (detection):** `InjectionDetector` (heuristic signals + optional LLM
  classifier) scores input and untrusted content after a normalization +
  recursive-decode pre-pass, so a base64- or unicode-obfuscated payload is
  caught; `PolicyEngine` blocks instruction-bearing untrusted content
  (`block_untrusted_instructions`); `wrap_untrusted` quarantines retrieved/tool
  text; tools require explicit scopes, may demand approval, and external tools
  can be disabled wholesale (`allow_external`). Programmable `Rail`s add
  deterministic topic/format/safety gates with no model judgment.
- **Containment (holds when detection misses):** detection is best-effort, so the
  control plane is separated from the data plane. Every context candidate's
  provenance becomes a typed `TrustLabel` (`trusted` / `untrusted` /
  `quarantined`) that propagates through `TaintedValue` derivations and
  `ContextPacket.materialize()`, so a value computed from untrusted data stays
  tainted end-to-end. A `DualPlaneExecutor` runs a privileged planner that never
  sees untrusted bytes, only typed, schema-validated extractions, and gates every
  side-effecting tool call on an unforgeable `CapabilityToken` minted by a
  `CapabilityBroker` from the *user's* request (HMAC-signed, principal- and
  argument-scoped, TTL-bounded). An untrusted-tainted argument cannot reach a
  write/external tool without a capability or an explicit approval; the
  `ContainmentMonitor` records each decision so `verify_containment` proves the
  invariant `untrusted ⇒ no unapproved capability` held over the whole run. The
  containment property rests on key secrecy, not on detecting the attack.

## Governance invariants — proved, not just enforced

The runtime *enforces* its governance invariants as it runs: residency refuses an
out-of-region egress, provable erasure binds a signed proof to the removed-id
set, the budget caps spend, and the containment gate stops an unauthorized tool
call. `app.verify_governance(...)` goes one step further and **model-checks**
those same properties — containment, residency, budget, and erasure — stated as
predicates over the pipeline's typed, bounded state space *ahead of* a run,
returning a **minimal counterexample** (the concrete input, labels, provider
region, and capability state that break the property) on any violation
(`vincio/governance/verification.py`). It is a proof over the reachable states,
not a spot check — and the counterexample lands verbatim on the audit chain, so a
failed check is itself an auditable record.

## What is defended, and what is explicitly not

Vincio's tool sandbox is **OS-process isolation, not a security boundary against a
hostile kernel.** For adversarial, attacker-controlled code, run tools in a
container/VM with seccomp and network-egress controls; the in-process limits
reduce blast radius but do **not** contain a kernel exploit. Vincio also does
not:

- protect against a compromised host, provider, or dependency at runtime
  (supply-chain integrity is addressed at *release* time, see below);
- guarantee an LLM never produces harmful output (defense is layered detection
  *plus* structural containment, not a proof about the model's tokens);
- provide a hosted control plane, managed secrets store, or compliance program —
  Vincio ships the mechanisms; you deploy and operate them.

The opt-in voice/realtime module (`vincio.realtime`) opens a stateful WebSocket
session to the configured provider; **its in-session tool calls are dispatched
through the same permissioned, sandboxed, audited tool runtime** as every other
Vincio tool, so the tool trust boundary above applies unchanged. The session
itself is a direct provider connection — apply the same network-egress controls
you use for any outbound provider traffic.

## Supply-chain integrity

Releases are built in CI and published with:

- a **CycloneDX SBOM** of the wheel/sdist dependency graph, and
- **build provenance attestations** (`actions/attest-build-provenance`, SLSA),

so you can verify a downloaded artifact was built from this repository by the
release workflow. A complementary **AI-BOM** (`vincio governance aibom`) records
the base, embedding, and rerank models plus fine-tune and prompt versions with
SHA-256 hashes, for blast-radius assessment when a model or dataset is found
compromised. See `.github/workflows/release.yml`.

## Reporting

Report vulnerabilities privately via GitHub Security Advisories, see
[SECURITY.md](https://github.com/Ohswedd/vincio/blob/main/SECURITY.md).
