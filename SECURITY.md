# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Report privately via GitHub's
[private vulnerability reporting](https://github.com/Ohswedd/vincio/security/advisories/new).

Include a description, steps to reproduce, and the affected version. We aim to acknowledge reports
within a few days and will keep you updated on remediation. Thank you for helping keep Vincio and its
users safe.

## Supported versions

Vincio follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). Security fixes land on
the latest minor of the current major; older majors are not maintained.

| Version | Supported |
| ------- | --------- |
| 7.x     | ✅ |
| < 7.0   | ❌ (upgrade to the latest 7.x release) |

## The security model, in one idea

Vincio is a **library you run on your own infrastructure**. Security, permissions, and validation are
enforced **deterministically in code — never gated on model output** — and every capability runs *in
your process*, not as a hosted control plane that widens your exposure surface. A model that is
jailbroken, prompt-injected, or simply wrong cannot turn that into a privilege escalation, because the
decision was never the model's to make.

The [threat model](docs/security/threat-model.md) is the authoritative reference — assets, trust
boundaries, and the STRIDE control behind each defense. The summary below pairs each defense with *how*
it is achieved so you can audit it in the source.

## The defenses, and how each is achieved

**Untrusted input & prompt injection.** All retrieved, ingested, MCP-served, and A2A-delegated content
is treated as untrusted: it is injection-scanned after a normalization + recursive-decode pre-pass
(so a base64- or unicode-obfuscated payload is caught), wrapped so it cannot instruct the model, and
never granted instruction authority. *Only* system and developer content may instruct the model —
enforced by the prompt compiler, not by asking the model to behave.

**Containment that holds when detection misses.** Detection is a filter, not a guarantee; Vincio adds a
*provable* one. The control plane is separated from the data plane: provenance becomes a typed trust
label that propagates through every tainted derivation; a `DualPlaneExecutor`'s privileged planner never
sees untrusted bytes; and every side-effecting tool call is gated on an unforgeable **capability token**
minted from the user's own request. An untrusted-tainted argument therefore *cannot* reach a write or
external tool without a capability or an explicit human approval — and `verify_containment(run)` proves
the invariant held across a completed run.

**Secret & PII leakage, and egress control.** Deterministic PII and secret detection/redaction
(multilingual) run in-process across context, tool I/O, and logs, with an **always-on egress DLP scan**
on outbound traffic — so a secret that entered as evidence cannot silently leave in a tool call.

**Multi-tenant isolation & access control.** RBAC scopes and ABAC rules gate every access; tenant
isolation is **fail-closed** in the engine — a missing or mismatched tenant denies, it doesn't
default-allow.

**Tool & code execution.** Tools are permissioned, may require approval, and run in a resource-limited
subprocess sandbox (wall-clock and output limits, plus — on POSIX — CPU, memory, and file-descriptor
caps). MCP tools and Agent Skill scripts run through the *same* permissioned, sandboxed, audited runtime;
there is no unsandboxed side door.

**Audit integrity.** Every run, retrieval, tool call, memory write, and access decision is recorded in
an append-only, SHA-256 **hash-chained, signed** audit log with Merkle checkpoints. `vincio audit
verify <path>` re-validates a persisted log **offline** and pinpoints the first broken line — tamper
evidence you can check without trusting the system that produced it.

**Governance proofs.** A governance-invariant verifier checks containment, residency, the budget cap,
and the erasure-proof binding across their whole bounded state space *ahead of* a run, returning a
minimal counterexample on a violation — a model check, not a spot check.

## Operator hardening checklist

Vincio ships the mechanisms; you deploy them. For a production deployment:

- [ ] **Run adversarial tools in a real boundary.** The built-in sandbox is OS-process isolation, which
  reduces blast radius but does *not* contain a kernel exploit. For attacker-controlled code, run tools
  in a container or microVM with seccomp and network-egress controls.
- [ ] **Vet plugins like dependencies.** The entry-point plugin system imports and runs code from any
  installed distribution advertising a `vincio.<kind>` entry point. Listing plugins is side-effect-free;
  only *loading* imports them, and a plugin that fails to import is isolated and reported.
- [ ] **Set residency, budgets, and consent explicitly** (`set_residency`, `set_cost_budget`,
  `ConsentLedger`) — the governance verifier can only prove invariants you've declared.
- [ ] **Govern exported training data.** The distillation flywheel writes a fine-tuning JSONL you own;
  full trace capture is opt-in and off by default. Apply your own retention/access controls before
  sending it to a provider.
- [ ] **Persist and periodically verify the audit log** (`vincio audit verify`), and keep the signing
  key out of the application's own trust domain.
- [ ] **Keep secrets in the environment or a manager**, never in `vincio.yaml`; the config layering
  reads `VINCIO_*` env vars last so they win.

## Supply-chain integrity

Releases are built in CI and published with a **CycloneDX SBOM** and **SLSA build-provenance
attestations** (`actions/attest-build-provenance`), so you can verify a downloaded wheel or sdist was
built from this repository by the release workflow (`.github/workflows/release.yml`):

```bash
gh attestation verify <artifact> --repo Ohswedd/vincio
```

The dependency SBOM is complemented by an **AI-BOM** (`vincio governance aibom`) recording the base
model, embedding and rerank models, and fine-tune and prompt versions with SHA-256 hashes — for
blast-radius assessment when a model or dataset is found compromised. And you can confirm the integrity
of any persisted audit log offline:

```bash
vincio audit verify .vincio/audit/audit.jsonl
```
