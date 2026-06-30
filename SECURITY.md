# Security Policy

## Supported versions

Vincio follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Security fixes land on the latest minor of the current major. Older majors are not
maintained — upgrade to the latest release.

| Version | Supported |
| ------- | --------- |
| 6.x     | ✅ |
| < 6.0   | ❌ (upgrade to the latest 6.x release) |

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Report privately
via GitHub's [private vulnerability reporting](https://github.com/Ohswedd/vincio/security/advisories/new).

Include a description of the issue, steps to reproduce, and the affected version. We aim
to acknowledge reports within a few days and will keep you updated on remediation
progress. Thank you for helping keep Vincio and its users safe.

## Security model

Vincio is a **library you run on your own infrastructure**. Security, permissions, and
validation are enforced deterministically in code — never gated on model output — and
every capability runs **in your process**, not as a hosted control plane that widens
your exposure surface.

The defenses below are summarized here; the [threat model](docs/security/threat-model.md)
is the authoritative reference, with the assets, trust boundaries, and STRIDE controls
behind each one.

- **Untrusted input and prompt injection.** All retrieved, ingested, MCP-served, and
  A2A-delegated content is treated as untrusted: it is injection-scanned (after a
  normalization and recursive-decode pre-pass), wrapped so it cannot instruct the model,
  and never granted instruction authority. Only system and developer content may instruct
  the model.
- **Containment that holds when detection misses.** The control plane is separated from
  the data plane. Provenance becomes a typed trust label that propagates through tainted
  derivations; a dual-plane executor's privileged planner never sees untrusted bytes, and
  every side-effecting tool call is gated on an unforgeable capability token minted from
  the user's request. An untrusted-tainted argument cannot reach a write or external tool
  without a capability or explicit approval, and `verify_containment` proves the invariant
  held across a run.
- **Secret and PII leakage, and egress control.** Deterministic PII and secret detection
  and redaction run in-process across context, tool I/O, and logs, with an always-on
  egress DLP scan on outbound traffic.
- **Multi-tenant isolation and access control.** RBAC scopes and ABAC rules gate every
  access, with fail-closed tenant isolation enforced in the engine.
- **Tool and code execution.** Tools are permissioned, may require approval, and run in a
  resource-limited subprocess sandbox (wall-clock, output, and — on POSIX — CPU, memory,
  and file-descriptor limits). MCP tools and Agent Skill scripts run through the same
  permissioned, sandboxed, audited runtime.
- **Audit integrity.** Every run, retrieval, tool call, memory write, and access decision
  is recorded in an append-only, SHA-256 hash-chained, signed audit log. `vincio audit
  verify <path>` re-validates a persisted log offline and pinpoints the first broken line.
- **Governance proofs.** A governance-invariant verifier checks containment, residency,
  the budget cap, and the erasure-proof binding across their whole bounded state space
  ahead of any run, returning a minimal counterexample on a violation.

### Operational notes

- **Third-party plugins execute in your process.** The entry-point plugin system imports
  and runs code from any installed distribution advertising a `vincio.<kind>` entry point;
  vet plugins like any dependency. Listing installed plugins is side-effect-free — only
  loading imports them, and a plugin that fails to import is isolated and reported.
- **The sandbox is OS-process isolation, not a boundary against a hostile kernel.** For
  adversarial, attacker-controlled code, run tools in a container or VM with seccomp and
  network-egress controls; the in-process limits reduce blast radius but do not contain a
  kernel exploit.
- **Exported training data is yours to govern.** The distillation flywheel writes a
  fine-tuning JSONL file you own; full trace capture is opt-in and off by default. Apply
  your own retention and access controls before sending it to a provider.

## Supply-chain integrity

Releases are built in CI and published with a **CycloneDX SBOM** and **SLSA
build-provenance attestations** (`actions/attest-build-provenance`), so you can verify a
downloaded wheel or sdist was built from this repository by the release workflow
(`.github/workflows/release.yml`):

```bash
gh attestation verify <artifact> --repo Ohswedd/vincio
```

The dependency SBOM is complemented by an **AI-BOM** (`vincio governance aibom`) recording
the base model, embedding and rerank models, and fine-tune and prompt versions with
SHA-256 hashes, for blast-radius assessment when a model or dataset is found compromised.

You can also confirm the integrity of a persisted audit log offline:

```bash
vincio audit verify .vincio/audit/audit.jsonl
```
