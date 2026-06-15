# Security Policy

## Supported versions

Vincio follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from
1.0. Security fixes land on the latest 1.x line.

| Version | Supported |
| ------- | --------- |
| 1.4.x   | ✅        |
| 1.3.x   | ✅        |
| 1.2.x   | ✅        |
| 1.1.x   | ✅        |
| 1.0.x   | ✅        |
| < 1.0   | ❌ (upgrade to 1.4.x) |

## Threat model

Vincio is a library you run on your own infrastructure. What it defends against
(prompt injection, secret/PII leakage, cross-tenant access, audit tampering,
runaway tool execution) and what it explicitly does not (kernel-level sandbox
escape, a compromised host/provider) is documented in the
[threat model](docs/security/threat-model.md). The tool sandbox is OS-process
isolation with `setrlimit` CPU/memory/fd limits — for adversarial code, run
tools in a container/VM.

The 1.1 interoperability protocols keep these guarantees at the boundary: MCP
tools (consumed from a server) run through the same permissioned, sandboxed,
audited runtime, and MCP resources enter as untrusted, injection-scanned
evidence; an MCP/A2A server you expose validates bearer tokens (OAuth 2.1
resource server) and enforces the policy engine + audit log on every inbound
call; Agent-Skill bundled scripts run only in the subprocess sandbox.

The 1.2 agentic-evaluation features stay in-process: online-eval scores, drift
baselines, and human annotations are written to your own metadata store (no
traffic mirrored to any external service), and the user simulator runs against
your app, not a third party.

The 1.3 cost/reliability features (batch, circuit breakers, key pools,
cascades, cost attribution, budgets, prompt caching, sharded indexing) are all
in-process and additive — they introduce no new external services. Budget
breaches surface as `PolicyViolation`s on the existing hash-chained audit path.

The 1.4 reflective-optimization and flywheel features run in-process and add no
external services: the reflective optimizer reuses the existing eval/registry/
audit path, and judge calibration uses your own labelled data. The distillation
flywheel writes a fine-tuning **JSONL file you own** from the runs/traces you
choose; treat it as you would any export of model inputs/outputs — it may
contain business data. Every exported example is grounding-checked, and the
grounding/feedback filters bound what is written. The trace-based path's full
capture (`enable_training_capture()`) is opt-in and off by default; the faithful
`runs=` path reads only the `RunResult`s you pass it. Apply your own retention
and access controls to the exported file before sending it to a provider's
fine-tuning API.

## Supply-chain integrity

Releases are built in CI and published with a **CycloneDX SBOM** and **SLSA
build-provenance attestations** (`actions/attest-build-provenance`), so you can
verify a downloaded wheel/sdist was built from this repository by the release
workflow (`.github/workflows/release.yml`). Verify with:

```bash
gh attestation verify <artifact> --repo Ohswedd/vincio
```

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
