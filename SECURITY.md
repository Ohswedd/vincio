# Security Policy

## Supported versions

Vincio follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from
1.0. Security fixes land on the latest 1.x line.

| Version | Supported |
| ------- | --------- |
| 1.6.x   | ✅        |
| 1.5.x   | ✅        |
| 1.4.x   | ✅        |
| 1.3.x   | ✅        |
| 1.2.x   | ✅        |
| 1.1.x   | ✅        |
| 1.0.x   | ✅        |
| < 1.0   | ❌ (upgrade to 1.6.x) |

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

The 1.6 enterprise-governance features are evidence *generated from data Vincio
already holds* — they add no external services. Model/system cards, the
compliance-framework coverage matrix (OWASP LLM 2025 / OWASP Agentic / NIST AI
RMF / MITRE ATLAS), and the AI-BOM are views over the live config, the audit
chain, eval reports, and the price table. Data-residency routing refuses egress
to disallowed provider regions as a blocking `PolicyViolation` on the audit
path; right-to-erasure-by-source (`app.erase_source`) purges a source from every
index, memory, and cache and is logged on the hash-chained audit chain.
Non-English PII locale packs and RAG-poisoning detection extend the existing
detectors. Residency resolves the provider region from a region-pinned endpoint
(`provider.base_urls`; AWS/GCP/Vertex/sovereign URLs) with jurisdiction-aware
matching, then refuses to *send* a request to a disallowed region — Vincio cannot
guarantee where a global provider runs it, so the strongest posture is a
region-pinned endpoint plus this client-side egress refusal. Synthetic-content
marking emits a C2PA-style manifest bound by SHA-256; it can be cryptographically
**signed** (built-in symmetric `HmacSigner`, or your own asymmetric
`ContentSigner`) and verified with `verify_manifest` (Vincio assumes no signing
authority — the key and any PKI are yours).

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
