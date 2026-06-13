# Security Policy

## Supported versions

Vincio follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from
1.0. Security fixes land on the latest 1.x line.

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✅        |
| 0.9.x   | ⚠️ critical fixes only |
| < 0.9   | ❌ (upgrade to 1.0.x) |

## Threat model

Vincio is a library you run on your own infrastructure. What it defends against
(prompt injection, secret/PII leakage, cross-tenant access, audit tampering,
runaway tool execution) and what it explicitly does not (kernel-level sandbox
escape, a compromised host/provider) is documented in the
[threat model](docs/security/threat-model.md). The tool sandbox is OS-process
isolation with `setrlimit` CPU/memory/fd limits — for adversarial code, run
tools in a container/VM.

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
