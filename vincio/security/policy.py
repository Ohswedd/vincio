"""Deterministic policy engine (Engineering Principle 10).

Evaluates run-level policies before and after model execution. Policy
enforcement never depends on model judgment: each check is plain code over
the run state, producing explainable :class:`PolicyViolation` records.

Programmable rails plug in here: a :class:`~vincio.security.rails.RailEngine`
attached to the policy engine evaluates topic / format / safety / custom
rails inside the same input and output checks, so rails are enforced
before and after every generation with zero extra wiring.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.types import ModelRequest, PolicySet, TrustLevel
from .injection import InjectionDetector
from .pii import PIIDetector, redact
from .rails import RailEngine
from .secrets import SecretScanner

__all__ = ["PolicyViolation", "PolicyCheckResult", "PolicyEngine"]

# Egress DLP blocks on outbound credentials and sensitive identifiers, not on
# ordinary PII (a user may legitimately send their own email). These are the
# PII types that should never leave the process unredacted at high confidence.
_EGRESS_BLOCK_PII_TYPES = frozenset({"api_key", "secret", "government_id", "credit_card", "iban"})


class PolicyViolation(BaseModel):
    policy: str
    severity: Literal["block", "warn"]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyCheckResult(BaseModel):
    allowed: bool
    violations: list[PolicyViolation] = Field(default_factory=list)
    transformed_text: str | None = None  # set when redaction was applied

    @property
    def blocking(self) -> list[PolicyViolation]:
        return [v for v in self.violations if v.severity == "block"]


class PolicyEngine:
    def __init__(
        self,
        policies: PolicySet | None = None,
        *,
        pii_detector: PIIDetector | None = None,
        injection_detector: InjectionDetector | None = None,
        rails: RailEngine | None = None,
        secret_scanner: SecretScanner | None = None,
        egress_dlp: Literal["off", "warn", "block"] = "warn",
    ) -> None:
        self.policies = policies or PolicySet()
        self.pii = pii_detector or PIIDetector()
        self.injection = injection_detector or InjectionDetector()
        self.rails = rails
        self.secrets = secret_scanner or SecretScanner()
        # always-on egress DLP mode for the assembled provider request.
        self.egress_dlp = egress_dlp

    def _check_rails(
        self, text: str, *, direction: str, violations: list[PolicyViolation]
    ) -> str | None:
        """Evaluate rails for one direction; returns redacted text if any."""
        if self.rails is None or not self.rails.rails:
            return None
        check = self.rails.check(text, direction=direction)
        for violation in check.violations:
            violations.append(
                PolicyViolation(
                    policy=f"rail:{violation.rail}",
                    severity="block" if violation.action == "block" else "warn",
                    message=violation.message,
                    details={"kind": violation.kind, **violation.details},
                )
            )
        return check.transformed_text

    # -- input-side checks ---------------------------------------------------------

    def check_input(self, text: str, *, trust: TrustLevel = TrustLevel.USER) -> PolicyCheckResult:
        violations: list[PolicyViolation] = []
        transformed: str | None = None

        if self.policies.safety != "minimal":
            verdict = self.injection.detect(text)
            if verdict.detected:
                severity: Literal["block", "warn"] = "block" if (
                    self.policies.safety == "strict" or not trust.allowed_to_instruct_model
                ) else "warn"
                violations.append(
                    PolicyViolation(
                        policy="injection_detection",
                        severity=severity,
                        message=f"prompt injection signals detected (risk={verdict.risk})",
                        details={"signals": [s.pattern for s in verdict.signals]},
                    )
                )

        if self.policies.redact_pii_in_context:
            matches = self.pii.detect(text)
            if matches:
                transformed = redact(text, matches)
                violations.append(
                    PolicyViolation(
                        policy="pii_redaction",
                        severity="warn",
                        message=f"redacted {len(matches)} PII span(s) from context",
                        details={"types": sorted({m.type for m in matches}), "count": len(matches)},
                    )
                )

        rail_transformed = self._check_rails(
            transformed if transformed is not None else text,
            direction="input",
            violations=violations,
        )
        if rail_transformed is not None:
            transformed = rail_transformed

        allowed = not any(v.severity == "block" for v in violations)
        return PolicyCheckResult(allowed=allowed, violations=violations, transformed_text=transformed)

    def check_untrusted_content(self, text: str, *, source: str) -> PolicyCheckResult:
        """Check retrieved/tool content. Untrusted instructions are blocked
        when block_untrusted_instructions is on."""
        violations: list[PolicyViolation] = []
        verdict = self.injection.detect(text)
        if verdict.detected:
            severity: Literal["block", "warn"] = (
                "block" if self.policies.block_untrusted_instructions else "warn"
            )
            violations.append(
                PolicyViolation(
                    policy="untrusted_instruction",
                    severity=severity,
                    message=f"instruction-like content in untrusted source {source!r} (risk={verdict.risk})",
                    details={"signals": [s.pattern for s in verdict.signals], "source": source},
                )
            )
        allowed = not any(v.severity == "block" for v in violations)
        return PolicyCheckResult(allowed=allowed, violations=violations)

    # -- output-side checks -----------------------------------------------------------

    def check_output(self, text: str, *, citations_present: bool | None = None) -> PolicyCheckResult:
        violations: list[PolicyViolation] = []
        if self.policies.require_citations and citations_present is False:
            violations.append(
                PolicyViolation(
                    policy="require_citations",
                    severity="block",
                    message="output contains no citations but citations are required",
                )
            )
        if self.policies.safety == "strict":
            matches = self.pii.detect(text)
            high = [m for m in matches if m.confidence >= 0.9 and m.type in ("api_key", "secret", "government_id", "credit_card")]
            if high:
                violations.append(
                    PolicyViolation(
                        policy="output_sensitive_data",
                        severity="block",
                        message="output contains credentials or government/financial identifiers",
                        details={"types": sorted({m.type for m in high})},
                    )
                )
        transformed = self._check_rails(text, direction="output", violations=violations)
        allowed = not any(v.severity == "block" for v in violations)
        return PolicyCheckResult(allowed=allowed, violations=violations, transformed_text=transformed)

    # -- egress DLP (last-mile, provider boundary) ------------------------------------

    def scan_egress(self, request: ModelRequest) -> PolicyCheckResult:
        """Deterministic DLP scan of the fully-assembled provider request.

        Scans system + messages + tool schemas — the exact bytes about to leave
        the process — for credentials, secrets, and sensitive identifiers,
        independent of how earlier (input/output) checks were wired. This is the
        always-on last line of defense: even a call site that bypassed every
        other check still passes through here before any provider dispatch.

        Mode ``off`` returns immediately; ``warn`` records findings without
        blocking; ``block`` blocks on a high-confidence credential/identifier
        leak. Residency is enforced separately at the run boundary.
        """
        if self.egress_dlp == "off":
            return PolicyCheckResult(allowed=True)

        violations: list[PolicyViolation] = []
        # Assemble the outbound text once: message content + tool descriptions.
        parts: list[str] = [message.text for message in request.messages]
        tool_payload: dict[str, Any] = {}
        for spec in request.tools:
            parts.append(spec.description or "")
            tool_payload[spec.name] = {
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
            }
        assembled = "\n".join(p for p in parts if p)

        # Secrets in free text and structured tool schemas.
        secret_findings = self.secrets.scan_text(assembled)
        secret_findings += self.secrets.scan(tool_payload, path="tools")
        if secret_findings:
            kinds = sorted({f.kind for f in secret_findings})
            violations.append(
                PolicyViolation(
                    policy="egress_secret",
                    severity="block" if self.egress_dlp == "block" else "warn",
                    message=f"outbound request carries {len(secret_findings)} secret-like value(s)",
                    details={"kinds": kinds, "paths": [f.path for f in secret_findings][:10]},
                )
            )

        # High-confidence credentials / identifiers in the assembled text.
        pii_matches = self.pii.detect(assembled)
        leaking = [
            m for m in pii_matches
            if m.type in _EGRESS_BLOCK_PII_TYPES and m.confidence >= 0.9
        ]
        if leaking:
            violations.append(
                PolicyViolation(
                    policy="egress_sensitive_identifier",
                    severity="block" if self.egress_dlp == "block" else "warn",
                    message="outbound request carries credentials or sensitive identifiers",
                    details={"types": sorted({m.type for m in leaking})},
                )
            )

        allowed = not any(v.severity == "block" for v in violations)
        return PolicyCheckResult(allowed=allowed, violations=violations)

    # -- memory-side checks -----------------------------------------------------------

    def check_memory_write(self, content: str) -> PolicyCheckResult:
        violations: list[PolicyViolation] = []
        if not self.policies.allow_memory_writes:
            violations.append(
                PolicyViolation(
                    policy="memory_writes_disabled",
                    severity="block",
                    message="memory writes are disabled by policy",
                )
            )
        matches = self.pii.detect(content)
        secrets = [m for m in matches if m.type in ("api_key", "secret")]
        if secrets:
            violations.append(
                PolicyViolation(
                    policy="memory_no_secrets",
                    severity="block",
                    message="refusing to store credentials in memory",
                    details={"types": sorted({m.type for m in secrets})},
                )
            )
        allowed = not any(v.severity == "block" for v in violations)
        return PolicyCheckResult(allowed=allowed, violations=violations)
