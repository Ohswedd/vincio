"""Programmable rails.

NeMo-Guardrails-style input/output rails expressed in the deterministic
policy engine. A rail is plain data — direction, kind, action, parameters —
and every check is plain code over the text: keyword/regex topic matching,
format constraints, and the security engine's own detectors (PII, secrets,
injection). No rail depends on model judgment, so enforcement is exact,
explainable, and free.

Rails run inside :meth:`PolicyEngine.check_input` /
:meth:`PolicyEngine.check_output`, i.e. before and after every generation,
and each violation becomes a :class:`PolicyViolation` that lands on the
trace and in the audit log like any other policy decision.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import SecurityError
from .injection import InjectionDetector
from .pii import PIIDetector, redact
from .secrets import SecretScanner

__all__ = ["Rail", "RailEngine"]

RailDirection = Literal["input", "output", "both"]
RailKind = Literal["topic", "format", "safety", "custom"]
RailAction = Literal["block", "warn", "redact"]


class Rail(BaseModel):
    """One programmable rail.

    Kinds:

    - ``topic`` — ``blocked_topics`` (deny when mentioned) and/or
      ``allowed_topics`` (deny when none is mentioned).
    - ``format`` — ``max_chars``, ``require_pattern`` (regex the text must
      match), ``forbid_pattern`` (regex the text must not match).
    - ``safety`` — ``detectors`` chosen from ``pii`` / ``secrets`` /
      ``injection``; ``redact`` action masks PII instead of blocking.
    - ``custom`` — a named predicate registered on the
      :class:`RailEngine`; the rail fires when the predicate returns a
      truthy value (a string return becomes the violation message).
    """

    name: str
    kind: RailKind = "custom"
    direction: RailDirection = "both"
    action: RailAction = "block"
    # topic
    blocked_topics: list[str] = Field(default_factory=list)
    allowed_topics: list[str] = Field(default_factory=list)
    # format
    max_chars: int | None = None
    require_pattern: str | None = None
    forbid_pattern: str | None = None
    # safety
    detectors: list[Literal["pii", "secrets", "injection"]] = Field(default_factory=list)
    # custom
    predicate: str | None = None  # name registered via RailEngine.register
    params: dict[str, Any] = Field(default_factory=dict)

    def applies(self, direction: str) -> bool:
        return self.direction in ("both", direction)


class RailViolation(BaseModel):
    rail: str
    kind: RailKind
    action: RailAction
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RailCheck(BaseModel):
    """Outcome of evaluating all applicable rails over one text."""

    allowed: bool = True
    violations: list[RailViolation] = Field(default_factory=list)
    transformed_text: str | None = None  # set when a redact rail fired


def _topic_hits(text: str, topics: list[str]) -> list[str]:
    lowered = text.lower()
    return [
        topic
        for topic in topics
        if re.search(rf"\b{re.escape(topic.lower())}\b", lowered)
    ]


class RailEngine:
    """Evaluates rails deterministically, reusing the security detectors."""

    def __init__(
        self,
        rails: list[Rail] | None = None,
        *,
        pii_detector: PIIDetector | None = None,
        secret_scanner: SecretScanner | None = None,
        injection_detector: InjectionDetector | None = None,
    ) -> None:
        self.rails: list[Rail] = list(rails or [])
        self.pii = pii_detector or PIIDetector()
        self.secrets = secret_scanner or SecretScanner()
        self.injection = injection_detector or InjectionDetector()
        self._predicates: dict[str, Callable[[str, dict[str, Any]], Any]] = {}

    def add(self, rail: Rail) -> Rail:
        if rail.kind == "custom" and rail.predicate and rail.predicate not in self._predicates:
            raise SecurityError(
                f"rail {rail.name!r} references unregistered predicate {rail.predicate!r}; "
                "register it via RailEngine.register first"
            )
        self.rails.append(rail)
        return rail

    def register(self, name: str, predicate: Callable[[str, dict[str, Any]], Any]) -> None:
        """Register a custom predicate: ``(text, params) -> falsy | message``."""
        self._predicates[name] = predicate

    def check(self, text: str, *, direction: str) -> RailCheck:
        result = RailCheck()
        current = text or ""
        for rail in self.rails:
            if not rail.applies(direction):
                continue
            violation = self._evaluate(rail, current)
            if violation is None:
                continue
            if rail.action == "redact" and rail.kind == "safety":
                matches = self.pii.detect(current)
                if matches:
                    current = redact(current, matches)
                    result.transformed_text = current
            result.violations.append(violation)
            if rail.action == "block":
                result.allowed = False
        return result

    def _evaluate(self, rail: Rail, text: str) -> RailViolation | None:
        if rail.kind == "topic":
            blocked = _topic_hits(text, rail.blocked_topics)
            if blocked:
                return self._violation(rail, f"blocked topic(s) mentioned: {blocked}", topics=blocked)
            if rail.allowed_topics and not _topic_hits(text, rail.allowed_topics):
                return self._violation(
                    rail,
                    "text addresses none of the allowed topics",
                    allowed=rail.allowed_topics,
                )
            return None
        if rail.kind == "format":
            if rail.max_chars is not None and len(text) > rail.max_chars:
                return self._violation(
                    rail, f"text exceeds {rail.max_chars} characters", length=len(text)
                )
            if rail.require_pattern and not re.search(rail.require_pattern, text):
                return self._violation(
                    rail, f"text does not match required pattern {rail.require_pattern!r}"
                )
            if rail.forbid_pattern:
                match = re.search(rail.forbid_pattern, text)
                if match:
                    return self._violation(
                        rail,
                        f"text matches forbidden pattern {rail.forbid_pattern!r}",
                        match=match.group(0)[:80],
                    )
            return None
        if rail.kind == "safety":
            findings: dict[str, Any] = {}
            if "pii" in rail.detectors:
                matches = self.pii.detect(text)
                if matches:
                    findings["pii"] = sorted({m.type for m in matches})
            if "secrets" in rail.detectors:
                secrets = self.secrets.scan_text(text)
                if secrets:
                    findings["secrets"] = sorted({s.kind for s in secrets})
            if "injection" in rail.detectors:
                verdict = self.injection.detect(text)
                if verdict.detected:
                    findings["injection"] = [s.pattern for s in verdict.signals]
            if findings:
                return self._violation(
                    rail, f"safety detector(s) fired: {sorted(findings)}", **findings
                )
            return None
        # custom
        if rail.predicate is None:
            return None
        predicate = self._predicates.get(rail.predicate)
        if predicate is None:
            return self._violation(rail, f"predicate {rail.predicate!r} is not registered")
        outcome = predicate(text, rail.params)
        if outcome:
            message = outcome if isinstance(outcome, str) else f"rail {rail.name!r} fired"
            return self._violation(rail, message)
        return None

    @staticmethod
    def _violation(rail: Rail, message: str, **details: Any) -> RailViolation:
        return RailViolation(
            rail=rail.name, kind=rail.kind, action=rail.action, message=message, details=details
        )
