"""Continuous assurance cases & production certification.

The platform already *produces* the evidence a production AI system is judged on —
eval and regression gates, the governance-invariant verifier, reasoning
certificates and runtime monitors, identity and delegation provenance, the signed
audit chain, C2PA media provenance, and SBOM / SLSA build attestations. This module
is the capstone that ties them together: it **assembles that evidence into one
structured, machine-checkable argument** that the system is fit for purpose, and
keeps that argument **continuously valid as the system changes** — the
assurance-case discipline (GSN / CAE) the safety and regulatory frontier demands.

* :class:`Claim` / :class:`Evidence` — an argument tree: a top claim (*this app is
  fit for purpose X under context Y*) decomposed into sub-claims, each discharged by
  evidence the platform already emits, bound by hash so the whole case
  :meth:`~AssuranceCase.verify`\\s offline and a missing or stale piece is pinpointed.
* :class:`AssuranceCase` (``app.assurance_case``) — the signed, content-bound case;
  :meth:`~AssuranceCase.check` re-evaluates it against the current evidence into an
  :class:`AssuranceReport`, and :func:`assurance_regression_gate` turns a falsified
  claim into a build failure — assurance as a living, CI-gated invariant.
* :class:`Incident` + :meth:`~AssuranceCase.learn_from` — a signed production
  failure ties to the sub-claim it falsified and the case **learns** (a remediation
  sub-claim demands fresh evidence before it re-validates).
* :class:`CertificationReport` (``app.certify``) — a portable, offline-verifiable
  certification (the case, its evidence, residual risks, build provenance) a
  downstream operator or auditor checks from the bytes.

Everything here is opt-in, additive, deterministic, and offline; it composes the
platform's existing verdicts rather than introducing a new runtime.

    case = app.assurance_case(
        "The assistant is fit for production",
        context="EU deployment, tier-1 traffic",
        subclaims=[Claim(id="governance", statement="Controls hold",
                         evidence=[Evidence.from_governance(app.verify_governance())])],
    )
    report = app.certify(case)
    assert report.certified and report.verify()
"""

from __future__ import annotations

from .case import (
    AssuranceCase,
    AssuranceReport,
    Claim,
    ClaimStatus,
    assurance_regression_gate,
)
from .certification import CertificationReport, certify
from .evidence import EVIDENCE_KINDS, Evidence
from .incident import Incident

__all__ = [
    "Evidence",
    "EVIDENCE_KINDS",
    "Claim",
    "ClaimStatus",
    "AssuranceCase",
    "AssuranceReport",
    "assurance_regression_gate",
    "Incident",
    "CertificationReport",
    "certify",
]
