"""Solvency, arbitration, reputation, admission, and cross-org capstone verbs — a private mixin of
:class:`~vincio.core.app.ContextApp`.

Extracted verbatim from ``vincio/core/app.py`` (v7.5 structure line): method
source, decorators, comments, and docstrings are unchanged. ``ContextApp``
composes this class, so every method here remains an ``app.*`` verb; the
``self: ContextApp`` annotations keep attribute access type-checked against
the composed app. The standing hygiene lints (:mod:`vincio._error_contract`,
:mod:`vincio._observable_failure`, :mod:`vincio._assert_robustness`)
deliberately keep ``vincio/core/_app_*.py`` in scope despite the private
filename, so the verb surface stays guarded after the split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..providers.base import run_sync
from ..stability import _resolve_renamed_kwarg
from .errors import (
    SettlementError,
)

if TYPE_CHECKING:
    from .app import ContextApp


def _resolve_verifier(verifier: Any | None, verify_with: Any | None, owner: str) -> Any | None:
    """Resolve the ``verifier=`` / deprecated ``verify_with=`` rename runway.

    Shared by the app's settlement verbs mid-rename (since 7.5, removed in 8.0):
    the old keyword warns and forwards, passing both raises
    :class:`~vincio.core.errors.SettlementError`.
    """
    return _resolve_renamed_kwarg(
        verifier,
        verify_with,
        new_name="verifier",
        old_name="verify_with",
        owner=owner,
        since="7.5",
        removed_in="8.0",
        error=SettlementError,
        stacklevel=4,
    )


class _SettlementVerbs:
    """Solvency, arbitration, reputation, admission, and cross-org capstone verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        _issued_revocations: list[Any]
        imported_reputation: Any


    def attest_liabilities(  # type: ignore[misc]
        self: ContextApp,
        poster: str,
        liabilities: Any,
        *,
        attestor: str | None = None,
        as_of: Any | None = None,
        prior: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Attest a poster's total obligations into a signed, content-bound proof-of-liabilities.

        Issues a :class:`~vincio.settlement.LiabilityAttestation` over the obligations ``poster``
        owes — itemized ``liabilities`` (a number, a mapping of ``creditor -> amount``, or
        :class:`~vincio.settlement.LiabilityLine` items) whose total re-derives on every verify —
        the liability side of a proof-of-solvency. ``attestor`` defaults to this app (a
        third-party attestor vouching), and when it is also the ``poster`` the attestation is
        self-attested. Pass ``prior`` (the preceding snapshot) to link it into a hash-linked history
        :meth:`check_history_consistency` can walk. Signs it as the attestor and, unless
        ``record_audit`` is off, records the issuance on the audit chain. Returns it::

            owed = app.attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
            proof = app.prove_solvency(reserves_proof, owed)
        """
        from ..settlement import attest_liabilities as _attest

        resolved_attestor = attestor or self.name
        attestation = _attest(
            poster, liabilities, attestor=resolved_attestor, as_of=as_of, prior=prior
        )
        if sign and self.name == attestation.attestor:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                attestation.sign(signer, party=attestation.attestor)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import LIABILITY_ACTION

            entry = self.audit.record(
                LIABILITY_ACTION,
                resource=attestation.poster,
                decision="self_attested" if attestation.self_attested else "attested",
                details=attestation.audit_details(),
            )
            attestation.audit_id = getattr(entry, "id", None)
        return attestation

    def inclusion_proof(self: ContextApp, liabilities: Any, creditor: str) -> Any:  # type: ignore[misc]
        """Build an offline-verifiable inclusion proof for one creditor's liability claim.

        Thin wrapper over :meth:`~vincio.settlement.LiabilityAttestation.inclusion_proof`: the
        :class:`~vincio.settlement.InclusionProof` shows ``creditor``'s obligation is a leaf of
        the attestation's signed Merkle root, so the creditor confirms its claim was counted in
        the attested total and a poster cannot quietly drop it. Returns it::

            owed = app.attest_liabilities("vendor", {"acme": 60.0, "globex": 40.0})
            proof = app.inclusion_proof(owed, "acme")
            proof.verify(owed).valid  # True
        """
        return liabilities.inclusion_proof(creditor)

    def check_completeness(  # type: ignore[misc]
        self: ContextApp,
        liabilities: Any,
        claims: Any,
        *,
        as_of: Any | None = None,
        sign: bool = True,
        verifier: Any | None = None,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Fold creditor claims against a liability attestation into a completeness check.

        Issues a :class:`~vincio.settlement.CompletenessProof`
        (:func:`~vincio.settlement.check_completeness`) folding what creditors can prove they are
        owed (``claims`` — a ``creditor -> amount`` mapping, or
        :class:`~vincio.settlement.LiabilityLine` / settlement-record / ``(creditor, amount)``
        items) against the attestation, pinpointing every omitted or under-stated claim as an
        :class:`~vincio.settlement.OmissionBreach` and raising the attested figure to a completed
        total :meth:`prove_solvency` reads (``completeness=``). Signs the check as this app and,
        unless ``record_audit`` is off, records it on the audit chain. A tampered attestation is
        refused (a forged attestor signature too, with ``verifier``). ``verify_with`` is a
        deprecated alias for ``verifier`` (since 7.5, removed in 8.0). Returns it::

            owed = app.attest_liabilities("vendor", {"acme": 60.0})
            check = app.check_completeness(owed, {"acme": 60.0, "globex": 40.0})
            check.require_complete()  # raises: globex is omitted
        """
        from ..settlement import check_completeness as _check

        verifier = _resolve_verifier(verifier, verify_with, "app.check_completeness")
        proof = _check(liabilities, claims, verifier=verifier, as_of=as_of)
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                proof.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import COMPLETENESS_ACTION

            entry = self.audit.record(
                COMPLETENESS_ACTION,
                resource=proof.poster,
                decision=proof.status,
                details=proof.audit_details(),
            )
            proof.audit_id = getattr(entry, "id", None)
        return proof

    def prove_solvency(  # type: ignore[misc]
        self: ContextApp,
        custody: Any,
        liabilities: Any,
        *,
        poster: str | None = None,
        completeness: Any | None = None,
        as_of: Any | None = None,
        sign: bool = True,
        verifier: Any | None = None,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Fold a reserve proof against a liability proof into a proof-of-solvency.

        Reconciles a proven :class:`~vincio.settlement.CustodyAttestation` (reserves) against a
        proven :class:`~vincio.settlement.LiabilityAttestation` (obligations) for the same poster
        into a bounded :class:`~vincio.settlement.SolvencyProof` — the proof-of-solvency the
        literature pairs with a proof-of-reserves (``reserves ≥ liabilities``). When the
        liabilities exceed the reserves the shortfall surfaces as a pinpointed
        :class:`~vincio.settlement.InsolvencyBreach`. Pass ``completeness`` (a
        :class:`~vincio.settlement.CompletenessProof` over this attestation) to bound the margin
        against the *completed* liability total — the attestor's figure raised by every
        obligation a creditor proved it omitted, not just the creditors the attestor listed.
        Signs the proof as this app and, unless ``record_audit`` is off, records it on the audit
        chain. A tampered or wrong-poster attestation (or completeness check) is refused (a
        forged signature too, with ``verifier``). ``verify_with`` is a deprecated alias for
        ``verifier`` (since 7.5, removed in 8.0). The proof's solvency-adjusted held figure
        reads into :meth:`guard_collateral` (``solvency=``). Returns the proof::

            reserves = app.attest_custody("vendor", {"omnibus": 80.0})
            owed = app.attest_liabilities("vendor", {"acme": 60.0})
            proof = app.prove_solvency(reserves, owed)
            ledger = app.guard_collateral([pool], solvency=proof)
            proof.require_solvent()  # raises if liabilities exceed reserves
        """
        from ..settlement import prove_solvency as _prove

        verifier = _resolve_verifier(verifier, verify_with, "app.prove_solvency")
        proof = _prove(
            custody,
            liabilities,
            poster=poster,
            completeness=completeness,
            as_of=as_of,
            verifier=verifier,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                proof.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import SOLVENCY_ACTION

            entry = self.audit.record(
                SOLVENCY_ACTION,
                resource=proof.poster,
                decision=proof.status,
                details=proof.audit_details(),
            )
            proof.audit_id = getattr(entry, "id", None)
        return proof

    def check_root_consistency(  # type: ignore[misc]
        self: ContextApp,
        attestations: Any,
        *,
        verifier: Any | None = None,
        record_reputation: bool = True,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Compare liability attestations across creditors for cross-org non-equivocation.

        Folds a set of liability attestations a group of creditors hold — each the attestation a
        poster signed *for it* — into a :class:`~vincio.settlement.RootConsistencyReport`
        (:func:`~vincio.settlement.check_root_consistency`), surfacing every poster that signed
        **different** roots for the same ``(poster, attestor, as_of)`` as a non-repudiable
        :class:`~vincio.settlement.EquivocationProof`. Where :meth:`check_completeness` catches an
        omission only when the omitted creditor folds its own claim, this catches the counterparty
        that **equivocates** — showing each creditor a root on which its own claim *is* present
        while the totals disagree. ``attestations`` items may be bare attestations or
        ``(creditor, attestation)`` pairs to record which creditor saw each root. With
        ``verifier`` only attestor-signed roots are admitted, so a forged root cannot
        manufacture a false accusation. ``verify_with`` is a deprecated alias for ``verifier``
        (since 7.5, removed in 8.0). Unless ``record_audit`` is off, records each proven
        equivocation on the audit chain; unless ``record_reputation`` is off, credits a failure
        against the equivocating poster on this app's reputation ledger (when one is attached).
        Returns the report::

            owed_acme = vendor.attest_liabilities("vendor", {"acme": 60.0}, as_of=t)
            owed_globex = vendor.attest_liabilities("vendor", {"globex": 40.0}, as_of=t)
            report = auditor.check_root_consistency([("acme", owed_acme), ("globex", owed_globex)])
            report.require_consistent()  # raises: vendor signed two roots for one instant
        """
        from ..settlement import check_root_consistency as _check

        verifier = _resolve_verifier(verifier, verify_with, "app.check_root_consistency")
        report = _check(attestations, verifier=verifier)
        dinged: set[str] = set()
        for proof in report.equivocations:
            if record_audit and self.audit is not None:
                from ..settlement.solvency import EQUIVOCATION_ACTION

                entry = self.audit.record(
                    EQUIVOCATION_ACTION,
                    resource=proof.poster,
                    decision="equivocation",
                    details=proof.audit_details(),
                )
                proof.audit_id = getattr(entry, "id", None)
            # Every distinct pairwise conflict is audited, but a poster's reputation is debited
            # once per check — three conflicting roots are one equivocating counterparty.
            if (
                record_reputation
                and self.reputation_ledger is not None
                and proof.poster not in dinged
            ):
                self.reputation_ledger.record_outcome(
                    proof.poster,
                    passed=False,
                    round_id=proof.id,
                    details={"kind": "liability_equivocation", "attestor": proof.attestor},
                )
                dinged.add(proof.poster)
        return report

    def discharge_liability(  # type: ignore[misc]
        self: ContextApp,
        poster: str,
        amount_usd: float,
        *,
        creditor: str | None = None,
        as_of: Any | None = None,
        note: str = "",
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Issue a signed, content-bound :class:`~vincio.settlement.Discharge` of what ``poster`` owes.

        Releases ``amount_usd`` of the obligation ``poster`` owes this app — the **creditor** issues
        the discharge, so ``creditor`` defaults to this app and it is signed with this app's key.
        Folded into :meth:`check_history_consistency` (``discharges=``) to explain a legitimate
        reduction in the poster's liabilities between two snapshots, so the matching drop is not
        treated as a debt that silently vanished. Unless ``record_audit`` is off, records the
        issuance on the audit chain. Returns it::

            settled = acme.discharge_liability("vendor", 70.0)  # acme releases $70 of vendor's debt
            report = auditor.check_history_consistency(snapshots, discharges=[settled])
        """
        from ..settlement import discharge_liability as _discharge

        resolved_creditor = creditor or self.name
        discharge = _discharge(poster, resolved_creditor, amount_usd, as_of=as_of, note=note)
        if sign and self.name == discharge.creditor:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                discharge.sign(signer, party=discharge.creditor)
        if record_audit and self.audit is not None:
            from ..settlement.solvency import DISCHARGE_ACTION

            entry = self.audit.record(
                DISCHARGE_ACTION,
                resource=discharge.poster,
                decision=discharge.status,
                details=discharge.audit_details(),
            )
            discharge.audit_id = getattr(entry, "id", None)
        return discharge

    def check_history_consistency(  # type: ignore[misc]
        self: ContextApp,
        attestations: Any,
        *,
        discharges: Any | None = None,
        verifier: Any | None = None,
        record_reputation: bool = True,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Walk a poster's liability snapshots for cross-time monotonicity (no debt silently dropped).

        Folds a set of liability snapshots into a
        :class:`~vincio.settlement.HistoryConsistencyReport`
        (:func:`~vincio.settlement.check_history_consistency`), surfacing every poster that let a
        creditor's obligation **drop** between snapshots without a signed
        :class:`~vincio.settlement.Discharge` (``discharges``) explaining the release as a pinpointed
        :class:`~vincio.settlement.MonotonicityBreach`. Where :meth:`check_root_consistency` catches a
        counterparty signing different roots for the *same* instant, this catches one quietly dropping
        a past obligation in a *later* snapshot. With ``verifier`` only attestor-signed snapshots
        and creditor-signed discharges are admitted as evidence. ``verify_with`` is a deprecated
        alias for ``verifier`` (since 7.5, removed in 8.0). Unless ``record_audit`` is off,
        records each inconsistent history on the audit chain; unless ``record_reputation`` is off,
        credits a failure against the breaching poster on this app's reputation ledger (when one is
        attached). Returns the report::

            s1 = vendor.attest_liabilities("vendor", {"acme": 100.0}, as_of=t1)
            s2 = vendor.attest_liabilities("vendor", {"acme": 30.0}, as_of=t2, prior=s1)
            report = auditor.check_history_consistency([s1, s2])
            report.require_consistent()  # raises: $70 owed to acme vanished without a discharge
        """
        from ..settlement import check_history_consistency as _check

        verifier = _resolve_verifier(verifier, verify_with, "app.check_history_consistency")
        report = _check(attestations, discharges=discharges, verifier=verifier)
        signer = self._resolve_contract_signer(None, True)
        for proof in report.proofs:
            if signer is not None:
                proof.sign(signer, party=self.name)
            if record_audit and self.audit is not None:
                from ..settlement.solvency import HISTORY_ACTION

                entry = self.audit.record(
                    HISTORY_ACTION,
                    resource=proof.poster,
                    decision=proof.status,
                    details=proof.audit_details(),
                )
                proof.audit_id = getattr(entry, "id", None)
            if record_reputation and self.reputation_ledger is not None and not proof.monotone:
                self.reputation_ledger.record_outcome(
                    proof.poster,
                    passed=False,
                    round_id=proof.id,
                    details={"kind": "liability_history", "attestor": proof.attestor},
                )
        return report

    def build_set_off_statement(  # type: ignore[misc]
        self: ContextApp,
        poster: str,
        creditor: str,
        owed_usd: float,
        owing_usd: float,
        *,
        references: Any | None = None,
        as_of: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Collapse the mutual obligations between a poster and one creditor into a statement.

        Builds a :class:`~vincio.settlement.SetOffStatement`
        (:func:`~vincio.settlement.build_set_off_statement`) stating the obligations running *both
        ways* between ``poster`` and ``creditor`` — ``owed_usd`` the poster owes the creditor,
        ``owing_usd`` the creditor owes the poster back — and computing the poster's bounded net
        liability (``max(0, owed − owing)``). Signs it as this app (one side of the mutually-agreed
        close-out — the counterparty co-signs its copy) and, unless ``record_audit`` is off, records
        the issuance on the audit chain. Returns it::

            statement = vendor.build_set_off_statement("vendor", "acme", 30.0, 12.0)
            resolution = auditor.resolve_insolvency(reserves, owed, set_off=[statement])
        """
        from ..settlement import build_set_off_statement as _build

        statement = _build(
            poster, creditor, owed_usd, owing_usd, references=references, as_of=as_of
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                statement.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.setoff import SETOFF_ACTION

            entry = self.audit.record(
                SETOFF_ACTION,
                resource=statement.poster,
                decision=statement.direction,
                details=statement.audit_details(),
            )
            statement.audit_id = getattr(entry, "id", None)
        return statement

    def build_seniority_schedule(  # type: ignore[misc]
        self: ContextApp,
        poster: str,
        tranches: Any,
        *,
        as_of: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Rank a poster's obligations into a signed, content-bound seniority schedule.

        Builds a :class:`~vincio.settlement.SenioritySchedule`
        (:func:`~vincio.settlement.build_seniority_schedule`) ranking the creditors ``poster`` owes
        into priority tranches — rank ``0`` most senior — an :meth:`resolve_insolvency` waterfall
        pays out in. ``tranches`` is an ordered spec — its simplest form is a list of creditor-name
        lists where **position is priority** (``[["bank"], ["acme", "globex"]]``) — or
        :class:`~vincio.settlement.SeniorityTranche` items for explicit ranks and labels. Signs the
        schedule as this app and, unless ``record_audit`` is off, records the issuance on the audit
        chain. Returns it::

            schedule = app.build_seniority_schedule("vendor", [["bank"], ["acme", "globex"]])
            resolution = app.resolve_insolvency(reserves, owed, schedule)
        """
        from ..settlement import build_seniority_schedule as _build

        schedule = _build(poster, tranches, as_of=as_of)
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                schedule.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.waterfall import SENIORITY_ACTION

            entry = self.audit.record(
                SENIORITY_ACTION,
                resource=schedule.poster,
                decision="self_ranked" if schedule.poster == self.name else "ranked",
                details=schedule.audit_details(),
            )
            schedule.audit_id = getattr(entry, "id", None)
        return schedule

    def resolve_insolvency(  # type: ignore[misc]
        self: ContextApp,
        custody: Any,
        liabilities: Any,
        schedule: Any | None = None,
        *,
        poster: str | None = None,
        completeness: Any | None = None,
        solvency: Any | None = None,
        set_off: Any | None = None,
        as_of: Any | None = None,
        verifier: Any | None = None,
        sign: bool = True,
        record_reputation: bool = True,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Distribute a poster's proven reserves across its ranked liabilities into a resolution.

        Folds a proven :class:`~vincio.settlement.CustodyAttestation` against a proven
        :class:`~vincio.settlement.LiabilityAttestation` and distributes the reserves across the
        obligations **by seniority then pari-passu within a tranche** (``schedule``) into a
        content-bound :class:`~vincio.settlement.InsolvencyResolution`
        (:func:`~vincio.settlement.resolve_insolvency`), pinpointing each creditor's bounded
        :class:`~vincio.settlement.CreditorRecovery` and the shortfall it bears — so an insolvency a
        :class:`~vincio.settlement.SolvencyProof` only *flagged* is *resolved* into who-gets-what.
        With no ``schedule`` the whole liability set is one pari-passu tranche; pass
        ``completeness`` to distribute against the *completed* liability set, and ``set_off`` (a list
        of mutually-signed :class:`~vincio.settlement.SetOffStatement`\\ s) to **close-out net** each
        creditor to its net claim before the waterfall. Reuses
        :func:`~vincio.settlement.prove_solvency`, so a tampered, forged, or wrong-poster
        attestation (or a malformed/wrong-poster schedule, or a one-sided/over-stated set-off) is
        refused (a forged signature too, with ``verifier``). ``verify_with`` is a deprecated
        alias for ``verifier`` (since 7.5, removed in 8.0). Signs the resolution as this app;
        unless ``record_audit`` is off, records it on the audit chain; unless ``record_reputation``
        is off, credits a failure against the poster on this app's reputation ledger (when one is
        attached) when the reserves could not make every creditor whole. Returns the resolution::

            owed = app.attest_liabilities("vendor", {"bank": 50.0, "acme": 50.0})
            reserves = app.attest_custody("vendor", {"omnibus": 50.0})
            schedule = app.build_seniority_schedule("vendor", [["bank"], ["acme"]])
            resolution = app.resolve_insolvency(reserves, owed, schedule)
            resolution.require_fully_recovered()  # raises: acme bears the $50 shortfall
        """
        from ..settlement import resolve_insolvency as _resolve

        verifier = _resolve_verifier(verifier, verify_with, "app.resolve_insolvency")
        resolution = _resolve(
            custody,
            liabilities,
            schedule,
            poster=poster,
            completeness=completeness,
            solvency=solvency,
            set_off=set_off,
            as_of=as_of,
            verifier=verifier,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                resolution.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.waterfall import INSOLVENCY_ACTION

            entry = self.audit.record(
                INSOLVENCY_ACTION,
                resource=resolution.poster,
                decision=resolution.status,
                details=resolution.audit_details(),
            )
            resolution.audit_id = getattr(entry, "id", None)
        if record_reputation and self.reputation_ledger is not None and not resolution.solvent:
            self.reputation_ledger.record_outcome(
                resolution.poster,
                passed=False,
                round_id=resolution.id,
                details={"kind": "insolvency_resolution", "attestor": resolution.attestor},
            )
        return resolution

    def guard_collateral(  # type: ignore[misc]
        self: ContextApp,
        pools: list[Any],
        *,
        poster: str | None = None,
        held: float | None = None,
        custody: Any | None = None,
        solvency: Any | None = None,
        sign: bool = True,
        verifier: Any | None = None,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Fold a counterparty's collateral pools into a bounded re-use guard.

        Reconciles what ``pools`` collectively pledge against the capital the poster holds
        into a single, content-bound :class:`~vincio.settlement.CollateralLedger` — the
        rehypothecation analogue of :meth:`clear_settlements`. The same capital pledged across
        more than one pool is pinpointed as a :class:`~vincio.settlement.ReuseBreach`, and
        each beneficiary's claim is bounded to its deterministic, pari-passu share of the held
        capital, so a scarce stake is apportioned by priority rather than over-promised. Signs
        the ledger as this app and, unless ``record_audit`` is off, records the guard on the
        audit chain. The ledger verifies offline from the bytes alone, and a tampered pool is
        **refused** (with ``verifier`` a forged pool signature is too) rather than folded
        silently. ``verify_with`` is a deprecated alias for ``verifier`` (since 7.5, removed
        in 8.0).

        The held figure comes from a ``solvency``
        :class:`~vincio.settlement.SolvencyProof` (the solvency-adjusted ``max(0, reserves −
        liabilities)``, bounding pledges against capital not already owed elsewhere and exposing
        :attr:`~vincio.settlement.CollateralLedger.insolvent`), a ``custody``
        :class:`~vincio.settlement.CustodyAttestation` **proving** the reserves (a tampered or
        forged one is refused, and an :class:`~vincio.settlement.UnderReservedBreach` surfaces
        when the proven reserves fall below the pledges), an explicit *asserted* ``held``, or
        — by default — the gross pledge minus the provably double-pledged capital. Returns the
        ledger::

            proof = app.attest_custody("vendor", {"omnibus": 80.0})
            ledger = app.guard_collateral([pool_a, pool_b], custody=proof)
            ledger.under_reserved          # proven reserves below the pledges
            ledger.require_within_bounds()  # raises if over-committed
        """
        from ..settlement import guard_collateral as _guard

        verifier = _resolve_verifier(verifier, verify_with, "app.guard_collateral")
        ledger = _guard(
            pools,
            poster=poster,
            held=held,
            custody=custody,
            solvency=solvency,
            verifier=verifier,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                ledger.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.rehypothecation import REHYPOTHECATION_ACTION

            entry = self.audit.record(
                REHYPOTHECATION_ACTION,
                resource=ledger.id,
                decision=ledger.status,
                details=ledger.audit_details(),
            )
            ledger.audit_id = getattr(entry, "id", None)
        return ledger

    def settle_saga(  # type: ignore[misc]
        self: ContextApp,
        result: Any,
        *,
        contracts: dict[str, Any],
        run_id: str | None = None,
        party: str | None = None,
        sign: bool = True,
        record_reputation: bool = True,
    ) -> list[Any]:
        """Close the books on every contract a cross-org saga ran under.

        Meters each contracted forward step from the saga's durable journal and
        reconciles the per-step delivery against the matching contract in
        ``contracts`` (keyed by contract id), appending one signed, hash-chained
        :class:`~vincio.settlement.SettlementRecord` per contract to the settlement
        book — so a whole cross-org engagement reconciles in one call. Returns the
        records, in contract-id order.
        """
        return self._settlement_book().settle_saga(
            result,
            contracts=contracts,
            run_id=run_id,
            party=party,
            sign=sign,
            record_reputation=record_reputation,
        )

    def settlement_report(self: ContextApp, counterparty: str | None = None) -> Any:  # type: ignore[misc]
        """Per-counterparty settlement roll-up — beside the cost report.

        Each row totals what was owed, what was delivered, and the net balance with
        a counterparty, with the settled / breached tally behind it. Returns an
        empty :class:`~vincio.settlement.SettlementReport` when no book is attached.
        """
        if self.settlement_book is None:
            from ..settlement import SettlementReport

            return SettlementReport(owner=self.name)
        return self.settlement_book.report(counterparty)

    def clear_settlements(  # type: ignore[misc]
        self: ContextApp,
        *,
        books: list[Any] | None = None,
        records: list[Any] | None = None,
        sign: bool = True,
        verifier: Any | None = None,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Net a fleet's settlement books into one minimal cleared set.

        Folds the bilateral balances across ``books`` (and/or loose ``records``) —
        or, by default, this app's own attached settlement book — into a single,
        content-bound :class:`~vincio.settlement.NettingSet`: each org's many
        positions collapsed to the minimal set of net obligations, the same web of
        contracts cleared once. Signs the set as this app (the clearer) and, unless
        ``record_audit`` is off, records the clearing on the audit chain. The set
        verifies offline from the bytes alone — the positions balance and the cleared
        obligations reproduce them — and pinpoints any disputed contract rather than
        netting it silently. ``verify_with`` is a deprecated alias for ``verifier``
        (since 7.5, removed in 8.0). Returns the set::

            netting = app.clear_settlements(books=[acme_book, vendor_book])
            netting.verify().valid  # offline-verifiable
            netting.print_summary()
        """
        from ..settlement import net_settlements

        all_records: list[Any] = list(records or [])
        sources = books
        if sources is None and not all_records and self.settlement_book is not None:
            sources = [self.settlement_book]
        for book in sources or []:
            all_records.extend(book.records)
        verifier = _resolve_verifier(verifier, verify_with, "app.clear_settlements")
        netting = net_settlements(all_records, owner=self.name, verifier=verifier)
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                netting.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.netting import NETTING_ACTION

            entry = self.audit.record(
                NETTING_ACTION,
                resource=netting.id,
                decision="clean" if netting.clean else "disputed",
                details=netting.audit_details(),
            )
            netting.audit_id = getattr(entry, "id", None)
        return netting

    def arbitrate(  # type: ignore[misc]
        self: ContextApp,
        records: list[Any],
        *,
        contract_id: str | None = None,
        sign: bool = True,
        verifier: Any | None = None,
        record_audit: bool = True,
        record_reputation: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Adjudicate a disputed contract from the records its parties submit.

        Resolves a pinpointed disagreement (a
        :class:`~vincio.settlement.NettingDispute`, or two records that do not
        reconcile) into a content-bound :class:`~vincio.settlement.Resolution`:
        deterministically decides which figure stands — a reconciliation hash both
        parties co-signed is upheld, a contradicting unilateral claim is rejected and
        pinpointed — reading only the submitted signed records and asserting nothing
        it cannot recompute. Signs the resolution as this app (the arbiter) and,
        unless ``record_audit`` is off, records it on the audit chain; unless
        ``record_reputation`` is off, closes the reputation loop by debiting the
        party whose claim did not stand. The resolution verifies offline from the
        bytes alone. ``verify_with`` is a deprecated alias for ``verifier`` (since
        7.5, removed in 8.0). Returns it::

            resolution = app.arbitrate([buyer_record, seller_record])
            resolution.verify().valid  # offline-verifiable
            resolution.print_summary()
        """
        from ..settlement import arbitrate

        verifier = _resolve_verifier(verifier, verify_with, "app.arbitrate")
        resolution = arbitrate(
            records,
            contract_id=contract_id,
            arbiter=self.name,
            verifier=verifier,
        )
        if sign:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                resolution.sign(signer, party=self.name)
        if record_audit and self.audit is not None:
            from ..settlement.arbitration import ARBITRATION_ACTION

            entry = self.audit.record(
                ARBITRATION_ACTION,
                resource=resolution.contract_id,
                decision=resolution.status,
                details=resolution.audit_details(),
            )
            resolution.audit_id = getattr(entry, "id", None)
        if record_reputation and self.reputation_ledger is not None:
            for party in resolution.dissenters:
                self.reputation_ledger.record_outcome(
                    party,
                    passed=False,
                    round_id=resolution.contract_id,
                    details={
                        "kind": "arbitration",
                        "resolution_id": resolution.id,
                        "contract_id": resolution.contract_id,
                        "reason": "claim did not stand",
                    },
                )
        return resolution

    def attest_reputation(  # type: ignore[misc]
        self: ContextApp,
        subject: str,
        *,
        book: Any | None = None,
        resolutions: Any | None = None,
        config: Any | None = None,
        horizon_days: float | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Issue a signed, portable attestation of a counterparty's earned standing.

        Reads this app's own settlement book (``book``, else the attached one) and
        any arbitration ``resolutions`` for ``subject`` and summarizes how its
        delivery fared — fulfilled settlements as successes, breaches and arbitration
        dissents as failures — into a content-bound
        :class:`~vincio.settlement.ReputationAttestation`, signed as this app (the
        issuer). A prospective counterparty verifies it from the bytes alone (a
        tampered score or a forged issuer is caught) and folds several issuers'
        attestations into a bounded prior with :meth:`import_reputation`.
        ``horizon_days`` optionally declares a validity window after which an
        as-of-aware import treats the attestation as stale. Unless ``record_audit`` is
        off, the issuance lands on the audit chain. Raises
        :class:`~vincio.core.errors.SettlementError` when this app has no admissible
        history with the subject to attest. Returns the attestation::

            att = app.attest_reputation("vendor")
            att.verify(app.contract_signer).valid  # offline-verifiable
        """
        from ..settlement.attestation import ATTESTATION_ACTION

        source = book if book is not None else self._settlement_book()
        signer = self._resolve_contract_signer(None, sign)
        attestation = source.attest(
            subject,
            resolutions=resolutions,
            config=config,
            sign=sign and signer is not None,
            verifier=None,
            horizon_days=horizon_days,
        )
        if sign and signer is not None and source.signer is None:
            # Sign as the issuer (the book's owner), the identity book.attest would
            # use, so the signature party matches the attestation's issuer and the
            # attestation verifies against its own default require=[issuer].
            attestation.sign(signer, party=attestation.issuer)
        if record_audit and self.audit is not None:
            entry = self.audit.record(
                ATTESTATION_ACTION,
                resource=attestation.subject,
                decision="issued",
                details=attestation.audit_details(),
            )
            attestation.audit_id = getattr(entry, "id", None)
        return attestation

    def revoke_attestation(  # type: ignore[misc]
        self: ContextApp,
        attestation: Any,
        *,
        book: Any | None = None,
        replacement: Any | None = None,
        reason: str = "",
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Withdraw a prior attestation, by its hash, as a signed revocation.

        Builds a content-bound
        :class:`~vincio.settlement.AttestationRevocation` that supersedes or withdraws
        ``attestation`` — which this app (the issuer) must have issued — signed as this
        app and, unless ``record_audit`` is off, recorded on the audit chain.
        ``replacement`` optionally names the attestation that supersedes it. A
        prospective counterparty passes the revocation to :meth:`import_reputation` so
        the withdrawn claim is excluded from the combination, pinpointed, never
        silently honored. Returns the revocation::

            rev = app.revoke_attestation(att, reason="vendor regressed")
            rev.verify(app.contract_signer).valid  # offline-verifiable
        """
        from ..settlement.attestation import REVOCATION_ACTION

        source = book if book is not None else self._settlement_book()
        signer = self._resolve_contract_signer(None, sign)
        revocation = source.revoke(
            attestation,
            replacement=replacement,
            reason=reason,
            sign=sign and signer is not None,
        )
        if sign and signer is not None and source.signer is None:
            # Sign as the issuer (the book's owner), matching how revoke would, so the
            # signature party matches the revocation's issuer and it verifies against
            # its own default require=[issuer].
            revocation.sign(signer, party=revocation.issuer)
        if record_audit and self.audit is not None:
            entry = self.audit.record(
                REVOCATION_ACTION,
                resource=revocation.subject,
                decision="superseded" if revocation.is_supersession else "withdrawn",
                details=revocation.audit_details(),
            )
            revocation.audit_id = getattr(entry, "id", None)
        # Retain it so ``serve_attestations`` can return it to a peer that pulls this
        # app's standing about the subject, superseding any cached copy of the claim.
        self._issued_revocations = [
            r for r in self._issued_revocations if r.content_hash != revocation.content_hash
        ]
        self._issued_revocations.append(revocation)
        return revocation

    def import_reputation(  # type: ignore[misc]
        self: ContextApp,
        attestations: list[Any],
        *,
        subject: str | None = None,
        config: Any | None = None,
        verifier: Any | None = None,
        allow_self: bool = False,
        revocations: list[Any] | None = None,
        as_of: Any | None = None,
        trust: Any | None = None,
        trust_config: Any | None = None,
        weight: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Combine other orgs' attestations into a prior that weights negotiation.

        Verifies each :class:`~vincio.settlement.ReputationAttestation` offline,
        refusing and pinpointing a tampered or forged one, and pools the admissible
        evidence across issuers into a bounded, evidence-weighted
        :class:`~vincio.settlement.PortableReputation` prior under ``config`` — never
        a single self-asserted number (an issuer that vouches for itself is refused).
        Any signed :class:`~vincio.settlement.AttestationRevocation` in ``revocations``
        excludes the attestation its issuer withdrew, and with an ``as_of`` clock a
        stale attestation (past its issuer-declared validity window) decays out of the
        prior rather than anchoring it forever — so the imported standing reflects
        *current* standing, not a frozen snapshot. Pass a ``trust`` source or a
        ``trust_config`` to weigh each issuer's evidence by this app's **own trust in
        that issuer** (rooted in the attached
        :class:`~vincio.optimize.reputation.ReputationLedger`, composed transitively
        over the attestations), so corroboration from a trusted issuer counts for more
        than volume from an unknown one and a Sybil cluster cannot manufacture standing.
        With ``weight`` (the default) the prior is attached so the next negotiation
        weights a counterparty with no local history by what its past counterparties
        attest, under the same bounded ``[floor, 1]`` rule a local reputation uses; the
        attached local ledger stays the source of truth for a counterparty this app
        already knows. ``verify_with`` is a deprecated alias for ``verifier`` (since 7.5,
        removed in 8.0). Returns the prior::

            prior = app.import_reputation([att_a, att_b], revocations=[rev], as_of=now)
            result = app.negotiate("transcribe calls", buyer=..., seller=...)
        """
        from ..settlement.attestation import combine_attestations

        verifier = _resolve_verifier(verifier, verify_with, "app.import_reputation")
        prior = combine_attestations(
            attestations,
            subject=subject,
            config=config,
            verifier=verifier,
            base=self.reputation_ledger,
            allow_self=allow_self,
            revocations=revocations,
            as_of=as_of,
            trust=trust,
            trust_config=trust_config,
        )
        if weight:
            self.imported_reputation = prior
        return prior

    def admit(  # type: ignore[misc]
        self: ContextApp,
        subject: str,
        *,
        reputation: Any | None = None,
        policy: Any | None = None,
        config: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Decide a counterparty's admitted exposure from its earned standing.

        Reads ``subject``'s standing from the same source the negotiation path weights
        by — an imported :class:`~vincio.settlement.PortableReputation` if one is attached
        (:meth:`import_reputation`), else the local
        :class:`~vincio.optimize.reputation.ReputationLedger` — or an explicit
        ``reputation`` source — and maps it to a bounded
        :class:`~vincio.settlement.AdmissionDecision`: a maximum contract value (the
        exposure ceiling), a required escrow fraction, and an SLA-strictness factor. A
        thin or low-trust standing is admitted on *conservative* terms rather than
        refused — discounted exposure, never a hard gate — and as the counterparty
        accrues settled, corroborated history its ceiling **ramps** toward parity, a
        regression walking it back. Pass an :class:`~vincio.settlement.AdmissionPolicy` as
        ``policy`` (or an :class:`~vincio.settlement.AdmissionConfig` as ``config``) to set
        the parity ceiling and ramp. Unless ``record_audit`` is off, the decision lands on
        the hash-chained audit log, binding the standing it read and the ceiling it set.
        The decision verifies offline from the bytes alone — the terms re-derive from the
        bound standing — and folds into the existing negotiation / contracting path
        (:meth:`~vincio.settlement.AdmissionDecision.bound_position` /
        :meth:`~vincio.settlement.AdmissionDecision.apply_to_terms`). Returns it::

            decision = app.admit("vendor")
            buyer = decision.bound_position(buyer_position(max_price_usd=5.0, max_sla_seconds=5.0))
            result = app.negotiate("transcribe", buyer=buyer, seller=..., seller_id="vendor")
        """
        from ..settlement.admission import ADMISSION_ACTION, AdmissionPolicy

        engine = policy if isinstance(policy, AdmissionPolicy) else AdmissionPolicy(config)
        source = reputation if reputation is not None else self._reputation_view()
        decision = engine.admit(subject, reputation=source)
        if record_audit and self.audit is not None:
            entry = self.audit.record(
                ADMISSION_ACTION,
                resource=decision.subject,
                decision="parity" if decision.at_parity else "graduated",
                details=decision.audit_details(),
            )
            decision.audit_id = getattr(entry, "id", None)
        return decision

    def serve_attestations(  # type: ignore[misc]
        self: ContextApp,
        *,
        book: Any | None = None,
        revocations: list[Any] | None = None,
        attestations: list[Any] | None = None,
        config: Any | None = None,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose this app's earned standing as a queryable attestation peer over A2A.

        Returns an :class:`~vincio.a2a.A2AServer` whose Agent Card advertises an
        ``attestation-exchange`` skill; an importer pulls from it with
        :meth:`gather_reputation`. Answering a query for a subject, the peer returns a
        :class:`~vincio.settlement.ReputationBundle` of its **own** signed artifacts —
        the *current* attestation it can issue from its settlement ``book`` (else the
        attached one) and the revocations it has issued (``revocations``, else the ones
        this app has signed via :meth:`revoke_attestation`). Pass an explicit
        ``attestations`` list to serve a fixed signed snapshot instead of re-issuing.
        **Pull, never push:** the peer only ever answers a query, and only with
        artifacts it signed.
        """
        from ..settlement.exchange import attestation_a2a_server

        return attestation_a2a_server(
            book if book is not None else self._settlement_book(),
            revocations=revocations if revocations is not None else self._issued_revocations,
            attestations=attestations,
            config=config,
            name=name,
            url=url,
            description=description,
            token_validator=token_validator,
            audit=self.audit,
        )

    async def agather_reputation(  # type: ignore[misc]
        self: ContextApp,
        subject: str,
        *,
        peers: Any,
        directory: Any | None = None,
        principal: Any | None = None,
        config: Any | None = None,
        verifier: Any | None = None,
        allow_self: bool = False,
        held_attestations: list[Any] | None = None,
        held_revocations: list[Any] | None = None,
        as_of: Any | None = None,
        trust: Any | None = None,
        trust_config: Any | None = None,
        max_peers: int | None = None,
        weight: bool = True,
        record_audit: bool = True,
        verify_with: Any | None = None,
    ) -> Any:
        """Assemble a current prior by pulling signed artifacts from a bounded peer set.

        The gossip analogue of :meth:`import_reputation`: instead of being *handed* a
        bundle, this app **queries** a bounded set of ``peers`` (each an
        :class:`~vincio.settlement.AttestationExchange`, an in-process
        :class:`~vincio.a2a.A2AServer`, or an :class:`~vincio.a2a.A2AClient`) for the
        signed attestations and revocations they hold about ``subject``, governs each
        through ``directory`` (an :class:`~vincio.registry.AgentDirectory`'s
        allow-list, audited), verifies every fetched artifact from the bytes,
        deduplicates by content hash, and folds them — with any ``held_attestations`` /
        ``held_revocations`` already on hand — into a bounded, evidence-weighted
        :class:`~vincio.settlement.PortableReputation` under the same freshness,
        revocation, and ``[floor, 1]`` discipline :meth:`import_reputation` uses. Pass a
        ``trust`` source or a ``trust_config`` to weigh each gathered issuer's evidence
        by this app's own trust in it (rooted in the attached ledger, composed
        transitively), so a cluster of unknown peers cannot out-evidence a few it
        trusts. Every peer visited and artifact fetched lands on the audit chain. With
        ``weight`` (the default) the assembled prior is attached so the next negotiation
        weights an unknown counterparty by what its peers attest. ``verify_with`` is a
        deprecated alias for ``verifier`` (since 7.5, removed in 8.0). Returns a
        :class:`~vincio.settlement.GatheredReputation`::

            result = await app.agather_reputation("vendor", peers={"acme": acme_server})
            result.weight("vendor")  # drops into the negotiation path
        """
        from ..settlement.exchange import gather_reputation

        verifier = _resolve_verifier(verifier, verify_with, "app.agather_reputation")
        result = await gather_reputation(
            subject,
            peers=peers,
            directory=directory,
            principal=principal,
            config=config,
            verifier=verifier,
            base=self.reputation_ledger,
            allow_self=allow_self,
            held_attestations=held_attestations,
            held_revocations=held_revocations,
            as_of=as_of,
            trust=trust,
            trust_config=trust_config,
            max_peers=max_peers,
            audit=self.audit,
            record_audit=record_audit,
        )
        if weight:
            self.imported_reputation = result.reputation
        return result

    def gather_reputation(self: ContextApp, subject: str, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Synchronous wrapper around :meth:`agather_reputation`."""
        return run_sync(self.agather_reputation(subject, **kwargs))

    def cross_org_engagement(  # type: ignore[misc]
        self: ContextApp,
        *,
        buyer: str = "",
        seller: str = "",
        scope: str = "",
        coordinator: str | None = None,
    ) -> Any:
        """Thread the whole cross-org settlement & credit fabric behind one call-path.

        Returns a :class:`~vincio.settlement.CrossOrgEngagement` — the capstone facade
        that composes the entire pipeline (discover → negotiate → contract →
        choreograph delivery → meter → settle → net → arbitrate → attest and port
        reputation → admit → post and pool collateral under a rehypothecation guard →
        prove reserves, solvency, completeness, non-equivocation, and history → and, on
        default, resolve the insolvency by seniority waterfall with close-out set-off)
        into one governed, audited, hash-linked narrative. Each lifecycle method
        delegates to the *same* entry point on this app a caller would use directly, so
        the primitives stay unchanged and usable on their own; the facade only captures
        and **narrates** them.

        :meth:`~vincio.settlement.CrossOrgEngagement.seal` mints the content-bound,
        signed :class:`~vincio.settlement.EngagementNarrative`, and
        :meth:`~vincio.settlement.CrossOrgEngagement.verify` proves the whole chain —
        and every captured artifact — verifies offline, so a tamper introduced anywhere
        is caught::

            eng = app.cross_org_engagement(buyer="acme", seller="vendor", scope="transcribe")
            contract = eng.negotiate(buyer=buyer_pos, seller=seller_pos)
            eng.choreograph(saga, participants=parts)
            eng.settle_saga(contracts={contract.id: contract})
            narrative = eng.seal()
            narrative.verify(app.contract_signer).valid  # offline-verifiable
        """
        from ..settlement.engagement import CrossOrgEngagement

        return CrossOrgEngagement(
            self, buyer=buyer, seller=seller, scope=scope, coordinator=coordinator or self.name
        )
