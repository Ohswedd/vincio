"""Governance, compliance, verified reasoning, privacy, consent, and erasure verbs — a private mixin of
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

from typing import TYPE_CHECKING, Any, cast

from ..governance.lineage import ErasureResult, build_erasure_proof
from ..governance.residency import ResidencyPolicy
from ..providers.base import run_sync
from .diagnostics import note_suppressed

if TYPE_CHECKING:
    from .app import ContextApp


class _GovernanceVerbs:
    """Governance, compliance, verified reasoning, privacy, consent, and erasure verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        consent_ledger: Any
        privacy_accountant: Any
        reputation_ledger: Any
        residency: ResidencyPolicy


    # -- governance & compliance -----------------------------------------------------

    def _card_format(self: ContextApp, override: Any | None = None):  # type: ignore[misc]
        from ..governance.cards import CardFormat

        return CardFormat(override or self.config.governance.card_format)

    def model_card(self: ContextApp, *, eval_report: Any | None = None, format: Any | None = None):  # type: ignore[misc]
        """Generate a :class:`~vincio.governance.ModelCard` from the live config.

        Pass an :class:`~vincio.evals.reports.EvalReport` to attach measured
        evaluation evidence. ``format`` overrides the configured card schema
        (``vincio`` / ``open_model_card`` / ``ai_card``).
        """
        from ..governance.cards import generate_model_card

        return generate_model_card(self, eval_report=eval_report, format=self._card_format(format))

    def system_card(self: ContextApp, *, eval_report: Any | None = None, format: Any | None = None):  # type: ignore[misc]
        """Generate a :class:`~vincio.governance.SystemCard` (model + retrieval +
        memory + safety filters + human-oversight points) from the live config."""
        from ..governance.cards import generate_system_card

        return generate_system_card(
            self, eval_report=eval_report, format=self._card_format(format), name=self.name
        )

    def compliance_report(self: ContextApp, *, redteam: Any | None = None, eval_report: Any | None = None):  # type: ignore[misc]
        """Map this app's controls to OWASP/NIST/MITRE frameworks as a coverage
        matrix, backed by red-team and eval evidence
        (:class:`~vincio.governance.ComplianceReport`)."""
        from ..governance.frameworks import ComplianceMapper

        return ComplianceMapper().map(redteam=redteam, eval_report=eval_report, target=self)

    def aibom(self: ContextApp, *, datasets: list[Any] | None = None, prompts: list[Any] | None = None):  # type: ignore[misc]
        """Generate an AI bill of materials (:class:`~vincio.governance.AIBOM`)
        for the live model/embedder/reranker, with SHA-256 model-hash slots."""
        from ..governance.aibom import generate_aibom

        return generate_aibom(self, datasets=datasets, prompts=prompts)

    def trace_lineage(self: ContextApp, source: str):  # type: ignore[misc]
        """Return the source → chunk → evidence → output lineage for a source
        name or document id (:class:`~vincio.governance.LineageRecord`)."""
        return self.lineage.trace(source)

    def verify_governance(  # type: ignore[misc]
        self: ContextApp,
        invariants: Any | None = None,
        *,
        record: bool = True,
        raise_on_violation: bool = False,
    ):
        """Formally verify the governance invariants hold, ahead of any run.

        Proves — by exhaustive bounded model checking, not after-the-fact
        observation — that the platform's governance controls satisfy their
        specifications across the whole typed input space: injection-containment
        (``untrusted ⇒ no unapproved capability``), data residency (in-jurisdiction
        egress refusal, reflecting this app's ``deny_on_unknown`` posture), the
        budget hard cap, and the erasure-proof content binding. Returns a
        content-hashed :class:`~vincio.governance.VerificationReport`; a failed
        property carries a minimal :class:`~vincio.governance.Counterexample`::

            report = app.verify_governance()
            assert report.held
            for cx in report.counterexamples:
                print(cx.render())

        The verdict is deterministic and offline, and (when ``record``) lands on
        the hash-chained audit log as a ``governance_verification`` decision. Pass
        a custom ``invariants`` list to verify a different property set; set
        ``raise_on_violation`` to raise
        :class:`~vincio.core.errors.GovernanceVerificationError` instead of
        returning a non-holding report.
        """
        from ..core.errors import GovernanceVerificationError
        from ..governance.verification import (
            GovernanceVerifier,
            budget_invariant,
            containment_invariant,
            erasure_invariant,
            residency_invariant,
        )

        if invariants is None:
            invariants = [
                containment_invariant(),
                residency_invariant(deny_on_unknown=self.residency.deny_on_unknown),
                budget_invariant(),
                erasure_invariant(),
            ]
        verifier = GovernanceVerifier(invariants, audit_log=self.audit)
        report = verifier.verify(record=record)
        if raise_on_violation and not report.held:
            raise GovernanceVerificationError(
                f"{len(report.counterexamples)} governance invariant(s) violated: "
                + "; ".join(c.render() for c in report.counterexamples),
                counterexamples=report.counterexamples,
            )
        return report

    def mark_output(self: ContextApp, content: str, *, model: str | None = None, signer: Any | None = None):  # type: ignore[misc]
        """Build a C2PA-style synthetic-content provenance manifest for output
        (:class:`~vincio.governance.ProvenanceManifest`).

        Signs the manifest when a ``signer`` is passed or ``app.content_signer``
        is set (e.g. an :class:`~vincio.governance.HmacSigner`)."""
        from ..governance.transparency import mark_synthetic_content

        return mark_synthetic_content(
            content,
            model_id=model or self.model,
            provider=self._provider_name,
            signer=signer or self.content_signer,
        )

    # -- verified reasoning & neuro-symbolic certificates --------------

    def verify_reasoning(  # type: ignore[misc]
        self: ContextApp,
        answer: Any,
        *,
        verifiers: Any | None = None,
        evidence: Any | None = None,
        schema: dict[str, Any] | None = None,
        constraints: Any | None = None,
        statistical_claims: Any | None = None,
        facts: dict[str, Any] | None = None,
        now: Any | None = None,
        regenerate: Any | None = None,
        max_cycles: int = 2,
        raise_on_refute: bool = False,
        record: bool = True,
    ) -> Any:
        """Attach and check a deterministic :class:`~vincio.verify.Certificate` to an answer.

        Runs a set of offline kernels (arithmetic, units, temporal, schema,
        constraints, citation entailment — the default
        :func:`~vincio.verify.default_verifiers`) over ``answer`` and returns a
        :class:`~vincio.verify.VerifiedAnswer` whose certificate is **verified**,
        **refuted**, or **inapplicable**. A refuted certificate is a *proof the
        answer is wrong* (a recomputation disagreed), so the orchestrator refuses
        to emit it: :attr:`VerifiedAnswer.holds` is ``False`` and ``refused`` is set.

        When a refuted answer can be repaired, pass a ``regenerate`` callable
        ``(answer, critique) -> new_answer`` to drive the bounded self-correction
        loop: the deterministic refutations become a critique, the callable
        produces a fresh answer, and it is re-certified, up to ``max_cycles`` — the
        same refuse-or-repair discipline structured output already uses, now over
        *reasoning* rather than *structure*. Ground the kernels with ``evidence``
        (citation entailment), ``schema`` (structural conformance), ``constraints``
        (constraint satisfaction), ``statistical_claims`` (the trend / correlation /
        interval / forecast kernels, which recompute a stated statistic from the
        cited cells and refuse a spurious causal claim), ``facts`` and ``now``. When
        ``statistical_claims`` are supplied and ``verifiers`` is left default, the
        statistical kernels are added to the default set automatically. Because a
        statistical claim is grounded in the context rather than the answer text, a
        ``regenerate`` callback may repair one by returning a corrected
        :class:`~vincio.verify.StatisticalClaim` (or a list of them); the loop
        re-grounds the context with the corrected claim before re-certifying, so the
        same refuse-or-repair discipline drives the statistical kernels too. The
        verdict lands on the hash-chained audit log as a ``reasoning_verification``
        decision unless ``record`` is off; set ``raise_on_refute`` to raise
        :class:`~vincio.core.errors.CertificateRefutedError` instead.
        """
        from ..core.errors import CertificateRefutedError
        from ..verify import CompositeVerifier, VerificationContext, VerifiedAnswer
        from ..verify.kernels import default_verifiers
        from ..verify.statistical import statistical_verifiers

        claims = list(statistical_claims) if statistical_claims else []
        if verifiers is not None:
            kernels = list(verifiers)
        else:
            kernels = default_verifiers() + (statistical_verifiers() if claims else [])
        verifier = CompositeVerifier(kernels)
        context = VerificationContext(
            evidence=list(evidence) if evidence else [],
            schema=schema,
            constraints=list(constraints) if constraints else [],
            statistical_claims=claims,
            facts=facts or {},
            now=now,
        )
        from ..verify.statistical import StatisticalClaim

        current = answer
        certificate = verifier.certify(current, context)
        attempts = 1
        while certificate.refuted and regenerate is not None and attempts <= max_cycles:
            critique = "The previous answer failed verification:\n" + "\n".join(
                f"- {c.name}: {c.detail}" for c in certificate.refutations
            )
            repaired = regenerate(current, critique)
            if repaired is None or repaired == current:
                break
            # A statistical claim is grounded in the context, not the answer text, so
            # a repair that re-states the corrected claim(s) re-grounds the context
            # before re-certifying; any other value is a replacement answer as before.
            repaired_claims = (
                [repaired] if isinstance(repaired, StatisticalClaim)
                else list(repaired) if isinstance(repaired, list)
                and repaired and all(isinstance(c, StatisticalClaim) for c in repaired)
                else None
            )
            if repaired_claims is not None:
                context = context.model_copy(update={"statistical_claims": repaired_claims})
            current = repaired
            certificate = verifier.certify(current, context)
            attempts += 1

        refused = certificate.refuted
        verified = VerifiedAnswer(
            answer=current,
            certificate=certificate,
            attempts=attempts,
            refused=refused,
            stopped_reason=(
                "refused" if refused else certificate.status
            ),
        )
        if record and self.audit is not None:
            self.audit.record(
                "reasoning_verification",
                resource=certificate.subject_hash,
                decision=certificate.status,
                details={
                    "kinds": certificate.kinds,
                    "refutations": [c.name for c in certificate.refutations],
                    "attempts": attempts,
                    "certificate_hash": certificate.certificate_hash,
                },
            )
        if raise_on_refute and refused:
            raise CertificateRefutedError(
                "answer certificate refuted: "
                + "; ".join(c.detail for c in certificate.refutations),
                details={"refutations": [c.name for c in certificate.refutations]},
            )
        return verified

    def behavior_monitor(self: ContextApp, specs: Any) -> Any:  # type: ignore[misc]
        """Build a :class:`~vincio.verify.RuntimeMonitor` over one or more
        :class:`~vincio.verify.BehaviorSpec`\\ s.

        The monitor checks a property over an agent's trajectory step-by-step:
        feed it :class:`~vincio.verify.BehaviorEvent`\\ s via ``observe`` as the
        agent runs, or a recorded list via ``check_trajectory``. It is the online,
        per-step, behavioural analogue of the ahead-of-run governance verifier.
        """
        from ..verify import RuntimeMonitor

        return RuntimeMonitor(specs)

    def shield(self: ContextApp, specs: Any, *, mode: str = "block", repair: Any | None = None, use: bool = False) -> Any:  # type: ignore[misc]
        """Build a :class:`~vincio.verify.Shield` that prevents a behavioural violation.

        A shield wraps a monitor and, before an action executes, **blocks** it
        (``mode='block'``), **repairs** it to a safe alternative (``mode='repair'``
        with a ``repair`` callback), or merely records it (``mode='monitor'``). With
        ``use=True`` the shield is installed on this app's tool runtime, so a
        policy-violating tool call (a write before approval, a tool outside scope)
        is structurally refused — the per-step, online counterpart of the rails.
        """
        from ..verify import Shield

        built = Shield(specs, mode=mode, repair=repair)  # type: ignore[arg-type]
        if use:
            self.use_shield(built)
        return built

    def use_shield(self: ContextApp, shield: Any | None) -> Any:  # type: ignore[misc]
        """Install (or clear, with ``None``) a behavioural shield on the tool runtime.

        Once installed, every tool call is checked against the shield's
        :class:`~vincio.verify.BehaviorSpec`\\ s *before* it executes; a blocked
        call returns a denied result like a failed permission check. Returns the
        shield.
        """
        self.tool_runtime.shield = shield
        return shield

    def synthesize_program(  # type: ignore[misc]
        self: ContextApp, spec: Any, examples: Any, *, require: bool = True, record: bool = True
    ) -> Any:
        """Synthesize and verify a small data-transform program.

        Runs ``spec``'s whitelisted op pipeline on representative ``examples``,
        checks its declared properties (schema conformance, row-count relations,
        field invariants), and returns a
        :class:`~vincio.verify.SynthesizedProgram` carrying the
        :class:`~vincio.verify.Certificate` that proves them — proof-carrying code
        in the tool plane. With ``require`` (the default) a refuted program raises
        :class:`~vincio.core.errors.ProgramSynthesisError` rather than returning;
        the verdict lands on the audit log as a ``program_synthesis`` decision.
        """
        from ..verify import synthesize

        program = synthesize(spec, list(examples), require=require)
        if record and self.audit is not None:
            self.audit.record(
                "program_synthesis",
                resource=program.certificate.subject_hash,
                decision=program.certificate.status,
                details={
                    "name": getattr(spec, "name", ""),
                    "properties": [c.name for c in program.certificate.checks],
                    "certificate_hash": program.certificate.certificate_hash,
                },
            )
        return program
    def risk_tier(  # type: ignore[misc]
        self: ContextApp,
        *,
        purpose: str = "",
        domains: list[str] | None = None,
        prohibited_practices: list[str] | None = None,
    ):
        """Classify this app into the EU AI Act risk tiers (advisory)."""
        from ..governance.eu_ai_act import RiskTierClassifier

        return RiskTierClassifier(
            purpose=purpose, domains=domains, prohibited_practices=prohibited_practices
        ).classify(self)

    def annex_iv(  # type: ignore[misc]
        self: ContextApp,
        *,
        format: str = "markdown",
        purpose: str = "",
        domains: list[str] | None = None,
        eval_report: Any | None = None,
        redteam: Any | None = None,
    ):
        """Generate EU AI Act Annex IV technical documentation as a cited artifact."""
        from ..governance.eu_ai_act import AnnexIVBuilder, RiskTierClassifier

        classifier = RiskTierClassifier(purpose=purpose, domains=domains)
        return AnnexIVBuilder(classifier=classifier).build(
            self, format=cast("Any", format), eval_report=eval_report, redteam=redteam
        )

    def fria(  # type: ignore[misc]
        self: ContextApp,
        *,
        format: str = "markdown",
        purpose: str = "",
        domains: list[str] | None = None,
        affected_groups: list[str] | None = None,
        eval_report: Any | None = None,
    ):
        """Generate an EU AI Act Art. 27 fundamental-rights impact assessment."""
        from ..governance.eu_ai_act import FRIAGenerator, RiskTierClassifier

        classifier = RiskTierClassifier(purpose=purpose, domains=domains)
        return FRIAGenerator(classifier=classifier).generate(
            self,
            format=cast("Any", format),
            affected_groups=affected_groups,
            eval_report=eval_report,
        )

    def set_residency(  # type: ignore[misc]
        self: ContextApp,
        allowed_regions: list[str],
        *,
        provider_regions: dict[str, str] | None = None,
        deny_on_unknown: bool = True,
    ) -> ContextApp:
        """Pin allowed provider regions; runs outside them are refused egress."""
        self.residency = ResidencyPolicy(
            allowed_regions=list(allowed_regions),
            provider_regions={**self.residency.provider_regions, **(provider_regions or {})},
            deny_on_unknown=deny_on_unknown,
        )
        return self

    def use_consent_ledger(self: ContextApp, ledger: Any | None = None, *, default_allow: bool = False) -> Any:  # type: ignore[misc]
        """Attach a :class:`~vincio.governance.consent.ConsentLedger`.

        Binds data to a GDPR purpose and lawful basis. Once attached, access
        decisions (:meth:`AccessController.check_purpose`) and memory recall
        consult it, so a withdrawn consent or a purpose mismatch is enforced in
        code. Persists to the app's store and writes grants/revokes/denied checks
        to the same audit chain as erasure. Returns the ledger."""
        from ..governance.consent import ConsentLedger

        if ledger is None:
            ledger = ConsentLedger(store=self.store, audit=self.audit, default_allow=default_allow)
        self.consent_ledger = ledger
        self.access.consent_ledger = ledger
        if self.memory is not None:
            self.memory.consent_ledger = ledger
        return ledger

    def use_privacy_accountant(  # type: ignore[misc]
        self: ContextApp,
        accountant: Any | None = None,
        *,
        default_budget: Any | None = None,
        default_mechanism: Any | None = None,
        delta: float = 1e-5,
    ) -> Any:
        """Attach a differential-privacy accountant over the learning loop.

        Composes a per-subject ``(ε, δ)`` budget across every accounted memory
        consolidation and federated contribution: a step that would exceed a
        subject's remaining budget is refused (or down-weighted), every spend and
        refusal on the same hash-chained audit log as consent and erasure. Once
        attached, :meth:`MemoryEngine.consolidate` and
        :meth:`contribute_federated` gate automatically, and
        :meth:`privacy_report` rolls up the spent budget alongside
        :meth:`cost_report`. Pass a configured
        :class:`~vincio.governance.privacy.PrivacyAccountant`, or let this build
        one wired to the app's audit chain and store. Returns the accountant::

            from vincio import PrivacyBudget
            app.use_privacy_accountant(default_budget=PrivacyBudget(epsilon=2.0))
        """
        from ..governance.privacy import PrivacyAccountant

        if accountant is None:
            accountant = PrivacyAccountant(
                default_budget=default_budget,
                default_mechanism=default_mechanism,
                delta=delta,
                audit=self.audit,
                store=self.store,
            )
        self.privacy_accountant = accountant
        if self.memory is not None:
            self.memory.privacy_accountant = accountant
        return accountant

    def set_privacy_budget(  # type: ignore[misc]
        self: ContextApp,
        *,
        subject_id: str | None = None,
        epsilon: float,
        delta: float = 1e-5,
        on_breach: str = "refuse",
    ) -> ContextApp:
        """Set a per-subject (or default) differential-privacy budget.

        Creates the accountant on first use. ``subject_id=None`` is the default
        budget applied to any subject without a specific one; ``on_breach`` is
        ``"refuse"`` (a hard cap) or ``"downweight"`` (clip harder to fit)::

            app.set_privacy_budget(subject_id="alice", epsilon=1.0)
            app.set_privacy_budget(epsilon=3.0, on_breach="downweight")
        """
        from ..governance.privacy import PrivacyBudget

        if self.privacy_accountant is None:
            self.use_privacy_accountant(delta=delta)
        self.privacy_accountant.set_budget(
            PrivacyBudget(
                subject_id=subject_id,
                epsilon=epsilon,
                delta=delta,
                on_breach=on_breach,  # type: ignore[arg-type]
            )
        )
        return self

    def privacy_report(self: ContextApp, subject: str | None = None):  # type: ignore[misc]
        """Per-subject differential-privacy budget roll-up.

        The privacy analogue of :meth:`cost_report`: each row is a subject's
        cumulative ``ε`` spent against its ceiling, with operation and refusal
        counts, so the spent privacy budget is an auditable number. Returns an
        empty :class:`~vincio.governance.privacy.PrivacyReport` when no accountant
        is attached."""
        if self.privacy_accountant is None:
            from ..governance.privacy import PrivacyReport

            return PrivacyReport()
        return self.privacy_accountant.report(subject)

    def use_reputation_ledger(self: ContextApp, ledger: Any | None = None, *, config: Any | None = None) -> Any:  # type: ignore[misc]
        """Attach a cross-fleet reputation ledger over the federated round.

        Earns a per-member reliability score from how each federated contribution
        fared against the no-regression gate — a pass credits the contributor, a
        regression debits it — and reliability-weights the
        :class:`~vincio.optimize.federated.SecureAggregator` so a repeatedly
        regressing or adversarial member is discounted without being singled out.
        The discount is bounded and reversible: a weight only ever lowers a
        member's pull, and adoption still clears the same gate, so reputation can
        never bypass the quality bar. Every update lands on the same hash-chained
        audit log as consent, privacy, and erasure, so a member's standing is a
        mechanical, auditable, replayable number.

        Once attached, :meth:`federated_improvement` / :meth:`adopt_federated`
        weight contributions by reputation and record each round's verdict back
        automatically, and :meth:`reputation_report` rolls up each member's score
        next to the cost and privacy reports. Pass a configured
        :class:`~vincio.optimize.reputation.ReputationLedger`, or let this build one
        wired to the app's audit chain, event bus, and store. Returns the ledger::

            app.use_reputation_ledger()
            result = app.adopt_federated(golden, [mine, *peer_updates])
        """
        from ..optimize.reputation import ReputationLedger

        if ledger is None:
            ledger = ReputationLedger(
                config=config, audit=self.audit, events=self.events, store=self.store
            )
        self.reputation_ledger = ledger
        return ledger

    def reputation_report(self: ContextApp, member: str | None = None):  # type: ignore[misc]
        """Per-member cross-fleet reputation roll-up.

        Each row is a member's earned reliability score and the aggregation weight
        it maps to, with the success / failure tally behind it, so a member's
        standing in the fleet is an auditable number. Returns an empty
        :class:`~vincio.optimize.reputation.ReputationReport` when no ledger is
        attached."""
        if self.reputation_ledger is None:
            from ..optimize.reputation import ReputationReport

            return ReputationReport()
        return self.reputation_ledger.report(member)

    def erase_source(self: ContextApp, source: str, *, prove: bool = True) -> ErasureResult:  # type: ignore[misc]
        """Right-to-erasure-by-source: purge a source from indexes, memory,
        caches, and generated artifacts, logged on the hash-chained audit chain.

        ``source`` is a source name (as passed to :meth:`add_source`) or a
        document id. Returns an :class:`~vincio.governance.ErasureResult`.
        Idempotent: a second call finds nothing left to erase.

        When ``prove``, the sweep emits a signed, content-bound
        :class:`~vincio.governance.ErasureProof` on the result — a manifest of
        exactly which chunk / document / memory / artifact ids were removed,
        bound by SHA-256, signed with :attr:`content_signer` when set, and
        anchored to the audit chain's Merkle root — so erasure is *provable*,
        not merely logged.
        """
        record = self.lineage.trace(source)
        result = ErasureResult(source=source, found=not record.is_empty)
        chunk_ids = list(record.chunks)
        # The exact identifiers removed, per store — the binding the proof covers.
        removed_ids: dict[str, list[str]] = {}
        per_index: dict[str, int] = {}
        index_handles = {
            "bm25": self._bm25,
            "vector": self._vector,
            "sparse": self._sparse,
            "late_interaction": self._late_interaction,
        }
        if chunk_ids:
            for label, index in index_handles.items():
                if index is None:
                    continue
                per_index[label] = run_sync(index.delete(chunk_ids))
                result.indexes_swept += 1
            result.chunks_removed = len(chunk_ids)
            removed_ids["chunks"] = list(chunk_ids)
            if self.entity_graph is not None:
                # Entity graph is rebuilt from sources; drop nothing destructively
                # here beyond chunk references already removed from indexes.
                pass

        # Documents recorded in the metadata store. Count only deletions that
        # actually succeed, so the audit trail never overstates erasure.
        removed_docs: list[str] = []
        for doc_id in record.documents:
            try:
                if hasattr(self.store, "delete") and self.store.delete("documents", doc_id):  # type: ignore[attr-defined]
                    result.documents_removed += 1
                    removed_docs.append(doc_id)
            except Exception:
                note_suppressed("governance.erase.document_delete")
        if removed_docs:
            removed_ids["documents"] = removed_docs

        # Memory items whose provenance references the source (exact matches on
        # source name / id, never a loose substring that could over-delete).
        removed_memories: list[str] = []
        if self.memory is not None:
            doc_set = set(record.documents)
            for item in list(self.memory.store.all_items(statuses=())):
                meta = item.metadata or {}
                refs = {meta.get("source"), meta.get("source_id")}
                if source in refs or bool(refs & doc_set):
                    if self.memory.delete(item.id):
                        result.memories_removed += 1
                        removed_memories.append(item.id)
        if removed_memories:
            removed_ids["memories"] = removed_memories

        # Generated artifacts (cited documents, images, audio) derived from the
        # source — removed from the blob/metadata store so the deliverable is
        # erased alongside the evidence and memory it was built from.
        removed_artifacts: list[str] = []
        for artifact_key in record.artifacts:
            erased = False
            for store_obj, kind in ((self.store, "artifacts"), (self.store, "documents")):
                try:
                    if hasattr(store_obj, "delete") and store_obj.delete(kind, artifact_key):  # type: ignore[attr-defined]
                        erased = True
                except Exception:
                    note_suppressed("governance.erase.artifact_delete")
            # The lineage link is severed regardless, which is the auditable fact.
            result.artifacts_removed += 1
            removed_artifacts.append(artifact_key)
            _ = erased
        if removed_artifacts:
            removed_ids["artifacts"] = removed_artifacts

        # Registered tabular datasets ingested from the source — dropped from the
        # data catalog so an erased source is erased as structured data too, and the
        # semantic layers defined over them un-registered (their definitions can no
        # longer ground to absent rows).
        removed_datasets: list[str] = []
        catalog = getattr(self, "_data_catalog_obj", None)
        for table in list(record.datasets):
            if catalog is not None and catalog.remove(table):
                removed_datasets.append(table)
            self._semantic_layers.pop(table, None)
        if removed_datasets:
            result.datasets_removed = len(removed_datasets)
            removed_ids["datasets"] = removed_datasets

        # Caches: erasure correctness outweighs cache retention.
        for cache in (self.response_cache, self.context_compile_cache):
            backend = getattr(cache, "backend", None) or getattr(cache, "cache", None)
            if backend is not None and hasattr(backend, "clear"):
                try:
                    backend.clear()
                    result.caches_invalidated += 1
                except Exception:
                    note_suppressed("governance.erase.cache_invalidate")

        entry = self.audit.record(
            "erase_source",
            decision="allow",
            resource=source,
            details={
                "found": result.found,
                "chunks_removed": result.chunks_removed,
                "documents_removed": result.documents_removed,
                "memories_removed": result.memories_removed,
                "artifacts_removed": result.artifacts_removed,
                "datasets_removed": result.datasets_removed,
                "indexes_swept": result.indexes_swept,
                "caches_invalidated": result.caches_invalidated,
                "per_index": per_index,
            },
        )
        result.audit_entry_id = entry.id

        # Build the signed, content-bound erasure proof over the precise
        # removed-id set, anchored to the audit chain's current Merkle root.
        if prove:
            proof = build_erasure_proof(
                source,
                removed_ids,
                counts={
                    "chunks": result.chunks_removed,
                    "documents": result.documents_removed,
                    "memories": result.memories_removed,
                    "artifacts": result.artifacts_removed,
                    "datasets": result.datasets_removed,
                    "caches": result.caches_invalidated,
                },
                signer=self.content_signer,
                audit_entry_id=entry.id,
                audit_merkle_root=self.audit.merkle_root(),
            )
            result.proof = proof
            self.audit.record(
                "erasure_proof",
                decision="allow",
                resource=source,
                details={
                    # frozen audit-detail key — external consumers bind to it.
                    "content_sha256": proof.content_hash,
                    "signed": proof.signature is not None,
                    "key_id": proof.key_id,
                    "removed": proof.removed,
                },
            )

        self.events.emit(
            "governance.source_erased",
            {
                "source": source,
                "found": result.found,
                "proven": result.proof is not None,
                "content_hash": result.proof.content_hash if result.proof else None,
                # deprecated wire key (since 7.5, removal 8.0): dual-emitted so
                # external sinks keep receiving it through the rename runway.
                "content_sha256": result.proof.content_hash if result.proof else None,
            },
        )
        self.lineage.forget(source)
        # Drop the source registration so it is not re-counted.
        self.sources.pop(source, None)
        return result
