"""Negotiation, choreography, settlement-core, identity, and custody verbs — a private mixin of
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
from .errors import (
    ConfigError,
)

if TYPE_CHECKING:
    from .app import ContextApp


class _CrossOrgVerbs:
    """Negotiation, choreography, settlement-core, identity, and custody verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        _contract_signer: Any
        _identity: Any
        content_signer: Any
        settlement_book: Any


    # -- agent negotiation & contracting --------------------------------------

    def _negotiation_party(self: ContextApp, spec: Any, role: str, member_id: str) -> Any:  # type: ignore[misc]
        """Coerce a position or a party into a negotiating :class:`Party`."""
        from ..negotiation import LocalParty, NegotiationPosition
        from ..negotiation.engine import Party

        if isinstance(spec, NegotiationPosition):
            if spec.role != role:
                raise ConfigError(
                    f"negotiate {role}= expects a {role} position; got role={spec.role!r}"
                )
            return LocalParty(member_id, spec, reputation=self._reputation_view())
        if isinstance(spec, Party):
            return spec
        raise ConfigError(f"negotiate {role}= must be a NegotiationPosition or a negotiation Party")

    def _reputation_view(self: ContextApp) -> Any:  # type: ignore[misc]
        """The reputation an offer is weighted by: imported prior over local ledger.

        Returns the imported :class:`~vincio.settlement.PortableReputation` when one
        is attached (it already falls back to the local ledger for a counterparty
        this app knows), else the local
        :class:`~vincio.optimize.reputation.ReputationLedger`, else ``None`` (offers
        are weighted at parity). So a negotiation against a brand-new counterparty is
        weighted by what its past counterparties attest, while one this app has lived
        through keeps its own earned standing.
        """
        if self.imported_reputation is not None:
            return self.imported_reputation
        return self.reputation_ledger

    def _resolve_contract_signer(self: ContextApp, signer: Any | None, sign: bool) -> Any | None:  # type: ignore[misc]
        """Pick the signer for a contract: explicit → audit signer → per-app key."""
        if signer is not None:
            return signer
        if not sign:
            return None
        audit_signer = getattr(self.audit, "signer", None)
        if audit_signer is not None:
            return audit_signer
        if self._contract_signer is None:
            from ..core.utils import new_id
            from ..security.audit import HMACSigner

            self._contract_signer = HMACSigner(
                new_id("contract-key"), key_id=f"{self.name}-contracts"
            )
        return self._contract_signer

    async def anegotiate(  # type: ignore[misc]
        self: ContextApp,
        scope: str,
        *,
        buyer: Any,
        seller: Any,
        budget: Any | None = None,
        signer: Any | None = None,
        sign: bool = True,
        buyer_id: str = "buyer",
        seller_id: str = "seller",
    ) -> Any:
        """Run a bounded buyer/seller negotiation; return a :class:`NegotiationResult`.

        ``buyer`` / ``seller`` are each a
        :class:`~vincio.negotiation.NegotiationPosition` (run as a local,
        deterministic party) or an already-built
        :class:`~vincio.negotiation.Party` — e.g. an
        :class:`~vincio.negotiation.A2ANegotiator` reaching a counterparty over the
        A2A fabric. The bargain is bounded by ``budget`` (a
        :class:`~vincio.negotiation.NegotiationBudget` or a kwargs dict);
        termination is guaranteed, returning a partial result on a deadline. On
        agreement a :class:`~vincio.negotiation.Contract` is minted and signed by
        both parties (with ``signer``, else the audit-chain signer, else a per-app
        key), and the outcome is recorded on the hash-chained audit log. When a
        reputation ledger is attached (:meth:`use_reputation_ledger`) it weights
        each local party's view of the counterparty's offers — a regressing agent
        is discounted without being singled out::

            from vincio.negotiation import buyer_position, seller_position

            result = app.negotiate(
                "transcribe 1k support calls",
                buyer=buyer_position(max_price_usd=0.10, max_sla_seconds=5.0),
                seller=seller_position(min_price_usd=0.04, ideal_price_usd=0.12),
            )
            if result.agreed:
                result.contract.verify(app.contract_signer)  # offline-verifiable
        """
        from ..negotiation import Negotiation, NegotiationBudget

        nbudget = (
            budget if isinstance(budget, NegotiationBudget) else NegotiationBudget(**(budget or {}))
        )
        buyer_party = self._negotiation_party(buyer, "buyer", buyer_id)
        seller_party = self._negotiation_party(seller, "seller", seller_id)
        contract_signer = self._resolve_contract_signer(signer, sign)
        negotiation = Negotiation(
            buyer_party,
            seller_party,
            budget=nbudget,
            signer=contract_signer,
            audit=self.audit,
            events=self.events,
        )
        return await negotiation.run(scope)

    def negotiate(self: ContextApp, scope: str, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Synchronous wrapper around :meth:`anegotiate`."""
        return run_sync(self.anegotiate(scope, **kwargs))

    @property
    def contract_signer(self: ContextApp) -> Any | None:  # type: ignore[misc]
        """The signer this app uses to sign/verify contracts (may build one)."""
        return self._resolve_contract_signer(None, True)

    def serve_negotiation(  # type: ignore[misc]
        self: ContextApp,
        party: Any,
        *,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose a local negotiating :class:`~vincio.negotiation.Party` over A2A.

        Returns an :class:`~vincio.a2a.A2AServer` whose Agent Card advertises a
        ``negotiate`` skill; a remote engine reaches it with an
        :class:`~vincio.negotiation.A2ANegotiator`. Each offer exchange is a
        bounded, audited A2A task on this app's hash-chained log.
        """
        from ..negotiation.fabric import negotiation_a2a_server

        return negotiation_a2a_server(
            party,
            name=name,
            url=url,
            description=description,
            token_validator=token_validator,
            audit=self.audit,
        )

    def enforce_contract(  # type: ignore[misc]
        self: ContextApp,
        contract: Any,
        *,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        quality: float | None = None,
        raise_on_breach: bool = False,
        record_reputation: bool = True,
    ) -> Any:
        """Check delivered work against a contract and record the verdict.

        Compares the delivered cost / latency / quality against the agreed terms
        (:meth:`~vincio.negotiation.Contract.check`), records a
        ``contract_fulfillment`` decision on the audit chain, and — when a
        reputation ledger is attached and ``record_reputation`` is set — credits
        the seller on fulfilment or debits it on a breach, so a breached SLA
        discounts the seller's future offers. Returns a
        :class:`~vincio.negotiation.ContractFulfillment`.
        """
        fulfillment = contract.check(
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality=quality,
            raise_on_breach=raise_on_breach,
        )
        self.audit.record(
            "contract_fulfillment",
            resource=getattr(contract, "id", None),
            decision="fulfilled" if fulfillment.fulfilled else "breached",
            details={
                "seller": getattr(contract, "seller", None),
                "buyer": getattr(contract, "buyer", None),
                "breaches": fulfillment.breaches,
            },
        )
        if record_reputation and self.reputation_ledger is not None:
            self.reputation_ledger.record_outcome(
                contract.seller,
                passed=fulfillment.fulfilled,
                round_id=getattr(contract, "id", "contract"),
                details={"kind": "contract_fulfillment"},
            )
        return fulfillment

    # -- cross-org workflow choreography --------------------------------------

    def _capability_binder(self: ContextApp, saga: Any, directory: Any, binder: Any, weights: Any) -> Any:  # type: ignore[misc]
        """Resolve the binder for a saga: explicit binder, else built from a directory.

        Returns ``None`` for a fully statically-wired saga (no discovered steps), so
        the engine path is unchanged unless discovery is actually used. When the
        saga has capability steps, an explicit ``binder`` wins; otherwise a
        :class:`~vincio.choreography.CapabilityBinder` is built over ``directory``,
        this app's reputation ledger, and its settlement book, so discovery is
        ranked by the same reputation and settlement signals the rest of the fabric
        uses.
        """
        if binder is not None:
            return binder
        if not any(getattr(s, "is_discovered", False) for s in saga.steps):
            return None
        if directory is None:
            raise ConfigError(
                "this saga declares capability steps; pass directory= (a governed "
                "AgentDirectory) or binder= so the participant can be resolved at "
                "dispatch time"
            )
        from ..choreography import CapabilityBinder

        return CapabilityBinder(
            directory,
            reputation=self.reputation_ledger,
            settlement_book=self.settlement_book,
            weights=weights,
        )

    def _choreography(  # type: ignore[misc]
        self: ContextApp, saga: Any, participants: dict[str, Any], signer: Any, binder: Any = None
    ) -> Any:
        """Build a :class:`~vincio.choreography.Choreography` bound to this app."""
        from ..choreography import Choreography

        return Choreography(
            saga,
            participants,
            coordinator=self.name,
            store=self.store,
            audit=self.audit,
            events=self.events,
            signer=signer,
            binder=binder,
        )

    async def achoreograph(  # type: ignore[misc]
        self: ContextApp,
        saga: Any,
        *,
        participants: dict[str, Any],
        input: dict[str, Any] | None = None,
        saga_id: str | None = None,
        signer: Any | None = None,
        sign: bool = True,
        directory: Any | None = None,
        binder: Any | None = None,
        binding_weights: Any | None = None,
        interrupt_after: int | None = None,
    ) -> Any:
        """Run a durable, compensating cross-org saga; return a :class:`SagaResult`.

        ``saga`` is a :class:`~vincio.choreography.Saga`; ``participants`` maps each
        org id the saga dispatches to onto a
        :class:`~vincio.choreography.Participant` — a
        :class:`~vincio.choreography.RemoteParticipant` reaching a counterparty over
        the A2A fabric, or (as a convenience) a ``dict`` of handler callables run
        in-process. The :class:`~vincio.choreography.SagaJournal` is checkpointed to
        this app's metadata store after every step, so the saga **survives a
        restart** — continue it with :meth:`aresume_choreography` — and is recorded,
        hash-chained, on this app's audit log. A forward step that fails, raises, or
        breaches its step contract triggers deterministic compensation of the
        completed steps in reverse order. ``interrupt_after`` cooperatively pauses
        the forward pass into a resumable state::

            from vincio.choreography import Saga

            saga = (
                Saga(name="fulfil-order")
                .step("reserve", participant="warehouse", action="reserve",
                      compensation="release")
                .step("charge", participant="payments", action="charge",
                      compensation="refund")
            )
            result = app.choreograph(saga, participants={
                "warehouse": warehouse_client, "payments": payments_handlers,
            })
            assert result.journal.verify().intact  # offline-verifiable

        A step may instead declare the *capability* it needs and have its
        counterparty **resolved at dispatch time** from a governed
        :class:`~vincio.registry.AgentDirectory` passed as ``directory=`` — ranked
        by reputation and prior settlement fit, under the same allow-list, contract,
        and per-org audit a statically-wired step runs under. The candidate set is
        the orgs registered in both the directory and ``participants``; pass a
        prepared :class:`~vincio.choreography.CapabilityBinder` as ``binder=`` (or
        :class:`~vincio.choreography.BindingWeights` as ``binding_weights=``) to
        tune the ranking::

            saga = Saga(name="fulfil").step(
                "transcribe", capability="transcription", action="run",
            )
            result = app.choreograph(
                saga, participants={"vendor-a": a, "vendor-b": b}, directory=directory,
            )
            result.bindings["transcribe"].org  # the counterparty discovery chose
        """
        engine = self._choreography(
            saga,
            participants,
            self._resolve_contract_signer(signer, sign),
            self._capability_binder(saga, directory, binder, binding_weights),
        )
        return await engine.arun(input, saga_id=saga_id, interrupt_after=interrupt_after)

    def choreograph(self: ContextApp, saga: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Synchronous wrapper around :meth:`achoreograph`."""
        return run_sync(self.achoreograph(saga, **kwargs))

    async def aresume_choreography(  # type: ignore[misc]
        self: ContextApp,
        saga: Any,
        saga_id: str,
        *,
        participants: dict[str, Any],
        signer: Any | None = None,
        sign: bool = True,
        directory: Any | None = None,
        binder: Any | None = None,
        binding_weights: Any | None = None,
        interrupt_after: int | None = None,
    ) -> Any:
        """Resume a saga from this app's durable store after a restart.

        Rebuild the same :class:`~vincio.choreography.Saga` and ``participants`` in
        code (only the journal is persisted) and pass the ``saga_id``; completed
        steps keep their outputs and are not re-run, and a saga interrupted
        mid-rollback finishes compensating. A terminal saga is returned unchanged.
        A discovered step that already ran keeps the org it was bound to (recorded
        on the journal); one not yet reached binds at dispatch time on resume, so
        pass the same ``directory=`` / ``binder=`` used for the original run.
        """
        engine = self._choreography(
            saga,
            participants,
            self._resolve_contract_signer(signer, sign),
            self._capability_binder(saga, directory, binder, binding_weights),
        )
        return await engine.aresume(saga_id, interrupt_after=interrupt_after)

    def resume_choreography(self: ContextApp, saga: Any, saga_id: str, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Synchronous wrapper around :meth:`aresume_choreography`."""
        return run_sync(self.aresume_choreography(saga, saga_id, **kwargs))

    def serve_choreography(  # type: ignore[misc]
        self: ContextApp,
        handlers: Any,
        *,
        org_id: str | None = None,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose this org's choreography handlers over A2A.

        Returns an :class:`~vincio.a2a.A2AServer` whose Agent Card advertises a
        ``choreograph`` skill; a remote coordinator dispatches steps to it with a
        :class:`~vincio.choreography.RemoteParticipant`. Each step this org performs
        or compensates is recorded on **this app's** hash-chained audit log — its
        self-governance of the steps that cross into it.
        """
        from ..choreography.fabric import choreography_a2a_server

        return choreography_a2a_server(
            handlers,
            org_id=org_id or self.name,
            name=name,
            url=url,
            description=description,
            token_validator=token_validator,
            audit=self.audit,
        )

    # -- agent-to-agent settlement & metering ---------------------------------

    def use_settlement_book(self: ContextApp, book: Any | None = None, *, owner: str | None = None) -> Any:  # type: ignore[misc]
        """Attach a durable, hash-chained ledger of cross-org settlements.

        Closing the books on contracted work — :meth:`settle` and
        :meth:`settle_saga` — appends a typed, signed, offline-verifiable
        :class:`~vincio.settlement.SettlementRecord` to this book, links it into the
        book's hash chain, records the verdict on this app's audit chain, and (when
        a reputation ledger is attached) credits or debits the seller, so a settled
        overrun or shortfall weights the next negotiation. Pass a configured
        :class:`~vincio.settlement.SettlementBook`, or let this build one wired to
        the app's contract signer, audit chain, event bus, store, and reputation
        ledger. Returns the book::

            app.use_settlement_book()
            record = app.settle(contract, cost_usd=0.08, latency_ms=1200, quality=0.9)
            app.settlement_report().print_summary()
        """
        from ..settlement import SettlementBook

        if book is None:
            book = SettlementBook(
                owner or self.name,
                signer=self._resolve_contract_signer(None, True),
                audit=self.audit,
                events=self.events,
                store=self.store,
                reputation=self.reputation_ledger,
            )
        self.settlement_book = book
        return book

    def _settlement_book(self: ContextApp) -> Any:  # type: ignore[misc]
        """The attached book, or a transient one wired to this app for one call."""
        if self.settlement_book is not None:
            return self.settlement_book
        from ..settlement import SettlementBook

        return SettlementBook(
            self.name,
            signer=self._resolve_contract_signer(None, True),
            audit=self.audit,
            events=self.events,
            reputation=self.reputation_ledger,
        )

    def meter(self: ContextApp, contract: Any, *, run_id: str | None = None) -> Any:  # type: ignore[misc]
        """A :class:`~vincio.settlement.Meter` accruing usage against a contract.

        Accrue a :class:`~vincio.settlement.UsageEvent` as each unit of contracted
        work completes; :meth:`settle` reconciles the resulting reading against the
        agreed terms. Metering is pure accumulation — it records what was delivered,
        attributed to the contract and the run, the way the cost report attributes
        spend; the contract's budget is what enforces a cap.
        """
        from ..settlement import Meter

        return Meter(contract.id, run_id=run_id)

    def settle(  # type: ignore[misc]
        self: ContextApp,
        contract: Any,
        *,
        reading: Any | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
        quality: float | None = None,
        run_id: str | None = None,
        party: str | None = None,
        sign: bool = True,
        record_reputation: bool = True,
        escrow: Any | None = None,
        escrow_config: Any | None = None,
        pool: Any | None = None,
    ) -> Any:
        """Close the books on contracted work: reconcile, sign, audit, and record.

        Reconciles the delivered work — a metered
        :class:`~vincio.settlement.MeterReading` (``reading``) or explicit
        ``cost_usd`` / ``latency_ms`` / ``quality`` figures — against the contract's
        agreed price / SLA / quality into a typed
        :class:`~vincio.settlement.SettlementRecord`, signs it as this app's side of
        the contract, appends it to the attached settlement book (hash-chained,
        checkpointed) or a transient one, records the verdict on the audit chain,
        and — unless ``record_reputation`` is off — credits the seller on fulfilment
        or debits it on a breach. The record verifies offline from the bytes alone;
        the counterparty's independently-produced record reconciles against it with
        :func:`~vincio.settlement.reconcile`. Returns the record::

            record = app.settle(contract, cost_usd=0.08, latency_ms=1200, quality=0.92)
            record.verify(app.contract_signer)  # offline-verifiable

        Pass an ``escrow`` posted against the contract (:meth:`post_escrow`) to settle the
        collateral in the same call: it is resolved against the record — the whole stake
        released on a fulfilled delivery, a bounded proportional slice forfeited on a
        breach — signed, and audited in place, so the collateral closes the same loop the
        settlement does. ``escrow_config`` overrides the forfeiture policy.

        Pass a ``pool`` the contract is backed by (:meth:`post_collateral_pool`) to draw the
        same settlement against a shared
        :class:`~vincio.settlement.CollateralPool` instead — the forfeiture drawn from the
        pooled stake and the rest released back to the available balance, re-signed and
        audited in place.
        """
        return self._settlement_book().settle(
            contract,
            reading=reading,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            quality=quality,
            run_id=run_id,
            party=party,
            sign=sign,
            record_reputation=record_reputation,
            escrow=escrow,
            escrow_config=escrow_config,
            pool=pool,
        )

    def post_escrow(  # type: ignore[misc]
        self: ContextApp,
        contract: Any,
        *,
        decision: Any | None = None,
        fraction: float | None = None,
        amount: float | None = None,
        poster: str | None = None,
        beneficiary: str | None = None,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Post collateral against a contract as a signed, offline-verifiable escrow.

        Binds the admission-required collateral — read from an
        :class:`~vincio.settlement.AdmissionDecision` (``decision``), an explicit
        ``fraction`` / ``amount``, or the admission posture
        :meth:`~vincio.settlement.AdmissionDecision.apply_to_terms` stamped onto the
        contract's terms — to the specific contract and counterparty into an
        :class:`~vincio.settlement.Escrow`, signs it as this app's side of the contract,
        appends the posting to the attached settlement book's audit chain, and returns
        it. The escrow verifies offline from the bytes alone (the held amount re-derives
        from the admission posture); :meth:`settle` (with ``escrow=``) or
        :meth:`settle_escrow` resolves it against delivery::

            decision = app.admit("vendor")
            escrow = app.post_escrow(contract, decision=decision)
            escrow.verify().valid  # offline-verifiable
        """
        return self._settlement_book().post_escrow(
            contract,
            decision=decision,
            fraction=fraction,
            amount=amount,
            poster=poster,
            beneficiary=beneficiary,
            config=config,
            party=party,
            sign=sign,
        )

    def settle_escrow(  # type: ignore[misc]
        self: ContextApp,
        escrow: Any,
        record: Any,
        *,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Resolve a posted escrow against a settlement record (release or forfeit).

        Settles ``escrow`` against the contract's
        :class:`~vincio.settlement.SettlementRecord` (from :meth:`settle`): releases the
        whole stake on a fulfilled delivery and forfeits a bounded slice proportional to
        the shortfall on a breach — driven by the same settlement verdict — re-signs the
        resolved escrow as this app, and records the release / forfeiture on the audit
        chain. ``config`` overrides the forfeiture policy. Returns the resolved escrow::

            record = app.settle(contract, cost_usd=0.20)   # a cost overrun: a breach
            app.settle_escrow(escrow, record)              # forfeits a proportional slice
        """
        return self._settlement_book().settle_escrow(
            escrow, record, config=config, party=party, sign=sign
        )

    def post_collateral_pool(  # type: ignore[misc]
        self: ContextApp,
        contracts: Any,
        *,
        poster: str | None = None,
        posted: float | None = None,
        decisions: Any | None = None,
        fraction: float | None = None,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Post one stake backing many contracts as a signed, offline-verifiable margin account.

        Binds a counterparty's single posted stake to the set of ``contracts`` it backs into
        a :class:`~vincio.settlement.CollateralPool`, allocating each a per-contract share
        proportional to its admission-required collateral — read from a matching
        :class:`~vincio.settlement.AdmissionDecision` in ``decisions``, a uniform
        ``fraction``, or the admission posture stamped onto each contract's terms. Signs it
        as this app's side and appends the posting to the settlement book's audit chain. A
        clean delivery frees capital for the next contract and a breach is covered from the
        shared stake; :meth:`settle` (with ``pool=``) or :meth:`draw_pool` draws an open
        contract's settlement against it::

            pool = app.post_collateral_pool([c1, c2, c3], decisions={c1.id: d1, ...})
            pool.verify().valid  # offline-verifiable — allocations re-derive, balance reconciles
        """
        return self._settlement_book().post_collateral_pool(
            contracts,
            poster=poster,
            posted=posted,
            decisions=decisions,
            fraction=fraction,
            config=config,
            party=party,
            sign=sign,
        )

    def draw_pool(  # type: ignore[misc]
        self: ContextApp,
        pool: Any,
        record: Any,
        *,
        config: Any | None = None,
        party: str | None = None,
        sign: bool = True,
    ) -> Any:
        """Draw one backed contract's settlement against a collateral pool (draw or release).

        Settles the matching contract against its
        :class:`~vincio.settlement.SettlementRecord` (from :meth:`settle`): draws a bounded
        slice proportional to the shortfall from the shared stake on a breach and releases
        the rest back to the available balance on a clean delivery — driven by the same
        settlement verdict — re-signs the pool as this app, and records the draw on the
        audit chain. ``config`` overrides the forfeiture policy. Returns the pool::

            record = app.settle(contract, cost_usd=140.0)   # a cost overrun: a breach
            app.draw_pool(pool, record)                     # draws a proportional slice
        """
        return self._settlement_book().draw_pool(
            pool, record, config=config, party=party, sign=sign
        )

    def identity(  # type: ignore[misc]
        self: ContextApp,
        name: str | None = None,
        *,
        controller: str = "",
        capabilities: Any | None = None,
        seed: Any | None = None,
        use: bool = False,
        record_audit: bool = True,
    ) -> Any:
        """Mint a portable, self-certifying :class:`~vincio.security.AgentIdentity`.

        The identity is built on an Ed25519 key whose **DID is derived from the public
        key** (``did:vincio:ed25519:<hex>``), so the identifier resolves to the
        verifying key offline with no registry. ``name`` labels it (defaults to this
        app's name), ``controller`` names the operating org, ``capabilities`` are the
        capabilities it advertises, and ``seed`` (32 bytes) makes the key deterministic
        for tests. With ``use=True`` the identity also becomes this app's signer (see
        :meth:`use_identity`). Unless ``record_audit`` is off, the issuance lands on the
        hash-chained audit log. The identity satisfies the
        :class:`~vincio.security.audit.ChainSigner` protocol, so it drops into every
        signing slot the platform already exposes::

            agent = app.identity("billing-agent", capabilities=["retrieve", "summarize"])
            grant = agent.delegate("did:vincio:ed25519:...", capabilities=["retrieve"])
        """
        from ..security.identity import AgentIdentity

        identity = AgentIdentity.generate(
            name or self.name,
            controller=controller,
            capabilities=list(capabilities) if capabilities else None,
            seed=seed,
        )
        # Bind first (when requested) so the identity adopts the audit signer before
        # the mint entry is recorded — the mint then lands signed by its own DID.
        if use:
            self.use_identity(identity, record_audit=False)
        if record_audit and self.audit is not None:
            from ..security.identity import IDENTITY_ACTION

            entry = self.audit.record(
                IDENTITY_ACTION,
                resource=identity.did,
                decision="minted",
                details=identity.document.audit_details(),
            )
            identity.document.audit_id = getattr(entry, "id", None)
        return identity

    def use_identity(self: ContextApp, identity: Any, *, record_audit: bool = True) -> Any:  # type: ignore[misc]
        """Bind ``identity`` as this app's signer so every artifact carries its DID.

        Sets the identity as the content signer and the contract/settlement signer —
        and, when the audit log has not yet recorded anything, as the audit-chain
        signer too — so subsequent audit entries, negotiated contracts, settlement
        records, and signed manifests all record the identity's **DID** as their
        ``key_id``. Accountability becomes mechanical: a verifier resolves the signer
        from the DID and checks the signature from the bytes, rather than trusting an
        out-of-band ``key_id`` string. Returns the identity.
        """
        self._identity = identity
        self.content_signer = identity
        self._contract_signer = identity
        # Adopt the audit signer only on a fresh log, so the chain stays verifiable
        # under one key (mixing signers mid-chain would break offline verification).
        if self.audit is not None and not self.audit.entries:
            self.audit.signer = identity
        if record_audit and self.audit is not None:
            from ..security.identity import IDENTITY_ACTION

            self.audit.record(
                IDENTITY_ACTION,
                resource=getattr(identity, "did", None),
                decision="bound",
                details={"did": getattr(identity, "did", None), "name": getattr(identity, "name", "")},
            )
        return identity

    def issue_credential(  # type: ignore[misc]
        self: ContextApp,
        subject: Any,
        claims: dict[str, str],
        *,
        as_identity: Any | None = None,
        not_after: Any | None = None,
        expires_in: Any | None = None,
        record_audit: bool = True,
    ) -> Any:
        """Issue a signed, offline-verifiable :class:`~vincio.security.AgentCredential`.

        The issuer signs a verifiable claim about ``subject`` (an agent DID or
        :class:`~vincio.security.AgentIdentity`) — e.g.
        ``{"admitted_capability": "retrieve", "operated_by": "org-acme"}`` — that an
        importer verifies offline and folds into the admission / registry path
        (:meth:`~vincio.security.AgentCredential.admits`). The issuer is ``as_identity``
        or this app's bound identity (:meth:`use_identity`); raises
        :class:`~vincio.core.errors.IdentityError` if neither is set. Records the
        issuance on the audit chain unless ``record_audit`` is off. Returns the
        credential::

            org = app.identity("org-acme", use=True)
            cred = app.issue_credential(agent, {"admitted_capability": "retrieve"})
            cred.verify().valid  # True, from the bytes alone
        """
        from ..core.errors import IdentityError

        issuer = as_identity or self._identity
        if issuer is None:
            raise IdentityError(
                "no issuing identity: pass as_identity= or bind one with app.use_identity(...)",
                details={"app": self.name},
            )
        credential = issuer.issue_credential(
            subject, claims, not_after=not_after, expires_in=expires_in
        )
        if record_audit and self.audit is not None:
            from ..security.identity import CREDENTIAL_ACTION

            entry = self.audit.record(
                CREDENTIAL_ACTION,
                resource=credential.subject,
                decision="issued",
                details=credential.audit_details(),
            )
            credential.audit_id = getattr(entry, "id", None)
        return credential

    def attest_custody(  # type: ignore[misc]
        self: ContextApp,
        poster: str,
        reserves: Any,
        *,
        custodian: str | None = None,
        as_of: Any | None = None,
        sign: bool = True,
        record_audit: bool = True,
    ) -> Any:
        """Attest a poster's proven reserves into a signed, content-bound proof-of-reserves.

        Issues a :class:`~vincio.settlement.CustodyAttestation` over the capital ``poster``
        actually holds — itemized ``reserves`` (a number, a mapping of ``account -> amount``,
        or :class:`~vincio.settlement.ReserveLine` items) whose total re-derives on every
        verify — so the held figure :meth:`guard_collateral` bounds the pledges against is
        **evidence-backed** rather than asserted. ``custodian`` defaults to this app (a
        third-party custodian vouching), and when it is also the ``poster`` the attestation is
        self-custody. Signs it as the custodian and, unless ``record_audit`` is off, records
        the issuance on the audit chain. The attestation verifies offline from the bytes
        alone — a tampered reserve figure or a forged custodian is caught. Returns it::

            proof = app.attest_custody("vendor", {"omnibus": 80.0})
            ledger = app.guard_collateral([pool_a, pool_b], custody=proof)
            ledger.require_reserved()  # raises if proven reserves < pledged
        """
        from ..settlement import attest_custody as _attest

        resolved_custodian = custodian or self.name
        attestation = _attest(poster, reserves, custodian=resolved_custodian, as_of=as_of)
        if sign and self.name == attestation.custodian:
            signer = self._resolve_contract_signer(None, True)
            if signer is not None:
                attestation.sign(signer, party=attestation.custodian)
        if record_audit and self.audit is not None:
            from ..settlement.custody import CUSTODY_ACTION

            entry = self.audit.record(
                CUSTODY_ACTION,
                resource=attestation.poster,
                decision="self_custody" if attestation.self_custody else "custodied",
                details=attestation.audit_details(),
            )
            attestation.audit_id = getattr(entry, "id", None)
        return attestation
