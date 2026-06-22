"""Cross-org workflow discovery: run-time capability binding for sagas.

A discovered saga step declares the *capability* it needs and the engine resolves
the participant at dispatch time from the governed agent directory — ranked by
reputation and prior settlement fit, under the same allow-list, contract, and
per-org audit a statically-wired step runs under. These tests hold two guarantees:
**binding correctness** (the best-available allowed candidate is chosen,
deterministically, and recorded) and **governance preservation** (an unlisted or
unreachable candidate is never bound; the decision is audited; compensation,
durability, and contract enforcement behave exactly as for a static step).
"""

from __future__ import annotations

import pytest

from vincio import ContextApp
from vincio.a2a import connect_a2a_in_process
from vincio.a2a.protocol import AgentCard, AgentSkill
from vincio.choreography import (
    BindingWeights,
    CapabilityBinder,
    Choreography,
    RemoteParticipant,
    Saga,
    StepOutcome,
)
from vincio.core.errors import ChoreographyError, ConfigError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.storage.base import InMemoryMetadataStore


def _app(name: str = "coord") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _directory(app: ContextApp, vendors, *, capability: str = "transcription", allow=("vendor-*",)):
    """Register each vendor in a governed directory advertising ``capability``."""
    directory = app.agent_directory(allow=list(allow))
    for name in vendors:
        directory.register(
            AgentCard(
                name=name,
                description=f"{name} does {capability}",
                skills=[
                    AgentSkill(
                        id="run",
                        name="run",
                        description=f"perform {capability}",
                        tags=[capability],
                    )
                ],
            )
        )
    return directory


def _handlers(log: list[str] | None = None, *, fail: set[str] | None = None):
    """A vendor's handler dict: forward ``run`` and compensating ``discard``."""
    fail = fail or set()

    def mk(org: str):
        def run(payload):
            if log is not None:
                log.append(f"{org}:run")
            if org in fail:
                return StepOutcome(ok=False, error=f"{org} declined")
            return {"text": org}

        def discard(payload):
            if log is not None:
                log.append(f"{org}:discard")
            return {"discarded": org}

        return {"run": run, "discard": discard}

    return mk


# -- step definition & validation ---------------------------------------------


def test_step_requires_exactly_one_of_participant_or_capability():
    with pytest.raises(ChoreographyError):
        Saga(name="s").step("a", action="run", participant="o", capability="c")
    with pytest.raises(ChoreographyError):
        Saga(name="s").step("a", action="run")  # neither


def test_discovered_step_flag():
    saga = Saga(name="s").step("a", action="run", capability="transcription")
    assert saga.steps[0].is_discovered
    static = Saga(name="s").step("a", action="run", participant="o")
    assert not static.steps[0].is_discovered


# -- binding correctness -------------------------------------------------------


def test_binds_highest_reputation_candidate():
    app = _app()
    app.use_reputation_ledger()
    app.reputation_ledger.record_outcome("vendor-a", passed=True, round_id="r1")
    app.reputation_ledger.record_outcome("vendor-b", passed=False, round_id="r1")
    app.reputation_ledger.record_outcome("vendor-b", passed=False, round_id="r2")
    directory = _directory(app, ["vendor-a", "vendor-b"])
    mk = _handlers()
    parts = {"vendor-a": mk("vendor-a"), "vendor-b": mk("vendor-b")}

    saga = Saga(name="job").step("t", action="run", capability="transcription")
    result = app.choreograph(saga, participants=parts, directory=directory)

    assert result.status == "completed"
    binding = result.bindings["t"]
    assert binding.org == "vendor-a"
    assert binding.eligible == 2 and binding.considered == 2
    # vendor-a outranks vendor-b on the reputation signal.
    by_org = {c.org: c for c in binding.candidates}
    assert by_org["vendor-a"].score > by_org["vendor-b"].score
    assert result.output_of("t") == {"text": "vendor-a"}


def test_ties_break_deterministically_by_org_id():
    app = _app()
    directory = _directory(app, ["vendor-b", "vendor-a"])  # equal standing
    mk = _handlers()
    parts = {"vendor-a": mk("vendor-a"), "vendor-b": mk("vendor-b")}
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    first = app.choreograph(saga, participants=parts, directory=directory)
    second = app.choreograph(saga, participants=parts, directory=directory)
    assert first.bindings["t"].org == second.bindings["t"].org == "vendor-a"


def test_settlement_reliability_ranks_candidates():
    app = _app()
    book = app.use_settlement_book()
    # vendor-a honoured its prior deals; vendor-b breached.
    good = Contract(buyer="coord", seller="vendor-a", terms=ContractTerms(scope="x", price_usd=0.10)).seal()
    bad = Contract(buyer="coord", seller="vendor-b", terms=ContractTerms(scope="x", price_usd=0.10)).seal()
    book.settle(good, cost_usd=0.08)  # within price -> settled
    book.settle(bad, cost_usd=0.50)  # overrun -> breached
    directory = _directory(app, ["vendor-a", "vendor-b"])
    mk = _handlers()
    parts = {"vendor-a": mk("vendor-a"), "vendor-b": mk("vendor-b")}
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    result = app.choreograph(saga, participants=parts, directory=directory)
    assert result.bindings["t"].org == "vendor-a"
    by_org = {c.org: c for c in result.bindings["t"].candidates}
    assert by_org["vendor-a"].settlement_reliability == 1.0
    assert by_org["vendor-b"].settlement_reliability == 0.0


def test_contract_fit_prefers_within_budget_history():
    app = _app()
    book = app.use_settlement_book()
    # Both honoured (no breach), but vendor-b historically delivers near the cap.
    a = Contract(buyer="coord", seller="vendor-a", terms=ContractTerms(scope="x", price_usd=1.0)).seal()
    b = Contract(buyer="coord", seller="vendor-b", terms=ContractTerms(scope="x", price_usd=1.0)).seal()
    book.settle(a, cost_usd=0.10)  # well under
    book.settle(b, cost_usd=1.0)  # right at the cap
    directory = _directory(app, ["vendor-a", "vendor-b"])
    mk = _handlers()
    parts = {"vendor-a": mk("vendor-a"), "vendor-b": mk("vendor-b")}
    # A new step under a tight contract: the candidate whose history fits it best wins.
    deal = Contract(buyer="coord", seller="*", terms=ContractTerms(scope="x", price_usd=0.20)).seal()
    saga = Saga(name="job").step("t", action="run", capability="transcription", contract=deal)
    result = app.choreograph(saga, participants=parts, directory=directory)
    by_org = {c.org: c for c in result.bindings["t"].candidates}
    assert by_org["vendor-a"].contract_fit == 1.0
    assert by_org["vendor-b"].contract_fit < 1.0
    assert result.bindings["t"].org == "vendor-a"


# -- governance preservation ---------------------------------------------------


def test_unlisted_candidate_is_never_bound():
    app = _app()
    # Only vendor-a is allow-listed; vendor-evil advertises the capability but is denied.
    directory = _directory(app, ["vendor-a", "vendor-evil"], allow=["vendor-a"])
    mk = _handlers()
    parts = {"vendor-a": mk("vendor-a"), "vendor-evil": mk("vendor-evil")}
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    result = app.choreograph(saga, participants=parts, directory=directory)
    assert result.bindings["t"].org == "vendor-a"
    by_org = {c.org: c for c in result.bindings["t"].candidates}
    assert by_org["vendor-evil"].allowed is False
    assert by_org["vendor-evil"].rejected_reason
    assert by_org["vendor-evil"].score == 0.0
    # The governed resolution of each candidate is on the audit chain.
    assert app.audit.query(action="agent_resolve")


def test_unreachable_candidate_is_rejected():
    app = _app()
    directory = _directory(app, ["vendor-a", "vendor-b"])
    mk = _handlers()
    # vendor-b advertises the capability but has no participant binding.
    parts = {"vendor-a": mk("vendor-a")}
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    result = app.choreograph(saga, participants=parts, directory=directory)
    assert result.bindings["t"].org == "vendor-a"
    by_org = {c.org: c for c in result.bindings["t"].candidates}
    assert by_org["vendor-b"].reachable is False
    assert by_org["vendor-b"].eligible is False


def test_no_eligible_candidate_raises():
    app = _app()
    directory = _directory(app, ["vendor-evil"], allow=["vendor-a"])  # only a denied one
    parts = {"vendor-evil": _handlers()("vendor-evil")}
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    with pytest.raises(ChoreographyError):
        app.choreograph(saga, participants=parts, directory=directory)


def test_binding_decision_is_audited():
    app = _app()
    directory = _directory(app, ["vendor-a"])
    parts = {"vendor-a": _handlers()("vendor-a")}
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    app.choreograph(saga, participants=parts, directory=directory)
    binds = app.audit.query(action="choreography_bind")
    assert binds and binds[0].details["bound_org"] == "vendor-a"
    assert app.audit.verify_chain()


def test_missing_directory_raises_config_error():
    app = _app()
    parts = {"vendor-a": _handlers()("vendor-a")}
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    with pytest.raises(ConfigError):
        app.choreograph(saga, participants=parts)


def test_engine_without_binder_rejects_capability_step():
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    with pytest.raises(ChoreographyError):
        Choreography(saga, {"vendor-a": {"run": lambda p: {}}})


# -- discovery preserves compensation / durability / contract ------------------


def test_compensation_dispatches_to_the_bound_org():
    app = _app()
    app.use_reputation_ledger()
    app.reputation_ledger.record_outcome("vendor-b", passed=False, round_id="r1")
    directory = _directory(app, ["vendor-a", "vendor-b"])
    log: list[str] = []
    mk = _handlers(log)
    parts = {"vendor-a": mk("vendor-a"), "vendor-b": mk("vendor-b")}
    # The discovered step binds vendor-a (best reputation); the second step fails,
    # so the discovered step must compensate at the org it was actually bound to.
    parts["vendor-a"]["fail"] = lambda p: StepOutcome(ok=False, error="boom")
    saga = (
        Saga(name="job")
        .step("t", action="run", capability="transcription", compensation="discard")
        .step("z", action="fail", participant="vendor-a")
    )
    result = app.choreograph(saga, participants=parts, directory=directory)
    assert result.status == "compensated"
    assert result.bindings["t"].org == "vendor-a"
    # vendor-a's discard ran (the bound org); vendor-b's never did.
    assert "vendor-a:discard" in log
    assert "vendor-b:discard" not in log


def test_contract_breach_compensates_discovered_step():
    app = _app()
    directory = _directory(app, ["vendor-a"])
    terms = ContractTerms(scope="x", price_usd=0.10, quality_floor=0.8)
    deal = Contract(buyer="coord", seller="*", terms=terms).seal()
    comp: list[str] = []
    parts = {
        "vendor-a": {
            "run": lambda p: StepOutcome(ok=True, cost_usd=0.50, quality=0.9),  # overrun
            "discard": lambda p: comp.append("discard") or {},
        }
    }
    saga = Saga(name="job").step(
        "t", action="run", capability="transcription", compensation="discard", contract=deal
    )
    result = app.choreograph(saga, participants=parts, directory=directory)
    assert result.status == "compensated"
    failed = [r for r in result.journal.forward_records() if r.status == "failed"]
    assert failed and failed[0].fulfilled is False
    assert any("price" in b for b in failed[0].breaches)


def test_durable_resume_does_not_rebind_completed_step():
    app = _app()
    store = InMemoryMetadataStore()
    directory = _directory(app, ["vendor-a", "vendor-b"])
    app.use_reputation_ledger()
    app.reputation_ledger.record_outcome("vendor-b", passed=False, round_id="r1")
    runs: dict[str, int] = {}

    def counted(org: str):
        def run(payload):
            runs[org] = runs.get(org, 0) + 1
            return {"text": org}

        return {"run": run, "discard": lambda p: {}}

    parts = {"vendor-a": counted("vendor-a"), "vendor-b": counted("vendor-b")}
    binder = CapabilityBinder(directory, reputation=app.reputation_ledger)
    saga = (
        Saga(name="job")
        .step("t1", action="run", capability="transcription")
        .step("t2", action="run", capability="transcription")
    )
    engine1 = Choreography(saga, parts, store=store, binder=binder)
    paused = engine1.run(saga_id="d1", interrupt_after=1)
    assert paused.status == "interrupted"
    engine2 = Choreography(saga, parts, store=store, binder=binder)  # fresh engine
    resumed = engine2.resume("d1")
    assert resumed.status == "completed"
    # vendor-a (best reputation) bound for both steps; t1 ran exactly once.
    assert runs == {"vendor-a": 2}
    assert resumed.bindings["t1"].org == "vendor-a"
    assert resumed.bindings["t2"].org == "vendor-a"


# -- mixed static + discovered + A2A -------------------------------------------


def test_mixed_static_and_discovered_steps():
    app = _app()
    directory = _directory(app, ["vendor-a"])
    log: list[str] = []
    parts = {
        "warehouse": {"reserve": lambda p: log.append("reserve") or {"ticket": 1}},
        "vendor-a": _handlers(log)("vendor-a"),
    }
    saga = (
        Saga(name="job")
        .step("reserve", action="reserve", participant="warehouse")
        .step("transcribe", action="run", capability="transcription")
    )
    result = app.choreograph(saga, participants=parts, directory=directory)
    assert result.status == "completed"
    assert result.completed_steps == ["reserve", "transcribe"]
    assert result.bindings["transcribe"].org == "vendor-a"
    assert "transcribe" in result.bindings and "reserve" not in result.bindings


async def test_discovery_binds_remote_participant_over_a2a():
    coord = _app("coord")
    vendor = _app("vendor")
    directory = coord.agent_directory(allow=["vendor-a"])
    directory.register(
        AgentCard(
            name="vendor-a",
            description="remote transcriber",
            skills=[AgentSkill(id="run", name="run", description="transcription", tags=["transcription"])],
        )
    )
    server = vendor.serve_choreography({"run": lambda p: {"text": "remote"}}, org_id="vendor-a")
    client = connect_a2a_in_process(server)
    remote = RemoteParticipant(client, org_id="vendor-a")
    saga = Saga(name="job").step("t", action="run", capability="transcription")
    result = await coord.achoreograph(saga, participants={"vendor-a": remote}, directory=directory)
    assert result.status == "completed"
    assert result.output_of("t") == {"text": "remote"}
    assert result.bindings["t"].org == "vendor-a"


# -- the binder directly -------------------------------------------------------


def test_binding_weights_validation():
    with pytest.raises(ChoreographyError):
        BindingWeights(reputation=-1.0).validate_coherent()
    with pytest.raises(ChoreographyError):
        BindingWeights(reputation=0, settlement=0, contract_fit=0).validate_coherent()
    with pytest.raises(ChoreographyError):
        BindingWeights(unknown_settlement_score=2.0).validate_coherent()


def test_binder_rank_includes_all_candidates():
    app = _app()
    directory = _directory(app, ["vendor-a", "vendor-b"], allow=["vendor-a"])
    binder = CapabilityBinder(directory)
    ranked = binder.rank("transcription", available={"vendor-a", "vendor-b"})
    assert [c.org for c in ranked] == ["vendor-a", "vendor-b"]  # eligible first
    assert ranked[0].eligible and not ranked[1].eligible


def test_binder_bind_requires_capability():
    app = _app()
    directory = _directory(app, ["vendor-a"])
    binder = CapabilityBinder(directory)
    static_step = Saga(name="s").step("a", action="run", participant="vendor-a").steps[0]
    with pytest.raises(ChoreographyError):
        binder.bind(static_step)


def test_step_binding_audit_details_shape():
    app = _app()
    directory = _directory(app, ["vendor-a"])
    binder = CapabilityBinder(directory)
    step = Saga(name="s").step("t", action="run", capability="transcription").steps[0]
    binding = binder.bind(step, available={"vendor-a"})
    details = binding.audit_details()
    assert details["bound_org"] == "vendor-a"
    assert details["capability"] == "transcription"
    assert isinstance(details["candidates"], list)
