"""Cross-org workflow choreography: durable saga, compensation, per-org governance."""

from __future__ import annotations

import pytest

from vincio import ContextApp
from vincio.a2a import connect_a2a_in_process
from vincio.choreography import (
    Choreography,
    LocalParticipant,
    RemoteParticipant,
    Saga,
    SagaContext,
    SagaJournal,
    StepOutcome,
    StepRequest,
)
from vincio.core.errors import ChoreographyError, CompensationError
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider
from vincio.security.audit import HMACSigner
from vincio.storage.base import InMemoryMetadataStore


def _app(name: str = "acme") -> ContextApp:
    return ContextApp(name=name, provider=MockProvider(default_text="ok"), model="mock-1")


def _recorder() -> tuple[dict[str, int], dict[str, list]]:
    counts: dict[str, int] = {}
    order: dict[str, list] = {"forward": [], "compensation": []}
    return counts, order


def _handlers(counts: dict[str, int], order: dict[str, list]) -> dict:
    def mk(name: str, *, comp: bool = False, fail: bool = False, raises: bool = False):
        def handler(payload):
            counts[name] = counts.get(name, 0) + 1
            order["compensation" if comp else "forward"].append(name)
            if raises:
                raise RuntimeError(f"{name} crashed")
            if fail:
                return StepOutcome(ok=False, error=f"{name} declined")
            return {"step": name, "payload": payload}

        return handler

    return {"mk": mk}


# -- saga definition -----------------------------------------------------------


def test_saga_builder_and_validation():
    saga = (
        Saga(name="order")
        .step("a", participant="o", action="do_a", compensation="undo_a")
        .step("b", participant="o", action="do_b")
    )
    assert [s.name for s in saga.steps] == ["a", "b"]
    assert saga.by_name("a").compensation == "undo_a"
    assert saga.validate_coherent() is saga


def test_saga_rejects_duplicate_and_empty():
    saga = Saga(name="x").step("a", participant="o", action="do")
    with pytest.raises(ChoreographyError):
        saga.step("a", participant="o", action="again")
    with pytest.raises(ChoreographyError):
        Saga(name="empty").validate_coherent()


def test_unknown_participant_raises():
    saga = Saga(name="x").step("a", participant="ghost", action="do")
    with pytest.raises(ChoreographyError):
        Choreography(saga, {"real": {"do": lambda p: {}}})


def test_bad_participant_spec_raises():
    saga = Saga(name="x").step("a", participant="o", action="do")
    with pytest.raises(ChoreographyError):
        Choreography(saga, {"o": object()})


# -- forward execution ---------------------------------------------------------


async def test_forward_success_completes_in_order():
    counts, order = _recorder()
    mk = _handlers(counts, order)["mk"]
    saga = (
        Saga(name="ok")
        .step("a", participant="o", action="do_a")
        .step("b", participant="o", action="do_b")
        .step("c", participant="o", action="do_c")
    )
    parts = {"o": {"do_a": mk("a"), "do_b": mk("b"), "do_c": mk("c")}}
    result = await _app().achoreograph(saga, participants=parts)
    assert result.status == "completed"
    assert result.ok and not result.unwound
    assert result.completed_steps == ["a", "b", "c"]
    assert order["forward"] == ["a", "b", "c"]
    assert result.output == {"step": "c", "payload": {}}


async def test_build_payload_threads_prior_outputs():
    captured = {}

    def producer(payload):
        return {"value": 21}

    def consumer(payload):
        captured.update(payload)
        return {"doubled": payload["value"] * 2}

    saga = (
        Saga(name="chain")
        .step("produce", participant="o", action="produce")
        .step(
            "consume",
            participant="o",
            action="consume",
            build=lambda ctx: {"value": ctx["produce"]["value"], "seed": ctx.input["seed"]},
        )
    )
    parts = {"o": {"produce": producer, "consume": consumer}}
    result = await _app().achoreograph(saga, participants=parts, input={"seed": 5})
    assert captured == {"value": 21, "seed": 5}
    assert result.output_of("consume") == {"doubled": 42}


async def test_build_must_return_dict():
    saga = Saga(name="x").step("a", participant="o", action="do", build=lambda ctx: 5)
    with pytest.raises(ChoreographyError):
        await _app().achoreograph(saga, participants={"o": {"do": lambda p: {}}})


# -- compensation (saga rollback) ----------------------------------------------


async def test_failure_compensates_in_reverse_order():
    counts, order = _recorder()
    mk = _handlers(counts, order)["mk"]
    saga = (
        Saga(name="rollback")
        .step("a", participant="o", action="do_a", compensation="undo_a")
        .step("b", participant="o", action="do_b", compensation="undo_b")
        .step("c", participant="o", action="do_c", compensation="undo_c")
    )
    parts = {
        "o": {
            "do_a": mk("a"),
            "do_b": mk("b"),
            "do_c": mk("c", fail=True),
            "undo_a": mk("a", comp=True),
            "undo_b": mk("b", comp=True),
        }
    }
    result = await _app().achoreograph(saga, participants=parts)
    assert result.status == "compensated"
    assert result.unwound
    assert result.failed_step == "c"
    # b and a completed; c failed; compensate b then a (reverse), c has nothing done.
    assert result.compensated_steps == ["b", "a"]
    assert order["compensation"] == ["b", "a"]


async def test_raising_handler_is_a_failure_not_a_crash():
    counts, order = _recorder()
    mk = _handlers(counts, order)["mk"]
    saga = (
        Saga(name="r")
        .step("a", participant="o", action="do_a", compensation="undo_a")
        .step("b", participant="o", action="do_b")
    )
    parts = {"o": {"do_a": mk("a"), "do_b": mk("b", raises=True), "undo_a": mk("a", comp=True)}}
    result = await _app().achoreograph(saga, participants=parts)
    assert result.status == "compensated"
    assert result.compensated_steps == ["a"]


async def test_step_without_compensation_is_skipped_on_rollback():
    counts, order = _recorder()
    mk = _handlers(counts, order)["mk"]
    saga = (
        Saga(name="r")
        .step("a", participant="o", action="do_a")  # no compensation
        .step("b", participant="o", action="do_b", compensation="undo_b")
        .step("c", participant="o", action="do_c")
    )
    parts = {
        "o": {
            "do_a": mk("a"),
            "do_b": mk("b"),
            "do_c": mk("c", fail=True),
            "undo_b": mk("b", comp=True),
        }
    }
    result = await _app().achoreograph(saga, participants=parts)
    assert result.compensated_steps == ["b"]  # a had no undo
    assert order["compensation"] == ["b"]


async def test_compensation_failure_ends_failed_and_can_raise():
    saga = (
        Saga(name="r")
        .step("a", participant="o", action="do_a", compensation="undo_a")
        .step("b", participant="o", action="do_b")
    )

    def boom(payload):
        raise RuntimeError("undo failed")

    parts = {"o": {"do_a": lambda p: {}, "do_b": lambda p: StepOutcome(ok=False), "undo_a": boom}}
    app = _app()
    result = await app.achoreograph(saga, participants=parts)
    assert result.status == "failed"
    comp = [r for r in result.journal.records if r.kind == "compensation"][0]
    assert comp.status == "compensation_failed"

    # With raise_on_compensation_failure the engine surfaces the residue.
    engine = Choreography(
        saga, parts, store=None, raise_on_compensation_failure=True
    )
    with pytest.raises(CompensationError) as exc:
        await engine.arun()
    assert "a" in exc.value.failures


# -- contract governance -------------------------------------------------------


async def test_resume_retries_failed_compensation_to_clean_unwind():
    store = InMemoryMetadataStore()
    saga = (
        Saga(name="r")
        .step("a", participant="o", action="do_a", compensation="undo_a")
        .step("b", participant="o", action="do_b")
    )
    # First engine: the compensation of 'a' fails, leaving the saga "failed".
    flaky = {
        "o": {
            "do_a": lambda p: {"a": 1},
            "do_b": lambda p: StepOutcome(ok=False, error="boom"),
            "undo_a": lambda p: (_ for _ in ()).throw(RuntimeError("undo down")),
        }
    }
    first = await Choreography(saga, flaky, store=store).arun(saga_id="r1")
    assert first.status == "failed"

    # Second engine: the participant is healthy now; resuming retries the
    # outstanding compensation and the saga unwinds cleanly.
    healed = {
        "o": {
            "do_a": lambda p: {"a": 1},
            "do_b": lambda p: StepOutcome(ok=False, error="boom"),
            "undo_a": lambda p: {"undone": True},
        }
    }
    second = await Choreography(saga, healed, store=store).aresume("r1")
    assert second.status == "compensated"
    assert "a" in second.compensated_steps


async def test_contract_breach_triggers_compensation():
    terms = ContractTerms(scope="x", price_usd=0.10, sla_seconds=3.0, quality_floor=0.8)
    contract = Contract(buyer="a", seller="b", terms=terms).seal()
    comp = []
    saga = (
        Saga(name="c")
        .step("pre", participant="o", action="pre", compensation="undo_pre")
        .step("work", participant="o", action="work", contract=contract)
    )
    parts = {
        "o": {
            "pre": lambda p: {"pre": 1},
            "undo_pre": lambda p: comp.append("pre") or {},
            "work": lambda p: StepOutcome(ok=True, cost_usd=0.50, quality=0.9),
        }
    }
    result = await _app().achoreograph(saga, participants=parts)
    assert result.status == "compensated"
    assert comp == ["pre"]
    failed = [r for r in result.journal.forward_records() if r.status == "failed"][0]
    assert failed.fulfilled is False
    assert any("price" in b for b in failed.breaches)


async def test_contract_fulfilled_completes():
    terms = ContractTerms(scope="x", price_usd=0.10, sla_seconds=3.0, quality_floor=0.8)
    contract = Contract(buyer="a", seller="b", terms=terms).seal()
    saga = Saga(name="c").step("work", participant="o", action="work", contract=contract)
    parts = {"o": {"work": lambda p: StepOutcome(ok=True, cost_usd=0.05, latency_ms=2000, quality=0.95)}}
    result = await _app().achoreograph(saga, participants=parts)
    assert result.status == "completed"
    done = result.journal.completed_forward()[0]
    assert done.fulfilled is True and done.contract_id == contract.id


# -- durability & resume -------------------------------------------------------


async def test_interrupt_then_resume_survives_a_fresh_engine():
    counts, order = _recorder()
    mk = _handlers(counts, order)["mk"]
    store = InMemoryMetadataStore()
    saga = (
        Saga(name="two")
        .step("a", participant="o", action="do_a")
        .step("b", participant="o", action="do_b")
    )
    parts = {"o": {"do_a": mk("a"), "do_b": mk("b")}}

    engine1 = Choreography(saga, parts, store=store)
    r1 = await engine1.arun(saga_id="s1", interrupt_after=1)
    assert r1.status == "interrupted"
    assert counts == {"a": 1}

    # A brand-new engine, same durable store: resume continues, a not re-run.
    engine2 = Choreography(saga, parts, store=store)
    r2 = await engine2.aresume("s1")
    assert r2.status == "completed"
    assert counts == {"a": 1, "b": 1}
    assert r2.completed_steps == ["a", "b"]


async def test_resume_unknown_saga_raises():
    saga = Saga(name="x").step("a", participant="o", action="do")
    engine = Choreography(saga, {"o": {"do": lambda p: {}}}, store=InMemoryMetadataStore())
    with pytest.raises(ChoreographyError):
        await engine.aresume("missing")


async def test_resume_terminal_saga_is_idempotent():
    store = InMemoryMetadataStore()
    saga = Saga(name="x").step("a", participant="o", action="do")
    calls = {"n": 0}

    def once(payload):
        calls["n"] += 1
        return {}

    engine = Choreography(saga, {"o": {"do": once}}, store=store)
    r1 = await engine.arun(saga_id="done")
    assert r1.status == "completed"
    r2 = await engine.aresume("done")
    assert r2.status == "completed"
    assert calls["n"] == 1  # not re-run


async def test_app_resume_after_restart():
    store_app = _app()
    saga = (
        Saga(name="t")
        .step("a", participant="o", action="do_a")
        .step("b", participant="o", action="do_b")
    )
    calls = {}
    parts = {
        "o": {
            "do_a": lambda p: calls.__setitem__("a", calls.get("a", 0) + 1) or {"x": 1},
            "do_b": lambda p: calls.__setitem__("b", calls.get("b", 0) + 1) or {"y": 2},
        }
    }
    r1 = await store_app.achoreograph(saga, participants=parts, saga_id="ord", interrupt_after=1)
    assert r1.status == "interrupted"
    r2 = await store_app.aresume_choreography(saga, "ord", participants=parts)
    assert r2.status == "completed" and calls == {"a": 1, "b": 1}


# -- journal integrity (offline-verifiable) ------------------------------------


async def test_journal_hash_chain_verifies_offline():
    saga = (
        Saga(name="v")
        .step("a", participant="o", action="do_a")
        .step("b", participant="o", action="do_b")
    )
    parts = {"o": {"do_a": lambda p: {"r": 1}, "do_b": lambda p: {"r": 2}}}
    result = await _app().achoreograph(saga, participants=parts)
    assert result.journal.verify().intact


async def test_journal_tamper_is_detected():
    saga = Saga(name="v").step("a", participant="o", action="do")
    parts = {"o": {"do": lambda p: {"r": 1}}}
    result = await _app().achoreograph(saga, participants=parts)
    journal = SagaJournal.from_record(result.journal.to_record())
    journal.records[0].output = {"r": 999}  # tamper after the fact
    verdict = journal.verify()
    assert not verdict.intact and verdict.broken_at == 0


async def test_journal_signature_verifies_and_tamper_fails():
    signer = HMACSigner("k", key_id="coord")
    saga = Saga(name="v").step("a", participant="o", action="do")
    parts = {"o": {"do": lambda p: {"r": 1}}}
    engine = Choreography(saga, parts, store=InMemoryMetadataStore(), signer=signer)
    result = await engine.arun()
    assert result.journal.verify(signer).intact
    forged = HMACSigner("other", key_id="coord")
    assert not result.journal.verify(forged).intact


# -- per-org governance & A2A fabric -------------------------------------------


async def test_per_org_audit_on_separate_chains():
    coord = _app("coord")
    vendor = _app("vendor")
    server = vendor.serve_choreography({"do": lambda p: {"ok": 1}}, org_id="vendor")
    client = connect_a2a_in_process(server)
    remote = RemoteParticipant(client, org_id="vendor")
    saga = Saga(name="x").step("s", participant="vendor", action="do")
    result = await coord.achoreograph(saga, participants={"vendor": remote})
    assert result.status == "completed"
    # The coordinator audits the handoff on its chain; the vendor audits its
    # execution on its own — no shared control plane.
    assert coord.audit.query(action="choreography_step")
    assert vendor.audit.query(action="choreography_step")
    assert coord.audit.verify_chain() and vendor.audit.verify_chain()


async def test_a2a_parity_with_local():
    counts_local, order_local = _recorder()
    mk = _handlers(counts_local, order_local)["mk"]

    def build_saga():
        return (
            Saga(name="p")
            .step("a", participant="o", action="do_a")
            .step("b", participant="o", action="do_b", build=lambda ctx: {"prev": ctx["a"]["step"]})
        )

    local_parts = {"o": {"do_a": mk("a"), "do_b": mk("b")}}
    local = await _app().achoreograph(build_saga(), participants=local_parts)

    org = _app("org")
    server = org.serve_choreography(
        {"do_a": lambda p: {"step": "a", "payload": p}, "do_b": lambda p: {"step": "b", "payload": p}},
        org_id="o",
    )
    client = connect_a2a_in_process(server)
    remote = RemoteParticipant(client, org_id="o")
    over_a2a = await _app().achoreograph(build_saga(), participants={"o": remote})

    assert local.status == over_a2a.status == "completed"
    assert local.output_of("b")["step"] == over_a2a.output_of("b")["step"] == "b"


async def test_a2a_remote_failure_compensates():
    coord = _app("coord")
    vendor = _app("vendor")
    comp = []
    server = vendor.serve_choreography(
        {
            "reserve": lambda p: {"id": "r1"},
            "release": lambda p: comp.append("released") or {},
            "charge": lambda p: StepOutcome(ok=False, error="card declined"),
        },
        org_id="vendor",
    )
    client = connect_a2a_in_process(server)
    remote = RemoteParticipant(client, org_id="vendor")
    saga = (
        Saga(name="x")
        .step("reserve", participant="vendor", action="reserve", compensation="release")
        .step("charge", participant="vendor", action="charge")
    )
    result = await coord.achoreograph(saga, participants={"vendor": remote})
    assert result.status == "compensated"
    assert result.compensated_steps == ["reserve"]
    assert comp == ["released"]


# -- events & wire envelopes ---------------------------------------------------


async def test_completion_emits_event():
    app = _app()
    seen = []
    app.events.subscribe("choreography.completed", lambda evt: seen.append(evt))
    saga = Saga(name="x").step("a", participant="o", action="do")
    await app.achoreograph(saga, participants={"o": {"do": lambda p: {}}})
    assert seen and seen[0].payload["status"] == "completed"


def test_step_request_outcome_roundtrip():
    req = StepRequest(saga_id="s", step="a", action="do", payload={"n": 1}, contract_id="c1")
    assert StepRequest.from_wire(req.to_wire()) == req
    out = StepOutcome(ok=True, output={"r": 2}, cost_usd=0.1, quality=0.9)
    assert StepOutcome.from_wire(out.to_wire()) == out


async def test_local_participant_unknown_action_is_failed_outcome():
    part = LocalParticipant("o", {"known": lambda p: {}})
    req = StepRequest(saga_id="s", step="a", action="unknown")
    outcome = await part.perform(req)
    assert not outcome.ok and "unknown" in outcome.error


def test_saga_context_accessors():
    ctx = SagaContext(input={"a": 1}, outputs={"step1": {"x": 2}})
    assert ctx.input["a"] == 1
    assert ctx["step1"] == {"x": 2}
    assert ctx.output_of("missing") == {}
