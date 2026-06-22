"""Cross-org workflow discovery — bind the best counterparty for a step at run time.

Vincio runs durable, compensating sagas across organizations (example 68), under
negotiated contracts (67), settled and reconciled afterwards (69). This example
adds the next rung: a saga step that declares the **capability** it needs and lets
the engine **resolve the participant at dispatch time** from the governed agent
directory — ranked by reputation and prior settlement fit — rather than a hard-coded
org id. Discovery changes *who* runs a step; the allow-list, contract, per-org
audit, compensation, and durability are exactly those of a statically-wired step.

Five steps, all offline and deterministic:

  1. Discovered binding: a capability step binds the best-available allowed vendor
     and runs — the binding decision is recorded on the saga journal.
  2. Reputation ranks: a regressing vendor is discounted without being singled out,
     so the reliable one wins a close race.
  3. Governed: a vendor that advertises the capability but is not allow-listed is
     never bound, and the governed resolution lands on the audit chain.
  4. Same governance: a discovered step under a contract is enforced, and a breach
     compensates exactly as a static step would.
  5. Over the A2A fabric: discovery binds a remote participant identically.

This is a library capability inside your process, never a hosted matching service.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp
from vincio.a2a import connect_a2a_in_process
from vincio.a2a.protocol import AgentCard, AgentSkill
from vincio.choreography import RemoteParticipant, Saga, StepOutcome
from vincio.negotiation import Contract, ContractTerms
from vincio.providers import MockProvider


def _vendor_card(name: str, capability: str = "transcription") -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} — performs {capability}",
        skills=[AgentSkill(id="run", name="run", description=capability, tags=[capability])],
    )


async def main() -> None:
    app = ContextApp(name="acme", provider=MockProvider(default_text="ok"), model="mock-1")
    app.use_reputation_ledger()

    # Two vendors advertise "transcription" in a governed directory (allow-listed).
    directory = app.agent_directory(allow=["vendor-*"])
    directory.register(_vendor_card("vendor-a"))
    directory.register(_vendor_card("vendor-b"))

    # vendor-b has a worse track record against the no-regression gate.
    app.reputation_ledger.record_outcome("vendor-a", passed=True, round_id="r1")
    app.reputation_ledger.record_outcome("vendor-b", passed=False, round_id="r1")
    app.reputation_ledger.record_outcome("vendor-b", passed=False, round_id="r2")

    def vendor(org: str):
        return {
            "run": lambda p: {"text": f"transcribed by {org}"},
            "discard": lambda p: {"discarded": org},
        }

    participants = {"vendor-a": vendor("vendor-a"), "vendor-b": vendor("vendor-b")}

    # 1 + 2. A discovered step binds the best-ranked allowed vendor ---------
    saga = Saga(name="job").step(
        "transcribe", action="run", capability="transcription", compensation="discard"
    )
    result = app.choreograph(saga, participants=participants, directory=directory)
    binding = result.bindings["transcribe"]
    print(f"1. status={result.status}; bound {binding.capability!r} → {binding.org}")
    print(
        "2. ranked: "
        + ", ".join(f"{c.org}(score={c.score:.3f})" for c in binding.candidates)
    )

    # 3. Governance: an unlisted vendor is never bound ----------------------
    gov = ContextApp(name="gov", provider=MockProvider(default_text="ok"), model="mock-1")
    gdir = gov.agent_directory(allow=["vendor-a"])  # only vendor-a is allowed
    gdir.register(_vendor_card("vendor-a"))
    gdir.register(_vendor_card("vendor-evil"))
    gres = gov.choreograph(
        Saga(name="job").step("transcribe", action="run", capability="transcription"),
        participants={"vendor-a": vendor("vendor-a"), "vendor-evil": vendor("vendor-evil")},
        directory=gdir,
    )
    evil = next(c for c in gres.bindings["transcribe"].candidates if c.org == "vendor-evil")
    print(
        f"3. bound={gres.bindings['transcribe'].org}; "
        f"vendor-evil allowed={evil.allowed} ({evil.rejected_reason}); "
        f"resolutions audited={bool(gov.audit.query(action='agent_resolve'))}"
    )

    # 4. Same governance: a discovered step's contract breach compensates ---
    deal = Contract(
        buyer="acme", seller="*", terms=ContractTerms(scope="transcribe", price_usd=0.10)
    ).seal()
    comp: list[str] = []
    breach_parts = {
        "warehouse": {"prep": lambda p: {"ready": True}, "undo": lambda p: comp.append("prep") or {}},
        "vendor-a": {"run": lambda p: StepOutcome(ok=True, cost_usd=0.50)},  # 5× over the price
    }
    breach_saga = (
        Saga(name="contracted")
        .step("prep", action="prep", participant="warehouse", compensation="undo")
        .step("transcribe", action="run", capability="transcription", contract=deal)
    )
    breached = app.choreograph(breach_saga, participants=breach_parts, directory=directory)
    failed = [r for r in breached.journal.forward_records() if r.status == "failed"][0]
    print(
        f"4. contract breach → status={breached.status}; "
        f"compensated={comp}; breach={failed.breaches}"
    )

    # 5. Over the A2A fabric, discovery binds a remote participant ----------
    coord = ContextApp(name="coord", provider=MockProvider(default_text="ok"), model="mock-1")
    remote_org = ContextApp(name="remote", provider=MockProvider(default_text="ok"), model="mock-1")
    rdir = coord.agent_directory(allow=["vendor-a"])
    rdir.register(_vendor_card("vendor-a"))
    server = remote_org.serve_choreography({"run": lambda p: {"text": "remote"}}, org_id="vendor-a")
    remote = RemoteParticipant(connect_a2a_in_process(server), org_id="vendor-a")
    over_a2a = await coord.achoreograph(
        Saga(name="job").step("transcribe", action="run", capability="transcription"),
        participants={"vendor-a": remote},
        directory=rdir,
    )
    print(
        f"5. A2A discovery status={over_a2a.status}; "
        f"bound={over_a2a.bindings['transcribe'].org}; "
        f"output={over_a2a.output_of('transcribe')}"
    )


if __name__ == "__main__":
    asyncio.run(main())
