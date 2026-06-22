"""Cross-org workflow choreography — a durable, compensating saga across orgs.

Vincio lets agents discover, negotiate, and contract across organizations over the
A2A fabric. This example adds the next rung: the **durable work** they coordinate —
a long-running, compensating workflow that spans more than one organization's agent
fabric. Each org governs and audits its own steps on its own hash-chained chain;
only a typed contract and audited handoffs cross a trust boundary; and a failure on
one side triggers deterministic compensation across the whole choreography.

Five steps, all offline and deterministic:

  1. Forward: an ordered cross-org saga dispatches each step to a participant org
     and completes them in order — the choreography analogue of a durable graph.
  2. Compensation: a failure unwinds the already-completed steps in reverse order,
     so a half-completed cross-org transaction rolls back cleanly.
  3. Durable: the saga journal is checkpointed after every step, so a fresh engine
     resumes it after a restart and never re-runs a completed step.
  4. Verifiable: the hash-chained journal verifies offline from the bytes alone;
     a tampered record is caught.
  5. Fabric: a participant reached over A2A runs the same as a local one, and each
     side audits its own steps on its own chain — no shared control plane.

Everything here is opt-in and additive; this is a library capability inside your
process, never a hosted control plane.
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp
from vincio.a2a import connect_a2a_in_process
from vincio.choreography import Choreography, RemoteParticipant, Saga, SagaJournal, StepOutcome
from vincio.providers import MockProvider
from vincio.storage.base import InMemoryMetadataStore


def build_saga() -> Saga:
    # Reserve stock, charge the card, then ship — each step names the org that
    # performs it and the compensation that undoes it on rollback.
    return (
        Saga(name="fulfil-order")
        .step("reserve", participant="warehouse", action="reserve", compensation="release")
        .step("charge", participant="payments", action="charge", compensation="refund")
        .step("ship", participant="warehouse", action="ship")
    )


async def main() -> None:
    app = ContextApp(name="acme", provider=MockProvider(default_text="ok"), model="mock-1")

    # 1. Forward path -------------------------------------------------------
    log: list[str] = []
    healthy = {
        "warehouse": {
            "reserve": lambda p: log.append("reserve") or {"ticket": "WH-1"},
            "release": lambda p: log.append("release") or {"released": True},
            "ship": lambda p: log.append("ship") or {"tracking": "TRK-9"},
        },
        "payments": {
            "charge": lambda p: log.append("charge") or {"receipt": "PAY-7"},
            "refund": lambda p: log.append("refund") or {"refunded": True},
        },
    }
    ok = await app.achoreograph(build_saga(), participants=healthy, input={"sku": "A1"})
    print(f"1. status={ok.status} completed={ok.completed_steps}")

    # 2. Compensation: a failure unwinds the completed steps in reverse -----
    comp: list[str] = []
    failing = {
        "warehouse": {
            "reserve": lambda p: {"ticket": "WH-2"},
            "release": lambda p: comp.append("release") or {},
            "ship": lambda p: StepOutcome(ok=False, error="carrier unavailable"),
        },
        "payments": {
            "charge": lambda p: {"receipt": "PAY-8"},
            "refund": lambda p: comp.append("refund") or {},
        },
    }
    rolled = await app.achoreograph(build_saga(), participants=failing)
    print(
        f"2. ship failed → status={rolled.status} "
        f"compensated={rolled.compensated_steps} (reverse order: {comp})"
    )

    # 3. Durable: interrupt, then resume on a fresh engine sharing the store -
    store = InMemoryMetadataStore()
    runs: dict[str, int] = {}

    def counted(name: str):
        return lambda p: runs.__setitem__(name, runs.get(name, 0) + 1) or {"step": name}

    two = Saga(name="two").step("a", participant="o", action="do_a").step(
        "b", participant="o", action="do_b"
    )
    parts = {"o": {"do_a": counted("a"), "do_b": counted("b")}}
    paused = await Choreography(two, parts, store=store).arun(saga_id="ord", interrupt_after=1)
    resumed = await Choreography(two, parts, store=store).aresume("ord")  # fresh engine, same store
    print(
        f"3. paused={paused.status} → resumed={resumed.status}; "
        f"step 'a' ran {runs['a']}x (never re-run on resume)"
    )

    # 4. Offline-verifiable journal -----------------------------------------
    print(f"4. journal verifies offline={ok.journal.verify().intact}")
    tampered = SagaJournal.from_record(ok.journal.to_record())
    tampered.records[0].output = {"forged": True}
    print(f"   tampered journal verifies={tampered.verify().intact}")

    # 5. Over the A2A fabric, per-org governance ----------------------------
    coord = ContextApp(name="coord", provider=MockProvider(default_text="ok"), model="mock-1")
    vendor = ContextApp(name="vendor", provider=MockProvider(default_text="ok"), model="mock-1")
    server = vendor.serve_choreography(
        {"transcribe": lambda p: {"text": "..."}}, org_id="vendor"
    )
    client = connect_a2a_in_process(server)
    remote = RemoteParticipant(client, org_id="vendor")
    saga = Saga(name="remote").step("transcribe", participant="vendor", action="transcribe")
    over_a2a = await coord.achoreograph(saga, participants={"vendor": remote})
    print(
        f"5. A2A saga status={over_a2a.status}; "
        f"coordinator chain intact={coord.audit.verify_chain()}, "
        f"vendor chain intact={vendor.audit.verify_chain()} (separate chains, no shared plane)"
    )


if __name__ == "__main__":
    asyncio.run(main())
