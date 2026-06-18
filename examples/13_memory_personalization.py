"""Memory & personalization: scoped remember/recall over user, agent,
and session memory, hybrid vector+graph recall, episodic→semantic
consolidation with provenance, forgetting & GDPR-style hygiene through the
audit log, and the memory eval harness — fully offline.
"""

import asyncio

from vincio.memory import (
    MemoryConsolidator,
    MemoryEngine,
    evaluate_memory,
    personalization_dataset,
)
from vincio.retrieval import LocalHashEmbedder
from vincio.security.audit import AuditLog


def scoped_personalization(engine: MemoryEngine) -> None:
    print("— scoped remember/recall (user, agent, session) —")
    user = engine.for_user("u1")
    user.remember("User prefers concise technical answers")
    user.remember("User works in the compliance department")
    agent = engine.for_agent("support-bot")
    agent.remember("Agent escalates billing disputes to the finance team")
    session = engine.for_session("s1")
    session.remember("We agreed to migrate the billing stack to Postgres")
    session.remember("The rollout deadline is March 15 next quarter")

    for item in user.recall("how should answers be written", top_k=2):
        print(f"  user recall:  {item.content}")
    for item in agent.recall("billing dispute handling", top_k=1):
        print(f"  agent recall: {item.content}")


def hybrid_recall(engine: MemoryEngine) -> None:
    print("\n— hybrid recall: lexical + vector + graph in one query —")
    engine.remember(
        "User decided to consolidate Acme invoices quarterly",
        user_id="u1",
        entities=["Acme"],
    )
    results = engine.search("invoicing cadence", user_id="u1", task_entities=["Acme"], top_k=1)
    for result in results:
        parts = {k: result.components[k] for k in ("lexical", "vector", "graph")}
        print(f"  {result.item.content}\n  components: {parts}")


async def consolidation(engine: MemoryEngine) -> None:
    print("\n— episodic→semantic consolidation with provenance —")
    report = await engine.consolidate("s1", user_id="u1")
    print(f"  examined={report.examined} promoted={report.promoted} archived={report.archived}")
    for item in report.items:
        print(f"  promoted [{item.scope.value}/{item.type.value}]: {item.content[:80]}")
        print(f"  provenance: consolidated_from={item.metadata['consolidated_from']}")


def hygiene(engine: MemoryEngine, audit: AuditLog) -> None:
    print("\n— forgetting & GDPR-style hygiene through the audit log —")
    stale = engine.remember("User timezone is UTC+1", user_id="u1", confidence=0.6)
    engine.remember("User timezone is UTC-5", user_id="u1", confidence=0.9)
    print(f"  contradiction superseded: {engine.store.get(stale.id).status == 'archived'}")
    exported = engine.export_owner_data("u1")
    print(f"  exported {len(exported)} memorie(s) for u1")
    target = exported[0]["id"]
    engine.forget(target, reason="user_request")
    print(f"  forgot {target}")
    print(f"  decay pass: {engine.decay_pass()}")
    actions = sorted({entry.action for entry in audit.entries})
    print(f"  audit actions: {actions} (chain valid: {audit.verify_chain()})")


async def eval_harness(engine: MemoryEngine) -> None:
    print("\n— memory eval harness (VincioBench `memory` family) —")
    engine.remember("User prefers detailed walkthroughs", user_id="u2")
    engine.remember("User timezone is UTC-5", user_id="u3", confidence=0.9)
    report = await evaluate_memory(engine, personalization_dataset(), top_k=3)
    for name, value in report.metrics.items():
        print(f"  {name}: {value}")


async def main() -> None:
    audit = AuditLog(directory=None)
    engine = MemoryEngine(
        embedder=LocalHashEmbedder(),
        ttl_days={"session": 30.0},
        audit=audit,
    )
    scoped_personalization(engine)
    hybrid_recall(engine)
    await consolidation(engine)
    hygiene(engine, audit)
    await eval_harness(engine)
    # Deduplication is part of every consolidation pass; it can also run alone:
    merged = MemoryConsolidator(engine).dedup(owner_id="u1")
    print(f"\n— standalone dedup pass merged {merged} near-duplicate(s) —")
    print(f"memory stats: {engine.stats()}")


if __name__ == "__main__":
    asyncio.run(main())
