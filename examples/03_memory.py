"""Layered memory & personalization.

A single, runnable tour of Vincio's memory engine: how an agent remembers
and recalls facts across user / agent / session / team scopes, why every
write goes through a guarded policy pipeline (so contradictions supersede
instead of pile up), how confidence decays over time, how bi-temporal
memory lets you ask "what did we believe *then*", how per-memory ACLs keep
team memory on a need-to-know basis, how episodic chatter is consolidated
into durable semantic facts, and how GDPR-style edit/forget/export/erase
flow through a tamper-evident audit log. Runs fully offline.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from vincio.core.utils import utcnow
from vincio.governance.consent import ConsentLedger, Purpose
from vincio.memory import MemoryConsolidator, MemoryEngine
from vincio.retrieval import LocalHashEmbedder
from vincio.security.audit import AuditLog


def banner(title: str) -> None:
    print(f"\n— {title} —")


# 1) SCOPED REMEMBER / RECALL ------------------------------------------------
# Memory is partitioned by scope. The same engine serves a per-user profile,
# a per-agent operating manual, a per-session scratchpad, and a per-team
# shared brain. `for_*` returns a scoped handle so you never have to repeat
# the owner id on every call; recall is scope-isolated by construction.
def scoped_personalization(engine: MemoryEngine) -> None:
    banner("scoped remember / recall (user, agent, session)")
    user = engine.for_user("u1")
    user.remember("User prefers concise technical answers")
    user.remember("User works in the compliance department")

    agent = engine.for_agent("support-bot")
    agent.remember("Agent escalates billing disputes to the finance team")

    session = engine.for_session("s1")
    session.remember("We agreed to migrate the billing stack to Postgres")
    session.remember("The rollout deadline is March 15 next quarter")

    for item in user.recall("how should answers be written", top_k=1):
        print(f"  user:    {item.content}")
    for item in agent.recall("billing dispute handling", top_k=1):
        print(f"  agent:   {item.content}")
    for item in session.recall("database migration", top_k=1):
        print(f"  session: {item.content}")


# 2) THE GUARDED WRITE PIPELINE + CONTRADICTION RESOLUTION -------------------
# Writes are not blind inserts. Every candidate passes a write policy that can
# reject junk, dedupe near-restatements (which instead *confirm* the existing
# memory), and resolve a contradiction by superseding the stale value. The
# higher-confidence, newer fact wins; the old one is archived, not duplicated.
def guarded_writes(engine: MemoryEngine) -> None:
    banner("guarded write pipeline + contradiction resolution")
    stale = engine.remember("User timezone is UTC+1", user_id="u1", confidence=0.6)
    fresh = engine.remember("User timezone is UTC-5", user_id="u1", confidence=0.9)

    # A near-identical restatement does not create a duplicate; it confirms the
    # existing memory (note the returned id matches and confirmations went up).
    again = engine.remember("User timezone is UTC-5", user_id="u1")
    print(f"  superseded old fact archived: {engine.store.get(stale.id).status == 'archived'}")
    print(f"  restatement deduped to same id: {again.id == fresh.id} "
          f"(confirmations={again.confirmations})")
    print(f"  live recall returns winner: "
          f"{engine.recall('what timezone is the user in', user_id='u1', top_k=1)[0].content}")


# 3) CONFIDENCE DECAY --------------------------------------------------------
# Memories lose confidence over time (exponential decay on age). A periodic
# decay pass demotes faded, never-reinforced items toward `decayed`/`archived`
# so recall favors fresh, reinforced knowledge — but importance (usage,
# confirmations, stability) buys an item a longer leash before it fades.
def confidence_decay(engine: MemoryEngine) -> None:
    banner("confidence decay (importance-weighted retention)")
    # Forge an old memory by back-dating it well past the decay horizon (the
    # write policy admits it at a normal confidence; age is what erodes the
    # *effective* confidence), then run the maintenance pass.
    old = engine.remember("User once mentioned a beta feature flag", user_id="u1", confidence=0.6)
    # Decay is measured from `updated_at`; back-date it past the horizon. With
    # decay_lambda=0.01, ~600 days erodes effective confidence below the floor.
    old.updated_at = utcnow() - timedelta(days=600)
    engine.store.put(old)
    report = engine.decay_pass()
    print(f"  decay pass: {report}")
    print(f"  faded item status now: {engine.store.get(old.id).status}")


# 4) BI-TEMPORAL MEMORY: valid_from / valid_to + as-of recall + correct() ----
# Memory records two timelines: when the system *learned* a fact and when the
# fact was *true in the world* (valid_from / valid_to). `correct()` closes the
# old fact's validity interval and opens a new one — so an as-of recall before
# the correction still returns what we believed true *then*. This is how you
# answer "what was the user's plan last quarter?" without losing history.
def bitemporal_memory(engine: MemoryEngine) -> None:
    banner("bi-temporal memory: valid_from / valid_to + as-of recall + correct()")
    t0 = utcnow() - timedelta(days=90)
    original = engine.remember(
        "User is on the Basic plan", user_id="u1", confidence=0.9, valid_from=t0
    )
    # 30 days ago the user upgraded. correct() preserves the old value.
    upgrade_moment = utcnow() - timedelta(days=30)
    engine.correct(original.id, "User is on the Pro plan", valid_from=upgrade_moment)

    now_hits = engine.recall("what plan is the user on", user_id="u1", top_k=1)
    past_hits = engine.recall(
        "what plan is the user on",
        user_id="u1",
        top_k=1,
        as_of=utcnow() - timedelta(days=60),  # before the upgrade
    )
    print(f"  as-of now:        {now_hits[0].content if now_hits else '(none)'}")
    print(f"  as-of 60d ago:    {past_hits[0].content if past_hits else '(none)'}")


# 5) PER-MEMORY ACLs ---------------------------------------------------------
# Team-shared memory lives under one team owner, but each item can carry an
# ACL listing who may recall it. Recall passes a `reader`; an item with a
# populated ACL surfaces only to listed members, while un-ACL'd team memory
# stays visible to everyone on the team. Need-to-know, enforced at recall.
def per_memory_acls(engine: MemoryEngine) -> None:
    banner("per-memory ACLs on team-shared memory")
    engine.remember(
        "Q3 roadmap: ship the audit-log export",
        team_id="team-eng",  # readable by the whole team (no ACL)
    )
    engine.remember(
        "Severance terms for the contractor offboarding",
        team_id="team-eng",
        acl=["alice", "manager"],  # restricted
        privacy_class="confidential",
    )
    # alice is on the ACL, so she sees both; bob sees only the un-ACL'd roadmap.
    for reader in ("alice", "bob"):
        hits = engine.recall(
            "roadmap offboarding severance", team_id="team-eng", reader=reader, top_k=5
        )
        print(f"  reader={reader:<6} sees: {[h.content[:30] for h in hits]}")


# 6) EPISODIC -> SEMANTIC CONSOLIDATION --------------------------------------
# Session chatter is *episodic* (cheap, transient). Consolidation reviews a
# session and promotes durable conclusions into *semantic* user memory with
# provenance back to the episodes they came from — the same move a human makes
# turning "what we just discussed" into "what I now know about this user".
async def consolidation(engine: MemoryEngine) -> None:
    banner("episodic -> semantic consolidation with provenance")
    report = await engine.consolidate("s1", user_id="u1")
    print(f"  examined={report.examined} promoted={report.promoted} archived={report.archived}")
    for item in report.items[:2]:
        origin = item.metadata.get("consolidated_from", "?")
        print(f"  promoted [{item.scope.value}/{item.type.value}]: {item.content[:60]}")
        print(f"            provenance consolidated_from={origin}")
    # Dedup is part of every consolidation pass, but can also run standalone.
    merged = MemoryConsolidator(engine).dedup(owner_id="u1")
    print(f"  standalone dedup merged {merged} near-duplicate(s)")


# 7) AUDITED GDPR EDIT / FORGET / EXPORT / ERASE -----------------------------
# Every privacy-relevant mutation is recorded on a hash-chained audit log, so
# you can prove what happened and that the record was not tampered with. This
# demonstrates the data-subject rights: rectification (edit), erasure of a
# single memory (forget), portability (export), and full erasure of a subject.
# A ConsentLedger additionally drops any memory whose purpose lost consent.
def gdpr_lifecycle(engine: MemoryEngine, audit: AuditLog, consent: ConsentLedger) -> None:
    banner("audited GDPR edit / forget / export / erase")

    # Bind a memory to a purpose the subject consented to, then withdraw it.
    consent.grant("u9", [Purpose.PERSONALIZATION])
    engine.remember(
        "User u9 likes dark mode",
        user_id="u9",
        purpose=Purpose.PERSONALIZATION.value,
    )
    before = engine.recall("ui preferences", user_id="u9", top_k=3)
    consent.revoke("u9", purpose=Purpose.PERSONALIZATION)
    after = engine.recall("ui preferences", user_id="u9", top_k=3)
    print(f"  consent-gated recall: before={len(before)} after-revoke={len(after)}")

    # Rectification: edit re-passes the write policy and is audited.
    target = engine.remember("User u9 email is old@example.com", user_id="u9")
    engine.edit(target.id, content="User u9 email is new@example.com")
    print(f"  rectified -> {engine.store.get(target.id).content}")

    # Portability: every stored memory for the subject as plain dicts.
    exported = engine.export_owner_data("u9")
    print(f"  exported {len(exported)} memorie(s) for u9")

    # Erasure of one memory, then the whole subject.
    engine.forget(target.id, reason="user_request")
    erased = engine.erase_owner_data("u9")
    print(f"  forgot 1 + erased remaining {erased} for u9")

    actions = sorted({entry.action for entry in audit.entries})
    print(f"  audit actions: {actions}")
    print(f"  audit chain valid: {audit.verify_chain()}")


async def main() -> None:
    # An audit log (in-memory, directory=None) and a consent ledger wire the
    # engine for governance; the LocalHashEmbedder gives deterministic offline
    # vector recall; session memory expires after 30 days unless consolidated.
    audit = AuditLog(directory=None)
    consent = ConsentLedger(audit=audit)
    engine = MemoryEngine(
        embedder=LocalHashEmbedder(),
        ttl_days={"session": 30.0},
        audit=audit,
        consent_ledger=consent,
    )

    scoped_personalization(engine)
    guarded_writes(engine)
    confidence_decay(engine)
    bitemporal_memory(engine)
    per_memory_acls(engine)
    await consolidation(engine)
    gdpr_lifecycle(engine, audit, consent)

    print(f"\nmemory stats: {engine.stats()}")


if __name__ == "__main__":
    asyncio.run(main())
