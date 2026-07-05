"""Layered memory & personalization.

How an agent remembers and recalls across user / agent / session / team scopes,
why every write passes a guarded policy (so contradictions supersede instead of
piling up), how confidence decays with age, how bi-temporal memory answers "what
did we believe *then*", how per-memory ACLs enforce need-to-know, how episodic
chatter consolidates into durable semantic facts, and how GDPR edit/forget/
export/erase flow through a tamper-evident audit log. Runs fully offline.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from vincio.core.utils import utcnow
from vincio.governance.consent import ConsentLedger, Purpose
from vincio.memory import MemoryConsolidator, MemoryEngine
from vincio.retrieval import LocalHashEmbedder
from vincio.security.audit import AuditLog


async def main() -> None:
    # In-memory audit + consent wire the engine for governance; LocalHashEmbedder
    # gives deterministic offline vector recall; session memory expires in 30 days.
    audit = AuditLog(directory=None)
    consent = ConsentLedger(audit=audit)
    engine = MemoryEngine(embedder=LocalHashEmbedder(), ttl_days={"session": 30.0},
                          audit=audit, consent_ledger=consent)

    # 1. Memory is partitioned by scope. for_*() returns a scoped handle so you
    #    never repeat the owner id, and recall is scope-isolated by construction:
    #    a per-user profile, per-agent operating manual, per-session scratchpad.
    engine.for_user("u1").remember("User prefers concise technical answers")
    engine.for_agent("support-bot").remember("Agent escalates billing disputes to finance")
    engine.for_session("s1").remember("We agreed to migrate the billing stack to Postgres")
    print("1. scoped recall:",
          engine.for_user("u1").recall("how should answers be written", top_k=1)[0].content)

    # 2. Writes are not blind inserts. Every candidate passes a write policy that
    #    rejects junk, DEDUPES a near-restatement (which instead *confirms* the
    #    existing memory), and resolves a contradiction by SUPERSEDING the stale
    #    value — the higher-confidence, newer fact wins; the old one is archived.
    stale = engine.remember("User timezone is UTC+1", user_id="u1", confidence=0.6)
    fresh = engine.remember("User timezone is UTC-5", user_id="u1", confidence=0.9)
    again = engine.remember("User timezone is UTC-5", user_id="u1")  # restatement, not a dup
    print(f"2. guarded write: stale archived={engine.store.get(stale.id).status == 'archived'},"
          f" restatement deduped={again.id == fresh.id} (confirmations={again.confirmations}),"
          f" winner={engine.recall('user timezone', user_id='u1', top_k=1)[0].content!r}")

    # 3. Confidence decays exponentially with age. Back-date a memory past the
    #    horizon and a maintenance pass demotes the faded, never-reinforced item —
    #    recall then favours fresh, reinforced knowledge. (Importance buys a leash.)
    old = engine.remember("User once mentioned a beta feature flag", user_id="u1", confidence=0.6)
    old.updated_at = utcnow() - timedelta(days=600)  # decay is measured from updated_at
    engine.store.put(old)
    engine.decay_pass()
    print("3. decay: faded item status now", engine.store.get(old.id).status)

    # 4. Bi-temporal memory records two timelines: when we LEARNED a fact and when
    #    it was TRUE in the world (valid_from/valid_to). correct() closes the old
    #    interval and opens a new one, so an as_of recall before the correction
    #    still returns what we believed true then — history is never lost.
    t0 = utcnow() - timedelta(days=90)
    original = engine.remember("User is on the Basic plan", user_id="u1", confidence=0.9, valid_from=t0)
    engine.correct(original.id, "User is on the Pro plan", valid_from=utcnow() - timedelta(days=30))
    now = engine.recall("what plan is the user on", user_id="u1", top_k=1)
    past = engine.recall("what plan is the user on", user_id="u1", top_k=1,
                         as_of=utcnow() - timedelta(days=60))  # before the upgrade
    print(f"4. bi-temporal: now={now[0].content!r} vs 60d-ago={past[0].content!r}")

    # 5. Team memory lives under one team owner, but each item can carry an ACL of
    #    who may recall it. An ACL'd item surfaces only to listed readers; un-ACL'd
    #    team memory stays visible to everyone. Need-to-know, enforced at recall.
    engine.remember("Q3 roadmap: ship the audit-log export", team_id="team-eng")  # open
    engine.remember("Severance terms for the contractor offboarding", team_id="team-eng",
                    acl=["alice", "manager"], privacy_class="confidential")  # restricted
    for reader in ("alice", "bob"):
        hits = engine.recall("roadmap offboarding severance", team_id="team-eng", reader=reader, top_k=5)
        print(f"5. ACL reader={reader:<6} sees {[h.content[:24] for h in hits]}")

    # 6. Session chatter is episodic (cheap, transient). Consolidation promotes
    #    durable conclusions into semantic user memory WITH provenance back to the
    #    episodes — the human move of turning "what we discussed" into "what I know".
    report = await engine.consolidate("s1", user_id="u1")
    merged = MemoryConsolidator(engine).dedup(owner_id="u1")
    swept = await engine.promote_aged_episodes(min_age_days=7.0, user_id="u1")  # periodic sweep
    print(f"6. consolidation: promoted={report.promoted} deduped={merged} aged-sweep={len(swept)}")

    # 7. Every privacy-relevant mutation is recorded on the hash-chained audit log,
    #    so you can prove what happened and that it was not tampered with. This is
    #    the data-subject rights surface: consent gating, rectify, forget, export,
    #    erase — and a ConsentLedger drops memory whose purpose lost consent.
    consent.grant("u9", [Purpose.PERSONALIZATION])
    engine.remember("User u9 likes dark mode", user_id="u9", purpose=Purpose.PERSONALIZATION.value)
    before = len(engine.recall("ui preferences", user_id="u9", top_k=3))
    consent.revoke("u9", purpose=Purpose.PERSONALIZATION)
    after = len(engine.recall("ui preferences", user_id="u9", top_k=3))
    target = engine.remember("User u9 email is old@example.com", user_id="u9")
    engine.edit(target.id, content="User u9 email is new@example.com")  # rectify (re-audited)
    exported = engine.export_owner_data("u9")  # portability
    engine.forget(target.id, reason="user_request")  # erase one
    erased = engine.erase_owner_data("u9")  # erase the subject
    print(f"7. GDPR: consent recall {before}->{after}, exported={len(exported)}, erased={1 + erased}")
    print(f"   audit actions {sorted({e.action for e in audit.entries})}, chain valid={audit.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
