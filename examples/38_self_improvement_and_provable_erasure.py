"""The 3.0 breaking culmination — one self-improvement contract, provable
erasure & the async-canonical core (3.0).

The 3.0 surface, all offline on the deterministic mock:

  1. Unified self-improvement contract: one ``SelfImprovementPolicy`` composes
     proposal, meta-optimization (successive-halving + learned weights),
     active-learning, and canary/rollback. ``app.self_improvement(...).astream()``
     emits the cycle as observe → proposal → meta → reeval → canary →
     promote/rollback events — every promotion on the same gated path the loop
     always used.
  2. Canary-gated deploy: ``app.deploy(candidate, dataset=...)`` promotes a
     prompt/policy live only if it clears a no-regression canary verdict (offline
     gated comparison); a failing gate rolls back to the last known-good version.
     The live form — ``app.deploy(live_inputs=..., score_fn=...)`` — ramps a
     fraction of real runs onto the candidate and auto-rolls-back a regression.
  3. Provable erasure: ``app.erase_source(...)`` emits a signed, content-bound
     ``ErasureProof`` over the exact removed-id set across indexes, memory, and
     generated artifacts — erasure that *verifies*, not merely logs.
  4. Consent & purpose: a ``ConsentLedger`` binds data to a GDPR purpose and
     lawful basis; withdrawing consent stops the affected memories surfacing.
  5. Bi-temporal memory: a corrected fact closes its valid interval, so as-of
     recall still returns what was believed true then; per-memory ACLs gate
     team-shared memory.

Everything is opt-in behind ``@experimental(since="3.0")`` on the frozen 2.0
surface; the flat ``app.<method>`` API stays fully supported.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp
from vincio.evals import Dataset, EvalCase
from vincio.governance import (
    ConsentLedger,
    HmacSigner,
    LawfulBasis,
    Purpose,
    verify_erasure_proof,
)
from vincio.memory import MemoryEngine
from vincio.optimize import CanarySpec, MetaSpec, SelfImprovementPolicy


def banner(title: str) -> None:
    print(f"\n== {title} ==")


def build_app(tmp) -> ContextApp:
    provider, model = example_provider(
        citing_responder("The Pro plan refund window is 30 days. [{ref}]")
    )
    app = ContextApp(name="self_improving", provider=provider, model=model)
    app.add_source("docs", path=str(write_sample_docs(tmp / "docs")), retrieval="bm25")
    return app


DATASET = Dataset(
    name="refunds",
    cases=[
        EvalCase(id=f"c{i}", input=q, expected="The Pro plan refund window is 30 days.")
        for i, q in enumerate(
            [
                "What is the refund window for the Pro plan?",
                "How long do Pro customers have to request a refund?",
                "Within how many days can a Pro plan be refunded?",
                "Pro plan refund period?",
                "Refund window for Pro?",
                "How many days to refund a Pro subscription?",
            ]
        )
    ],
)


async def self_improvement_contract(tmp) -> None:
    banner("1. Unified self-improvement contract (one streaming controller)")
    app = build_app(tmp)
    policy = SelfImprovementPolicy(
        metrics=["lexical_overlap", "cost", "latency"],
        meta=MetaSpec(strategies=["evolution"], budgets=[4]),
        canary=CanarySpec(metric="lexical_overlap"),
    )
    controller = app.self_improvement(policy, dataset=DATASET)
    async for event in controller.astream():
        print(f"  {event.phase:9s} | {event.action or '-':10s} | {event.reason[:64]}")
    print(f"  budget spent: {controller.events[-1].budget_spent}")


async def canary_gated_deploy(tmp) -> None:
    banner("2. Canary-gated deploy (promote only if it clears the verdict)")
    app = build_app(tmp)
    result = app.deploy(
        app.prompt_spec, dataset=DATASET, canary=CanarySpec(metric="lexical_overlap")
    )
    print(f"  deployed={result.deployed} ref={result.ref}")
    print(f"  verdict: {result.verdict.reason}")
    refused = app.deploy(
        app.prompt_spec,
        dataset=DATASET,
        canary=CanarySpec(metric="lexical_overlap"),
        gates={"lexical_overlap": ">= 0.999"},
    )
    print(f"  unreachable-gate deploy -> deployed={refused.deployed}, "
          f"rolled_back_to={refused.rolled_back_to}")


def live_canary_deploy(tmp) -> None:
    banner("2b. Live-traffic canary (ramp real runs, auto-rollback)")
    from vincio.prompts.compiler import CompilerOptions
    from vincio.prompts.optimizers import PromptVariant

    app = build_app(tmp)
    # The candidate renders XML; the citing mock answers either way, so a simple
    # length-based score stands in for an online quality signal here.
    candidate = PromptVariant(
        name="xml", spec=app.prompt_spec, compiler_options=CompilerOptions(format="xml")
    )
    score_fn = lambda result: float(len(str(result.output)) > 0)  # noqa: E731
    result = app.deploy(
        candidate,
        live_inputs=["refund window?"] * 12,   # a sampled live stream
        score_fn=score_fn,
        canary=CanarySpec(metric="answered", percent=50.0, min_samples=4),
    )
    print(f"  ramped {12} live runs at 50% -> deployed={result.deployed}")
    print(f"  verdict: {result.verdict.reason}")


def provable_erasure(tmp) -> None:
    banner("3. Provable erasure (signed, content-bound manifest)")
    app = build_app(tmp)
    app.content_signer = HmacSigner("erasure-secret", key_id="erase")
    app.lineage.record_artifact("docs", "reports/board-memo.pdf")  # a generated deliverable
    result = app.erase_source("docs")
    proof = result.proof
    print(f"  removed: chunks={result.chunks_removed} artifacts={result.artifacts_removed}")
    print(f"  proof signed={proof.signature is not None} sha256={proof.content_sha256[:16]}…")
    print(f"  verifies with the signing key: {verify_erasure_proof(proof, signer=app.content_signer)}")
    print(f"  verifies with a wrong key:     {verify_erasure_proof(proof, signer=HmacSigner('nope'))}")


def consent_and_purpose() -> None:
    banner("4. Consent & purpose (withdraw consent → recall stops)")
    ledger = ConsentLedger()
    ledger.grant("u1", [Purpose.PERSONALIZATION], lawful_basis=LawfulBasis.CONSENT)
    engine = MemoryEngine(consent_ledger=ledger)
    engine.write_fact(
        "User prefers concise answers", scope="user", owner_id="u1",
        type="preference", purpose="personalization",
    )
    print(f"  recall while consented: {[m.content for m in engine.recall('style', user_id='u1')]}")
    ledger.revoke("u1")
    print(f"  recall after withdrawal: {[m.content for m in engine.recall('style', user_id='u1')]}")


def bitemporal_memory() -> None:
    banner("5. Bi-temporal memory (as-of recall) + per-memory ACL")
    engine = MemoryEngine()
    moved_in = datetime(2026, 1, 1, tzinfo=UTC)
    located = engine.write_fact("User lives in Berlin", scope="user", owner_id="u1", valid_from=moved_in)
    engine.correct(located.id, "User lives in Munich", valid_from=datetime(2026, 3, 1, tzinfo=UTC))
    now = [m.content for m in engine.recall("where does the user live", user_id="u1")]
    feb = [
        m.content
        for m in engine.recall(
            "where does the user live", user_id="u1", as_of=datetime(2026, 2, 1, tzinfo=UTC)
        )
    ]
    print(f"  current recall: {now}")
    print(f"  as-of Feb 2026: {feb}")
    team = engine.for_team("eng")
    team.remember("Rotated the prod deploy key", acl=["alice"])
    print(f"  team recall (alice): {bool(engine.recall('deploy key', team_id='eng', reader='alice'))}")
    print(f"  team recall (bob):   {bool(engine.recall('deploy key', team_id='eng', reader='bob'))}")


async def main() -> None:
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    await self_improvement_contract(tmp)
    await canary_gated_deploy(tmp)
    live_canary_deploy(tmp)
    provable_erasure(tmp)
    consent_and_purpose()
    bitemporal_memory()
    print("\nFewer, truer abstractions: one self-improvement contract, provable erasure, "
          "an async-canonical core.")


if __name__ == "__main__":
    asyncio.run(main())
