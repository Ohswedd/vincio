"""Differential-privacy memory & training.

The federated round (example 59) bounds a *single member's per-round influence*
with clipping and a Gaussian mechanism. But a single bounded round is not a
guarantee about a *subject* whose data is touched again and again — by every memory
consolidation that folds their episodes into a durable summary, by every federated
contribution that learns from their traffic. This example shows the rung above it:
a **provable, composing, per-subject privacy budget** over memory consolidation and
the whole learning loop.

A Rényi / moments accountant tracks the cumulative ``(ε, δ)`` a subject's data has
spent. Privacy composes across rounds far more tightly than naively adding each
step's ``ε``, and a budget gates a learning step the way the cost report gates a
dollar: a consolidation or contribution that would exceed the subject's remaining
budget is **refused** (a hard cap) or **down-weighted** (clipped harder so its
sensitivity, and therefore its privacy cost, fits). Five steps, all offline and
deterministic:

  1. Account: each memory consolidation of a subject's episodes composes their
     privacy budget; the cumulative ε grows but stays under the naive sum.
  2. Refuse: once the budget is spent, the next consolidation is refused outright —
     the subject's episodes simply stay in their short-lived episodic form.
  3. Federate: a federated contribution learning from the same subject composes the
     *same* budget — the accountant spans memory and the learning loop.
  4. Down-weight: a budget set to down-weight admits a clipped-harder release that
     lands within the ceiling instead of refusing.
  5. Report: a per-subject privacy report sits alongside the cost report, and every
     spend and refusal is on the verifiable audit chain.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    FederatedPolicy,
    PrivacyBudget,
    PrivacyBudgetError,
    PrivacyMechanism,
    VincioConfig,
)
from vincio.core.types import MemoryScope, MemoryType
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.optimize.federated import PrivacyConfig
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

DELTA = 1e-5


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    # Keep the demo self-contained and deterministic on re-run: an in-memory
    # metadata store, so recorded privacy spends don't persist to disk.
    config.storage.metadata = "memory://"
    return config


# Distinct episodic observations, so the write policy keeps them as separate
# memories (near-duplicate content would be collapsed before consolidation).
_FACTS = [
    "the user prefers metric units and a dark theme",
    "the user's home airport is SFO and they fly on Tuesdays",
    "the user is allergic to penicillin",
    "the user manages a team of six engineers in Berlin",
    "the user's renewal date is the first of each quarter",
    "the user escalates billing issues to finance, not support",
]


def _seed_episodes(app: ContextApp, session_id: str, round_no: int) -> None:
    """Write a session's worth of distinct episodic memories for one subject."""
    for i, fact in enumerate(_FACTS):
        app.memory.write_fact(
            f"{fact} (note {round_no}.{i})",
            scope=MemoryScope.SESSION,
            owner_id=session_id,
            type=MemoryType.FACT,
            confidence=0.9,
        )


async def main() -> None:
    app = ContextApp(name="dp-demo", provider=MockProvider(default_text="ok"), config=_config())
    # A per-subject budget of ε = 2.0; each accounted release is a unit-sensitivity
    # Gaussian mechanism with noise multiplier 4.0 (≈ 1.23 ε on its own at δ=1e-5).
    app.use_privacy_accountant(
        default_budget=PrivacyBudget(epsilon=2.0, delta=DELTA),
        default_mechanism=PrivacyMechanism(noise_multiplier=4.0),
    )
    app.add_memory()

    print("1. Account — each consolidation composes the subject's privacy budget")
    for round_no in range(2):
        _seed_episodes(app, "sess-alice", round_no)
        report = await app.memory.consolidate("sess-alice", user_id="alice")
        charged = f"{report.privacy_epsilon:.3f}" if report.privacy_epsilon is not None else "—"
        print(f"   consolidation {round_no}: promoted={report.promoted}  cumulative ε={charged}")
    spent = app.privacy_accountant.spent("alice")
    single = PrivacyMechanism(noise_multiplier=4.0).epsilon(delta=DELTA)
    print(f"   composed ε after 2 rounds: {spent:.3f}  (naive sum would be {2 * single:.3f})")

    print("\n2. Refuse — once the budget is spent, the next consolidation is refused")
    _seed_episodes(app, "sess-alice", 2)
    refused = await app.memory.consolidate("sess-alice", user_id="alice")
    print(
        f"   consolidation 2: promoted={refused.promoted}  "
        f"privacy_refused={refused.privacy_refused}  (episodes stay episodic)"
    )

    print("\n3. Federate — a federated contribution composes the SAME budget")
    emb = LocalHashEmbedder(dim=64)
    fed_app = ContextApp(name="org-a", provider=MockProvider(default_text="x"), config=_config())
    fed_app.embedder = emb
    fed_app.use_privacy_accountant(default_budget=PrivacyBudget(epsilon=1.5, delta=DELTA))
    training = TrainingSet(
        name="fed",
        examples=[
            TrainingExample(
                messages=[
                    {"role": "user", "content": f"q {i}"},
                    {"role": "assistant", "content": f"a {i}"},
                ]
            )
            for i in range(4)
        ],
    )
    policy = FederatedPolicy(
        privacy=PrivacyConfig(min_contributors=2, clip_norm=1.0, dp_epsilon=0.8, dp_delta=DELTA),
        consent_subject="alice",
    )
    controller = fed_app.federated_improvement(policy)
    for attempt in range(5):
        try:
            await controller.build_contribution(
                member_id="org-a", participants=["org-a", "org-b"], training_set=training
            )
            print(f"   contribution {attempt}: ε spent={fed_app.privacy_accountant.spent('alice'):.3f}")
        except PrivacyBudgetError as exc:
            print(f"   contribution {attempt}: refused — {exc.code} ({exc.remediation[:48]}…)")
            break

    print("\n4. Down-weight — a more-private (noisier) release fits the remaining budget")
    dw_app = ContextApp(name="org-b", provider=MockProvider(default_text="x"), config=_config())
    dw_app.embedder = emb
    dw_app.use_privacy_accountant(
        default_budget=PrivacyBudget(epsilon=1.5, delta=DELTA, on_breach="downweight")
    )
    dw_controller = dw_app.federated_improvement(policy)
    for attempt in range(5):
        try:
            contribution = await dw_controller.build_contribution(
                member_id="org-b", participants=["org-a", "org-b"], training_set=training
            )
        except PrivacyBudgetError:
            print("   budget fully spent — no further release possible")
            break
        weight = dw_app.privacy_accountant.spends("alice")[-1].downweight
        print(
            f"   contribution {attempt}: noise ε×{weight:.2f}  "
            f"spent={dw_app.privacy_accountant.spent('alice'):.3f} / 1.5"
        )

    print("\n5. Report — the spent budget is auditable, alongside the cost report")
    app.privacy_report().print_summary()
    privacy_entries = [e for e in app.audit.entries if "privacy" in e.action]
    print(
        f"   audit chain: {len(privacy_entries)} privacy entries, "
        f"verified={app.audit.verify_chain()}"
    )

    print("\nA provable per-subject privacy budget — composed, gated, and auditable.")


if __name__ == "__main__":
    asyncio.run(main())
