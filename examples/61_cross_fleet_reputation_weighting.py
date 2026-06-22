"""Cross-fleet reputation & reliability-weighted federated aggregation.

The federated round (example 59) merges every member's contribution with **equal
weight**: a member whose contributions repeatedly regress the no-regression gate
still pulls the shared consensus geometry as hard as one whose contributions
consistently help. This example adds the missing rung — a per-member
**reputation**, earned only from how each past contribution fared against the
gate, that discounts an unreliable or adversarial member's pull on the consensus.

Five steps, all offline and deterministic:

  1. Earn: a reputation ledger scores each member from its gate track record — a
     Beta-Bernoulli posterior over passes and failures, bounded into a weight band.
  2. Weight: the secure aggregator weights a member's contribution by its
     reputation, so a regressor is discounted without being singled out.
  3. Discount: with equal weight the regressor pulls the consensus astray; weighted
     by reputation, the consensus leans toward the reliable member.
  4. Bounded & reversible: a weight only ever lowers a member's pull, and adoption
     still clears the very same no-regression gate — reputation can never bypass it.
  5. Auditable: every reputation update lives on the signed audit chain and replays
     from it exactly, so a member's standing is a mechanical, verifiable number.

Everything here is opt-in and additive; without a ledger the federated round
behaves exactly as before (every member weighted equally).
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    ContributionBuilder,
    FederatedPolicy,
    PrivacyConfig,
    ReputationConfig,
    ReputationLedger,
    SecureAggregator,
    VincioConfig,
)
from vincio.evals.datasets import Dataset, EvalCase
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.optimize.federated import _top_eigenvectors
from vincio.optimize.reputation import REPUTATION_ACTION
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

# Two members of a support-assistant fleet. "good" has a clean gate record; "bad"
# has repeatedly regressed — neither ever shares its raw traffic.
QA_GOOD = [
    ("what is the refund policy", "Refunds are processed within 30 days."),
    ("how do I reset my password", "Use the reset link on the login page."),
]
QA_BAD = [
    ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
    ("how do I contact support", "Email support@example.com any time."),
]
QA_ALL = QA_GOOD + QA_BAD
FLEET = ["good", "bad"]
DIM = 64


def _training_set(qa: list[tuple[str, str]]) -> TrainingSet:
    return TrainingSet(
        name="federated-adapter",
        examples=[
            TrainingExample(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            )
            for q, a in qa
        ],
    )


def _golden(qa: list[tuple[str, str]]) -> Dataset:
    return Dataset(
        name="golden",
        cases=[EvalCase(id=f"c{i}", input=q, expected=a) for i, (q, a) in enumerate(qa)],
    )


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


async def main() -> None:
    emb = LocalHashEmbedder(dim=DIM)

    print("1. Earn — a member's reputation is its gate track record, nothing else")
    config = ReputationConfig(weight_floor=0.05)
    ledger = ReputationLedger(config)
    print(f"   a fresh member starts at the prior: weight={ledger.weight('newcomer'):.3f}")
    for i in range(30):
        ledger.record_outcome("bad", passed=False, round_id=f"r{i}")  # a persistent regressor
        ledger.record_outcome("good", passed=True, round_id=f"r{i}")  # a reliable member
    print(
        f"   after 30 rounds: good weight={ledger.weight('good'):.3f}, "
        f"bad weight={ledger.weight('bad'):.3f}  (discounted toward the floor)"
    )

    print("\n2. Weight — the aggregator scales each member by its reputation")
    off = ContributionBuilder(embedder=emb, privacy=PrivacyConfig(secure_aggregation=False))
    good_c = await off.build(_training_set(QA_GOOD), "gguf-local", member_id="good", participants=None)
    bad_c = await off.build(_training_set(QA_BAD), "gguf-local", member_id="bad", participants=None)
    weighted = SecureAggregator(
        privacy=PrivacyConfig(secure_aggregation=False), rank=1, reputation=ledger
    ).aggregate([good_c, bad_c])
    print(f"   round weights: {weighted.provenance['reputation_weights']}")

    print("\n3. Discount — the consensus leans toward the reliable member")
    plain = SecureAggregator(privacy=PrivacyConfig(secure_aggregation=False), rank=1).aggregate(
        [good_c, bad_c]
    )
    good_dir = _top_eigenvectors(good_c.scatter, 1)[0][0]

    def align(subspace):
        return abs(sum(a * b for a, b in zip(subspace.basis[0], good_dir, strict=True)))

    print(
        f"   alignment with the reliable member's geometry: "
        f"equal-weight={align(plain):.3f}  ->  reputation-weighted={align(weighted):.3f}"
    )

    print("\n4. Bounded & reversible — reputation never bypasses the no-regression gate")
    app = ContextApp(
        name="good", provider=MockProvider(default_text="I am not sure about that."), config=_config()
    )
    app.embedder = emb
    app.use_reputation_ledger()
    # Seed the live ledger so the round is reliability-weighted.
    for _ in range(6):
        app.reputation_ledger.record_outcome("bad", passed=False, round_id="seed")
    ctl = app.federated_improvement(
        FederatedPolicy(min_examples=4, min_samples=4, require_significance=False),
        dataset=_golden(QA_ALL),
    )
    ca = await ctl.build_contribution(member_id="good", participants=FLEET, training_set=_training_set(QA_GOOD))
    cb = await ctl.build_contribution(member_id="bad", participants=FLEET, training_set=_training_set(QA_BAD))
    result = await ctl.aadopt(contributions=[ca, cb], training_set=_training_set(QA_ALL))
    print(
        f"   discounted contribution: good weight={ca.reputation_weight:.3f}, "
        f"bad weight={cb.reputation_weight:.3f}"
    )
    print(
        f"   adopted={result.adopted}  (Δ={result.verdict.delta:+.2f}, still at-least-as-good) — "
        "a bad reputation only lowers pull, never bypasses the gate"
    )

    print("\n5. Auditable — reputation lives on the chain and replays from it exactly")
    replayed = ReputationLedger.from_audit(app.audit)
    updates = app.audit.query(action=REPUTATION_ACTION)
    print(
        f"   {len(updates)} reputation updates on the signed chain; replayed bad weight="
        f"{replayed.weight('bad'):.3f} (matches {app.reputation_ledger.weight('bad'):.3f})"
    )
    app.reputation_report().print_summary()

    print("\nA fleet that trusts what works — earned, bounded, reversible, auditable.")


if __name__ == "__main__":
    asyncio.run(main())
