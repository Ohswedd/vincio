"""Federated / cross-org self-improvement.

On-device local adaptation (example 58) improves a model on its own traffic, but
always *within one trust boundary*. This example shows the rung above it: a fleet
of organizations improving **together without sharing the raw traffic**. Each
member fits a low-rank subspace on its own private data and contributes a numeric,
raw-text-free summary of where that adaptation lives; a secure aggregation merges
the fleet's contributions into a shared subspace; and the adopting member re-fits
its *own* adapter against that shared geometry, behind the same no-regression gate
a local promotion clears.

Five steps, all offline and deterministic (the mock provider stands in for each
member's in-process model, and the dependency-free local embedder builds the
subspaces):

  1. Contribute: each member builds a numeric contribution — the clipped, masked
     subspace scatter — that carries no prompt and no response, only geometry.
  2. Private: a serialized contribution contains none of the member's raw traffic,
     and the secure-aggregation masks hide each individual update.
  3. Aggregate: the masks cancel exactly in the sum, so the aggregator recovers the
     fleet subspace without ever seeing one member's update — and refuses a round
     below the k-anonymity contributor floor.
  4. Gate: the adopting member re-fits its own adapter against the shared subspace
     and adopts it only when at-least-as-good as its base on a held-out set.
  5. Reversible: a regressing federated adapter is refused outright, and unloading
     an adopted one restores the base model exactly.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import (
    ContextApp,
    ContributionBuilder,
    FederatedPolicy,
    PrivacyConfig,
    SecureAggregator,
    VincioConfig,
)
from vincio.evals.datasets import Dataset, EvalCase
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.optimize.federated import _add_into, _frobenius, _zeros
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

# Two members of a support-assistant fleet. Each has served a different slice of
# private traffic — neither will ever see the other's prompts or responses.
QA_A = [
    ("what is the refund policy", "Refunds are processed within 30 days."),
    ("how do I reset my password", "Use the reset link on the login page."),
]
QA_B = [
    ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
    ("how do I contact support", "Email support@example.com any time."),
]
QA_ALL = QA_A + QA_B
FLEET = ["org-a", "org-b"]
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
    privacy = PrivacyConfig(min_contributors=2, secure_aggregation=True, clip_norm=1.0)
    builder = ContributionBuilder(embedder=emb, privacy=privacy)

    print("1. Contribute — each member shares numeric geometry, never its traffic")
    contribution_a = await builder.build(
        _training_set(QA_A), "gguf-local", member_id="org-a", participants=FLEET
    )
    contribution_b = await builder.build(
        _training_set(QA_B), "gguf-local", member_id="org-b", participants=FLEET
    )
    print(
        f"   org-a contribution: {contribution_a.embed_dim}x{contribution_a.embed_dim} scatter, "
        f"{contribution_a.n_examples} examples, masked={contribution_a.masked}, "
        f"digest={contribution_a.digest[:12]}"
    )

    print("\n2. Private — no prompt or response appears in a contribution")
    blob = contribution_a.model_dump_json()
    leaked = [q for q, a in QA_A if q in blob or a in blob]
    print(f"   raw traffic strings found in the wire artifact: {leaked}  (none)")

    print("\n3. Aggregate — masks cancel; the aggregator never sees one update")
    unmasked = ContributionBuilder(embedder=emb, privacy=PrivacyConfig(secure_aggregation=False))
    plain_a = await unmasked.build(
        _training_set(QA_A), "gguf-local", member_id="org-a", participants=FLEET
    )
    plain_b = await unmasked.build(
        _training_set(QA_B), "gguf-local", member_id="org-b", participants=FLEET
    )
    masked_sum = _zeros(DIM, DIM)
    _add_into(masked_sum, contribution_a.scatter)
    _add_into(masked_sum, contribution_b.scatter)
    plain_sum = _zeros(DIM, DIM)
    _add_into(plain_sum, plain_a.scatter)
    _add_into(plain_sum, plain_b.scatter)
    residual = _frobenius(
        [[masked_sum[i][j] - plain_sum[i][j] for j in range(DIM)] for i in range(DIM)]
    )
    subspace = SecureAggregator(privacy=privacy, rank=8).aggregate(
        [contribution_a, contribution_b]
    )
    print(
        f"   mask-cancellation residual: {residual:.2e}  ->  merged rank-{subspace.rank} "
        f"subspace from {subspace.contributor_count} contributors"
    )
    try:
        SecureAggregator(privacy=privacy).aggregate([contribution_a])
    except Exception as exc:  # noqa: BLE001 - demonstrating the k-anonymity refusal
        print(f"   a single-member round is refused: {type(exc).__name__}")

    print("\n4. Gate — the adopting member re-fits its own adapter, gated for no regression")
    app = ContextApp(
        name="org-a", provider=MockProvider(default_text="I am not sure about that."), config=_config()
    )
    app.embedder = emb
    policy = FederatedPolicy(min_examples=4, min_samples=4, require_significance=False)
    result = app.adopt_federated(
        _golden(QA_ALL), [contribution_a, contribution_b], training_set=_training_set(QA_ALL), policy=policy
    )
    print(
        f"   adopted={result.adopted}  base={result.verdict.baseline:.2f} -> "
        f"adapted={result.verdict.candidate:.2f}  (Δ={result.verdict.delta:+.2f})  "
        f"secure_aggregation={result.privacy.secure_aggregation}"
    )
    print(f"   org-a now answers the way the fleet taught it: {app.run(QA_A[0][0]).raw_text!r}")

    print("\n5. Reversible — unload restores the base model, a regression is refused")
    app.use_local_adapter(None)
    print(f"   after unload: {app.run(QA_A[0][0]).raw_text!r}")

    def echo(req):
        return "GOOD answer " + req.messages[-1].text.split()[-1]

    reg_qa = [(f"q item {w}", f"GOOD answer {w}") for w in ("alpha", "beta", "gamma", "delta")]
    reg_app = ContextApp(name="org-r", provider=MockProvider(responder=echo), config=_config())
    reg_app.embedder = emb
    reg_a = await builder.build(
        _training_set(reg_qa[:2]), "gguf-local", member_id="org-a", participants=FLEET
    )
    reg_b = await builder.build(
        _training_set(reg_qa[2:]), "gguf-local", member_id="org-b", participants=FLEET
    )
    bad_local = TrainingSet(
        name="federated-adapter",
        examples=[
            TrainingExample(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": "wrong"}]
            )
            for q, _ in reg_qa
        ],
    )
    bad = reg_app.adopt_federated(
        _golden(reg_qa),
        [reg_a, reg_b],
        training_set=bad_local,
        policy=FederatedPolicy(min_examples=4, gate=0.6, min_samples=4, require_significance=False),
    )
    print(f"   regressing federated adapter: adopted={bad.adopted}  applied={reg_app.local_adapter is not None}")

    print("\nA fleet that improves together — private contributions, gated and reversible.")


if __name__ == "__main__":
    asyncio.run(main())
