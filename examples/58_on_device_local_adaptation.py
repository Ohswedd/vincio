"""On-device fine-tuning & continual local adaptation.

The distillation flywheel already turns production traces into executed *hosted*
fine-tune jobs, and the in-process GGUF provider already runs a quantized model
air-gapped. This example shows the rung above them: a LoRA-class adapter fit
*on-device* from the same grounded data and applied to the in-process model —
the run never leaves the process, so an air-gapped or edge deployment improves on
its own traffic with no hosted training round-trip.

Five steps, all offline and deterministic (driven by the mock provider standing
in for the in-process model, and the dependency-free local embedder):

  1. Fit: train a parameter-efficient, low-rank adapter on-device from a grounded
     training set — pure Python, no network, no SDK.
  2. Bounded: the adapter answers in-distribution traffic the grounded way it was
     taught, and stays inert off-distribution — it never reshapes traffic it has
     not seen.
  3. Gate: the continual loop promotes the adapter only when the locally-adapted
     model is at-least-as-good as its base on a held-out set — the same
     no-regression discipline a hosted fine-tune job clears.
  4. Reversible: unloading the adapter restores the base model exactly, and a
     regressing adapter is refused outright — never promoted, never applied.
  5. Versioned: every adapter is content-addressed and versioned in a registry
     that rolls a head back to an earlier version.

Everything here is opt-in and additive; nothing below is required to run Vincio.
"""

from __future__ import annotations

import asyncio

from vincio import (
    AdaptedProvider,
    AdapterRegistry,
    ContextApp,
    LocalAdaptationPolicy,
    LocalLoRATrainer,
    VincioConfig,
)
from vincio.core.types import Message, ModelRequest
from vincio.evals.datasets import Dataset, EvalCase
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

# A small slice of grounded traffic the edge model has served.
QA = [
    ("what is the refund policy", "Refunds are processed within 30 days."),
    ("how do I reset my password", "Use the reset link on the login page."),
    ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
    ("how do I contact support", "Email support@example.com any time."),
]


def _config() -> VincioConfig:
    config = VincioConfig()
    config.observability.exporter = "memory"
    return config


def _training_set() -> TrainingSet:
    return TrainingSet(
        name="local-adapter",
        examples=[
            TrainingExample(
                messages=[
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            )
            for q, a in QA
        ],
    )


def _golden() -> Dataset:
    return Dataset(
        name="golden",
        cases=[EvalCase(id=f"c{i}", input=q, expected=a) for i, (q, a) in enumerate(QA)],
    )


async def fit_on_device() -> None:
    print("1. Fit — train a low-rank adapter on-device from grounded traffic")
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=8, gate=0.85).fit(
        _training_set(), "gguf-local"
    )
    print(
        f"   rank={adapter.rank}  examples={len(adapter)}  size={adapter.size_bytes} bytes  "
        f"digest={adapter.digest[:12]}"
    )

    print("\n2. Bounded — adapts in-distribution traffic, inert off-distribution")
    provider = AdaptedProvider(MockProvider(default_text="(base model answer)"), adapter, embedder=emb)
    seen = await provider.generate(
        ModelRequest(model="gguf-local", messages=[Message(role="user", content=QA[0][0])])
    )
    unseen = await provider.generate(
        ModelRequest(
            model="gguf-local",
            messages=[Message(role="user", content="tell me a joke about giraffes in space")],
        )
    )
    print(f"   in-distribution:  {QA[0][0]!r} -> {seen.text!r}")
    print(f"   off-distribution: 'tell me a joke...' -> {unseen.text!r}  (base model, untouched)")


def gated_continual_loop() -> None:
    print("\n3. Gate — promote only when the adapted model is at-least-as-good")
    app = ContextApp(
        name="edge",
        provider=MockProvider(default_text="I am not sure about that."),
        config=_config(),
    )
    registry = AdapterRegistry()
    policy = LocalAdaptationPolicy(min_examples=4, min_samples=4, require_significance=False)
    result = app.adapt_locally(_golden(), training_set=_training_set(), policy=policy, registry=registry)
    print(
        f"   promoted={result.promoted}  base={result.verdict.baseline:.2f} -> "
        f"adapted={result.verdict.candidate:.2f}  (Δ={result.verdict.delta:+.2f})"
    )
    print(f"   live run now answers the grounded way: {app.run(QA[0][0]).raw_text!r}")

    print("\n4. Reversible — unload restores the base model, a regression is refused")
    app.use_local_adapter(None)
    print(f"   after unload: {app.run(QA[0][0]).raw_text!r}")

    def echo(req):
        return "GOOD answer " + req.messages[-1].text.split()[-1]

    reg_qa = [(f"q item {w}", f"GOOD answer {w}") for w in ("alpha", "beta", "gamma", "delta")]
    reg_app = ContextApp(name="edge2", provider=MockProvider(responder=echo), config=_config())
    bad = TrainingSet(
        name="local-adapter",
        examples=[
            TrainingExample(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": "wrong"}]
            )
            for q, _ in reg_qa
        ],
    )
    bad_ds = Dataset(
        name="g", cases=[EvalCase(id=f"r{i}", input=q, expected=a) for i, (q, a) in enumerate(reg_qa)]
    )
    refused = AdapterRegistry()
    bad_result = reg_app.adapt_locally(
        bad_ds,
        training_set=bad,
        policy=LocalAdaptationPolicy(
            min_examples=4, gate=0.6, min_samples=4, require_significance=False
        ),
        registry=refused,
    )
    print(
        f"   regressing adapter: promoted={bad_result.promoted}  "
        f"registry versions={[a.version for a in refused.versions('local-adapter')]}"
    )

    print("\n5. Versioned — content-addressed adapters with rollback")
    registry.register(registry.active("local-adapter"))  # register a second version
    registry.rollback("local-adapter", 1)
    print(
        f"   versions={[a.version for a in registry.versions('local-adapter')]}  "
        f"active=v{registry.active('local-adapter').version}"
    )


async def main() -> None:
    await fit_on_device()
    gated_continual_loop()
    print("\nAn edge model that improves on its own traffic — gated, reversible, in-process.")


if __name__ == "__main__":
    asyncio.run(main())
