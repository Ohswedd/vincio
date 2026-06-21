"""Tests for on-device fine-tuning & continual local adaptation.

All deterministic and offline: a LoRA-class adapter is fit in-process from a
grounded training set, applied through the deterministic mock provider, gated for
no regression, versioned, and rolled back. No network and no real model are
involved — the in-process adapter shapes generation directly.
"""

from __future__ import annotations

import pytest

import vincio
from vincio import ContextApp, VincioConfig
from vincio.core.errors import OptimizationError
from vincio.core.types import Message, ModelRequest
from vincio.evals.datasets import Dataset, EvalCase
from vincio.optimize import (
    AdaptationError,
    AdaptedProvider,
    AdapterGate,
    AdapterRegistry,
    ContinualAdaptation,
    LocalAdaptationPolicy,
    LocalAdapter,
    LocalLoRATrainer,
)
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.providers.mock import MockProvider
from vincio.retrieval.embeddings import LocalHashEmbedder

QA = [
    ("what is the refund policy", "Refunds are processed within 30 days."),
    ("how do I reset my password", "Use the reset link on the login page."),
    ("what are the shipping options", "We ship worldwide via DHL in 5-7 days."),
    ("how do I contact support", "Email support@example.com any time."),
]


def training_set(name: str = "local-adapter") -> TrainingSet:
    return TrainingSet(
        name=name,
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


def golden() -> Dataset:
    return Dataset(
        name="golden",
        cases=[EvalCase(id=f"c{i}", input=q, expected=a) for i, (q, a) in enumerate(QA)],
    )


def make_app(default_text: str = "I am not sure about that.") -> ContextApp:
    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    return ContextApp(name="edge", provider=MockProvider(default_text=default_text), config=cfg)


# ---------------------------------------------------------------------------
# Trainer & adapter
# ---------------------------------------------------------------------------


async def test_trainer_fits_low_rank_adapter():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=8).fit(training_set(), "gguf-local")
    assert isinstance(adapter, LocalAdapter)
    assert adapter.base_model == "gguf-local"
    assert len(adapter) == len(QA)
    # Rank is bounded by the requested rank and the number of examples.
    assert 0 < adapter.rank <= min(8, len(QA))
    assert adapter.embed_dim == emb.dim
    assert adapter.size_bytes > 0
    assert adapter.training_set_hash
    # Low-rank structure: an r×d orthonormal basis plus n×r example codes.
    assert all(len(axis) == emb.dim for axis in adapter.basis)
    assert all(len(code) == adapter.rank for code in adapter.codes)
    assert len(adapter.codes) == len(QA)


async def test_low_rank_compresses_when_rank_below_examples():
    emb = LocalHashEmbedder()
    # Many near-duplicate examples collapse into a small subspace: rank << n, so
    # the r·d + n·r footprint is strictly smaller than the full n·d example set.
    examples = [
        TrainingExample(
            messages=[
                {"role": "user", "content": f"refund question variant number {i}"},
                {"role": "assistant", "content": f"Refund answer {i}."},
            ]
        )
        for i in range(40)
    ]
    ts = TrainingSet(name="local-adapter", examples=examples)
    adapter = await LocalLoRATrainer(embedder=emb, rank=6).fit(ts, "gguf")
    n, d, r = len(examples), emb.dim, adapter.rank
    assert r <= 6 < n
    assert (r * d + n * r) < (n * d)


async def test_trainer_refuses_too_little_data():
    small = TrainingSet(name="x", examples=training_set().examples[:1])
    with pytest.raises(AdaptationError):
        await LocalLoRATrainer().fit(small, "gguf", min_examples=4)
    # AdaptationError carries the optimization error code/remediation surface.
    assert issubclass(AdaptationError, OptimizationError)


async def test_adapter_fires_in_distribution_inert_off_distribution():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=8, gate=0.85).fit(training_set(), "gguf")
    provider = AdaptedProvider(MockProvider(default_text="GENERIC"), adapter, embedder=emb)

    # An exact in-distribution request is answered the grounded way it was taught.
    hit = await provider.generate(
        ModelRequest(model="m", messages=[Message(role="user", content=QA[0][0])])
    )
    assert hit.text == QA[0][1]
    assert hit.raw["adapter"]["similarity"] == pytest.approx(1.0, abs=1e-6)

    # An off-distribution request falls through to the base model untouched.
    miss = await provider.generate(
        ModelRequest(
            model="m",
            messages=[Message(role="user", content="tell me a joke about giraffes in space")],
        )
    )
    assert miss.text == "GENERIC"
    assert "adapter" not in (miss.raw or {})


async def test_adapter_scale_zero_neutralizes():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=8).fit(training_set(), "gguf")
    adapter.scale = 0.0  # the one-line reversibility knob
    provider = AdaptedProvider(MockProvider(default_text="GENERIC"), adapter, embedder=emb)
    out = await provider.generate(
        ModelRequest(model="m", messages=[Message(role="user", content=QA[0][0])])
    )
    assert out.text == "GENERIC"


async def test_adapter_dim_mismatch_raises():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=4).fit(training_set(), "gguf")
    with pytest.raises(AdaptationError):
        adapter.apply([0.1] * (emb.dim + 1))


async def test_adapter_stream_matches_generate():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=8).fit(training_set(), "gguf")
    provider = AdaptedProvider(MockProvider(default_text="GENERIC"), adapter, embedder=emb)
    req = ModelRequest(model="m", messages=[Message(role="user", content=QA[1][0])])
    chunks = []
    async for event in provider.stream(req):
        if event.type == "text_delta":
            chunks.append(event.text)
    assert "".join(chunks) == QA[1][1]


async def test_adapter_digest_and_roundtrip(tmp_path):
    emb = LocalHashEmbedder()
    a1 = await LocalLoRATrainer(embedder=emb, rank=8).fit(training_set(), "gguf")
    a2 = await LocalLoRATrainer(embedder=emb, rank=8).fit(training_set(), "gguf")
    # Deterministic: same data + hyper-params -> same content address.
    assert a1.digest == a2.digest
    path = a1.save(tmp_path / "adapter.json")
    loaded = LocalAdapter.load(path)
    assert loaded.digest == a1.digest
    assert loaded.targets == a1.targets


# ---------------------------------------------------------------------------
# Registry (versioned & reversible)
# ---------------------------------------------------------------------------


async def test_registry_versions_and_rollback():
    emb = LocalHashEmbedder()
    reg = AdapterRegistry()
    a1 = await LocalLoRATrainer(embedder=emb).fit(training_set(), "gguf")
    a2 = await LocalLoRATrainer(embedder=emb, gate=0.9).fit(training_set(), "gguf")
    reg.register(a1)
    reg.register(a2)
    assert [a.version for a in reg.versions("local-adapter")] == [1, 2]
    assert reg.active("local-adapter").version == 2
    reg.rollback("local-adapter", 1)
    assert reg.active("local-adapter").version == 1
    assert reg.get("local-adapter", 2).gate == 0.9
    with pytest.raises(AdaptationError):
        reg.get("local-adapter", 99)


async def test_registry_prune_bounds_resident_versions():
    emb = LocalHashEmbedder()
    reg = AdapterRegistry()
    for i in range(5):
        reg.register(await LocalLoRATrainer(embedder=emb, gate=0.5 + i * 0.05).fit(training_set(), "g"))
    dropped = reg.prune("local-adapter", 2)
    assert dropped == 3
    assert [a.version for a in reg.versions("local-adapter")] == [4, 5]
    # Pruning never drops below the requested floor and is a no-op when under it.
    assert reg.prune("local-adapter", 10) == 0


async def test_registry_persists_to_disk(tmp_path):
    emb = LocalHashEmbedder()
    reg = AdapterRegistry(directory=tmp_path)
    reg.register(await LocalLoRATrainer(embedder=emb).fit(training_set(), "gguf"))
    reg.register(await LocalLoRATrainer(embedder=emb, gate=0.7).fit(training_set(), "gguf"))
    reg.rollback("local-adapter", 1)
    # A fresh registry over the same directory restores versions and the head.
    reopened = AdapterRegistry(directory=tmp_path)
    assert [a.version for a in reopened.versions("local-adapter")] == [1, 2]
    assert reopened.active("local-adapter").version == 1


# ---------------------------------------------------------------------------
# Gate (the on-device swap gate)
# ---------------------------------------------------------------------------


async def test_gate_promotes_at_least_as_good_blocks_regression():
    app = make_app(default_text="I am not sure about that.")
    controller = ContinualAdaptation(
        app,
        LocalAdaptationPolicy(min_samples=4, require_significance=False),
        dataset=golden(),
    )
    base = app._base_provider()
    emb = app.embedder
    adapter = await LocalLoRATrainer(embedder=emb, gate=0.85).fit(training_set(), app.model)
    adapted = AdaptedProvider(base, adapter, embedder=emb)

    base_report = await controller._eval_report(base)
    adapted_report = await controller._eval_report(adapted)
    gate = AdapterGate(regression_threshold=0.0, require_significance=False, min_samples=4)
    verdict = gate.evaluate(base_report, adapted_report)
    assert verdict.passed
    assert verdict.candidate >= verdict.baseline
    # The reverse comparison (a regressing swap) is blocked.
    reverse = gate.evaluate(adapted_report, base_report)
    assert not reverse.passed


# ---------------------------------------------------------------------------
# Continual adaptation loop (end-to-end, gated, reversible)
# ---------------------------------------------------------------------------


async def test_adapt_locally_promotes_and_applies():
    app = make_app(default_text="I am not sure about that.")
    reg = AdapterRegistry()
    result = app.adapt_locally(
        golden(),
        training_set=training_set(),
        policy=LocalAdaptationPolicy(min_examples=4, min_samples=4, require_significance=False),
        registry=reg,
    )
    assert result.promoted
    assert result.adapter_version == 1
    assert result.verdict.candidate >= result.verdict.baseline
    assert app.local_adapter is not None
    # The live app now answers the grounded way, in-process.
    assert app.run(QA[0][0]).raw_text == QA[0][1]
    # Reversible: unloading restores the base model.
    app.use_local_adapter(None)
    assert app.local_adapter is None
    assert app.run(QA[0][0]).raw_text == "I am not sure about that."


async def test_adapt_locally_refuses_regression():
    def responder(req):
        return "GOOD answer " + req.messages[-1].text.split()[-1]

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge", provider=MockProvider(responder=responder), config=cfg)
    qa = [
        ("q one alpha", "GOOD answer alpha"),
        ("q two beta", "GOOD answer beta"),
        ("q three gamma", "GOOD answer gamma"),
        ("q four delta", "GOOD answer delta"),
    ]
    ds = Dataset(name="g", cases=[EvalCase(id=f"c{i}", input=q, expected=a) for i, (q, a) in enumerate(qa)])
    bad = TrainingSet(
        name="local-adapter",
        examples=[
            TrainingExample(
                messages=[{"role": "user", "content": q}, {"role": "assistant", "content": "wrong nonsense"}]
            )
            for q, _ in qa
        ],
    )
    reg = AdapterRegistry()
    result = app.adapt_locally(
        ds,
        training_set=bad,
        policy=LocalAdaptationPolicy(min_examples=4, gate=0.6, min_samples=4, require_significance=False),
        registry=reg,
    )
    assert not result.promoted
    assert reg.versions("local-adapter") == []
    assert app.local_adapter is None


async def test_adapt_locally_dry_run_does_not_promote():
    app = make_app()
    reg = AdapterRegistry()
    result = app.adapt_locally(
        golden(),
        training_set=training_set(),
        policy=LocalAdaptationPolicy(
            min_examples=4, min_samples=4, require_significance=False, dry_run=True
        ),
        registry=reg,
    )
    assert not result.promoted
    assert "dry run" in result.reason
    assert reg.versions("local-adapter") == []


async def test_astream_emits_phases_and_audits():
    app = make_app()
    controller = app.local_adaptation(
        LocalAdaptationPolicy(min_examples=4, min_samples=4, require_significance=False),
        dataset=golden(),
    )
    phases = [ev.phase async for ev in controller.astream(training_set=training_set())]
    assert phases[0] == "observe"
    assert "train" in phases
    assert "gate" in phases
    assert phases[-1] == "promote"
    # Every decision lands on the hash-chained audit log.
    actions = [e.action for e in app.audit.entries if e.action == "local_adaptation"]
    assert actions  # recorded


async def test_no_dataset_fits_but_does_not_promote():
    app = make_app()
    controller = app.local_adaptation(
        LocalAdaptationPolicy(min_examples=4), dataset=None
    )
    result = await controller.aadapt(training_set=training_set())
    assert not result.promoted
    assert "cannot gate" in result.reason


async def test_adapt_from_runs():
    app = make_app(default_text="I am not sure about that.")
    # Produce RunResults that carry input + output (no grounding required here).
    runs = [app.run(q) for q, _ in QA]
    result = app.adapt_locally(
        golden(),
        runs=runs,
        policy=LocalAdaptationPolicy(
            min_examples=1, min_samples=4, require_significance=False, require_grounding=False
        ),
    )
    # The adapter is fit from the run transcripts; gating decides promotion.
    assert result.training_examples >= 1
    assert result.verdict is not None


def test_public_surface_exported():
    for name in [
        "LocalAdapter",
        "LocalLoRATrainer",
        "AdaptedProvider",
        "AdapterRegistry",
        "AdapterGate",
        "LocalAdaptationPolicy",
        "AdaptationResult",
        "ContinualAdaptation",
    ]:
        assert hasattr(vincio, name), name
        assert name in vincio.__all__
