"""Coverage-hardening tests for on-device local adaptation.

These target the specific uncovered error paths, branches, and edge cases of
:mod:`vincio.optimize.local_adaptation`: zero-norm / collapsing example
embeddings, the inert-adapter branches, the default-embedder fallbacks, the
native backend, the transparent provider passthroughs (embed / list_models /
capabilities / aclose / stream-miss), registry persistence head-pointer and
``latest`` / ``prune`` boundaries, the trace-curation path of the continual
loop, the insufficient-data refusal, and the safety-gate overlay that blocks an
otherwise-passing adapter. All deterministic and offline via MockProvider and
the LocalHashEmbedder — no network, no mocks.
"""

from __future__ import annotations

import pytest

from vincio import ContextApp, VincioConfig
from vincio.core.types import Message, ModelRequest
from vincio.evals.datasets import Dataset, EvalCase
from vincio.optimize.distill import TrainingExample, TrainingSet
from vincio.optimize.local_adaptation import (
    AdaptationError,
    AdaptedProvider,
    AdapterRegistry,
    ContinualAdaptation,
    LocalAdaptationPolicy,
    LocalAdapter,
    LocalLoRATrainer,
    _orthonormal_basis,
    _unit,
)
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
# Pure linear algebra edge cases (lines 102, 124->116, 246)
# ---------------------------------------------------------------------------


def test_unit_zero_vector_returns_zeros():
    # A vector below the 1e-12 norm floor collapses to an all-zero unit vector
    # instead of dividing by ~zero.
    out = _unit([0.0, 0.0, 0.0])
    assert out == [0.0, 0.0, 0.0]


def test_orthonormal_basis_drops_collapsing_residual():
    # The second vector is identical to the first, so its Gram-Schmidt residual
    # collapses below 1e-9 and is discarded — basis stays rank 1 despite 3 inputs.
    v = [1.0, 0.0, 0.0]
    basis = _orthonormal_basis([v, list(v), list(v)], rank=8)
    assert len(basis) == 1
    assert basis[0] == pytest.approx([1.0, 0.0, 0.0])


def test_orthonormal_basis_stops_at_rank():
    # Two orthogonal directions but rank capped at 1 -> exactly one axis kept.
    basis = _orthonormal_basis([[1.0, 0.0], [0.0, 1.0]], rank=1)
    assert len(basis) == 1


def test_apply_returns_none_when_query_projects_to_zero():
    # An adapter whose basis is orthogonal to the query yields an all-zero code,
    # so `not any(code)` short-circuits to None (line 246) before any scoring.
    adapter = LocalAdapter(
        embed_dim=2,
        rank=1,
        gate=0.5,
        scale=1.0,
        basis=[[1.0, 0.0]],
        codes=[[1.0]],
        targets=["taught"],
        supports=[1.0],
    )
    # Query lies entirely along the axis NOT in the basis -> projection is 0.
    assert adapter.apply([0.0, 1.0]) is None


def test_apply_inert_when_scale_zero_or_empty():
    base = LocalAdapter(embed_dim=2, basis=[[1.0, 0.0]], codes=[[1.0]], targets=["t"])
    assert base.model_copy(update={"scale": 0.0}).apply([1.0, 0.0]) is None
    assert base.model_copy(update={"codes": []}).apply([1.0, 0.0]) is None
    assert base.model_copy(update={"basis": []}).apply([1.0, 0.0]) is None


def test_apply_dim_mismatch_message():
    adapter = LocalAdapter(embed_dim=4, basis=[[1.0, 0, 0, 0]], codes=[[1.0]], targets=["t"])
    with pytest.raises(AdaptationError, match=r"query embedding dim 2 != adapter dim 4"):
        adapter.apply([1.0, 0.0])


def test_apply_below_gate_returns_none():
    # A real fit, but a query far off-distribution scores below the gate.
    adapter = LocalAdapter(
        embed_dim=2, rank=1, gate=0.99, scale=1.0,
        basis=[[1.0, 0.0]], codes=[[1.0]], targets=["t"], supports=[1.0],
    )
    # Code along the axis but slightly off similarity won't matter here: use a
    # query whose unit projection inner product < gate by orthogonal mixing.
    assert adapter.apply([0.1, 1.0]) is None


def test_apply_support_defaults_when_missing():
    # supports shorter than codes -> falls back to 1.0 for the matched index.
    adapter = LocalAdapter(
        embed_dim=2, rank=1, gate=0.0, scale=1.0,
        basis=[[1.0, 0.0]], codes=[[1.0]], targets=["t"], supports=[],
    )
    hit = adapter.apply([1.0, 0.0])
    assert hit is not None
    assert hit.support == 1.0


# ---------------------------------------------------------------------------
# Trainer default-embedder & native backend (lines 358-360, 395-396)
# ---------------------------------------------------------------------------


async def test_fit_uses_default_embedder_when_none():
    # No embedder injected -> the trainer builds a LocalHashEmbedder itself.
    adapter = await LocalLoRATrainer(rank=4).fit(training_set(), "gguf")
    assert adapter.embed_dim > 0
    assert adapter.provenance["embedder"] == "LocalHashEmbedder"
    assert adapter.lora_path is None


async def test_fit_records_native_backend_path():
    class FakeBackend:
        name = "fake-gguf"

        async def train(self, training_jsonl: str, base_model: str) -> str:
            assert base_model == "gguf"
            assert "refund" in training_jsonl  # real grounded JSONL was passed
            return "/tmp/adapter.gguf"

    trainer = LocalLoRATrainer(embedder=LocalHashEmbedder(), rank=4, backend=FakeBackend())
    adapter = await trainer.fit(training_set(), "gguf")
    assert adapter.lora_path == "/tmp/adapter.gguf"
    assert adapter.provenance["native_backend"] == "fake-gguf"


async def test_fit_refuses_below_min_examples():
    small = TrainingSet(
        name="x",
        examples=[
            TrainingExample(messages=[{"role": "user", "content": "q"},
                                      {"role": "assistant", "content": "a"}])
        ],
    )
    with pytest.raises(AdaptationError, match=r"needs at least 3 grounded examples; got 1"):
        await LocalLoRATrainer(embedder=LocalHashEmbedder()).fit(small, "gguf", min_examples=3)


# ---------------------------------------------------------------------------
# AdaptedProvider: default embedder & passthroughs (433-435, 445, 448, 482-499)
# ---------------------------------------------------------------------------


async def test_adapted_provider_builds_default_embedder():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=4, gate=0.5).fit(training_set(), "gguf")
    # No embedder passed -> wrapper constructs a LocalHashEmbedder sized to dim.
    provider = AdaptedProvider(MockProvider(default_text="GENERIC"), adapter)
    assert provider.embedder.dim == adapter.embed_dim
    # The wrapper's own deterministic embedder matches what the adapter was fit
    # with, so the exact training query lands in-distribution and is answered the
    # grounded way it was taught (not the base model's GENERIC).
    req = ModelRequest(model="gguf", messages=[Message(role="user", content=QA[0][0])])
    resp = await provider.generate(req)
    assert resp.text == QA[0][1]
    assert resp.raw["adapter"]["name"] == adapter.name


async def test_match_returns_none_when_no_user_turn():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=4).fit(training_set(), "gguf")
    provider = AdaptedProvider(MockProvider(default_text="BASE"), adapter, embedder=emb)
    # Only a system message -> empty query -> falls through to the base model.
    req = ModelRequest(model="gguf", messages=[Message(role="system", content="you are a bot")])
    resp = await provider.generate(req)
    assert resp.text == "BASE"


async def test_match_returns_none_on_embedder_dim_mismatch():
    fit_emb = LocalHashEmbedder(dim=256)
    adapter = await LocalLoRATrainer(embedder=fit_emb, rank=4).fit(training_set(), "gguf")
    # Wrap with a DIFFERENT-dimensioned embedder: produced vector length != adapter
    # dim, so _match returns None (line 448) and we fall through to base.
    wrong_emb = LocalHashEmbedder(dim=128)
    provider = AdaptedProvider(MockProvider(default_text="BASE"), adapter, embedder=wrong_emb)
    req = ModelRequest(model="gguf", messages=[Message(role="user", content=QA[0][0])])
    resp = await provider.generate(req)
    assert resp.text == "BASE"


async def test_stream_emits_adapter_chunks_on_hit():
    # An in-distribution request streams the learned target in <=16-char chunks
    # plus a usage and a done event (lines 485-490).
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=4, gate=0.5).fit(training_set(), "gguf")
    provider = AdaptedProvider(MockProvider(default_text="BASE"), adapter, embedder=emb)
    req = ModelRequest(model="gguf", messages=[Message(role="user", content=QA[0][0])])
    events = [ev async for ev in provider.stream(req)]
    text = "".join(ev.text for ev in events if ev.type == "text_delta")
    assert text == QA[0][1]
    assert all(len(ev.text) <= 16 for ev in events if ev.type == "text_delta")
    assert events[-1].type == "done"
    assert events[-1].response.text == QA[0][1]
    assert any(ev.type == "usage" for ev in events)


async def test_stream_falls_through_on_miss():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=4).fit(training_set(), "gguf")
    provider = AdaptedProvider(MockProvider(default_text="BASE-STREAM"), adapter, embedder=emb)
    req = ModelRequest(
        model="gguf",
        messages=[Message(role="user", content="totally unrelated zzz qqq")],
    )
    texts = [ev.text async for ev in provider.stream(req) if ev.type == "text_delta"]
    assert "".join(texts) == "BASE-STREAM"


async def test_provider_passthroughs_delegate_to_base():
    emb = LocalHashEmbedder()
    adapter = await LocalLoRATrainer(embedder=emb, rank=4).fit(training_set(), "gguf")
    base = MockProvider(default_text="BASE")
    provider = AdaptedProvider(base, adapter, embedder=emb)
    # name + requires_api_key are taken from the base (transparency).
    assert provider.name == base.name
    assert provider.requires_api_key == getattr(base, "requires_api_key", False)
    # embed / capabilities / list_models / aclose all delegate.
    vecs = await provider.embed(["hello"])
    assert vecs == await base.embed(["hello"])
    assert provider.capabilities("mock") == base.capabilities("mock")
    assert await provider.list_models() == await base.list_models()
    await provider.aclose()  # must not raise


# ---------------------------------------------------------------------------
# Registry: latest / prune boundary / persistence reload (536, 542, 586-587, 607)
# ---------------------------------------------------------------------------


async def test_registry_latest_and_empty():
    reg = AdapterRegistry()
    assert reg.latest("missing") is None
    assert reg.active("missing") is None
    emb = LocalHashEmbedder()
    reg.register(await LocalLoRATrainer(embedder=emb).fit(training_set(), "gguf"))
    reg.register(await LocalLoRATrainer(embedder=emb, gate=0.7).fit(training_set(), "gguf"))
    latest = reg.latest("local-adapter")
    assert latest is not None
    assert latest.version == 2


async def test_prune_drops_oldest_and_keeps_recent():
    # keep < len(versions) -> the oldest are dropped, newest retained (611-613).
    reg = AdapterRegistry()
    emb = LocalHashEmbedder()
    for i in range(5):
        reg.register(
            await LocalLoRATrainer(embedder=emb, gate=0.5 + i * 0.05).fit(training_set(), "g")
        )
    dropped = reg.prune("local-adapter", 2)
    assert dropped == 3
    kept = reg.versions("local-adapter")
    assert [a.version for a in kept] == [4, 5]


def test_adapter_len_reports_target_count():
    # __len__ (line 186) is the number of learned targets.
    adapter = LocalAdapter(
        embed_dim=2, basis=[[1.0, 0.0]], codes=[[1.0], [0.0]], targets=["a", "b"]
    )
    assert len(adapter) == 2


async def test_registry_get_missing_version_raises():
    reg = AdapterRegistry()
    with pytest.raises(AdaptationError, match=r"no adapter 'local-adapter' version 5"):
        reg.get("local-adapter", 5)


async def test_prune_keep_non_positive_is_noop():
    reg = AdapterRegistry()
    emb = LocalHashEmbedder()
    reg.register(await LocalLoRATrainer(embedder=emb).fit(training_set(), "gguf"))
    # keep <= 0 returns 0 immediately and drops nothing (line 607).
    assert reg.prune("local-adapter", 0) == 0
    assert reg.prune("local-adapter", -3) == 0
    assert len(reg.versions("local-adapter")) == 1


async def test_registry_reload_restores_head_pointer(tmp_path):
    emb = LocalHashEmbedder()
    reg = AdapterRegistry(directory=tmp_path)
    reg.register(await LocalLoRATrainer(embedder=emb).fit(training_set(), "gguf"))
    reg.register(await LocalLoRATrainer(embedder=emb, gate=0.7).fit(training_set(), "gguf"))
    reg.rollback("local-adapter", 1)  # head pointer now v1, persisted to HEAD file
    # A fresh registry over the same directory reads HEAD (line 540) and the two
    # versioned artifacts (line 534).
    reloaded = AdapterRegistry(directory=tmp_path)
    assert [a.version for a in reloaded.versions("local-adapter")] == [1, 2]
    active = reloaded.active("local-adapter")
    assert active is not None
    assert active.version == 1


async def test_registry_reload_defaults_head_to_last(tmp_path):
    emb = LocalHashEmbedder()
    reg = AdapterRegistry(directory=tmp_path)
    reg.register(await LocalLoRATrainer(embedder=emb).fit(training_set(), "gguf"))
    reg.register(await LocalLoRATrainer(embedder=emb, gate=0.7).fit(training_set(), "gguf"))
    # Remove the HEAD file so the reload takes the `else` branch (line 542):
    # default head = last loaded version.
    (tmp_path / "local-adapter" / "HEAD").unlink()
    reloaded = AdapterRegistry(directory=tmp_path)
    active = reloaded.active("local-adapter")
    assert active is not None
    assert active.version == 2


def test_registry_reload_skips_dir_without_versions(tmp_path):
    # A name directory with no v*.json is skipped (continue, line 536).
    (tmp_path / "empty-name").mkdir()
    reg = AdapterRegistry(directory=tmp_path)
    assert reg.versions("empty-name") == []
    assert reg.active("empty-name") is None


# ---------------------------------------------------------------------------
# ContinualAdaptation: trace curation, insufficient data, safety-gate overlay
# (787-793, 851-858, 896-902, 942->944)
# ---------------------------------------------------------------------------


async def test_loop_curates_from_app_traces():
    # No runs and no training_set: the loop pulls captured traces off the app's
    # memory exporter (load_all path, lines 787-793). The mock answers each
    # golden question with its expected text, so the traces ground cleanly.
    answers = dict(QA)

    def responder(req):
        q = next((m.text for m in reversed(req.messages) if m.role == "user"), "")
        return answers.get(q, "unknown")

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge", provider=MockProvider(responder=responder), config=cfg)
    for q, _ in QA:
        app.run(q)
    policy = LocalAdaptationPolicy(min_examples=1, require_grounding=False)
    controller = ContinualAdaptation(app, policy)
    ts = controller._build_training_set(None, None)
    # Each captured trace became a curated example.
    assert len(ts) >= 1


async def test_loop_curates_via_load_all_exporter(tmp_path):
    # When the tracer's exporter exposes load_all (the JSONL/persistent path),
    # the loop pulls traces through it (line 790) rather than the in-memory
    # `traces` list.
    from vincio.observability.exporters import JSONLExporter

    answers = dict(QA)

    def responder(req):
        q = next((m.text for m in reversed(req.messages) if m.role == "user"), "")
        return answers.get(q, "unknown")

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge", provider=MockProvider(responder=responder), config=cfg)
    for q, _ in QA:
        app.run(q)
    # Swap to a load_all-capable exporter holding the just-produced traces.
    exporter = JSONLExporter(directory=tmp_path / "traces")
    for trace in app.tracer.exporter.traces:
        exporter.export(trace)
    app.tracer.exporter = exporter
    assert not hasattr(exporter, "traces")  # forces the load_all branch
    controller = ContinualAdaptation(app, LocalAdaptationPolicy(require_grounding=False))
    ts = controller._build_training_set(None, None)
    assert len(ts) >= 1


async def test_loop_with_barebones_exporter_yields_empty_corpus():
    # An exporter exposing neither load_all nor traces leaves the corpus empty
    # (the 791->793 fall-through): both attr branches are skipped.
    class BareExporter:
        def export(self, trace):  # noqa: ANN001, D401 - minimal sink
            pass

    app = make_app()
    app.tracer.exporter = BareExporter()
    controller = ContinualAdaptation(app, LocalAdaptationPolicy(require_grounding=False))
    ts = controller._build_training_set(None, None)
    assert len(ts) == 0


async def test_loop_refuses_when_corpus_too_small():
    app = make_app()
    policy = LocalAdaptationPolicy(min_examples=10)  # more than our 4 examples
    controller = ContinualAdaptation(app, policy, dataset=golden())
    phases = [ev.phase async for ev in controller.astream(training_set=training_set())]
    assert phases == ["observe", "exhausted"]
    assert "refusing to fit an adapter" in controller.result.reason
    assert controller.result.promoted is False


async def test_loop_no_dataset_fits_but_skips_gate():
    app = make_app()
    policy = LocalAdaptationPolicy(min_examples=4)
    controller = ContinualAdaptation(app, policy)  # no dataset bound
    result = await controller.aadapt(training_set=training_set())
    assert result.promoted is False
    assert "cannot gate" in result.reason
    phases = [e.phase for e in controller.events]
    assert phases == ["observe", "train", "gate"]


async def test_safety_gate_overlay_blocks_passing_adapter():
    # Metric gate would pass (the adapter answers the goldens verbatim), but an
    # impossible safety gate forces verdict.passed False (lines 896-902).
    answers = dict(QA)

    def responder(req):
        q = next((m.text for m in reversed(req.messages) if m.role == "user"), "")
        return answers.get(q, "GENERIC OFF DISTRIBUTION")

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge", provider=MockProvider(responder=responder), config=cfg)
    policy = LocalAdaptationPolicy(
        min_examples=4,
        min_samples=4,
        require_significance=False,
        gates={"lexical_overlap": ">= 2.0"},  # impossible to satisfy
    )
    controller = ContinualAdaptation(app, policy, dataset=golden())
    result = await controller.aadapt(training_set=training_set())
    assert result.promoted is False
    assert result.verdict is not None
    assert result.verdict.passed is False
    assert "safety gates failed" in result.verdict.reason
    # Refused -> a rollback event was emitted.
    assert any(e.phase == "rollback" for e in controller.events)


def test_adapt_sync_promotes_and_installs():
    # The sync wrapper (line 978) runs the loop to completion; apply=True installs
    # the adapter on the app (line 943) so subsequent runs are shaped.
    answers = dict(QA)

    def responder(req):
        q = next((m.text for m in reversed(req.messages) if m.role == "user"), "")
        return answers.get(q, "GENERIC")

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge", provider=MockProvider(responder=responder), config=cfg)
    # A passing safety gate (>= 0.0 always holds) exercises the no-failures branch.
    policy = LocalAdaptationPolicy(
        min_examples=4,
        min_samples=4,
        require_significance=False,
        gates={"lexical_overlap": ">= 0.0"},
    )
    controller = ContinualAdaptation(app, policy, dataset=golden())
    result = controller.adapt(training_set=training_set(), apply=True)
    assert result.promoted is True
    assert result.adapter_version == 1
    # apply=True swapped the live provider for an AdaptedProvider.
    assert isinstance(app._base_provider(), AdaptedProvider)


async def test_dry_run_passes_gate_but_does_not_promote():
    # A passing gate under dry_run yields a "would be promoted" event and leaves
    # the registry empty (lines 933-938).
    answers = dict(QA)

    def responder(req):
        q = next((m.text for m in reversed(req.messages) if m.role == "user"), "")
        return answers.get(q, "GENERIC")

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge", provider=MockProvider(responder=responder), config=cfg)
    policy = LocalAdaptationPolicy(
        min_examples=4, min_samples=4, require_significance=False, dry_run=True
    )
    registry = AdapterRegistry()
    controller = ContinualAdaptation(app, policy, dataset=golden(), registry=registry)
    result = await controller.aadapt(training_set=training_set())
    assert result.promoted is False
    assert "dry run" in result.reason
    assert registry.active("local-adapter") is None
    assert controller.events[-1].phase == "promote"
    assert controller.events[-1].action == "dry_run"


async def test_build_training_set_from_runs():
    # Passing runs (not a prebuilt set, not None) routes through
    # export_training_set_from_runs (line 779).
    app = make_app()
    runs = [app.run(q) for q, _ in QA]
    policy = LocalAdaptationPolicy(require_grounding=False, min_support=0.0)
    controller = ContinualAdaptation(app, policy)
    ts = controller._build_training_set(runs, None)
    assert len(ts) >= 1
    assert ts.name == policy.name


async def test_build_training_set_returns_prebuilt_unchanged():
    # When a training_set is passed it is returned verbatim (line 779) — no
    # curation, no traces consulted.
    app = make_app()
    controller = ContinualAdaptation(app, LocalAdaptationPolicy())
    prebuilt = training_set()
    assert controller._build_training_set(None, prebuilt) is prebuilt


async def test_promote_without_apply_does_not_install(tmp_path):
    # apply=False exercises the `if apply` false branch (942->944): adapter is
    # registered & promoted but NOT installed on the app.
    answers = dict(QA)

    def responder(req):
        q = next((m.text for m in reversed(req.messages) if m.role == "user"), "")
        return answers.get(q, "GENERIC")

    cfg = VincioConfig()
    cfg.observability.exporter = "memory"
    app = ContextApp(name="edge", provider=MockProvider(responder=responder), config=cfg)
    policy = LocalAdaptationPolicy(min_examples=4, min_samples=4, require_significance=False)
    registry = AdapterRegistry(directory=tmp_path)
    controller = ContinualAdaptation(app, policy, dataset=golden(), registry=registry)
    result = await controller.aadapt(training_set=training_set(), apply=False)
    assert result.promoted is True
    assert result.adapter_version == 1
    # Registered in the registry...
    assert registry.active("local-adapter") is not None
    # ...but the app's live provider was never swapped to an AdaptedProvider.
    assert not isinstance(app._base_provider(), AdaptedProvider)
