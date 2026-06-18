"""Executed distillation & provider fine-tune jobs: cassette-backed
submit/poll lifecycles, the executed StudentTrainer that registers the trained
model, export semantic-dedup + truncation guard, and swap-gate-backed
promotion. All offline and deterministic."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from vincio.core.errors import FineTuneError
from vincio.core.types import ModelCapabilities, ModelProfile
from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.reports import CaseResult, EvalReport
from vincio.optimize import (
    BootstrapFinetune,
    TrainingExample,
    TrainingSet,
    export_training_set_from_runs,
    provider_trainer,
    semantic_dedupe,
)
from vincio.providers import (
    AnthropicFineTuneBackend,
    FineTuneStatus,
    GoogleFineTuneBackend,
    OpenAIFineTuneBackend,
    make_finetune_backend,
    run_finetune,
)
from vincio.providers.anthropic import AnthropicProvider
from vincio.providers.google import GoogleProvider
from vincio.providers.openai import OpenAIProvider
from vincio.providers.registry import ModelRegistry


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _openai(handler) -> OpenAIProvider:
    return OpenAIProvider(api_key="sk-test", client=_client(handler))


def _openai_ok(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/files"):
        return httpx.Response(200, json={"id": "file-abc", "object": "file"})
    if path.endswith("/fine_tuning/jobs"):
        return httpx.Response(
            200,
            json={"id": "ftjob-1", "status": "queued", "model": "gpt-5.2-mini", "fine_tuned_model": None},
        )
    if "/fine_tuning/jobs/" in path:
        return httpx.Response(
            200,
            json={
                "id": "ftjob-1",
                "status": "succeeded",
                "model": "gpt-5.2-mini",
                "fine_tuned_model": "ft:gpt-5.2-mini:vincio:1",
                "trained_tokens": 4096,
            },
        )
    return httpx.Response(404, json={})


def _report(quality: float, *, n: int = 8, cost: float = 0.001) -> EvalReport:
    metrics = {
        "lexical_overlap": quality,
        "groundedness": quality,
        "schema_validity": 1.0,
        "safety": 1.0,
        "cost": cost,
        "latency": 50.0,
    }
    return EvalReport(cases=[CaseResult(case_id=f"c{i}", metrics=dict(metrics)) for i in range(n)])


def _dataset(n: int = 8) -> Dataset:
    return Dataset(name="d", cases=[EvalCase(id=f"c{i}", input="q", expected="a") for i in range(n)])


def _grounded_set() -> TrainingSet:
    return TrainingSet(
        name="g",
        examples=[
            TrainingExample(
                messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                grounded=True,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Provider fine-tune backends (cassette-backed)
# ---------------------------------------------------------------------------


class TestOpenAIFineTuneBackend:
    async def test_submit_then_poll_to_success(self):
        backend = OpenAIFineTuneBackend(_openai(_openai_ok))
        job = await backend.submit('{"messages":[]}', "gpt-5.2-mini", suffix="vincio")
        assert job.status is FineTuneStatus.RUNNING and not job.done
        done = await backend.poll(job)
        assert done.status is FineTuneStatus.SUCCEEDED
        assert done.fine_tuned_model == "ft:gpt-5.2-mini:vincio:1"
        assert done.trained_tokens == 4096

    async def test_run_finetune_drives_to_terminal(self):
        backend = OpenAIFineTuneBackend(_openai(_openai_ok))
        job = await run_finetune(backend, '{"messages":[]}', "gpt-5.2-mini", poll_interval_s=0)
        assert job.fine_tuned_model == "ft:gpt-5.2-mini:vincio:1"

    async def test_run_finetune_raises_on_failure(self):
        def handler(request):
            if request.url.path.endswith("/files"):
                return httpx.Response(200, json={"id": "file-x"})
            if request.url.path.endswith("/fine_tuning/jobs"):
                return httpx.Response(200, json={"id": "j", "status": "running"})
            return httpx.Response(200, json={"id": "j", "status": "failed", "error": {"message": "bad data"}})

        backend = OpenAIFineTuneBackend(_openai(handler))
        with pytest.raises(FineTuneError, match="failed"):
            await run_finetune(backend, "{}", "gpt-5.2-mini", poll_interval_s=0)

    async def test_run_finetune_raises_without_model_id(self):
        def handler(request):
            if request.url.path.endswith("/files"):
                return httpx.Response(200, json={"id": "file-x"})
            if request.url.path.endswith("/fine_tuning/jobs"):
                return httpx.Response(200, json={"id": "j", "status": "running"})
            return httpx.Response(200, json={"id": "j", "status": "succeeded", "fine_tuned_model": None})

        backend = OpenAIFineTuneBackend(_openai(handler))
        with pytest.raises(FineTuneError, match="no model id"):
            await run_finetune(backend, "{}", "gpt-5.2-mini", poll_interval_s=0)


class TestGoogleFineTuneBackend:
    async def test_tuned_model_lifecycle(self):
        def handler(request):
            path = request.url.path
            if path.endswith("/tunedModels"):
                return httpx.Response(
                    200,
                    json={"name": "tunedModels/abc-123", "state": "CREATING", "baseModel": "models/gemini-2.5-flash"},
                )
            if "/tunedModels/" in path:
                return httpx.Response(200, json={"name": "tunedModels/abc-123", "state": "ACTIVE"})
            return httpx.Response(404, json={})

        provider = GoogleProvider(api_key="g", client=_client(handler))
        backend = GoogleFineTuneBackend(provider)
        jsonl = '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}'
        job = await run_finetune(backend, jsonl, "gemini-2.5-flash", poll_interval_s=0)
        assert job.fine_tuned_model == "tunedModels/abc-123"
        assert job.status is FineTuneStatus.SUCCEEDED


class TestBackendFactory:
    def test_dispatch_by_provider_name(self):
        assert isinstance(make_finetune_backend(_openai(_openai_ok)), OpenAIFineTuneBackend)
        assert isinstance(
            make_finetune_backend(GoogleProvider(api_key="g")), GoogleFineTuneBackend
        )
        assert isinstance(
            make_finetune_backend(AnthropicProvider(api_key="a")), AnthropicFineTuneBackend
        )

    def test_unsupported_provider_raises(self):
        with pytest.raises(FineTuneError):
            make_finetune_backend(SimpleNamespace(name="mock"))


# ---------------------------------------------------------------------------
# Executed StudentTrainer: trains + registers + returns model id
# ---------------------------------------------------------------------------


async def test_provider_trainer_registers_trained_student():
    registry = ModelRegistry()
    registry.register(
        ModelProfile(
            name="gpt-5.2-mini",
            provider="openai",
            model="gpt-5.2-mini",
            capabilities=ModelCapabilities(structured_output=True),
            input_cost_per_mtok=0.4,
            output_cost_per_mtok=1.6,
        )
    )
    trainer = provider_trainer(
        OpenAIFineTuneBackend(_openai(_openai_ok)), registry=registry, poll_interval_s=0
    )
    model_id = await trainer(_grounded_set(), "gpt-5.2-mini")
    assert model_id == "ft:gpt-5.2-mini:vincio:1"
    profile = registry.get(model_id)
    assert profile is not None
    assert profile.tier == "fast"  # the student is the cheap rung
    assert profile.input_cost_per_mtok == 0.4  # inherited from the base
    assert profile.capabilities.structured_output is True


# ---------------------------------------------------------------------------
# Export guards: truncation + semantic dedup
# ---------------------------------------------------------------------------


def _run(inp: str, out: str) -> SimpleNamespace:
    return SimpleNamespace(
        status="succeeded", metadata={"input": inp}, raw_text=out, evidence=[], run_id="r", trace_id="t"
    )


def test_truncation_guard_drops_overlong_examples():
    runs = [_run("short", "ok answer"), _run("q2", "x" * 5000)]
    ts = export_training_set_from_runs(runs, require_grounding=False, max_example_chars=1000)
    assert ts.metadata["dropped_truncated"] == 1
    assert len(ts.examples) == 1


async def test_semantic_dedupe_drops_paraphrase_duplicates():
    ts = TrainingSet(
        examples=[
            TrainingExample(messages=[{"role": "user", "content": "What is the refund window?"},
                                      {"role": "assistant", "content": "30 days."}]),
            TrainingExample(messages=[{"role": "user", "content": "What is the refund window?"},
                                      {"role": "assistant", "content": "30 days."}]),
            TrainingExample(messages=[{"role": "user", "content": "How do I reset my password?"},
                                      {"role": "assistant", "content": "Use the reset link in settings."}]),
        ]
    )
    deduped = await semantic_dedupe(ts, threshold=0.97)
    assert len(deduped.examples) == 2
    assert deduped.metadata["semantic_dropped"] == 1


# ---------------------------------------------------------------------------
# Swap-gate-backed promotion
# ---------------------------------------------------------------------------


class _FakeSwapGate:
    def __init__(self, passed: bool, reason: str = "no regression") -> None:
        self._passed = passed
        self._reason = reason
        self.calls: list[tuple[str, str]] = []

    async def evaluate(self, *, candidate_model, baseline_model, dataset):
        self.calls.append((candidate_model, baseline_model))
        return SimpleNamespace(passed=self._passed, reason=self._reason)


def _evaluator(quality_by_model):
    async def ev(model, ds):
        q, cost = quality_by_model[model]
        return _report(q, n=len(ds), cost=cost)

    return ev


async def test_swap_gate_allows_promotion():
    gate = _FakeSwapGate(True)
    loop = BootstrapFinetune(
        _evaluator({"teacher": (0.95, 0.01), "student": (0.93, 0.002)}),
        min_quality_ratio=0.9,
        swap_gate=gate,
    )
    res = await loop.distill(_grounded_set(), _dataset(), teacher="teacher", student="student")
    assert res.promoted is True
    assert res.swap_passed is True
    assert gate.calls == [("student", "teacher")]


async def test_swap_gate_blocks_promotion():
    loop = BootstrapFinetune(
        _evaluator({"teacher": (0.95, 0.01), "student": (0.93, 0.002)}),
        min_quality_ratio=0.9,
        swap_gate=_FakeSwapGate(False, "lexical_overlap regressed (p=0.01)"),
    )
    res = await loop.distill(_grounded_set(), _dataset(), teacher="teacher", student="student")
    assert res.promoted is False
    assert res.swap_passed is False
    assert "swap gate" in res.reason and "regressed" in res.reason


async def test_distill_semantic_dedupe_threshold_applied():
    # Two identical grounded examples collapse to one before training.
    dup = TrainingExample(
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], grounded=True
    )
    ts = TrainingSet(examples=[dup, dup.model_copy()])
    loop = BootstrapFinetune(_evaluator({"teacher": (0.95, 0.01), "student": (0.93, 0.002)}), min_quality_ratio=0.9)
    res = await loop.distill(
        ts, _dataset(), teacher="teacher", student="student", semantic_dedupe_threshold=0.97
    )
    assert res.training_examples == 1
