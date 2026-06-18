"""Provider/model rotation & swap regression.

Covers the capability guard, the registry-backed router, the swap gate +
model-swap regression with flake control, the shadow & canary providers with
auto-rollback, the lifecycle watcher + migration proposals, and live model
discovery + Google batch parity — all offline against the deterministic mock.
"""

from __future__ import annotations

from datetime import date

import pytest

from vincio import ContextApp
from vincio.core.errors import CapabilityMismatchError, ModelRetiredError, ProviderUnavailableError
from vincio.core.types import (
    ContentPart,
    ImageRef,
    Message,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    RunConfig,
    TokenUsage,
)
from vincio.evals.datasets import Dataset, EvalCase
from vincio.evals.swap import SwapGate, behavioral_shapes, model_swap_regression
from vincio.optimize.routing import ModelCascade, Router
from vincio.providers import MockProvider
from vincio.providers.base import FailoverChain, is_lifecycle_error
from vincio.providers.batch import (
    BatchRequest,
    BatchRunner,
    GoogleBatchBackend,
    InProcessBatchBackend,
)
from vincio.providers.capabilities import capability_check, requirements_for
from vincio.providers.circuit import HealthAwareFailover
from vincio.providers.discovery import discover_models
from vincio.providers.lifecycle import LifecycleWatcher
from vincio.providers.registry import ModelRegistry, default_model_registry
from vincio.providers.shadow import CanaryRouter, ShadowProvider


def _text_req(text: str = "hello") -> ModelRequest:
    return ModelRequest(model="x", messages=[Message(role="user", content=text)])


def _vision_req() -> ModelRequest:
    return ModelRequest(
        model="x",
        messages=[Message(role="user", content=[
            ContentPart(type="image", image=ImageRef(url="data:image/png;base64,AAAA"))
        ])],
    )


@pytest.fixture()
def app(offline_config):
    def responder(req):
        return "The capital of France is Paris." if req.model != "gpt-5.2-nano" else "wrong"

    return ContextApp(
        name="rot", provider=MockProvider(responder=responder), model="gpt-5.2",
        config=offline_config,
    )


@pytest.fixture()
def geo_dataset():
    return Dataset(name="geo", cases=[
        EvalCase(id=f"c{i}", input="What is the capital of France?",
                 expected="The capital of France is Paris.")
        for i in range(6)
    ])


# --------------------------------------------------------------------------- #
# Capability preflight
# --------------------------------------------------------------------------- #


class TestCapabilityGuard:
    def test_requirements_detect_modalities_and_tools(self):
        needs = requirements_for(_vision_req())
        assert needs.vision is True
        req = ModelRequest(model="x", messages=[Message(role="user", content="hi")],
                           output_schema={"type": "object"}, reasoning_effort="high")
        needs2 = requirements_for(req, input_tokens=5000)
        assert needs2.structured_output and needs2.reasoning
        assert needs2.min_context_tokens == 5000

    def test_known_incapable_is_blocked(self):
        reg = default_model_registry()
        needs = requirements_for(_vision_req())
        verdict = capability_check(needs, reg.capabilities("mistral-small-latest"))
        assert not verdict.ok and "vision" in verdict.missing

    def test_capable_passes(self):
        reg = default_model_registry()
        verdict = capability_check(requirements_for(_vision_req()), reg.capabilities("gpt-5.2"))
        assert verdict.ok

    def test_unknown_model_never_blocked(self):
        verdict = capability_check(requirements_for(_vision_req()), None, model="who-knows")
        assert verdict.ok and verdict.known is False

    def test_context_window_requirement(self):
        reg = default_model_registry()
        needs = requirements_for(_text_req(), input_tokens=300_000)
        verdict = capability_check(needs, reg.capabilities("claude-sonnet-4-6"))
        assert not verdict.ok  # 200k window < 300k


# --------------------------------------------------------------------------- #
# Failover + lifecycle classification
# --------------------------------------------------------------------------- #


class TestFailoverGuard:
    async def test_failover_skips_incapable_model(self):
        chain = FailoverChain([
            (MockProvider(default_text="mistral"), "mistral-small-latest"),
            (MockProvider(default_text="vision-ok"), "claude-sonnet-4-6"),
        ])
        assert (await chain.generate(_vision_req())).text == "vision-ok"

    async def test_all_incapable_raises_capability_mismatch(self):
        chain = FailoverChain([
            (MockProvider(default_text="a"), "mistral-small-latest"),
            (MockProvider(default_text="b"), "mistral-large-latest"),
        ])
        with pytest.raises(CapabilityMismatchError):
            await chain.generate(_vision_req())

    async def test_retired_model_raises_rotate_now(self):
        reg = ModelRegistry([
            ModelProfile(name="old", provider="x", model="old-model", retirement_date="2020-01-01")
        ])
        chain = FailoverChain([(MockProvider(default_text="x"), "old-model")], registry=reg)
        with pytest.raises(ModelRetiredError):
            await chain.generate(_text_req())

    async def test_guard_off_attempts_everything(self):
        chain = FailoverChain(
            [(MockProvider(default_text="served"), "mistral-small-latest")],
            guard_capabilities=False,
        )
        assert (await chain.generate(_vision_req())).text == "served"

    def test_lifecycle_error_classification(self):
        assert is_lifecycle_error(ProviderUnavailableError("model_not_found: gpt-3"))
        assert is_lifecycle_error(ModelRetiredError("retired"))
        assert not is_lifecycle_error(ProviderUnavailableError("temporary overload"))

    async def test_health_aware_failover_guards_capabilities(self):
        chain = HealthAwareFailover([
            (MockProvider(default_text="mistral"), "mistral-small-latest"),
            (MockProvider(default_text="vision-ok"), "gpt-5.2"),
        ])
        assert (await chain.generate(_vision_req())).text == "vision-ok"


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


class TestRouter:
    def test_picks_cheapest_capable(self):
        router = Router.from_models(
            MockProvider(), ["gpt-5.2", "gpt-5.2-mini", "gpt-5.2-nano"], strategy="cheapest"
        )
        assert router.pick(_text_req()).model == "gpt-5.2-nano"

    def test_budget_downgrade(self):
        router = Router.from_models(MockProvider(), ["gpt-5.2", "gpt-5.2-nano"], strategy="fastest")
        decision = router.pick(_text_req(), budget_usd=0.0)
        assert decision.downgraded and decision.model == "gpt-5.2-nano"

    def test_capability_filter_skips_incapable(self):
        router = Router.from_models(
            MockProvider(), ["mistral-small-latest", "gpt-5.2"], strategy="cheapest"
        )
        decision = router.pick(_vision_req())
        assert decision.model == "gpt-5.2"
        assert "mistral-small-latest" in decision.skipped

    def test_no_capable_model_raises(self):
        router = Router.from_models(MockProvider(), ["mistral-small-latest"], strategy="cheapest")
        with pytest.raises(CapabilityMismatchError):
            router.pick(_vision_req())

    async def test_generate_dispatches_to_pick_and_emits(self):
        events = []

        class Bus:
            def emit(self, name, payload):
                events.append((name, payload))

        router = Router(
            [(MockProvider(default_text="cheap"), "gpt-5.2-nano"),
             (MockProvider(default_text="strong"), "gpt-5.2")],
            strategy="cheapest", events=Bus(),
        )
        resp = await router.generate(_text_req())
        assert resp.text == "cheap"
        assert router.last_decision.model == "gpt-5.2-nano"
        assert any(name == "model.routed" for name, _ in events)


# --------------------------------------------------------------------------- #
# Cascade capability awareness
# --------------------------------------------------------------------------- #


class TestCascadeCapability:
    def test_first_capable_skips_incapable(self):
        cascade = ModelCascade.from_models(["mistral-small-latest", "gpt-5.2"])
        reg = default_model_registry()
        needs = requirements_for(_vision_req())

        def is_capable(m):
            return capability_check(needs, reg.capabilities(m)).ok

        assert cascade.first_capable(is_capable).model == "gpt-5.2"

    def test_next_rung_capable_walks_past_incapable(self):
        cascade = ModelCascade.from_models(
            ["gpt-5.2-nano", "mistral-small-latest", "gpt-5.2"], min_confidence=0.9
        )
        reg = default_model_registry()
        needs = requirements_for(_vision_req())

        def is_capable(m):
            return capability_check(needs, reg.capabilities(m)).ok

        nxt = cascade.next_rung_capable("gpt-5.2-nano", confidence=0.0, is_capable=is_capable)
        assert nxt is not None and nxt.model == "gpt-5.2"

    async def test_runtime_cascade_starts_on_capable_rung(self, offline_config):
        app = ContextApp(
            name="casc", provider=MockProvider(responder=lambda r: r.model), model="gpt-5.2",
            config=offline_config,
        )
        # mistral-small lacks reasoning; a reasoning request must start on gpt-5.2.
        app.use_cascade(["mistral-small-latest", "gpt-5.2"])
        result = await app.arun("explain", config=RunConfig(reasoning_effort="high"))
        assert result.metadata["cascade"]["model"] == "gpt-5.2"


# --------------------------------------------------------------------------- #
# EvalRunner repeats + flake quarantine
# --------------------------------------------------------------------------- #


class TestRepeatsAndFlake:
    async def test_repeats_record_stdev(self):
        from vincio.evals.metrics import MetricResult, RunOutput
        from vincio.evals.runners import EvalRunner

        seq = iter([0.2, 0.9, 0.2, 0.9])

        async def target(case):
            return RunOutput(output="x")

        def flaky_metric(case, run):
            return MetricResult(name="quality", value=next(seq))

        runner = EvalRunner(target, metrics=[flaky_metric], repeats=2,
                            flake_quarantine=True, flake_threshold=0.1)
        ds = Dataset(name="f", cases=[EvalCase(id="a", input="q"), EvalCase(id="b", input="q")])
        report = await runner.arun(ds)
        assert all("flaky" in c.tags for c in report.cases)
        assert report.cases[0].details["repeats"]["n"] == 2

    async def test_flaky_cases_excluded_from_gates(self):
        from vincio.evals.metrics import MetricResult, RunOutput
        from vincio.evals.runners import EvalRunner

        values = {"steady": [0.95, 0.95], "noisy": [0.1, 0.9]}

        async def target(case):
            return RunOutput(output=case.id)

        def metric(case, run):
            v = values[case.id].pop(0)
            return MetricResult(name="quality", value=v)

        runner = EvalRunner(target, metrics=[metric], repeats=2, flake_quarantine=True,
                            flake_threshold=0.1, gates={"quality": ">= 0.9"})
        ds = Dataset(name="g", cases=[EvalCase(id="steady", input="q"), EvalCase(id="noisy", input="q")])
        report = await runner.arun(ds)
        # gate aggregates only the steady case (0.95), so it passes.
        assert report.gates["quality"]["passed"]
        assert report.metadata["flaky_excluded_from_gates"] == 1


# --------------------------------------------------------------------------- #
# Swap gate + regression
# --------------------------------------------------------------------------- #


class TestSwapGate:
    async def test_regression_detected(self, app, geo_dataset):
        report = await model_swap_regression(
            app, geo_dataset, baseline_model="gpt-5.2", candidate_model="gpt-5.2-nano", repeats=2
        )
        assert report.regressed and "lexical_overlap" in report.regressions
        assert report.cost["ratio"] < 1.0  # nano is cheaper

    async def test_gate_blocks_regression_passes_safe(self, app, geo_dataset):
        bad = await app.agate_swap("gpt-5.2-nano", baseline_model="gpt-5.2", dataset=geo_dataset)
        good = await app.agate_swap("gpt-5.2-mini", baseline_model="gpt-5.2", dataset=geo_dataset)
        assert not bad.passed and good.passed
        assert bad.regression is not None

    async def test_gate_over_replayed_traces(self, app):
        from vincio.evals.replay import _CaptureExporter

        cap = _CaptureExporter(app.tracer.exporter)
        app.tracer.exporter = cap
        run = await app.arun("What is the capital of France?")
        trace = cap.captured[run.trace_id]
        app.tracer.exporter = cap._inner
        verdict = await app.agate_swap("gpt-5.2", baseline_model="gpt-5.2", traces=[trace])
        assert verdict.replay is not None and verdict.replay["status_regressions"] == 0

    def test_behavioral_shapes(self):
        from vincio.evals.reports import CaseResult, EvalReport

        report = EvalReport(cases=[
            CaseResult(case_id="a", metrics={"refusal_rate": 1.0, "tool_call_rate": 0.0,
                                             "output_length": 100.0}),
            CaseResult(case_id="b", metrics={"refusal_rate": 0.0, "tool_call_rate": 1.0,
                                             "output_length": 200.0}),
        ])
        shapes = behavioral_shapes(report)
        assert shapes["refusal_rate"] == 0.5
        assert shapes["tool_call_rate"] == 0.5
        assert shapes["output_length_mean"] == 150.0
        assert shapes["n"] == 2

    def test_gate_constructs(self, app):
        assert SwapGate(app).quality_metric == "lexical_overlap"


# --------------------------------------------------------------------------- #
# Shadow + canary
# --------------------------------------------------------------------------- #


class TestShadowAndCanary:
    async def test_shadow_returns_primary_records_both(self):
        shadow = ShadowProvider(
            MockProvider(default_text="primary"), MockProvider(default_text="candidate"),
            candidate_model="gpt-5.2-mini", block=True,
        )
        resp = await shadow.generate(_text_req())
        assert resp.text == "primary"
        diff = shadow.diff()
        assert diff["observations"] == 1 and diff["paired"] == 1

    async def test_shadow_isolates_candidate_error(self):
        class Boom(MockProvider):
            name = "boom"

            async def generate(self, request):
                raise ProviderUnavailableError("candidate down", provider="boom")

        shadow = ShadowProvider(MockProvider(default_text="primary"), Boom(), block=True)
        resp = await shadow.generate(_text_req())
        assert resp.text == "primary"
        assert shadow.observations[0].candidate_error is not None

    async def test_canary_auto_rollback(self):
        good = MockProvider(responder=lambda r: ModelResponse(
            model=r.model, text="ok", finish_reason="stop",
            usage=TokenUsage(input_tokens=5, output_tokens=2)))
        bad = MockProvider(responder=lambda r: ModelResponse(
            model=r.model, text="", finish_reason="content_filter",
            usage=TokenUsage(input_tokens=5, output_tokens=0)))
        rolled = {}

        class Reg:
            def rollback(self, name):
                rolled["name"] = name

        canary = CanaryRouter(good, bad, percent=50.0, min_samples=4, regression_threshold=0.2,
                              prompt_registry=Reg(), prompt_name="p")
        for _ in range(40):
            await canary.generate(_text_req())
            if canary.rolled_back:
                break
        assert canary.rolled_back
        assert rolled.get("name") == "p"
        assert (await canary.generate(_text_req())).text == "ok"  # primary after rollback

    async def test_canary_no_rollback_when_healthy(self):
        good = MockProvider(default_text="ok")
        canary = CanaryRouter(good, MockProvider(default_text="ok"), percent=50.0, min_samples=4)
        for _ in range(20):
            await canary.generate(_text_req())
        assert not canary.rolled_back


# --------------------------------------------------------------------------- #
# Lifecycle watcher
# --------------------------------------------------------------------------- #


class TestLifecycle:
    def test_deprecated_alert_and_successor_proposal(self):
        watcher = LifecycleWatcher()
        alerts = watcher.scan(["gemini-2.0-flash"], as_of=date(2026, 6, 17))
        assert alerts and alerts[0].lifecycle == "deprecated"
        proposal = watcher.propose_migration("gemini-2.0-flash")
        assert proposal.to_model == "gemini-2.5-flash" and proposal.kind == "successor"

    def test_pareto_proposal_when_no_successor(self):
        reg = ModelRegistry([
            ModelProfile(name="pricey", provider="x", model="pricey", input_cost_per_mtok=10,
                         output_cost_per_mtok=30),
            ModelProfile(name="cheap", provider="x", model="cheap", input_cost_per_mtok=1,
                         output_cost_per_mtok=3),
        ])
        watcher = LifecycleWatcher(reg)
        proposal = watcher.propose_migration("pricey")
        assert proposal.kind == "pareto" and proposal.to_model == "cheap"
        assert proposal.savings_pct > 0

    def test_proposal_rewrites_cascade_and_policy(self):
        from vincio.optimize.routing import RoutingPolicy

        watcher = LifecycleWatcher()
        proposal = watcher.propose_migration("gemini-2.0-flash")
        cascade = ModelCascade.from_models(["gemini-2.0-flash", "gpt-5.2"])
        rewritten = proposal.apply_to_cascade(cascade)
        assert rewritten.rungs[0].model == "gemini-2.5-flash"
        policy = RoutingPolicy(cheap_model="gemini-2.0-flash", default_model="gpt-5.2",
                               strong_model="gpt-5.2")
        new_policy = proposal.apply_to_policy(policy)
        assert new_policy.cheap_model == "gemini-2.5-flash"

    def test_app_watch_lifecycle(self, offline_config):
        app = ContextApp(name="lc", provider=MockProvider(), model="gemini-2.0-flash",
                         config=offline_config)
        result = app.watch_lifecycle(as_of=date(2026, 6, 17))
        assert any(a.model == "gemini-2.0-flash" for a in result["alerts"])
        assert any(p.to_model == "gemini-2.5-flash" for p in result["proposals"])


# --------------------------------------------------------------------------- #
# Discovery + Google batch parity
# --------------------------------------------------------------------------- #


class TestDiscoveryAndBatch:
    def test_openai_parse_models_list(self):
        from vincio.providers.openai import OpenAIProvider

        profiles = OpenAIProvider._parse_models_list({"data": [{"id": "gpt-9"}, {"id": "gpt-10"}]})
        assert [p.model for p in profiles] == ["gpt-9", "gpt-10"]
        assert all(p.provider == "openai" for p in profiles)

    def test_google_parse_models_list_filters_embeddings(self):
        from vincio.providers.google import GoogleProvider

        data = {"models": [
            {"name": "models/gemini-9", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embed-1", "supportedGenerationMethods": ["embedContent"]},
        ]}
        profiles = GoogleProvider._parse_models_list(data)
        assert [p.model for p in profiles] == ["gemini-9"]

    def test_reconcile_adds_and_marks_missing(self):
        reg = ModelRegistry([
            ModelProfile(name="a", provider="openai", model="a"),
            ModelProfile(name="b", provider="openai", model="b"),
        ])
        summary = reg.reconcile(
            [ModelProfile(name="a", provider="openai", model="a"),
             ModelProfile(name="c", provider="openai", model="c")],
            provider="openai", mark_missing_deprecated=True, as_of=date(2026, 6, 17),
        )
        assert "c" in summary["added"]
        assert "b" in summary["deprecated_missing"]
        assert reg.get("b").deprecation_date == "2026-06-17"

    async def test_discover_models_offline_safe(self):
        # A provider with no list endpoint returns [] — catalog stays authoritative.
        reg = ModelRegistry([ModelProfile(name="a", provider="mock", model="a")])
        summary = await discover_models(MockProvider(), registry=reg)
        assert summary == {"added": [], "updated": [], "deprecated_missing": []}

    def test_google_batch_inlined_parse(self):
        op = {"name": "batches/abc", "done": True,
              "metadata": {"state": "BATCH_STATE_SUCCEEDED"},
              "response": {"inlinedResponses": {"inlinedResponses": [
                  {"metadata": {"key": "r0"}, "response": {}}]}}}
        assert len(GoogleBatchBackend._inlined_responses(op)) == 1

    async def test_google_batch_parity_half_cost(self):
        from vincio.observability.costs import PriceTable

        runner = BatchRunner(InProcessBatchBackend(MockProvider(default_text="ok")), discount=0.5)
        reqs = [BatchRequest(custom_id=f"g{i}", request=ModelRequest(
            model="gemini-2.5-flash", messages=[Message(role="user", content="hi")])) for i in range(3)]
        result = await runner.run(reqs)
        assert len(result.succeeded) == 3
        sync = PriceTable().cost("gemini-2.5-flash", result.succeeded[0].response.usage)
        assert result.succeeded[0].response.cost_usd <= sync * 0.5 + 1e-12

    def test_registry_google_batch_pricing(self):
        from vincio.observability.costs import PriceTable

        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        table = PriceTable()
        assert table.cost("gemini-2.5-flash", usage, batch=True) < table.cost("gemini-2.5-flash", usage)


class TestGoogleBatchCassette:
    """Verify the real GoogleBatchBackend wire format offline, against recorded
    Gemini-shaped responses served by an httpx mock transport — so the URL paths,
    inlined request/response envelope, status mapping, parsing, reconciliation,
    and half-cost billing are all exercised without a live endpoint."""

    @staticmethod
    def _provider(handler):
        import httpx

        from vincio.providers.google import GoogleProvider

        return GoogleProvider(api_key="test", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def test_submit_poll_results_lifecycle(self):
        import json as _json

        import httpx

        from vincio.observability.costs import PriceTable

        captured: dict = {}
        completed = {
            "name": "batches/abc", "done": True,
            "metadata": {"state": "BATCH_STATE_SUCCEEDED",
                         "requestCounts": {"succeeded": 1, "failed": 1}},
            "response": {"inlinedResponses": {"inlinedResponses": [
                {"metadata": {"key": "r0"}, "response": {
                    "candidates": [{"content": {"parts": [{"text": "Paris"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}},
                {"metadata": {"key": "r1"}, "error": {"code": 3, "message": "bad request"}},
            ]}},
        }

        def handler(request):
            path = request.url.path
            if request.method == "POST" and path.endswith(":batchGenerateContent"):
                captured["path"] = path
                captured["body"] = _json.loads(request.content)
                return httpx.Response(200, json={"name": "batches/abc",
                                                 "metadata": {"state": "BATCH_STATE_PENDING"}})
            if request.method == "GET" and path.endswith("/batches/abc"):
                return httpx.Response(200, json=completed)
            return httpx.Response(404, json={"error": {"message": f"unexpected {request.method} {path}"}})

        provider = self._provider(handler)
        runner = BatchRunner(GoogleBatchBackend(provider), discount=0.5, poll_interval_s=0.0)
        reqs = [
            BatchRequest(custom_id="r0", request=ModelRequest(
                model="gemini-2.5-flash", messages=[Message(role="user", content="capital of France?")])),
            BatchRequest(custom_id="r1", request=ModelRequest(
                model="gemini-2.5-flash", messages=[Message(role="user", content="x")])),
        ]
        result = await runner.run(reqs)

        # wire: model in the path + inlined envelope keyed by custom_id
        assert "gemini-2.5-flash:batchGenerateContent" in captured["path"]
        inlined = captured["body"]["batch"]["inputConfig"]["requests"]["requests"]
        assert [e["metadata"]["key"] for e in inlined] == ["r0", "r1"]
        assert "contents" in inlined[0]["request"]  # GoogleProvider._payload output

        # reconciliation + parse via the real provider response parser
        by_id = result.by_id()
        assert by_id["r0"].ok and by_id["r0"].response.text == "Paris"
        assert not by_id["r1"].ok and "bad request" in by_id["r1"].error

        # half-cost billed on the succeeded response
        sync = PriceTable().cost("gemini-2.5-flash", by_id["r0"].response.usage)
        assert by_id["r0"].response.cost_usd <= sync * 0.5 + 1e-12
        await runner.aclose()

    async def test_cancel(self):
        import httpx

        def handler(request):
            path = request.url.path
            if request.method == "POST" and path.endswith(":batchGenerateContent"):
                return httpx.Response(200, json={"name": "batches/xyz",
                                                 "metadata": {"state": "BATCH_STATE_PENDING"}})
            if request.method == "POST" and path.endswith(":cancel"):
                return httpx.Response(200, json={"name": "batches/xyz",
                                                 "metadata": {"state": "BATCH_STATE_CANCELLED"}})
            return httpx.Response(404, json={"error": {"message": "unexpected"}})

        backend = GoogleBatchBackend(self._provider(handler))
        job, _ = await BatchRunner(backend).submit([
            BatchRequest(custom_id="r0", request=ModelRequest(
                model="gemini-2.5-flash", messages=[Message(role="user", content="x")]))
        ])
        cancelled = await backend.cancel(job)
        from vincio.providers.batch import BatchStatus

        assert cancelled.status == BatchStatus.CANCELLED
        await backend.aclose()


# --------------------------------------------------------------------------- #
# ContextApp wiring
# --------------------------------------------------------------------------- #


class TestAppWiring:
    async def test_use_router(self, offline_config):
        app = ContextApp(name="r", provider=MockProvider(default_text="ok"), model="gpt-5.2",
                         config=offline_config)
        app.use_router(["gpt-5.2-nano", "gpt-5.2"], strategy="cheapest")
        result = await app.arun("hi")
        assert result.status.value == "succeeded"
        assert app._provider_instance.last_decision.model == "gpt-5.2-nano"

    async def test_shadow_method(self, app):
        shadow = app.shadow("gpt-5.2-mini", block=True)
        await app.arun("What is the capital of France?")
        assert shadow.diff()["observations"] >= 1

    def test_swap_regression_sync(self, app, geo_dataset):
        report = app.swap_regression(geo_dataset, candidate_model="gpt-5.2-nano",
                                     baseline_model="gpt-5.2")
        assert report.regressed


# --------------------------------------------------------------------------- #
# Review-hardening regressions (review of the diff)
# --------------------------------------------------------------------------- #


class TestReviewHardening:
    def test_reconcile_folds_snapshot_as_alias_not_sparse(self):
        # A discovered dated snapshot must resolve to the rich profile, so the
        # capability guard is not tricked into refusing a capable model.
        reg = ModelRegistry()
        reg.reconcile(
            [ModelProfile(name="gpt-4o-2099-09-09", provider="openai", model="gpt-4o-2099-09-09")],
            provider="openai",
        )
        caps = reg.capabilities("gpt-4o-2099-09-09")
        assert caps is not None and caps.tool_calling is True  # gpt-4o's real caps

    def test_reconcile_mark_missing_ignores_snapshots(self):
        # gpt-4o is present when only its snapshot is discovered — not deprecated.
        reg = ModelRegistry()
        summary = reg.reconcile(
            [ModelProfile(name="gpt-4o-2024-11-20", provider="openai", model="gpt-4o-2024-11-20")],
            provider="openai", mark_missing_deprecated=True, as_of=date(2026, 6, 17),
        )
        assert "gpt-4o" not in summary["deprecated_missing"]
        assert reg.resolve("gpt-4o").lifecycle(as_of=date(2026, 6, 17)) != "deprecated"

    def test_guard_capabilities_permissive_on_sparse_profile(self):
        # A registered profile with bare-default capabilities is unjudgeable, so
        # the guard must permit it rather than block it for capabilities we never
        # learned (live discovery can never make the guard stricter).
        reg = ModelRegistry([ModelProfile(name="brand-new", provider="x", model="brand-new")])
        assert reg.capabilities("brand-new") is not None  # registered
        assert reg.guard_capabilities("brand-new") is None  # but unjudgeable
        verdict = capability_check(requirements_for(_vision_req()),
                                   reg.guard_capabilities("brand-new"), model="brand-new")
        assert verdict.ok

    async def test_list_models_delegates_through_wrappers(self):
        from vincio.core.types import ModelProfile as MP
        from vincio.providers.base import RetryingProvider
        from vincio.providers.circuit import CircuitBreaker
        from vincio.providers.transport import CoalescingProvider

        class Lister(MockProvider):
            name = "lister"

            async def list_models(self):
                return [MP(name="m1", provider="lister", model="m1")]

        for wrapper in (
            RetryingProvider(Lister()),
            CircuitBreaker(Lister()),
            CoalescingProvider(Lister()),
            FailoverChain([(Lister(), None)]),
        ):
            models = await wrapper.list_models()
            assert [m.model for m in models] == ["m1"]

    def test_use_router_enforces_candidate_residency(self):
        from types import SimpleNamespace

        from vincio.core.config import VincioConfig
        from vincio.core.errors import ResidencyViolationError

        cfg = VincioConfig()
        cfg.observability.exporter = "memory"
        app = ContextApp(name="res", provider=MockProvider(), model="gpt-5.2", config=cfg)

        class _Deny:
            enforced = True

            def check(self, *, provider, model, base_url):
                if model == "gpt-5.2-nano":
                    return SimpleNamespace(message="region eu not allowed",
                                           details={"region": "eu", "allowed_regions": ["us"]})
                return None

        app.residency = _Deny()
        with pytest.raises(ResidencyViolationError):
            app.use_router(["gpt-5.2", "gpt-5.2-nano"])


class _DenyModelResidency:
    """A residency stub that denies one specific model id (region-agnostic)."""

    enforced = True

    def __init__(self, denied: str):
        self.denied = denied

    def check(self, *, provider, model, base_url):
        from types import SimpleNamespace

        if model == self.denied:
            return SimpleNamespace(message=f"{model} region not allowed",
                                   details={"region": "eu", "allowed_regions": ["us"]})
        return None


class TestResidencyRunBoundary:
    """Residency is a run-boundary choke point over EVERY reachable model, not
    only the primary — even for candidates added before residency was tightened."""

    def _app(self):
        from vincio.core.config import VincioConfig

        cfg = VincioConfig()
        cfg.observability.exporter = "memory"
        return ContextApp(name="resb", provider=MockProvider(default_text="ok"),
                          model="gpt-5.2", config=cfg)

    def test_router_candidate_caught_at_run_boundary(self):
        from vincio.core.errors import ResidencyViolationError

        app = self._app()
        app.use_router(["gpt-5.2", "gpt-5.2-nano"])  # wiring under no residency
        app.residency = _DenyModelResidency("gpt-5.2-nano")  # tighten after wiring
        # The choke point (resolve_provider) now refuses the reachable candidate.
        with pytest.raises(ResidencyViolationError):
            app.resolve_provider()

    def test_cascade_rung_caught_at_run_boundary(self):
        from vincio.core.errors import ResidencyViolationError

        app = self._app()
        app.use_cascade(["gpt-5.2-nano", "gpt-5.2"])  # wiring under no residency
        app.residency = _DenyModelResidency("gpt-5.2")  # the strong rung
        with pytest.raises(ResidencyViolationError):
            app.resolve_provider()

    def test_budget_degrade_model_caught_at_run_boundary(self):
        from vincio.core.errors import ResidencyViolationError

        app = self._app()
        app.set_cost_budget(limit_usd=10.0, scope="global", on_breach="degrade",
                            degrade_model="gpt-5.2-nano")
        app.residency = _DenyModelResidency("gpt-5.2-nano")
        with pytest.raises(ResidencyViolationError):
            app.resolve_provider()

    def test_primary_model_still_enforced(self):
        from vincio.core.errors import ResidencyViolationError

        app = self._app()
        app.residency = _DenyModelResidency("gpt-5.2")
        with pytest.raises(ResidencyViolationError):
            app.resolve_provider()

    def test_allowed_models_pass(self):
        app = self._app()
        app.use_router(["gpt-5.2", "gpt-5.2-nano"])
        app.residency = _DenyModelResidency("some-other-model")  # none reachable denied
        assert app.resolve_provider() is not None

    async def test_denied_run_is_surfaced_end_to_end(self):
        # A reachable disallowed candidate fails the run (not a silent egress).
        app = self._app()
        app.use_router(["gpt-5.2", "gpt-5.2-nano"])
        app.residency = _DenyModelResidency("gpt-5.2-nano")
        result = await app.arun("hi")
        assert result.status.value != "succeeded"
        assert "region" in (result.error or "").lower() or result.status.value in ("denied", "failed")

    async def test_repeats_cost_latency_aggregate_consistent(self):
        from vincio.evals.metrics import MetricResult, RunOutput
        from vincio.evals.runners import EvalRunner

        costs = iter([0.001, 0.002, 0.009])

        async def target(case):
            return RunOutput(output="x", cost_usd=next(costs), latency_ms=10)

        def quality(case, run):
            return MetricResult(name="quality", value=0.9)

        runner = EvalRunner(target, metrics=[quality, "cost"], repeats=3, repeat_aggregate="median")
        ds = Dataset(name="r", cases=[EvalCase(id="a", input="q")])
        report = await runner.arun(ds)
        case = report.cases[0]
        # headline cost_usd uses the same aggregator as metrics["cost"] (median=0.002)
        assert case.cost_usd == pytest.approx(case.metrics["cost"])
        assert case.cost_usd == pytest.approx(0.002)
