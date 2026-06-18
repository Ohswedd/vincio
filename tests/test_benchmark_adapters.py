"""Tests for the agentic benchmark adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from vincio.evals import (
    BenchmarkError,
    BenchmarkTask,
    BFCLAdapter,
    GAIAAdapter,
    SWEBenchAdapter,
    TauBenchAdapter,
    WebArenaAdapter,
    available_benchmarks,
    load_benchmark,
)

FIXTURES = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures"


def test_registry_lists_five_adapters():
    assert available_benchmarks() == ["bfcl", "gaia", "swebench_verified", "tau_bench", "webarena"]


@pytest.mark.asyncio
async def test_swebench_resolution_criterion():
    adapter = SWEBenchAdapter(
        [
            BenchmarkTask(
                id="t",
                gold={"fail_to_pass": ["a"], "pass_to_pass": ["b"]},
            )
        ]
    )
    resolved = await adapter.score(adapter.tasks()[0], {"tests": {"a": "passed", "b": "passed"}})
    assert resolved.success is True
    # A regressed pass-to-pass test means NOT resolved even if fail-to-pass passes.
    broke_p2p = await adapter.score(adapter.tasks()[0], {"tests": {"a": "passed", "b": "failed"}})
    assert broke_p2p.success is False


@pytest.mark.asyncio
async def test_tau_bench_scores_environment_end_state():
    adapter = TauBenchAdapter(
        [
            BenchmarkTask(
                id="t",
                inputs={"env": "retail", "env_task": "cancel_refund"},
                recorded=[
                    {"tool": "cancel_order", "arguments": {"order_id": "O1002"}},
                    {"tool": "refund_order", "arguments": {"order_id": "O1002"}},
                ],
            )
        ]
    )
    report = await adapter.replay()
    assert report.success_rate == 1.0


@pytest.mark.asyncio
async def test_gaia_normalized_exact_match():
    adapter = GAIAAdapter()
    t = BenchmarkTask(id="t", gold="1000")
    assert (await adapter.score(t, "1,000")).success is True
    assert (await adapter.score(t, "the 1000")).success is True  # article dropped
    assert (await adapter.score(t, "1001")).success is False


@pytest.mark.asyncio
async def test_webarena_functional_checks():
    adapter = WebArenaAdapter()
    inc = BenchmarkTask(id="i", gold={"type": "must_include", "value": ["order", "shipped"]})
    assert (await adapter.score(inc, "your order shipped")).success is True
    assert (await adapter.score(inc, "your order is pending")).success is False
    url = BenchmarkTask(id="u", gold={"type": "url", "value": "/x"})
    assert (await adapter.score(url, "now at https://s/x")).success is True


@pytest.mark.asyncio
async def test_bfcl_ast_match_and_relevance():
    adapter = BFCLAdapter()
    simple = BenchmarkTask(id="s", gold=[{"name": "f", "arguments": {"x": 1}}])
    assert (await adapter.score(simple, [{"name": "f", "arguments": {"x": 1}}])).success is True
    assert (await adapter.score(simple, [{"name": "g", "arguments": {"x": 1}}])).success is False
    # Relevance: gold is "no call", so abstaining is correct.
    rel = BenchmarkTask(id="r", gold=[], metadata={"category": "relevance"})
    assert (await adapter.score(rel, [])).success is True
    assert (await adapter.score(rel, [{"name": "f", "arguments": {}}])).success is False


@pytest.mark.asyncio
async def test_bfcl_parallel_order_independent():
    adapter = BFCLAdapter()
    gold = [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
    task = BenchmarkTask(id="p", gold=gold)
    swapped = [{"name": "b", "arguments": {}}, {"name": "a", "arguments": {}}]
    assert (await adapter.score(task, swapped)).success is True


def test_task_set_hash_is_stable_and_change_sensitive():
    a = SWEBenchAdapter([BenchmarkTask(id="t", gold={"fail_to_pass": ["x"]})])
    b = SWEBenchAdapter([BenchmarkTask(id="t", gold={"fail_to_pass": ["x"]})])
    assert a.task_set_hash() == b.task_set_hash()  # deterministic / pinned
    c = SWEBenchAdapter([BenchmarkTask(id="t", gold={"fail_to_pass": ["y"]})])
    assert a.task_set_hash() != c.task_set_hash()  # task-set drift is caught


def test_fixture_hash_mismatch_raises():
    # A fixture declaring the wrong hash is rejected on load.
    import json
    import tempfile

    bad = {
        "name": "gaia",
        "task_set_hash": "deadbeefdeadbeef",
        "tasks": [{"id": "x", "gold": "1"}],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(bad, fh)
        path = fh.name
    with pytest.raises(BenchmarkError):
        GAIAAdapter(fixture_path=path)


@pytest.mark.parametrize(
    "name", ["swebench_verified", "tau_bench", "gaia", "webarena", "bfcl"]
)
@pytest.mark.asyncio
async def test_shipped_fixtures_replay_deterministically(name):
    path = FIXTURES / f"{name}.json"
    a = load_benchmark(name, fixture_path=path)
    b = load_benchmark(name, fixture_path=path)
    ra = await a.replay()
    rb = await b.replay()
    assert ra.model_dump() == rb.model_dump()  # offline replay is deterministic
    assert ra.task_set_hash  # pinned
    # Each shipped fixture has at least one resolved/correct instance.
    assert ra.success_rate > 0.0


@pytest.mark.asyncio
async def test_report_projects_to_eval_report():
    adapter = load_benchmark("gaia", fixture_path=FIXTURES / "gaia.json")
    report = await adapter.replay()
    eval_report = report.to_eval_report()
    assert eval_report.summary()["success"]["mean"] == 1.0
    assert eval_report.metadata["task_set_hash"] == report.task_set_hash


@pytest.mark.asyncio
async def test_run_with_solver_scores_fresh_output():
    adapter = GAIAAdapter([BenchmarkTask(id="t", gold="Paris")])

    def solver(task):
        return "Paris"

    report = await adapter.run(solver)
    assert report.success_rate == 1.0
    assert report.replayed is False
