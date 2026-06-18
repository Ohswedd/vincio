"""The benchmark adapters' LIVE-run path (2.2): a real Vincio agent produces
fresh output that the *identical* benchmark scorer evaluates — not replay.

Together with `test_benchmark_adapters.py` (the offline recorded-fixture path),
this closes the loop: `score()` is exercised on both recorded and freshly-solved
output, and official benchmark exports load into `BenchmarkTask`s.
"""

from __future__ import annotations

import json

import pytest

from vincio.agents import AgentExecutor
from vincio.agents.planner import Planner
from vincio.evals import (
    BFCLAdapter,
    GAIAAdapter,
    TauBenchAdapter,
    bfcl_tasks_from_export,
    gaia_tasks_from_export,
    make_agent_solver,
    make_env_solver,
    scripted_policy,
    swebench_tasks_from_export,
    tasks_from_jsonl,
)
from vincio.evals.benchmarks import EnvAction
from vincio.providers.mock import MockProvider
from vincio.tools.registry import ToolRegistry
from vincio.tools.runtime import ToolRuntime


@pytest.mark.asyncio
async def test_gaia_live_run_with_a_real_agent():
    # Load an official-shape GAIA record, then SOLVE it with a real AgentExecutor.
    tasks = gaia_tasks_from_export(
        [{"task_id": "g1", "Question": "What is the capital of France?", "Final answer": "Paris", "Level": 1}]
    )
    assert tasks[0].prompt == "What is the capital of France?"
    assert tasks[0].gold == "Paris" and tasks[0].recorded is None  # no recorded output → live only

    executor = AgentExecutor(MockProvider(default_text="Paris"), model="mock-1", planner=Planner(mode="static"))
    report = await GAIAAdapter(tasks).run(make_agent_solver(executor, mode="text"))
    assert report.replayed is False                 # fresh solve, not replay
    assert report.success_rate == 1.0               # scored by the identical GAIA scorer


@pytest.mark.asyncio
async def test_bfcl_live_run_extracts_agent_tool_calls():
    # The agent genuinely calls a tool; the live solver extracts the call from its
    # trajectory and the identical BFCL AST scorer grades it.
    tasks = bfcl_tasks_from_export(
        [{"id": "b1", "question": "Weather in SF?", "function": [{"name": "get_weather"}],
          "ground_truth": [{"name": "get_weather", "arguments": {"city": "sf"}}], "category": "simple"}]
    )
    assert tasks[0].inputs["functions"] == [{"name": "get_weather"}]

    registry = ToolRegistry()

    @registry.register()
    def get_weather(city: str) -> dict:
        """Get weather."""
        return {"city": city}

    provider = MockProvider(script=[{"tool_call": {"name": "get_weather", "arguments": {"city": "sf"}}}, "done"])
    executor = AgentExecutor(
        provider, model="mock-1", planner=Planner(mode="react"),
        tool_runtime=ToolRuntime(registry, cache_enabled=False), tool_specs=registry.specs(),
    )
    report = await BFCLAdapter(tasks).run(make_agent_solver(executor, mode="calls"))
    assert report.replayed is False
    assert report.success_rate == 1.0


@pytest.mark.asyncio
async def test_tau_bench_live_run_via_env_policy():
    # A policy decides actions by interacting with the deterministic world; the
    # identical end-state oracle scores them.
    tasks = [
        {"id": "t1", "inputs": {"env": "retail", "env_task": "cancel_refund"}, "gold": {"oracle": "environment"}},
    ]
    policy = scripted_policy([
        EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}),
        EnvAction(tool="refund_order", arguments={"order_id": "O1002"}),
    ])
    report = await TauBenchAdapter(tasks).run(make_env_solver(policy))
    assert report.replayed is False
    assert report.success_rate == 1.0


@pytest.mark.asyncio
async def test_callable_solver_path():
    # A plain callable solver is the simplest live path.
    tasks = gaia_tasks_from_export([{"task_id": "g", "Question": "2+2?", "Final answer": "4"}])
    report = await GAIAAdapter(tasks).run(make_agent_solver(lambda prompt: "4"))
    assert report.success_rate == 1.0


def test_swebench_export_parses_json_string_fields():
    # The released SWE-bench format JSON-encodes FAIL_TO_PASS / PASS_TO_PASS.
    tasks = swebench_tasks_from_export(
        [{"instance_id": "django__django-123", "problem_statement": "fix it",
          "FAIL_TO_PASS": json.dumps(["test_a", "test_b"]), "PASS_TO_PASS": json.dumps(["test_c"]),
          "repo": "django/django", "base_commit": "abc"}]
    )
    assert tasks[0].id == "django__django-123"
    assert tasks[0].gold == {"fail_to_pass": ["test_a", "test_b"], "pass_to_pass": ["test_c"]}
    assert tasks[0].inputs["repo"] == "django/django"


@pytest.mark.asyncio
async def test_swebench_live_run_scores_test_outcome():
    # A "solver" here stands in for the harness that runs the patched repo's tests.
    tasks = swebench_tasks_from_export(
        [{"instance_id": "x", "FAIL_TO_PASS": ["t1"], "PASS_TO_PASS": ["t2"]}]
    )

    def run_tests(prompt):  # the agent+harness produced this test outcome
        return {"tests": {"t1": "passed", "t2": "passed"}}

    from vincio.evals import SWEBenchAdapter

    report = await SWEBenchAdapter(tasks).run(run_tests)
    assert report.success_rate == 1.0


def test_tasks_from_jsonl_roundtrip(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"id": "a", "prompt": "p", "gold": "g"}\n{"id": "b", "prompt": "q", "gold": "h"}\n'
    )
    tasks = tasks_from_jsonl(path)
    assert [t.id for t in tasks] == ["a", "b"]
    assert tasks[0].gold == "g"
