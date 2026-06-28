"""Tests for the agentic benchmark adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from vincio.evals import (
    AgentBenchAdapter,
    BenchmarkError,
    BenchmarkTask,
    BFCLAdapter,
    GAIAAdapter,
    LiveCodeBenchAdapter,
    MMLUProAdapter,
    SWEBenchAdapter,
    TauBenchAdapter,
    ToolBenchAdapter,
    WebArenaAdapter,
    agentbench_tasks_from_export,
    available_benchmarks,
    livecodebench_tasks_from_export,
    load_benchmark,
    mmlu_pro_tasks_from_export,
    toolbench_tasks_from_export,
)

FIXTURES = Path(__file__).resolve().parent.parent / "benchmarks" / "fixtures"


def test_registry_lists_all_adapters():
    assert available_benchmarks() == [
        "agentbench",
        "bfcl",
        "bird",
        "gaia",
        "livecodebench",
        "mmlu_pro",
        "spider",
        "swebench_verified",
        "tau_bench",
        "toolbench",
        "webarena",
    ]


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
    "name",
    [
        "swebench_verified",
        "tau_bench",
        "gaia",
        "webarena",
        "bfcl",
        "agentbench",
        "toolbench",
        "livecodebench",
        "mmlu_pro",
        "spider",
        "bird",
    ],
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


# -- AgentBench: per-environment verifiable end state ------------------------


@pytest.mark.asyncio
async def test_agentbench_match_types():
    adapter = AgentBenchAdapter()
    exact = BenchmarkTask(id="e", gold={"match": "exact_match", "value": "/etc/app.conf"})
    assert (await adapter.score(exact, "/etc/app.conf")).success is True
    assert (await adapter.score(exact, "/etc/other")).success is False
    # numeric tolerance
    num = BenchmarkTask(id="n", gold={"match": "numeric", "value": 1540.5, "tolerance": 0.01})
    assert (await adapter.score(num, "1540.50")).success is True
    assert (await adapter.score(num, "1600")).success is False
    # set match is order-independent; partial set scores < 1 and fails.
    kg = BenchmarkTask(id="k", gold={"match": "set_match", "value": ["spain", "germany", "italy"]})
    full = await adapter.score(kg, ["Italy", "Germany", "Spain"])
    assert full.success is True and full.score == 1.0
    partial = await adapter.score(kg, ["Spain", "Germany"])
    assert partial.success is False and partial.score < 1.0


@pytest.mark.asyncio
async def test_agentbench_unknown_match_raises():
    adapter = AgentBenchAdapter()
    with pytest.raises(BenchmarkError):
        await adapter.score(BenchmarkTask(id="x", gold={"match": "fuzzy", "value": "y"}), "y")


def test_agentbench_export_mapping():
    tasks = agentbench_tasks_from_export(
        [{"id": "a1", "description": "count files", "environment": "os",
          "match": "numeric", "answer": 3, "tolerance": 0.0}]
    )
    assert tasks[0].gold == {"match": "numeric", "value": 3, "tolerance": 0.0}
    assert tasks[0].inputs["env"] == "os"


# -- ToolBench: solvable pass rate over a call path --------------------------


@pytest.mark.asyncio
async def test_toolbench_solvable_pass_rate():
    adapter = ToolBenchAdapter()
    task = BenchmarkTask(
        id="t", inputs={"available_apis": ["search", "summarize"]},
        gold={"final_answer": "done"},
    )
    solved = [
        {"name": "search", "arguments": {"q": "x"}},
        {"name": "Finish", "arguments": {"return_type": "give_answer", "final_answer": "done"}},
    ]
    assert (await adapter.score(task, solved)).success is True
    # gave up -> not solved even though APIs are valid.
    gave_up = [
        {"name": "search", "arguments": {"q": "x"}},
        {"name": "Finish", "arguments": {"return_type": "give_up_and_restart"}},
    ]
    assert (await adapter.score(task, gave_up)).success is False
    # hallucinated tool -> apis invalid -> fail, score reflects the bad call.
    hallucinated = [
        {"name": "teleport", "arguments": {}},
        {"name": "give_answer", "arguments": {"final_answer": "done"}},
    ]
    result = await adapter.score(task, hallucinated)
    assert result.success is False and result.score == 0.0


@pytest.mark.asyncio
async def test_toolbench_wrong_final_answer_fails():
    adapter = ToolBenchAdapter()
    task = BenchmarkTask(id="t", gold={"final_answer": "expected"})
    path = [{"name": "give_answer", "arguments": {"final_answer": "something else"}}]
    assert (await adapter.score(task, path)).success is False


def test_toolbench_export_mapping():
    tasks = toolbench_tasks_from_export(
        [{"id": "q1", "query": "weather", "api_list": [{"api_name": "get_weather"}],
          "final_answer": "sunny"}]
    )
    assert tasks[0].inputs["available_apis"] == ["get_weather"]
    assert tasks[0].gold["final_answer"] == "sunny"


# -- LiveCodeBench: all-tests-pass ------------------------------------------


@pytest.mark.asyncio
async def test_livecodebench_all_tests_pass():
    adapter = LiveCodeBenchAdapter()
    task = BenchmarkTask(id="l", gold={"public": ["p0"], "hidden": ["h0", "h1"]})
    all_pass = {"results": {"p0": "passed", "h0": "passed", "h1": "passed"}}
    assert (await adapter.score(task, all_pass)).success is True
    one_fail = {"results": {"p0": "passed", "h0": "passed", "h1": "failed"}}
    result = await adapter.score(task, one_fail)
    assert result.success is False and result.score == round(2 / 3, 4)


def test_livecodebench_export_mapping():
    tasks = livecodebench_tasks_from_export(
        [{"question_id": "abc", "question_content": "solve", "release_date": "2025-05-01",
          "public_test_cases": [{"id": "p0"}], "private_test_cases": [{"id": "h0"}]}]
    )
    assert tasks[0].gold == {"public": ["p0"], "hidden": ["h0"]}
    assert tasks[0].metadata["release_date"] == "2025-05-01"


# -- MMLU-Pro: option-letter extraction --------------------------------------


@pytest.mark.asyncio
async def test_mmlu_pro_letter_extraction():
    adapter = MMLUProAdapter()
    letter = BenchmarkTask(id="m", gold="D")
    assert (await adapter.score(letter, "After reasoning, the answer is (D).")).success is True
    assert (await adapter.score(letter, "Answer: D")).success is True
    assert (await adapter.score(letter, "I'll go with D")).success is True
    assert (await adapter.score(letter, "The answer is (A).")).success is False
    # gold may be a 0-based index (3 -> 'D').
    indexed = BenchmarkTask(id="i", gold=3)
    assert (await adapter.score(indexed, "answer is D")).success is True


def test_mmlu_pro_export_mapping():
    tasks = mmlu_pro_tasks_from_export(
        [{"question_id": "q", "question": "?", "options": ["a", "b"], "answer_index": 1,
          "category": "math"}]
    )
    assert tasks[0].gold == 1
    assert tasks[0].inputs["options"] == ["a", "b"]
    assert tasks[0].metadata["category"] == "math"


# -- Spider / BIRD: text-to-SQL execution accuracy ---------------------------


def _sql_db():
    return {
        "orders": {
            "columns": ["order_id", "region", "revenue"],
            "rows": [[1, "NA", 1200.5], [2, "EU", 980.0], [3, "NA", 300.0]],
        }
    }


@pytest.mark.asyncio
async def test_spider_execution_accuracy_not_string_match():
    from vincio.evals import SpiderAdapter

    adapter = SpiderAdapter()
    task = BenchmarkTask(
        id="t",
        prompt="how many orders over 1000",
        gold="SELECT COUNT(*) FROM orders WHERE revenue > 1000",
        inputs={"tables": _sql_db()},
    )
    # A semantic variant (different SQL, same result set) is correct by execution.
    ok = await adapter.score(task, "SELECT COUNT(order_id) AS n FROM orders WHERE revenue >= 1000.01")
    assert ok.success is True and ok.score == 1.0
    # A wrong query (different result set) fails.
    bad = await adapter.score(task, "SELECT COUNT(*) FROM orders")
    assert bad.success is False


@pytest.mark.asyncio
async def test_text_to_sql_refuses_a_generated_write():
    from vincio.evals import BIRDAdapter

    adapter = BIRDAdapter()
    task = BenchmarkTask(
        id="w",
        gold="SELECT COUNT(*) FROM orders",
        inputs={"tables": _sql_db(), "evidence": ""},
    )
    # A generated write is governed: refused, scored failed, marked invalid.
    result = await adapter.score(task, "DROP TABLE orders")
    assert result.success is False
    assert result.details["valid"] is False


@pytest.mark.asyncio
async def test_spider_gold_failure_raises():
    from vincio.evals import SpiderAdapter

    adapter = SpiderAdapter()
    task = BenchmarkTask(id="g", gold="SELECT * FROM nope", inputs={"tables": _sql_db()})
    with pytest.raises(BenchmarkError):
        await adapter.score(task, "SELECT * FROM orders")


def test_spider_and_bird_export_mapping():
    from vincio.evals import bird_tasks_from_export, spider_tasks_from_export

    spider = spider_tasks_from_export(
        [{"id": "s1", "question": "count", "query": "SELECT 1", "tables": {"t": {"columns": ["a"], "rows": [[1]]}}}]
    )
    assert spider[0].gold == "SELECT 1"
    assert spider[0].recorded == "SELECT 1"
    assert spider[0].inputs["tables"]["t"]["columns"] == ["a"]

    bird = bird_tasks_from_export(
        [{"id": "b1", "question": "avg", "SQL": "SELECT AVG(a) FROM t", "evidence": "a in units",
          "tables": {"t": {"columns": ["a"], "rows": [[1]]}}}]
    )
    assert bird[0].gold == "SELECT AVG(a) FROM t"
    assert bird[0].inputs["evidence"] == "a in units"
