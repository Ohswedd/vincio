"""Targeted coverage for vincio.evals.benchmarks.

These tests hit the uncovered error paths, edge cases, and branch arms of the
benchmark adapters and their export/loader helpers — using the deterministic
MockProvider for any model interaction, never mock/patch.
"""

from __future__ import annotations

import json

import pytest

from vincio import ContextApp, VincioConfig
from vincio.evals.benchmarks import (
    AgentBenchAdapter,
    BenchmarkError,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTask,
    BFCLAdapter,
    EnvAction,
    GAIAAdapter,
    LiveCodeBenchAdapter,
    MMLUProAdapter,
    SWEBenchAdapter,
    TauBenchAdapter,
    ToolBenchAdapter,
    WebArenaAdapter,
    _as_numeric,
    _coerce_text,
    _livecodebench_tests,
    _maybe_json,
    _mmlu_extract,
    _mmlu_gold_letter,
    _norm_value,
    agentbench_tasks_from_export,
    available_benchmarks,
    bfcl_tasks_from_export,
    build_agent_solver,
    build_env_solver,
    gaia_tasks_from_export,
    livecodebench_tasks_from_export,
    load_benchmark,
    mmlu_pro_tasks_from_export,
    scripted_policy,
    swebench_tasks_from_export,
    tasks_from_jsonl,
    toolbench_tasks_from_export,
)
from vincio.providers import MockProvider

# ---------------------------------------------------------------------------
# BenchmarkReport properties (lines 120, 128) + empty-report branches
# ---------------------------------------------------------------------------


def test_report_n_and_mean_score_with_results():
    report = BenchmarkReport(
        name="x",
        results=[
            BenchmarkResult(task_id="a", success=True, score=1.0),
            BenchmarkResult(task_id="b", success=False, score=0.5),
        ],
    )
    assert report.n == 2
    assert report.mean_score == 0.75
    assert report.success_rate == 0.5


def test_empty_report_rates_are_zero_not_div_by_zero():
    report = BenchmarkReport(name="empty")
    assert report.n == 0
    assert report.success_rate == 0.0
    assert report.mean_score == 0.0


def test_to_eval_report_projects_cases_and_metadata():
    report = BenchmarkReport(
        name="gaia",
        variant="v2",
        task_set_hash="deadbeef",
        replayed=True,
        results=[BenchmarkResult(task_id="t1", success=True, score=1.0, details={"k": 1})],
    )
    eval_report = report.to_eval_report()
    assert eval_report.name == "gaia/v2"
    assert eval_report.dataset == "gaia"
    assert eval_report.metadata == {"task_set_hash": "deadbeef", "replayed": True}
    (case,) = eval_report.cases
    assert case.case_id == "t1"
    assert case.metrics == {"success": 1.0, "score": 1.0}
    assert case.tags == ["benchmark:gaia"]


# ---------------------------------------------------------------------------
# Fixture loading (193->201) + replay error (228)
# ---------------------------------------------------------------------------


def test_fixture_without_declared_hash_loads(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [{"id": "a", "gold": "x"}]}), encoding="utf-8")
    adapter = GAIAAdapter(fixture_path=path)
    assert [t.id for t in adapter.tasks()] == ["a"]


def test_fixture_with_matching_declared_hash_loads(tmp_path):
    probe = GAIAAdapter([BenchmarkTask(id="a", gold="x")])
    good_hash = probe.task_set_hash()
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps({"task_set_hash": good_hash, "tasks": [{"id": "a", "gold": "x"}]}),
        encoding="utf-8",
    )
    adapter = GAIAAdapter(fixture_path=path)
    assert adapter.task_set_hash() == good_hash


def test_fixture_hash_mismatch_raises(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps({"task_set_hash": "0000beef", "tasks": [{"id": "a", "gold": "x"}]}),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="task-set hash mismatch"):
        GAIAAdapter(fixture_path=path)


def test_fixture_top_level_list(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"id": "a", "gold": "x"}]), encoding="utf-8")
    adapter = GAIAAdapter(fixture_path=path)
    assert [t.id for t in adapter.tasks()] == ["a"]


@pytest.mark.asyncio
async def test_replay_without_recorded_raises():
    adapter = GAIAAdapter([BenchmarkTask(id="t", gold="Paris")])
    with pytest.raises(BenchmarkError, match="no recorded output to replay"):
        await adapter.replay()


@pytest.mark.asyncio
async def test_replay_scores_recorded_output():
    adapter = GAIAAdapter([BenchmarkTask(id="t", gold="Paris", recorded="paris")])
    report = await adapter.replay()
    assert report.replayed is True
    assert report.success_rate == 1.0


# ---------------------------------------------------------------------------
# TauBench unsupported env (301)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tau_bench_unsupported_env_raises():
    adapter = TauBenchAdapter([BenchmarkTask(id="t", inputs={"env": "spaceship"})])
    with pytest.raises(BenchmarkError, match="unsupported env 'spaceship'"):
        await adapter.score(adapter.tasks()[0], [])


@pytest.mark.asyncio
async def test_tau_bench_variant_recorded_in_details():
    adapter = TauBenchAdapter(
        [BenchmarkTask(id="t", inputs={"env": "retail", "env_task": "cancel_refund"})],
        variant="tau2",
    )
    result = await adapter.score(
        adapter.tasks()[0],
        [
            {"tool": "cancel_order", "arguments": {"order_id": "O1002"}},
            {"tool": "refund_order", "arguments": {"order_id": "O1002"}},
        ],
    )
    assert result.details["variant"] == "tau2"
    assert result.success is True


# ---------------------------------------------------------------------------
# WebArena gold types (must_include, url, unknown -> 384)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webarena_must_include_list():
    adapter = WebArenaAdapter([BenchmarkTask(id="t", gold={"type": "must_include", "value": ["foo", "bar"]})])
    ok = await adapter.score(adapter.tasks()[0], "the FOO and the BAR are here")
    assert ok.success is True
    miss = await adapter.score(adapter.tasks()[0], "only foo here")
    assert miss.success is False


@pytest.mark.asyncio
async def test_webarena_must_include_scalar_value():
    adapter = WebArenaAdapter([BenchmarkTask(id="t", gold={"type": "must_include", "value": "needle"})])
    ok = await adapter.score(adapter.tasks()[0], "haystack NEEDLE haystack")
    assert ok.success is True


@pytest.mark.asyncio
async def test_webarena_url_match():
    adapter = WebArenaAdapter([BenchmarkTask(id="t", gold={"type": "url", "value": "/checkout/done"})])
    ok = await adapter.score(adapter.tasks()[0], "https://shop.test/Checkout/Done?x=1")
    assert ok.success is True


@pytest.mark.asyncio
async def test_webarena_unknown_gold_type_raises():
    adapter = WebArenaAdapter([BenchmarkTask(id="t", gold={"type": "regex", "value": ".*"})])
    with pytest.raises(BenchmarkError, match="unknown gold type 'regex'"):
        await adapter.score(adapter.tasks()[0], "anything")


@pytest.mark.asyncio
async def test_webarena_exact_match_default_type():
    # No explicit type -> defaults to exact_match, case/space-insensitive.
    adapter = WebArenaAdapter([BenchmarkTask(id="t", gold={"value": "Done"})])
    ok = await adapter.score(adapter.tasks()[0], "  done  ")
    assert ok.success is True
    assert ok.details["check"] == "exact_match"
    miss = await adapter.score(adapter.tasks()[0], "not done")
    assert miss.success is False


# ---------------------------------------------------------------------------
# _norm_value bool/list branches (407, 412-414)
# ---------------------------------------------------------------------------


def test_norm_value_bool_preserved_not_stringified():
    assert _norm_value(True) is True
    assert _norm_value(False) is False


def test_norm_value_recurses_into_list_and_lowercases_strings():
    assert _norm_value(["AbC", 3, True]) == ["abc", "3", True]


def test_norm_value_passthrough_for_dict():
    payload = {"a": 1}
    assert _norm_value(payload) is payload


@pytest.mark.asyncio
async def test_bfcl_type_coercible_and_bool_arg_match():
    # int gold vs string output normalize equal; bool stays bool.
    adapter = BFCLAdapter(
        [BenchmarkTask(id="t", gold=[{"name": "f", "arguments": {"n": 5, "flag": True, "tags": ["A"]}}])]
    )
    out = [{"name": "f", "arguments": {"n": "5", "flag": True, "tags": ["a"]}}]
    result = await adapter.score(adapter.tasks()[0], out)
    assert result.success is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_bfcl_missing_gold_call_partial_match():
    # One of two gold calls is absent from the output -> not in remaining.
    adapter = BFCLAdapter(
        [
            BenchmarkTask(
                id="t",
                gold=[
                    {"name": "f", "arguments": {"x": 1}},
                    {"name": "g", "arguments": {"y": 2}},
                ],
            )
        ]
    )
    out = [{"name": "f", "arguments": {"x": 1}}]  # 'g' missing
    result = await adapter.score(adapter.tasks()[0], out)
    assert result.success is False
    assert result.score == 0.5
    assert result.details == {"matched": 1, "expected": 2, "category": "simple"}


@pytest.mark.asyncio
async def test_bfcl_relevance_category_abstain():
    adapter = BFCLAdapter([BenchmarkTask(id="t", gold=[], metadata={"category": "relevance"})])
    abstained = await adapter.score(adapter.tasks()[0], [])
    assert abstained.success is True
    assert abstained.details["category"] == "relevance"
    called = await adapter.score(adapter.tasks()[0], [{"name": "f", "arguments": {}}])
    assert called.success is False


# ---------------------------------------------------------------------------
# _as_numeric exception (468-469) + AgentBench match arms (502-506, etc.)
# ---------------------------------------------------------------------------


def test_as_numeric_handles_garbage_and_commas():
    assert _as_numeric("1,234.5") == 1234.5
    assert _as_numeric("not a number") is None
    assert _as_numeric(None) is None


@pytest.mark.asyncio
async def test_agentbench_contains_partial_score():
    adapter = AgentBenchAdapter(
        [BenchmarkTask(id="t", gold={"match": "contains", "value": ["alpha", "beta"]})]
    )
    partial = await adapter.score(adapter.tasks()[0], "alpha only")
    assert partial.success is False
    assert partial.score == 0.5
    full = await adapter.score(adapter.tasks()[0], "ALPHA and BETA")
    assert full.success is True
    assert full.score == 1.0


@pytest.mark.asyncio
async def test_agentbench_contains_scalar_value():
    adapter = AgentBenchAdapter([BenchmarkTask(id="t", gold={"match": "contains", "value": "kw"})])
    result = await adapter.score(adapter.tasks()[0], "has KW inside")
    assert result.success is True


@pytest.mark.asyncio
async def test_agentbench_set_match_recall():
    adapter = AgentBenchAdapter(
        [BenchmarkTask(id="t", gold={"match": "set_match", "value": ["x", "y"]})]
    )
    partial = await adapter.score(adapter.tasks()[0], ["X"])
    assert partial.success is False
    assert partial.score == 0.5
    exact = await adapter.score(adapter.tasks()[0], ["Y", "X"])
    assert exact.success is True
    assert exact.score == 1.0


@pytest.mark.asyncio
async def test_agentbench_numeric_tolerance():
    adapter = AgentBenchAdapter(
        [BenchmarkTask(id="t", gold={"match": "numeric", "value": 100, "tolerance": 2})]
    )
    within = await adapter.score(adapter.tasks()[0], "101")
    assert within.success is True
    outside = await adapter.score(adapter.tasks()[0], "105")
    assert outside.success is False


@pytest.mark.asyncio
async def test_agentbench_bare_gold_treated_as_exact_match():
    adapter = AgentBenchAdapter([BenchmarkTask(id="t", gold="Yes", inputs={"env": "os"})])
    result = await adapter.score(adapter.tasks()[0], "yes")
    assert result.success is True
    assert result.details == {"match": "exact_match", "env": "os"}


@pytest.mark.asyncio
async def test_agentbench_unknown_match_raises():
    adapter = AgentBenchAdapter([BenchmarkTask(id="t", gold={"match": "fuzzy", "value": "x"})])
    with pytest.raises(BenchmarkError, match="unknown match type 'fuzzy'"):
        await adapter.score(adapter.tasks()[0], "x")


# ---------------------------------------------------------------------------
# ToolBench give-up / answer paths (573->578) + answer mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toolbench_give_up_fails():
    adapter = ToolBenchAdapter([BenchmarkTask(id="t")])
    actions = [{"name": "give_up_and_restart", "arguments": {}}]
    result = await adapter.score(adapter.tasks()[0], actions)
    assert result.success is False
    assert result.details["gave_up"] is True
    assert result.details["answered"] is False


@pytest.mark.asyncio
async def test_toolbench_hallucinated_api_lowers_score():
    adapter = ToolBenchAdapter(
        [BenchmarkTask(id="t", inputs={"available_apis": ["search"]})]
    )
    actions = [
        {"name": "search", "arguments": {}},
        {"name": "ghost_tool", "arguments": {}},
        {"name": "finish", "arguments": {"final_answer": "done"}},
    ]
    result = await adapter.score(adapter.tasks()[0], actions)
    assert result.details["valid_calls"] == 1
    assert result.details["worker_calls"] == 2
    assert result.score == 0.5
    assert result.success is False  # apis not all valid


@pytest.mark.asyncio
async def test_toolbench_answer_mismatch_fails():
    adapter = ToolBenchAdapter([BenchmarkTask(id="t", gold={"final_answer": "42"})])
    actions = [{"name": "finish", "arguments": {"final_answer": "wrong"}}]
    result = await adapter.score(adapter.tasks()[0], actions)
    assert result.details["answer_ok"] is False
    assert result.success is False


@pytest.mark.asyncio
async def test_toolbench_clean_path_passes_no_worker_calls():
    adapter = ToolBenchAdapter([BenchmarkTask(id="t")])
    actions = [{"name": "finish", "arguments": {"answer": "ok"}}]
    result = await adapter.score(adapter.tasks()[0], actions)
    assert result.success is True
    assert result.score == 1.0  # no worker calls -> answered gives 1.0


@pytest.mark.asyncio
async def test_toolbench_no_finish_action_unanswered():
    # No finish/give-up action at all -> finish is None, never answered.
    adapter = ToolBenchAdapter([BenchmarkTask(id="t", inputs={"available_apis": ["search"]})])
    actions = [{"name": "search", "arguments": {}}]
    result = await adapter.score(adapter.tasks()[0], actions)
    assert result.details["answered"] is False
    assert result.details["gave_up"] is False
    assert result.success is False  # valid api but never answered


@pytest.mark.asyncio
async def test_toolbench_finish_with_give_up_return_type():
    adapter = ToolBenchAdapter([BenchmarkTask(id="t")])
    actions = [{"name": "finish", "arguments": {"return_type": "give_up_and_restart"}}]
    result = await adapter.score(adapter.tasks()[0], actions)
    assert result.details["gave_up"] is True


# ---------------------------------------------------------------------------
# _livecodebench_tests shapes (609-611) + adapter
# ---------------------------------------------------------------------------


def test_livecodebench_tests_from_dict_tests_key():
    assert _livecodebench_tests({"tests": ["a", "b"]}) == ["a", "b"]


def test_livecodebench_tests_from_public_hidden_split():
    assert _livecodebench_tests({"public": ["p1"], "hidden": ["h1"]}) == ["p1", "h1"]


def test_livecodebench_tests_from_bare_list():
    assert _livecodebench_tests(["t1", "t2"]) == ["t1", "t2"]


def test_livecodebench_tests_from_unknown_type_is_empty():
    assert _livecodebench_tests(42) == []


@pytest.mark.asyncio
async def test_livecodebench_partial_pass_not_success():
    adapter = LiveCodeBenchAdapter([BenchmarkTask(id="t", gold={"tests": ["a", "b"]})])
    result = await adapter.score(adapter.tasks()[0], {"results": {"a": "passed", "b": "failed"}})
    assert result.success is False
    assert result.score == 0.5
    assert result.details["passed"] == 1


@pytest.mark.asyncio
async def test_livecodebench_empty_required_scores_zero():
    adapter = LiveCodeBenchAdapter([BenchmarkTask(id="t", gold={"tests": []})])
    result = await adapter.score(adapter.tasks()[0], {"results": {}})
    assert result.success is False
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# MMLU extraction / gold (668, 674) + adapter
# ---------------------------------------------------------------------------


def test_mmlu_extract_no_letter_returns_empty():
    assert _mmlu_extract("there is no choice here 12345") == ""


def test_mmlu_extract_answer_is_pattern():
    assert _mmlu_extract("After reasoning, the answer is (D).") == "D"


def test_mmlu_extract_handles_none():
    assert _mmlu_extract(None) == ""


def test_mmlu_gold_letter_bool_is_empty():
    assert _mmlu_gold_letter(True) == ""


def test_mmlu_gold_letter_index_and_letter():
    assert _mmlu_gold_letter(3) == "D"
    assert _mmlu_gold_letter("d") == "D"
    assert _mmlu_gold_letter(99) == ""  # out of range
    assert _mmlu_gold_letter("ZZ") == ""  # not a single A-J letter


@pytest.mark.asyncio
async def test_mmlu_pro_scores_match():
    adapter = MMLUProAdapter([BenchmarkTask(id="t", gold=3, metadata={"category": "physics"})])
    hit = await adapter.score(adapter.tasks()[0], "I think the answer is D")
    assert hit.success is True
    assert hit.details == {"predicted": "D", "gold": "D", "category": "physics"}
    miss = await adapter.score(adapter.tasks()[0], "the answer is A")
    assert miss.success is False


# ---------------------------------------------------------------------------
# _coerce_text (723, 726-727) and _maybe_json (819-820)
# ---------------------------------------------------------------------------


def test_coerce_text_none_str_and_json():
    assert _coerce_text(None) == ""
    assert _coerce_text("hello") == "hello"
    assert _coerce_text({"a": 1}) == '{"a": 1}'


def test_maybe_json_parses_and_passes_through():
    assert _maybe_json('["a", "b"]') == ["a", "b"]
    assert _maybe_json("not json") == "not json"
    assert _maybe_json([1, 2]) == [1, 2]


# ---------------------------------------------------------------------------
# build_agent_solver mode + runner branches (749, 756-759, 776, 779, 782)
# ---------------------------------------------------------------------------


def test_make_agent_solver_unknown_mode_raises():
    with pytest.raises(BenchmarkError, match="unknown solver mode 'votes'"):
        build_agent_solver(lambda p: p, mode="votes")


@pytest.mark.asyncio
async def test_make_agent_solver_contextapp_arun_path(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    app = ContextApp(
        name="solver_app",
        provider=MockProvider(default_text="Paris"),
        model="mock-1",
        config=config,
    )
    solver = build_agent_solver(app, mode="text")
    out = await solver(BenchmarkTask(id="t", prompt="capital of France?"))
    assert out == "Paris"


@pytest.mark.asyncio
async def test_make_agent_solver_contextapp_rejects_calls_mode(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    app = ContextApp(name="a", provider=MockProvider(default_text="x"), model="mock-1", config=config)
    solver = build_agent_solver(app, mode="calls")
    with pytest.raises(BenchmarkError, match="mode='calls' requires an AgentExecutor"):
        await solver(BenchmarkTask(id="t", prompt="p"))


@pytest.mark.asyncio
async def test_make_agent_solver_callable_rejects_calls_mode():
    solver = build_agent_solver(lambda prompt: "x", mode="calls")
    with pytest.raises(BenchmarkError, match="mode='calls' requires an AgentExecutor"):
        await solver(BenchmarkTask(id="t", prompt="p"))


@pytest.mark.asyncio
async def test_make_agent_solver_callable_async_output():
    async def runner(prompt):
        return f"echo:{prompt}"

    solver = build_agent_solver(runner)
    out = await solver(BenchmarkTask(id="t", prompt="hi"))
    assert out == "echo:hi"


@pytest.mark.asyncio
async def test_make_agent_solver_prompt_key_reads_inputs():
    solver = build_agent_solver(lambda prompt: prompt)
    task = build_agent_solver  # placeholder to avoid lint unused; replaced below
    del task
    s2 = build_agent_solver(lambda prompt: prompt, prompt_key="q")
    out = await s2(BenchmarkTask(id="t", prompt="fallback", inputs={"q": "from-inputs"}))
    assert out == "from-inputs"
    # falls back to task.prompt when the keyed input is empty
    out2 = await solver(BenchmarkTask(id="t", prompt="theprompt"))
    assert out2 == "theprompt"


@pytest.mark.asyncio
async def test_make_agent_solver_agentexecutor_text_mode():
    from vincio.agents import AgentExecutor
    from vincio.agents.planner import Planner

    executor = AgentExecutor(
        MockProvider(default_text="Paris"), model="mock-1", planner=Planner(mode="static")
    )
    solver = build_agent_solver(executor, mode="text")
    out = await solver(BenchmarkTask(id="t", prompt="capital of France?"))
    assert out == "Paris"


@pytest.mark.asyncio
async def test_make_agent_solver_agentexecutor_calls_mode():
    from vincio.agents import AgentExecutor
    from vincio.agents.planner import Planner
    from vincio.tools.registry import ToolRegistry
    from vincio.tools.runtime import ToolRuntime

    registry = ToolRegistry()

    @registry.register()
    def get_weather(city: str) -> dict:
        """Get weather for a city."""
        return {"city": city}

    provider = MockProvider(
        script=[{"tool_call": {"name": "get_weather", "arguments": {"city": "sf"}}}, "done"]
    )
    executor = AgentExecutor(
        provider,
        model="mock-1",
        planner=Planner(mode="react"),
        tool_runtime=ToolRuntime(registry, cache_enabled=False),
        tool_specs=registry.specs(),
    )
    solver = build_agent_solver(executor, mode="calls")
    calls = await solver(BenchmarkTask(id="t", prompt="weather in SF?"))
    assert calls == [{"name": "get_weather", "arguments": {"city": "sf"}}]


@pytest.mark.asyncio
async def test_make_agent_solver_non_runner_object_raises():
    class Bogus:
        pass

    solver = build_agent_solver(Bogus())
    with pytest.raises(BenchmarkError, match="must be a ContextApp, AgentExecutor, or callable"):
        await solver(BenchmarkTask(id="t", prompt="p"))


# ---------------------------------------------------------------------------
# build_env_solver unsupported env (802)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_env_solver_unsupported_env_raises():
    solver = build_env_solver(lambda obs: None)
    with pytest.raises(BenchmarkError, match="unsupported env 'moon'"):
        await solver(BenchmarkTask(id="t", inputs={"env": "moon"}))


# ---------------------------------------------------------------------------
# tasks_from_jsonl skips blank lines (829->827)
# ---------------------------------------------------------------------------


def test_tasks_from_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(
        '{"id": "a", "gold": "g"}\n\n   \n{"id": "b", "gold": "h"}\n',
        encoding="utf-8",
    )
    tasks = tasks_from_jsonl(path)
    assert [t.id for t in tasks] == ["a", "b"]


# ---------------------------------------------------------------------------
# Export helpers: branch arms (879, 906->908, 933->935, 960, 982->984)
# ---------------------------------------------------------------------------


def test_bfcl_export_flattens_multi_turn_question():
    records = [
        {
            "id": "b1",
            "question": [[{"role": "user", "content": "first"}], [{"role": "user", "content": "second"}]],
            "function": [{"name": "f"}],
            "ground_truth": [{"name": "f", "arguments": {}}],
        }
    ]
    tasks = bfcl_tasks_from_export(records)
    assert tasks[0].prompt == "first second"


def test_bfcl_export_plain_string_question_passthrough():
    # A plain string question skips the multi-turn flattening branch.
    tasks = bfcl_tasks_from_export(
        [{"id": "b0", "prompt": "weather?", "ground_truth": [{"name": "f", "arguments": {}}]}]
    )
    assert tasks[0].prompt == "weather?"
    assert tasks[0].metadata["category"] == "simple"


def test_bfcl_export_flattens_flat_dict_turns():
    # Each "turn" is a single message dict (not a list of messages).
    records = [
        {
            "id": "b2",
            "question": [{"role": "user", "content": "alpha"}, {"role": "user", "content": "beta"}],
            "function": [{"name": "f"}],
            "ground_truth": [],
        }
    ]
    tasks = bfcl_tasks_from_export(records)
    assert tasks[0].prompt == "alpha beta"


def test_agentbench_export_carries_tolerance():
    tasks = agentbench_tasks_from_export(
        [{"id": "a", "answer": 10, "match": "numeric", "tolerance": 1.5, "environment": "db"}]
    )
    assert tasks[0].gold == {"match": "numeric", "value": 10, "tolerance": 1.5}
    assert tasks[0].inputs["env"] == "db"


def test_agentbench_export_without_tolerance_omits_key():
    tasks = agentbench_tasks_from_export([{"id": "a", "answer": "x"}])
    assert tasks[0].gold == {"match": "exact_match", "value": "x"}
    assert "tolerance" not in tasks[0].gold


def test_toolbench_export_with_final_answer_sets_gold():
    tasks = toolbench_tasks_from_export(
        [{"id": "t", "query": "do it", "api_list": [{"api_name": "search"}, "browse"], "final_answer": "done"}]
    )
    assert tasks[0].gold == {"final_answer": "done"}
    assert tasks[0].inputs["available_apis"] == ["search", "browse"]


def test_toolbench_export_without_answer_empty_gold():
    tasks = toolbench_tasks_from_export([{"query_id": "q9", "query": "x"}])
    assert tasks[0].gold == {}
    assert tasks[0].id == "q9"


def test_livecodebench_export_explicit_tests_key():
    tasks = livecodebench_tasks_from_export(
        [{"question_id": "q1", "tests": ["t1", "t2"], "release_date": "2025-01-01"}]
    )
    assert tasks[0].gold == {"tests": ["t1", "t2"]}
    assert tasks[0].metadata["release_date"] == "2025-01-01"


def test_livecodebench_export_public_private_split():
    tasks = livecodebench_tasks_from_export(
        [
            {
                "question_id": "q2",
                "public_test_cases": [{"id": "p0"}, {}],
                "private_test_cases": ["raw"],
            }
        ]
    )
    assert tasks[0].gold == {"public": ["p0", "public_test_cases-1"], "hidden": ["raw"]}


def test_mmlu_export_uses_answer_index_when_answer_absent():
    tasks = mmlu_pro_tasks_from_export(
        [{"question_id": "m1", "question": "q?", "answer_index": 2, "options": ["a", "b", "c"]}]
    )
    assert tasks[0].gold == 2
    assert tasks[0].inputs["options"] == ["a", "b", "c"]


def test_mmlu_export_prefers_explicit_answer():
    tasks = mmlu_pro_tasks_from_export([{"id": "m", "question": "q", "answer": "C", "answer_index": 9}])
    assert tasks[0].gold == "C"


# ---------------------------------------------------------------------------
# load_benchmark unknown (1017) + registry
# ---------------------------------------------------------------------------


def test_load_benchmark_unknown_raises():
    with pytest.raises(BenchmarkError, match="unknown benchmark 'nope'"):
        load_benchmark("nope")


def test_load_benchmark_constructs_adapter():
    adapter = load_benchmark("swebench_verified")
    assert isinstance(adapter, SWEBenchAdapter)
    assert adapter.name == "swebench_verified"


def test_load_benchmark_passes_variant_kwarg():
    adapter = load_benchmark("tau_bench", variant="tau2")
    assert isinstance(adapter, TauBenchAdapter)
    assert adapter.variant == "tau2"


def test_available_benchmarks_is_sorted_and_complete():
    names = available_benchmarks()
    assert names == sorted(names)
    assert "mmlu_pro" in names and "agentbench" in names


# ---------------------------------------------------------------------------
# SWEBench scorer + run() path (213-219, 257-266)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_swebench_unresolved_when_no_fail_to_pass():
    # An empty fail_to_pass set can never count as resolved.
    adapter = SWEBenchAdapter([BenchmarkTask(id="t", gold={"fail_to_pass": [], "pass_to_pass": ["b"]})])
    result = await adapter.score(adapter.tasks()[0], {"tests": {"b": "passed"}})
    assert result.success is False
    assert result.score == 1.0  # the one required test is green
    assert result.details == {"fail_to_pass_ok": True, "pass_to_pass_ok": True}


@pytest.mark.asyncio
async def test_swebench_run_path_sync_solver_scores_fresh_output():
    adapter = SWEBenchAdapter([BenchmarkTask(id="t", gold={"fail_to_pass": ["a"], "pass_to_pass": ["b"]})])

    def harness(task):
        return {"tests": {"a": "passed", "b": "passed"}}

    report = await adapter.run(harness)
    assert report.replayed is False
    assert report.success_rate == 1.0
    assert report.task_set_hash == adapter.task_set_hash()


@pytest.mark.asyncio
async def test_run_path_with_explicit_tasks_subset():
    adapter = GAIAAdapter([BenchmarkTask(id="a", gold="x"), BenchmarkTask(id="b", gold="y")])
    only_b = [adapter.tasks()[1]]
    report = await adapter.run(lambda task: "y", tasks=only_b)
    assert [r.task_id for r in report.results] == ["b"]
    assert report.success_rate == 1.0


# ---------------------------------------------------------------------------
# build_env_solver retail run extracts the action list (803-805)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_env_solver_retail_extracts_actions_and_scores():
    def fresh_policy():
        return scripted_policy(
            [
                EnvAction(tool="cancel_order", arguments={"order_id": "O1002"}),
                EnvAction(tool="refund_order", arguments={"order_id": "O1002"}),
            ]
        )

    task = BenchmarkTask(id="t", inputs={"env": "retail", "env_task": "cancel_refund"})
    actions = await build_env_solver(fresh_policy())(task)
    tool_names = [a["tool"] for a in actions]
    assert "cancel_order" in tool_names and "refund_order" in tool_names
    # The identical TauBench oracle scores the extracted actions to success.
    report = await TauBenchAdapter([task]).run(build_env_solver(fresh_policy()))
    assert report.success_rate == 1.0


# ---------------------------------------------------------------------------
# gaia / swebench export field fallbacks (837-868)
# ---------------------------------------------------------------------------


def test_gaia_export_field_fallbacks_and_default_id():
    tasks = gaia_tasks_from_export(
        [
            {"task_id": "g1", "Question": "Q?", "Final answer": "A", "Level": 2},
            {"question": "q2", "answer": "a2"},  # lowercase fallbacks + missing id
        ]
    )
    assert tasks[0].id == "g1"
    assert tasks[0].prompt == "Q?" and tasks[0].gold == "A"
    assert tasks[0].metadata["level"] == 2
    assert tasks[1].id == "gaia-1"  # synthesized index id
    assert tasks[1].prompt == "q2" and tasks[1].gold == "a2"


def test_swebench_export_list_fields_and_default_id():
    tasks = swebench_tasks_from_export(
        [{"problem_statement": "fix", "fail_to_pass": ["t1"], "pass_to_pass": ["t2"]}]
    )
    assert tasks[0].id == "swe-0"  # synthesized index id
    assert tasks[0].gold == {"fail_to_pass": ["t1"], "pass_to_pass": ["t2"]}
    assert tasks[0].inputs == {"repo": None, "base_commit": None}
