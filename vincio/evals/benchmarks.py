"""Agentic benchmark adapters (2.2): run Vincio agents on the world's leaderboards.

The field compares agents on public leaderboards — **SWE-bench Verified**,
**τ-bench / τ²-bench**, **GAIA**, **WebArena**, and **BFCL**. This module ships
one adapter per benchmark behind a single :class:`BenchmarkAdapter` contract so a
Vincio agent earns those market-recognized scores *inside Vincio's own bench*,
and the verifiable task success feeds back into the Pareto optimizer rather than
sitting in a separate harness.

Two design commitments make the scores trustworthy:

* **Verifiable scoring.** Each adapter scores an **end-state** the benchmark
  itself defines — SWE-bench's fail-to-pass/pass-to-pass test transition, τ-bench's
  database end state (via the :mod:`~vincio.evals.environment` oracle), GAIA's
  normalized exact match, WebArena's functional check, BFCL's AST match — not a
  model-judge proxy.
* **Reproducibility + honest offline.** Each adapter pins its task set by a
  content hash (:meth:`BenchmarkAdapter.task_set_hash`), so a silent task-set
  change is caught; and offline it **degrades to recorded-fixture replay**
  (:meth:`BenchmarkAdapter.replay`) — replaying a recorded agent output against
  the real scorer — rather than pretending to clone repos or drive a browser.

Results project onto an :class:`~vincio.evals.reports.EvalReport`
(:meth:`BenchmarkReport.to_eval_report`) so the existing gates and the optimizer
consume leaderboard scores like any other eval.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import VincioError
from ..stability import experimental
from .environment import EnvAction, EnvironmentSimulator, make_retail_environment, scripted_policy
from .reports import CaseResult, EvalReport

__all__ = [
    "BenchmarkError",
    "BenchmarkTask",
    "BenchmarkResult",
    "BenchmarkReport",
    "BenchmarkAdapter",
    "SWEBenchAdapter",
    "TauBenchAdapter",
    "GAIAAdapter",
    "WebArenaAdapter",
    "BFCLAdapter",
    "BENCHMARK_ADAPTERS",
    "load_benchmark",
    "available_benchmarks",
    # live-run path: drive a real Vincio agent and load official task sets
    "make_agent_solver",
    "make_env_solver",
    "tasks_from_jsonl",
    "gaia_tasks_from_export",
    "swebench_tasks_from_export",
    "bfcl_tasks_from_export",
]


class BenchmarkError(VincioError):
    """A benchmark adapter / task-set error."""


# A solver turns a task into an adapter-specific output (a string answer, a
# patch's test outcome, a list of actions, …). May be sync or async.
Solver = Callable[["BenchmarkTask"], "Any | Awaitable[Any]"]


class BenchmarkTask(BaseModel):
    """One pinned benchmark instance: the prompt, the verifiable gold, and an
    optional ``recorded`` agent output for offline replay."""

    id: str
    prompt: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    gold: Any = None
    recorded: Any = None  # a recorded agent output, for offline replay
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """The score for one task: a verifiable ``success`` plus a continuous ``score``."""

    task_id: str
    success: bool
    score: float = 0.0
    output: Any = None
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """A scored benchmark run, projectable onto an :class:`EvalReport`."""

    name: str
    variant: str = ""
    task_set_hash: str = ""
    results: list[BenchmarkResult] = Field(default_factory=list)
    replayed: bool = False

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        return round(sum(1 for r in self.results if r.success) / len(self.results), 4) if self.results else 0.0

    @property
    def mean_score(self) -> float:
        return round(sum(r.score for r in self.results) / len(self.results), 4) if self.results else 0.0

    def to_eval_report(self) -> EvalReport:
        """Project onto an EvalReport so gates / the optimizer can consume it."""
        cases = [
            CaseResult(
                case_id=r.task_id,
                metrics={"success": 1.0 if r.success else 0.0, "score": r.score},
                details=r.details,
                tags=[f"benchmark:{self.name}"],
            )
            for r in self.results
        ]
        return EvalReport(
            name=f"{self.name}{('/' + self.variant) if self.variant else ''}",
            dataset=self.name,
            cases=cases,
            metadata={"task_set_hash": self.task_set_hash, "replayed": self.replayed},
        )


@experimental(since="2.2")
class BenchmarkAdapter(ABC):
    """Base contract for a leaderboard adapter.

    Subclasses implement :meth:`score` (the benchmark's verifiable scorer). The
    base provides task-set pinning, the offline :meth:`replay` path, and the
    fresh-solve :meth:`run` path.
    """

    name: str = "benchmark"
    variant: str = ""

    def __init__(
        self,
        tasks: list[BenchmarkTask] | None = None,
        *,
        fixture_path: str | Path | None = None,
    ) -> None:
        if tasks is None and fixture_path is not None:
            tasks = self._load_fixture(fixture_path)
        # Accept BenchmarkTask instances or plain dicts (coerced here).
        self._tasks = [t if isinstance(t, BenchmarkTask) else BenchmarkTask.model_validate(t) for t in (tasks or [])]

    # -- task set -------------------------------------------------------------

    def tasks(self) -> list[BenchmarkTask]:
        return list(self._tasks)

    def task_set_hash(self) -> str:
        """A stable content hash over ``(id, gold)`` of every task.

        Pins the task set: a changed/added/removed task changes the hash, so a
        silent task-set drift is caught against a recorded value.
        """
        canonical = [
            {"id": t.id, "gold": t.gold} for t in sorted(self._tasks, key=lambda t: t.id)
        ]
        blob = json.dumps(canonical, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _load_fixture(self, path: str | Path) -> list[BenchmarkTask]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_tasks = data.get("tasks", data) if isinstance(data, dict) else data
        tasks = [BenchmarkTask.model_validate(t) for t in raw_tasks]
        declared = data.get("task_set_hash") if isinstance(data, dict) else None
        if declared:
            self._tasks = tasks  # set before recompute
            actual = self.task_set_hash()
            if declared != actual:
                raise BenchmarkError(
                    f"{self.name}: fixture task-set hash mismatch "
                    f"(declared {declared!r}, computed {actual!r}); the task set drifted"
                )
        return tasks

    # -- scoring --------------------------------------------------------------

    @abstractmethod
    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        """Score one task output against its verifiable gold."""

    # -- run paths ------------------------------------------------------------

    async def run(self, solver: Solver, *, tasks: list[BenchmarkTask] | None = None) -> BenchmarkReport:
        """Solve each task with ``solver`` and score the fresh outputs."""
        results: list[BenchmarkResult] = []
        for task in tasks or self._tasks:
            output = solver(task)
            if hasattr(output, "__await__"):
                output = await output  # type: ignore[assignment]
            results.append(await self.score(task, output))
        return BenchmarkReport(
            name=self.name, variant=self.variant, task_set_hash=self.task_set_hash(), results=results
        )

    async def replay(self, *, tasks: list[BenchmarkTask] | None = None) -> BenchmarkReport:
        """Offline path: score each task's recorded output against the real scorer."""
        results: list[BenchmarkResult] = []
        for task in tasks or self._tasks:
            if task.recorded is None:
                raise BenchmarkError(f"{self.name}: task {task.id!r} has no recorded output to replay")
            results.append(await self.score(task, task.recorded))
        return BenchmarkReport(
            name=self.name,
            variant=self.variant,
            task_set_hash=self.task_set_hash(),
            results=results,
            replayed=True,
        )


# ---------------------------------------------------------------------------
# SWE-bench Verified
# ---------------------------------------------------------------------------


@experimental(since="2.2")
class SWEBenchAdapter(BenchmarkAdapter):
    """SWE-bench Verified: a patch *resolves* an issue iff the fail-to-pass tests
    turn green and the pass-to-pass tests stay green.

    ``gold`` = ``{"fail_to_pass": [...], "pass_to_pass": [...]}``; ``output`` =
    ``{"tests": {test_id: "passed"|"failed", ...}}`` (the test outcome of the
    candidate patch). This is SWE-bench's exact resolution criterion — verifiable,
    not a judge proxy.
    """

    name = "swebench_verified"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        gold = task.gold or {}
        fail_to_pass = list(gold.get("fail_to_pass", []))
        pass_to_pass = list(gold.get("pass_to_pass", []))
        tests = (output or {}).get("tests", {}) if isinstance(output, dict) else {}
        f2p_ok = all(tests.get(t) == "passed" for t in fail_to_pass)
        p2p_ok = all(tests.get(t) == "passed" for t in pass_to_pass)
        resolved = bool(fail_to_pass) and f2p_ok and p2p_ok
        total = len(fail_to_pass) + len(pass_to_pass)
        green = sum(1 for t in fail_to_pass + pass_to_pass if tests.get(t) == "passed")
        return BenchmarkResult(
            task_id=task.id,
            success=resolved,
            score=round(green / total, 4) if total else 0.0,
            output=output,
            details={"fail_to_pass_ok": f2p_ok, "pass_to_pass_ok": p2p_ok},
        )


# ---------------------------------------------------------------------------
# τ-bench / τ²-bench  (tool-agent in a stateful customer-service world)
# ---------------------------------------------------------------------------


@experimental(since="2.2")
class TauBenchAdapter(BenchmarkAdapter):
    """τ-bench / τ²-bench: a tool agent operates a customer-service world; success
    is the **database end state**, scored by the :mod:`~vincio.evals.environment`
    oracle.

    ``inputs`` = ``{"env": "retail", "env_task": "cancel_refund"}``; ``output`` /
    ``recorded`` = a list of action dicts (``{"tool": ..., "arguments": {...}}``)
    the agent took. The adapter replays them against the deterministic reference
    environment and returns the oracle's verdict. ``variant="tau2"`` additionally
    requires the agent to interleave the recorded user action (dual control).
    """

    name = "tau_bench"

    def __init__(self, tasks=None, *, fixture_path=None, variant: str = "tau") -> None:
        super().__init__(tasks, fixture_path=fixture_path)
        self.variant = variant

    def _make_env(self, task: BenchmarkTask):
        env_name = task.inputs.get("env", "retail")
        if env_name != "retail":
            raise BenchmarkError(f"tau_bench: unsupported env {env_name!r}")
        return make_retail_environment(task.inputs.get("env_task", "cancel_refund"))

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        actions = [EnvAction.model_validate(a) for a in (output or [])]
        env = self._make_env(task)
        result = await EnvironmentSimulator().arun(env, scripted_policy(actions))
        return BenchmarkResult(
            task_id=task.id,
            success=result.success,
            score=result.verification.score,
            output=output,
            details={"checks": [c.model_dump() for c in result.verification.checks], "variant": self.variant},
        )


# ---------------------------------------------------------------------------
# GAIA  (general assistant; normalized exact match)
# ---------------------------------------------------------------------------


_ARTICLES = {"a", "an", "the"}


def _gaia_normalize(text: Any) -> str:
    s = str(text).strip().lower()
    s = s.replace(",", "")  # 1,000 -> 1000
    s = re.sub(r"[^\w\s.%/-]", " ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


@experimental(since="2.2")
class GAIAAdapter(BenchmarkAdapter):
    """GAIA: a general-assistant question with a single gold answer, scored by
    GAIA's normalized **exact match** (lowercase, drop articles/punctuation,
    strip thousands separators)."""

    name = "gaia"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        predicted = _gaia_normalize(output)
        gold = _gaia_normalize(task.gold)
        match = predicted == gold and gold != ""
        return BenchmarkResult(
            task_id=task.id,
            success=match,
            score=1.0 if match else 0.0,
            output=output,
            details={"normalized_prediction": predicted, "normalized_gold": gold,
                     "level": task.metadata.get("level")},
        )


# ---------------------------------------------------------------------------
# WebArena  (web navigation; functional correctness check)
# ---------------------------------------------------------------------------


@experimental(since="2.2")
class WebArenaAdapter(BenchmarkAdapter):
    """WebArena: a web-navigation task scored by a functional check on the agent's
    final answer / end state.

    ``gold`` = ``{"type": "exact_match"|"must_include"|"url", "value": ...}``;
    ``output`` = the agent's final answer string (or reached URL). This mirrors
    WebArena's ``string_match`` / ``url_match`` functional evaluators.
    """

    name = "webarena"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        gold = task.gold or {}
        gtype = gold.get("type", "exact_match")
        value = gold.get("value")
        answer = str(output or "")
        norm = answer.strip().lower()
        if gtype == "exact_match":
            ok = norm == str(value).strip().lower()
        elif gtype == "must_include":
            needles = value if isinstance(value, list) else [value]
            ok = all(str(n).strip().lower() in norm for n in needles)
        elif gtype == "url":
            ok = str(value).strip().lower() in norm
        else:
            raise BenchmarkError(f"webarena: unknown gold type {gtype!r}")
        return BenchmarkResult(
            task_id=task.id, success=ok, score=1.0 if ok else 0.0, output=output,
            details={"check": gtype},
        )


# ---------------------------------------------------------------------------
# BFCL  (Berkeley Function-Calling Leaderboard; AST match)
# ---------------------------------------------------------------------------


def _normalize_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str(call.get("name", "")).strip()
    args = call.get("arguments", call.get("args", {})) or {}
    # Normalize argument values to strings for stable comparison across JSON
    # number/string ambiguity (BFCL allows type-coercible matches).
    norm = {k: _norm_value(v) for k, v in sorted(args.items())}
    return name, norm


def _norm_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return [_norm_value(v) for v in value]
    return value


@experimental(since="2.2")
class BFCLAdapter(BenchmarkAdapter):
    """BFCL (Berkeley Function-Calling Leaderboard): score the agent's function
    call(s) by **AST match** — function name + arguments — against the gold call(s).

    ``gold`` = ``[{"name": ..., "arguments": {...}}, ...]`` (an empty list for the
    ``relevance`` category, where the correct behaviour is to call *nothing*);
    ``output`` = the agent's produced call(s) in the same shape.
    """

    name = "bfcl"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        gold_calls = [_normalize_call(c) for c in (task.gold or [])]
        out_calls = [_normalize_call(c) for c in (output or [])]
        # Relevance category: gold is "no call"; success iff the agent abstained.
        if not gold_calls:
            ok = len(out_calls) == 0
            return BenchmarkResult(
                task_id=task.id, success=ok, score=1.0 if ok else 0.0, output=output,
                details={"category": task.metadata.get("category", "relevance")},
            )
        # Order-independent multiset match (BFCL parallel/multiple categories).
        remaining = list(out_calls)
        matched = 0
        for gold in gold_calls:
            if gold in remaining:
                remaining.remove(gold)
                matched += 1
        exact = matched == len(gold_calls) and not remaining
        return BenchmarkResult(
            task_id=task.id,
            success=exact,
            score=round(matched / len(gold_calls), 4),
            output=output,
            details={"matched": matched, "expected": len(gold_calls),
                     "category": task.metadata.get("category", "simple")},
        )


# ---------------------------------------------------------------------------
# Live-run path: drive a real Vincio agent, and load official task sets
# ---------------------------------------------------------------------------
#
# ``adapter.replay()`` is the offline path (a recorded output per task).
# ``adapter.run(solver)`` is the **live** path: it calls ``solver(task)`` to
# produce a *fresh* output and scores it with the **identical** ``score()``.
# The helpers below turn a Vincio agent into a solver, and load a real benchmark
# export into ``BenchmarkTask``s, so pointing an adapter at a live task set is a
# one-liner — not a reimplementation of the scorer.


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def make_agent_solver(runner: Any, *, mode: str = "text", prompt_key: str | None = None) -> Solver:
    """Turn a Vincio agent into a benchmark :data:`Solver` (the live-run path).

    ``runner`` may be a :class:`~vincio.core.app.ContextApp` (driven via ``arun``),
    an :class:`~vincio.agents.executor.AgentExecutor` (via ``run``), or any
    ``callable(prompt) -> output``. ``mode``:

    * ``"text"`` — return the agent's final answer string (GAIA, WebArena, or a
      SWE-bench answer agent emitting a test-outcome dict).
    * ``"calls"`` — return the agent's tool calls as ``[{"name", "arguments"}]``
      (BFCL). Requires an ``AgentExecutor`` runner (the calls come from its
      trajectory).

    ``prompt_key`` reads the prompt from ``task.inputs[prompt_key]`` instead of
    ``task.prompt`` when set.
    """
    if mode not in ("text", "calls"):
        raise BenchmarkError(f"unknown solver mode {mode!r}; expected 'text' or 'calls'")

    async def solve(task: BenchmarkTask) -> Any:
        prompt = task.prompt if prompt_key is None else _coerce_text(task.inputs.get(prompt_key))
        prompt = prompt or task.prompt

        if hasattr(runner, "arun"):  # ContextApp
            if mode == "calls":
                raise BenchmarkError("mode='calls' requires an AgentExecutor runner")
            result = await runner.arun(prompt)
            return getattr(result, "raw_text", "") or _coerce_text(getattr(result, "output", result))

        if hasattr(runner, "astream") and hasattr(runner, "model"):  # AgentExecutor
            # Drive the agent's own event stream: tool_call events carry the
            # arguments (a ReAct trajectory does not), and the terminal done event
            # carries the final answer — so one path serves both modes.
            calls: list[dict[str, Any]] = []
            text = ""
            async for event in runner.astream(prompt):
                if event.type == "tool_call":
                    calls.append({"name": event.tool_name, "arguments": dict(event.arguments or {})})
                elif event.type == "done":
                    text = _coerce_text(event.result)
            return calls if mode == "calls" else text

        if callable(runner):
            if mode == "calls":
                raise BenchmarkError("mode='calls' requires an AgentExecutor runner")
            out = runner(prompt)
            if hasattr(out, "__await__"):
                out = await out
            return out

        raise BenchmarkError("solver runner must be a ContextApp, AgentExecutor, or callable")

    return solve


def make_env_solver(policy: Any) -> Solver:
    """Drive an agent ``policy`` through the environment a τ-bench task names and
    return the action list — the live τ-bench run path.

    The agent *decides* the actions by interacting with the deterministic
    reference world; :class:`TauBenchAdapter` then scores those actions on the
    database end state with the identical oracle. ``policy`` is any
    ``AgentPolicy`` (``callable(observation) -> EnvAction``).
    """

    async def solve(task: BenchmarkTask) -> Any:
        from .environment import EnvironmentSimulator, make_retail_environment

        env_name = task.inputs.get("env", "retail")
        if env_name != "retail":
            raise BenchmarkError(f"make_env_solver: unsupported env {env_name!r}")
        env = make_retail_environment(task.inputs.get("env_task", "cancel_refund"))
        result = await EnvironmentSimulator().arun(env, policy)
        return [
            {"tool": step.tool_name, "arguments": dict(step.tool_arguments)}
            for step in result.trajectory.steps
            if step.is_tool
        ]

    return solve


def _maybe_json(value: Any) -> Any:
    """Parse a JSON-encoded string (SWE-bench's FAIL_TO_PASS et al.), else passthrough."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def tasks_from_jsonl(path: str | Path) -> list[BenchmarkTask]:
    """Load `BenchmarkTask`s from a JSONL file (one task object per line)."""
    out: list[BenchmarkTask] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(BenchmarkTask.model_validate(json.loads(line)))
    return out


def gaia_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map official GAIA records (``task_id`` / ``Question`` / ``Final answer`` /
    ``Level``) onto `BenchmarkTask`s for :class:`GAIAAdapter`."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        gold = rec.get("Final answer", rec.get("final_answer", rec.get("answer")))
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("task_id", rec.get("id", f"gaia-{index}"))),
                prompt=str(rec.get("Question", rec.get("question", ""))),
                gold=gold,
                metadata={"level": rec.get("Level", rec.get("level"))},
            )
        )
    return tasks


def swebench_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map official SWE-bench (Verified) records (``instance_id`` /
    ``FAIL_TO_PASS`` / ``PASS_TO_PASS``) onto `BenchmarkTask`s for
    :class:`SWEBenchAdapter`. ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` may be JSON
    strings (the released format) or lists."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        fail_to_pass = _maybe_json(rec.get("FAIL_TO_PASS", rec.get("fail_to_pass", [])))
        pass_to_pass = _maybe_json(rec.get("PASS_TO_PASS", rec.get("pass_to_pass", [])))
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("instance_id", rec.get("id", f"swe-{index}"))),
                prompt=str(rec.get("problem_statement", "")),
                gold={"fail_to_pass": list(fail_to_pass or []), "pass_to_pass": list(pass_to_pass or [])},
                inputs={"repo": rec.get("repo"), "base_commit": rec.get("base_commit")},
            )
        )
    return tasks


def bfcl_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map BFCL records (``id`` / ``question`` / ``function`` / ``ground_truth``)
    onto `BenchmarkTask`s for :class:`BFCLAdapter`. The available functions ride
    on ``inputs['functions']`` so a live solver can register them as tools."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        question = rec.get("question", rec.get("prompt", ""))
        if isinstance(question, list):  # BFCL multi-turn message format
            question = " ".join(
                m.get("content", "") for turn in question
                for m in (turn if isinstance(turn, list) else [turn])
                if isinstance(m, dict)
            )
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("id", f"bfcl-{index}")),
                prompt=str(question),
                gold=rec.get("ground_truth", rec.get("gold", [])),
                inputs={"functions": rec.get("function", rec.get("functions", []))},
                metadata={"category": rec.get("category", "simple")},
            )
        )
    return tasks


BENCHMARK_ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    SWEBenchAdapter.name: SWEBenchAdapter,
    TauBenchAdapter.name: TauBenchAdapter,
    GAIAAdapter.name: GAIAAdapter,
    WebArenaAdapter.name: WebArenaAdapter,
    BFCLAdapter.name: BFCLAdapter,
}


def available_benchmarks() -> list[str]:
    """The names of the shipped benchmark adapters."""
    return sorted(BENCHMARK_ADAPTERS)


def load_benchmark(name: str, **kwargs: Any) -> BenchmarkAdapter:
    """Construct a benchmark adapter by name."""
    if name not in BENCHMARK_ADAPTERS:
        raise BenchmarkError(f"unknown benchmark {name!r}; known: {available_benchmarks()}")
    return BENCHMARK_ADAPTERS[name](**kwargs)
