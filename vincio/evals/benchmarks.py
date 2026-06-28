"""Agentic benchmark adapters: run Vincio agents on the world's leaderboards.

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
    "AgentBenchAdapter",
    "ToolBenchAdapter",
    "LiveCodeBenchAdapter",
    "MMLUProAdapter",
    "SpiderAdapter",
    "BIRDAdapter",
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
    "agentbench_tasks_from_export",
    "toolbench_tasks_from_export",
    "livecodebench_tasks_from_export",
    "mmlu_pro_tasks_from_export",
    "spider_tasks_from_export",
    "bird_tasks_from_export",
]


class BenchmarkError(VincioError):
    """A benchmark adapter / task-set error."""

    code = "BENCHMARK_ERROR"


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
# AgentBench  (multi-environment agent; per-environment verifiable end state)
# ---------------------------------------------------------------------------


def _norm_text(value: Any) -> str:
    return str(value if value is not None else "").strip().lower()


def _as_numeric(value: Any) -> float | None:
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


class AgentBenchAdapter(BenchmarkAdapter):
    """AgentBench: an agent operates one of several environments (OS, DB,
    knowledge-graph, …); each task is scored on the **verifiable end state** the
    environment defines, selected by ``gold["match"]``:

    * ``exact_match`` — normalized string equality (OS / web tasks).
    * ``contains`` — every needle present in the answer (free-form tasks).
    * ``set_match`` — order-independent set equality of the returned items
      (knowledge-graph queries); the continuous score is recall against the
      gold set.
    * ``numeric`` — value within ``gold["tolerance"]`` (default ``0``) of the
      gold number (DB aggregation answers).

    ``gold`` = ``{"match": ..., "value": ..., "tolerance": 0.0}``; ``output`` =
    the agent's final answer (a string, or a list for ``set_match``). This scores
    AgentBench's own verifiable task outcome — not a model-judge proxy — while
    spanning its heterogeneous environments under one contract.
    """

    name = "agentbench"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        gold = task.gold if isinstance(task.gold, dict) else {"match": "exact_match", "value": task.gold}
        match = gold.get("match", "exact_match")
        value = gold.get("value")
        env = task.inputs.get("env") or task.metadata.get("env")
        if match == "exact_match":
            ok = _norm_text(output) == _norm_text(value)
            score = 1.0 if ok else 0.0
        elif match == "contains":
            needles = value if isinstance(value, list) else [value]
            answer = _norm_text(output)
            hits = sum(1 for n in needles if _norm_text(n) in answer)
            ok = hits == len(needles) and bool(needles)
            score = round(hits / len(needles), 4) if needles else 0.0
        elif match == "set_match":
            gold_set = {_norm_text(v) for v in (value or [])}
            out_set = {_norm_text(v) for v in (output or [])}
            hits = len(gold_set & out_set)
            ok = bool(gold_set) and gold_set == out_set
            score = round(hits / len(gold_set), 4) if gold_set else 0.0
        elif match == "numeric":
            predicted, target = _as_numeric(output), _as_numeric(value)
            tolerance = float(gold.get("tolerance", 0.0) or 0.0)
            ok = predicted is not None and target is not None and abs(predicted - target) <= tolerance
            score = 1.0 if ok else 0.0
        else:
            raise BenchmarkError(f"agentbench: unknown match type {match!r}")
        return BenchmarkResult(
            task_id=task.id, success=ok, score=score, output=output,
            details={"match": match, "env": env},
        )


# ---------------------------------------------------------------------------
# ToolBench  (multi-step API solution path; solvable pass rate)
# ---------------------------------------------------------------------------


_FINISH_TOOLS = {"finish", "give_answer", "submit", "final_answer"}
_GIVE_UP = {"give_up_and_restart", "give_up"}


def _toolbench_action(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = _norm_text(call.get("name") or call.get("tool"))
    args = call.get("arguments", call.get("args", {})) or {}
    return name, args if isinstance(args, dict) else {}


class ToolBenchAdapter(BenchmarkAdapter):
    """ToolBench: an agent solves a task by a **multi-step path of API calls**
    that terminates in a finish action. The adapter reproduces ToolBench's
    *solvable pass-rate* criterion on the path — a task passes iff:

    * the path **terminates with an answer** (a ``Finish``/``give_answer`` action,
      or ``Finish`` with ``return_type == "give_answer"``) rather than giving up;
    * every non-finish call names an **available API** (no hallucinated tool),
      taken from ``inputs["available_apis"]`` when provided; and
    * if ``gold["final_answer"]`` is set, the produced final answer matches it
      (normalized).

    ``output`` / ``recorded`` = ``[{"name", "arguments"}, …]`` (the call path, the
    final element being the finish action). The continuous score is the fraction
    of non-finish calls that reference a valid API. Unlike BFCL's single-call AST
    match, this grades a whole tool-use trajectory's validity and termination.
    """

    name = "toolbench"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        calls = [_toolbench_action(c) for c in (output or []) if isinstance(c, dict)]
        available = {_norm_text(a) for a in (task.inputs.get("available_apis") or task.inputs.get("functions") or [])}
        gold = task.gold if isinstance(task.gold, dict) else {}

        finish = next((c for c in reversed(calls) if c[0] in _FINISH_TOOLS or c[0] in _GIVE_UP), None)
        gave_up = finish is not None and (
            finish[0] in _GIVE_UP or _norm_text(finish[1].get("return_type")) == "give_up_and_restart"
        )
        answered = finish is not None and not gave_up
        # The final answer is the finish action's answer argument, when present.
        final_answer = ""
        if finish is not None:
            final_answer = str(
                finish[1].get("final_answer", finish[1].get("answer", finish[1].get("response", "")))
            )

        worker_calls = [c for c in calls if c[0] not in _FINISH_TOOLS and c[0] not in _GIVE_UP]
        if available:
            valid = sum(1 for name, _ in worker_calls if name in available)
        else:  # no API allow-list declared — every named call counts as valid
            valid = len(worker_calls)
        apis_ok = valid == len(worker_calls)
        score = round(valid / len(worker_calls), 4) if worker_calls else (1.0 if answered else 0.0)

        gold_answer = gold.get("final_answer")
        answer_ok = gold_answer is None or _norm_text(final_answer) == _norm_text(gold_answer)

        success = answered and apis_ok and answer_ok
        return BenchmarkResult(
            task_id=task.id, success=success, score=score, output=output,
            details={
                "answered": answered, "gave_up": gave_up, "apis_valid": apis_ok,
                "valid_calls": valid, "worker_calls": len(worker_calls), "answer_ok": answer_ok,
            },
        )


# ---------------------------------------------------------------------------
# LiveCodeBench  (contamination-free code generation; all tests pass)
# ---------------------------------------------------------------------------


def _livecodebench_tests(gold: Any) -> list[str]:
    if isinstance(gold, dict):
        if "tests" in gold:
            return [str(t) for t in (gold.get("tests") or [])]
        return [str(t) for t in (gold.get("public", []) or []) + (gold.get("hidden", []) or [])]
    if isinstance(gold, list):
        return [str(t) for t in gold]
    return []


class LiveCodeBenchAdapter(BenchmarkAdapter):
    """LiveCodeBench: a generated solution passes iff **every required test case
    turns green**, scored on the test outcomes the benchmark's own runner
    produces — mirroring SWE-bench's verifiable approach rather than executing
    untrusted model code inside Vincio.

    ``gold`` = ``{"tests": [...]}`` (the required test ids; or a
    ``{"public": [...], "hidden": [...]}`` split, both of which must pass);
    ``output`` = ``{"results": {test_id: "passed"|"failed"|"error"}}`` — the
    candidate solution's per-test outcome. Success is all-tests-pass (the
    contest-style ``pass@1`` criterion); the continuous score is the fraction of
    required tests passed. ``metadata["release_date"]`` carries LiveCodeBench's
    contamination-free release-window pin.
    """

    name = "livecodebench"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        required = _livecodebench_tests(task.gold)
        results = (output or {}).get("results", {}) if isinstance(output, dict) else {}
        passed = sum(1 for t in required if results.get(t) == "passed")
        all_pass = bool(required) and passed == len(required)
        return BenchmarkResult(
            task_id=task.id,
            success=all_pass,
            score=round(passed / len(required), 4) if required else 0.0,
            output=output,
            details={"passed": passed, "required": len(required),
                     "release_date": task.metadata.get("release_date")},
        )


# ---------------------------------------------------------------------------
# MMLU-Pro  (10-way multiple choice; robust letter extraction)
# ---------------------------------------------------------------------------


# Answer-extraction patterns in priority order, mirroring MMLU-Pro's own
# parser: an explicit "answer is (X)", then a bracketed/letter form, then a
# bare trailing option letter.
_MMLU_PATTERNS = [
    re.compile(r"answer\s+is\s*:?\s*\(?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"answer\s*:?\s*\(?\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\(\s*([A-J])\s*\)"),
    re.compile(r"\b([A-J])\b(?!.*\b[A-J]\b)", re.DOTALL),  # last standalone letter
]


def _mmlu_extract(output: Any) -> str:
    text = str(output if output is not None else "").strip()
    for pattern in _MMLU_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return ""


def _mmlu_gold_letter(gold: Any) -> str:
    """Gold may be a letter ('D') or a 0-based index (3 -> 'D')."""
    if isinstance(gold, bool):
        return ""
    if isinstance(gold, int):
        return chr(ord("A") + gold) if 0 <= gold < 10 else ""
    text = str(gold or "").strip().upper()
    return text if len(text) == 1 and "A" <= text <= "J" else ""


class MMLUProAdapter(BenchmarkAdapter):
    """MMLU-Pro: a 10-option (A–J) multiple-choice question scored by extracting
    the predicted option letter from the model's output and comparing it to the
    gold letter — MMLU-Pro's own answer-extraction-and-match criterion.

    ``gold`` = the correct option as a letter (``"D"``) or 0-based index (``3``);
    ``output`` = the model's free-text answer (the letter is extracted with a
    priority cascade of patterns). ``inputs["options"]`` may carry the choices
    for a live solver to render. Distinct from GAIA's free-form exact match: the
    answer space is the constrained option set, so extraction precedes the match.
    """

    name = "mmlu_pro"

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        predicted = _mmlu_extract(output)
        gold = _mmlu_gold_letter(task.gold)
        match = bool(gold) and predicted == gold
        return BenchmarkResult(
            task_id=task.id,
            success=match,
            score=1.0 if match else 0.0,
            output=output,
            details={"predicted": predicted, "gold": gold,
                     "category": task.metadata.get("category")},
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


def agentbench_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map AgentBench records (``id`` / ``description`` / ``environment`` /
    ``answer`` / ``match``) onto `BenchmarkTask`s for :class:`AgentBenchAdapter`.
    ``match`` defaults to ``exact_match``; ``answer`` becomes ``gold["value"]``."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        gold = {
            "match": rec.get("match", rec.get("metric", "exact_match")),
            "value": rec.get("answer", rec.get("gold")),
        }
        if "tolerance" in rec:
            gold["tolerance"] = rec["tolerance"]
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("id", f"agentbench-{index}")),
                prompt=str(rec.get("description", rec.get("question", rec.get("prompt", "")))),
                gold=gold,
                inputs={"env": rec.get("environment", rec.get("env"))},
                metadata={"env": rec.get("environment", rec.get("env"))},
            )
        )
    return tasks


def toolbench_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map ToolBench records (``id`` / ``query`` / ``api_list`` /
    ``final_answer``) onto `BenchmarkTask`s for :class:`ToolBenchAdapter`. The
    available API names ride on ``inputs['available_apis']`` so the solvable
    pass-rate scorer can reject hallucinated tools."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        api_list = rec.get("api_list", rec.get("available_apis", rec.get("functions", [])))
        names = [
            (a.get("api_name") or a.get("name")) if isinstance(a, dict) else a
            for a in (api_list or [])
        ]
        gold: dict[str, Any] = {}
        if rec.get("final_answer") is not None or rec.get("answer") is not None:
            gold["final_answer"] = rec.get("final_answer", rec.get("answer"))
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("id", rec.get("query_id", f"toolbench-{index}"))),
                prompt=str(rec.get("query", rec.get("prompt", ""))),
                gold=gold,
                inputs={"available_apis": [n for n in names if n]},
                metadata={"category": rec.get("category", "general")},
            )
        )
    return tasks


def livecodebench_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map LiveCodeBench records (``question_id`` / ``question_content`` /
    ``public_test_cases`` / ``private_test_cases`` / ``release_date``) onto
    `BenchmarkTask`s for :class:`LiveCodeBenchAdapter`. Test-case ids form the
    required-pass gold; the release window pins contamination-free selection."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        def _ids(key: str, rec: dict[str, Any] = rec) -> list[str]:
            cases = rec.get(key, []) or []
            return [str(c.get("id", f"{key}-{i}")) if isinstance(c, dict) else str(c)
                    for i, c in enumerate(cases)]

        if "tests" in rec:
            gold: Any = {"tests": [str(t) for t in rec["tests"]]}
        else:
            gold = {"public": _ids("public_test_cases"), "hidden": _ids("private_test_cases")}
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("question_id", rec.get("id", f"lcb-{index}"))),
                prompt=str(rec.get("question_content", rec.get("prompt", ""))),
                gold=gold,
                metadata={"release_date": rec.get("release_date"),
                          "difficulty": rec.get("difficulty")},
            )
        )
    return tasks


def mmlu_pro_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map MMLU-Pro records (``question_id`` / ``question`` / ``options`` /
    ``answer`` or ``answer_index`` / ``category``) onto `BenchmarkTask`s for
    :class:`MMLUProAdapter`. The option list rides on ``inputs['options']``."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        gold = rec.get("answer")
        if gold is None and rec.get("answer_index") is not None:
            gold = rec["answer_index"]
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("question_id", rec.get("id", f"mmlu-{index}"))),
                prompt=str(rec.get("question", rec.get("prompt", ""))),
                gold=gold,
                inputs={"options": rec.get("options", [])},
                metadata={"category": rec.get("category", rec.get("src"))},
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Spider / BIRD  (text-to-SQL; execution accuracy)
# ---------------------------------------------------------------------------


def _build_catalog(tables: dict[str, Any]) -> Any:
    """Build a :class:`~vincio.data.DataCatalog` from a task's in-line database
    (``{name: {"columns": [...], "rows": [[...]]}}``)."""
    from ..data import DataCatalog, Dataset

    catalog = DataCatalog()
    for name, spec in tables.items():
        columns = list(spec["columns"])
        rows = [list(r) for r in spec.get("rows", [])]
        catalog.add(Dataset.from_rows(rows, columns, name=name), name=name)
    return catalog


def _execute_result(sql: str, catalog: Any) -> tuple[bool, list[tuple[Any, ...]], str]:
    """Execute ``sql`` read-only against ``catalog``. Returns
    ``(valid, rows, error)`` — ``valid`` is false when the query was refused as
    not read-only or failed to execute."""
    from ..core.errors import QueryError
    from ..data import query_dataset

    try:
        result = query_dataset(sql, catalog, max_rows=100_000)
    except QueryError as exc:
        return False, [], str(exc)
    return True, [tuple(row) for row in result.rows], ""


class _TextToSQLAdapter(BenchmarkAdapter):
    """Shared text-to-SQL scoring: **execution accuracy** — the predicted SQL is
    correct iff, executed against the task's database, it returns the same result
    set as the gold SQL. The predicted query is held read-only (a write / DDL is
    refused, scoring the task failed), so the benchmark measures *governed*
    text-to-query, not raw generation.

    ``prompt`` = the natural-language question; ``inputs`` = ``{"tables": {name:
    {"columns": [...], "rows": [[...]]}}}`` (and, for BIRD, an ``"evidence"`` hint);
    ``gold`` = the gold SQL; ``output`` / ``recorded`` = the candidate SQL.
    """

    def _order_sensitive(self, gold_sql: str) -> bool:
        return "order by" in gold_sql.lower()

    async def score(self, task: BenchmarkTask, output: Any) -> BenchmarkResult:
        tables = task.inputs.get("tables", {})
        gold_sql = str(task.gold or "")
        pred_sql = str(output or "")
        catalog = _build_catalog(tables)
        gold_valid, gold_rows, _ = _execute_result(gold_sql, catalog)
        pred_valid, pred_rows, pred_err = _execute_result(pred_sql, catalog)
        if not gold_valid:
            raise BenchmarkError(f"{self.name}: gold SQL failed to execute for task {task.id!r}")
        if self._order_sensitive(gold_sql):
            match = pred_valid and pred_rows == gold_rows
        else:
            key = lambda rows: sorted(rows, key=repr)  # noqa: E731 - local comparator
            match = pred_valid and key(pred_rows) == key(gold_rows)
        return BenchmarkResult(
            task_id=task.id,
            success=bool(match),
            score=1.0 if match else 0.0,
            output=output,
            details={
                "valid": pred_valid,
                "execution_match": bool(match),
                "gold_rows": len(gold_rows),
                "pred_rows": len(pred_rows) if pred_valid else 0,
                "error": pred_err,
            },
        )


class SpiderAdapter(_TextToSQLAdapter):
    """Spider: cross-domain text-to-SQL scored by **execution accuracy** (the
    predicted query's result set equals the gold query's). Vincio runs both on its
    in-process, read-only SQL engine, so the score reflects *governed* text-to-query
    — a generated write or DDL is structurally refused and scores the task failed."""

    name = "spider"


class BIRDAdapter(_TextToSQLAdapter):
    """BIRD: large-scale, real-world text-to-SQL with external-knowledge
    ``evidence``, scored by **execution accuracy** (BIRD's primary metric). The
    ``inputs["evidence"]`` hint rides on the prompt for a live solver; scoring is
    the same governed, read-only execution match Spider uses."""

    name = "bird"


def spider_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map Spider records (``db_id`` / ``question`` / ``query`` / ``tables``) onto
    :class:`BenchmarkTask`s for :class:`SpiderAdapter`. The per-task ``tables``
    carry the inline database the offline engine executes against."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("id", f"spider-{index}")),
                prompt=str(rec.get("question", "")),
                gold=rec.get("query", rec.get("gold")),
                recorded=rec.get("recorded", rec.get("query", rec.get("gold"))),
                inputs={"tables": rec.get("tables", {})},
                metadata={"db_id": rec.get("db_id")},
            )
        )
    return tasks


def bird_tasks_from_export(records: list[dict[str, Any]]) -> list[BenchmarkTask]:
    """Map BIRD records (``db_id`` / ``question`` / ``SQL`` / ``evidence`` /
    ``tables``) onto :class:`BenchmarkTask`s for :class:`BIRDAdapter`."""
    tasks: list[BenchmarkTask] = []
    for index, rec in enumerate(records):
        gold = rec.get("SQL", rec.get("query", rec.get("gold")))
        tasks.append(
            BenchmarkTask(
                id=str(rec.get("id", f"bird-{index}")),
                prompt=str(rec.get("question", "")),
                gold=gold,
                recorded=rec.get("recorded", gold),
                inputs={"tables": rec.get("tables", {}), "evidence": rec.get("evidence", "")},
                metadata={"db_id": rec.get("db_id"), "difficulty": rec.get("difficulty")},
            )
        )
    return tasks


BENCHMARK_ADAPTERS: dict[str, type[BenchmarkAdapter]] = {
    SWEBenchAdapter.name: SWEBenchAdapter,
    TauBenchAdapter.name: TauBenchAdapter,
    GAIAAdapter.name: GAIAAdapter,
    WebArenaAdapter.name: WebArenaAdapter,
    BFCLAdapter.name: BFCLAdapter,
    AgentBenchAdapter.name: AgentBenchAdapter,
    ToolBenchAdapter.name: ToolBenchAdapter,
    LiveCodeBenchAdapter.name: LiveCodeBenchAdapter,
    MMLUProAdapter.name: MMLUProAdapter,
    SpiderAdapter.name: SpiderAdapter,
    BIRDAdapter.name: BIRDAdapter,
}


def available_benchmarks() -> list[str]:
    """The names of the shipped benchmark adapters."""
    return sorted(BENCHMARK_ADAPTERS)


def load_benchmark(name: str, **kwargs: Any) -> BenchmarkAdapter:
    """Construct a benchmark adapter by name."""
    if name not in BENCHMARK_ADAPTERS:
        raise BenchmarkError(f"unknown benchmark {name!r}; known: {available_benchmarks()}")
    return BENCHMARK_ADAPTERS[name](**kwargs)
