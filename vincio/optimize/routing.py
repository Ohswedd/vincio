"""Model routing optimization.

- :class:`RoutingPolicy` — deterministic difficulty/risk-based routing.
- :class:`RoutingOptimizer` — learns the difficulty threshold offline from
  per-tier eval reports.
- :class:`EpsilonGreedyBandit` and :class:`UCB1Bandit` — live routing
  bandits, to be used only behind offline eval gates.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from ..core.types import ModelEvent, ModelRequest, ModelResponse, TaskType, TokenUsage
from ..evals.reports import EvalReport
from .search import FitnessWeights

__all__ = [
    "RoutingPolicy",
    "estimate_difficulty",
    "RoutingOptimizer",
    "EpsilonGreedyBandit",
    "UCB1Bandit",
    "CascadeRung",
    "ModelCascade",
    "response_confidence",
    "RouteStrategy",
    "RoutingDecision",
    "Router",
]

_REASONING_RE = re.compile(
    r"(?i)\b(why|prove|derive|step[- ]by[- ]step|trade-?offs?|compare and|multi-?hop|implications?|root cause)\b"
)


def estimate_difficulty(text: str, *, task_type: TaskType = TaskType.GENERAL, evidence_count: int = 0) -> float:
    """Deterministic difficulty estimate in [0,1]."""
    score = 0.15
    words = len(text.split())
    score += min(0.3, words / 400)
    if _REASONING_RE.search(text):
        score += 0.25
    if task_type in (TaskType.AGENT_WORKFLOW, TaskType.PLANNING, TaskType.COMPLIANCE_REVIEW, TaskType.DOCUMENT_COMPARISON):
        score += 0.2
    elif task_type in (TaskType.CLASSIFICATION, TaskType.EXTRACTION):
        score -= 0.1
    score += min(0.2, evidence_count / 40)
    return max(0.0, min(1.0, score))


class RoutingPolicy(BaseModel):
    """Routing policy."""

    cheap_model: str
    default_model: str
    strong_model: str
    difficulty_threshold_low: float = 0.3
    difficulty_threshold_high: float = 0.65

    def route(
        self,
        *,
        difficulty: float,
        risk: str = "low",
        requires_reasoning: bool = False,
        validation_failed: bool = False,
    ) -> str:
        if validation_failed or requires_reasoning or risk == "high":
            return self.strong_model
        if difficulty < self.difficulty_threshold_low and risk == "low":
            return self.cheap_model
        if difficulty > self.difficulty_threshold_high:
            return self.strong_model
        return self.default_model


def response_confidence(response: ModelResponse, *, expects_schema: bool = False) -> float:
    """Default runtime confidence signal for a model response, in [0, 1].

    A clean stop is high confidence; a truncated or content-filtered answer, or
    a structured request that failed to parse, is low — exactly the cases worth
    escalating to a stronger model. Apps can supply a custom signal (e.g. a
    confidence metric) to :meth:`ContextApp.use_cascade`.
    """
    if response.finish_reason in ("length", "content_filter", "error"):
        return 0.0
    if expects_schema and response.structured is None:
        return 0.2
    if not (response.text or response.structured or response.tool_calls):
        return 0.0
    return 1.0


class CascadeRung(BaseModel):
    """One step of a runtime cascade: a model and the confidence below which a
    response is escalated to the next rung."""

    model: str
    provider: str | None = None
    min_confidence: float = 0.5


class ModelCascade(BaseModel):
    """An ordered cheap→strong model ladder for confidence-based escalation.

    At run time the cascade starts on the first (cheapest) rung and escalates to
    the next only when a response's confidence falls below the current rung's
    threshold — so most runs finish cheap and only the hard ones pay for the
    stronger model. The offline :class:`RoutingOptimizer` keeps tuning the
    thresholds; this is its runtime counterpart.
    """

    rungs: list[CascadeRung]
    max_escalations: int | None = None  # default: walk the whole ladder

    def model_post_init(self, _ctx: Any) -> None:
        if not self.rungs:
            raise ValueError("ModelCascade requires at least one rung")
        models = [rung.model for rung in self.rungs]
        if len(set(models)) != len(models):
            raise ValueError(f"ModelCascade rungs must have unique model names, got {models}")

    @classmethod
    def from_models(
        cls, models: list[str], *, min_confidence: float = 0.5, max_escalations: int | None = None
    ) -> ModelCascade:
        """Build a cascade from a cheap→strong list of model names."""
        return cls(
            rungs=[CascadeRung(model=m, min_confidence=min_confidence) for m in models],
            max_escalations=max_escalations,
        )

    @property
    def escalation_cap(self) -> int:
        ladder = len(self.rungs) - 1
        return ladder if self.max_escalations is None else min(self.max_escalations, ladder)

    def first(self) -> CascadeRung:
        return self.rungs[0]

    def _index(self, model: str) -> int:
        for i, rung in enumerate(self.rungs):
            if rung.model == model:
                return i
        return -1

    def next_rung(self, model: str, confidence: float) -> CascadeRung | None:
        """The next stronger rung when ``confidence`` is below ``model``'s
        threshold, else ``None`` (stay where we are)."""
        i = self._index(model)
        if i < 0 or i + 1 >= len(self.rungs):
            return None
        if confidence >= self.rungs[i].min_confidence:
            return None
        return self.rungs[i + 1]

    def next_rung_capable(
        self, model: str, confidence: float, is_capable: Any
    ) -> CascadeRung | None:
        """Like :meth:`next_rung` but skips rungs whose model cannot serve the
        request (capability guard, 1.8): once escalation is warranted, walk up
        past any incapable rung to the first capable stronger model."""
        nxt = self.next_rung(model, confidence)
        while nxt is not None and not is_capable(nxt.model):
            i = self._index(nxt.model)
            nxt = self.rungs[i + 1] if 0 <= i < len(self.rungs) - 1 else None
        return nxt

    def first_capable(self, is_capable: Any) -> CascadeRung:
        """The cheapest rung that can serve the request, or the first rung when
        none is known-capable (unknown models are never blocked)."""
        for rung in self.rungs:
            if is_capable(rung.model):
                return rung
        return self.rungs[0]


class RoutingOptimizer:
    """Learn the low/high thresholds from per-tier eval reports.

    Provide eval reports of the SAME dataset run with cheap/default/strong
    models, each case annotated with its difficulty in
    ``case.details['difficulty']`` (the app's eval target records it).
    """

    def __init__(self, weights: FitnessWeights | None = None) -> None:
        self.weights = weights or FitnessWeights()

    def optimize(
        self,
        policy: RoutingPolicy,
        reports: dict[str, EvalReport],  # tier name -> report ("cheap"/"default"/"strong")
        *,
        quality_metric: str = "lexical_overlap",
        min_quality_ratio: float = 0.97,
    ) -> RoutingPolicy:
        cheap = reports.get("cheap")
        default = reports.get("default")
        if cheap is None or default is None:
            return policy
        cheap_by_id = {c.case_id: c for c in cheap.cases}
        # Find the highest difficulty bucket where the cheap model keeps
        # >= min_quality_ratio of the default model's quality.
        buckets: dict[int, list[tuple[float, float]]] = {}
        for case in default.cases:
            cheap_case = cheap_by_id.get(case.case_id)
            if cheap_case is None:
                continue
            difficulty = float(
                case.details.get("difficulty")
                or cheap_case.details.get("difficulty")
                or 0.5
            )
            quality_default = case.metrics.get(quality_metric)
            quality_cheap = cheap_case.metrics.get(quality_metric)
            if quality_default is None or quality_cheap is None:
                continue
            buckets.setdefault(int(difficulty * 10), []).append((quality_cheap, quality_default))
        best_low = policy.difficulty_threshold_low
        for bucket in sorted(buckets):
            pairs = buckets[bucket]
            cheap_quality = sum(p[0] for p in pairs) / len(pairs)
            default_quality = sum(p[1] for p in pairs) / len(pairs)
            if default_quality <= 0:
                continue
            if cheap_quality / default_quality >= min_quality_ratio:
                best_low = max(best_low, (bucket + 1) / 10)
            else:
                break
        updated = policy.model_copy(update={"difficulty_threshold_low": min(best_low, policy.difficulty_threshold_high)})
        return updated


class EpsilonGreedyBandit:
    """Live routing bandit. Arms are model names; reward is the
    run-level fitness (caller computes it)."""

    def __init__(self, arms: list[str], *, epsilon: float = 0.1, seed: int | None = None) -> None:
        if not arms:
            raise ValueError("bandit requires at least one arm")
        self.arms = list(arms)
        self.epsilon = epsilon
        self.counts: dict[str, int] = {arm: 0 for arm in arms}
        self.values: dict[str, float] = {arm: 0.0 for arm in arms}
        self._rng = random.Random(seed)

    def select(self) -> str:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.arms)
        return max(self.arms, key=lambda arm: self.values[arm])

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.counts:
            raise ValueError(f"unknown arm {arm!r}")
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def snapshot(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "values": {k: round(v, 6) for k, v in self.values.items()},
            "total": sum(self.counts.values()),
        }

    def load(self, snapshot: dict[str, Any]) -> None:
        for arm, count in (snapshot.get("counts") or {}).items():
            if arm in self.counts:
                self.counts[arm] = int(count)
        for arm, value in (snapshot.get("values") or {}).items():
            if arm in self.values:
                self.values[arm] = float(value)


class UCB1Bandit:
    def __init__(self, arms: list[str]) -> None:
        if not arms:
            raise ValueError("bandit requires at least one arm")
        self.arms = list(arms)
        self.counts: dict[str, int] = {arm: 0 for arm in arms}
        self.values: dict[str, float] = {arm: 0.0 for arm in arms}
        self.total = 0

    def select(self) -> str:
        for arm in self.arms:  # play each arm once first
            if self.counts[arm] == 0:
                return arm
        return max(
            self.arms,
            key=lambda arm: self.values[arm]
            + math.sqrt(2 * math.log(self.total) / self.counts[arm]),
        )

    def update(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        self.total += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def snapshot(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "values": {k: round(v, 6) for k, v in self.values.items()},
            "total": self.total,
        }

    def load(self, snapshot: dict[str, Any]) -> None:
        for arm, count in (snapshot.get("counts") or {}).items():
            if arm in self.counts:
                self.counts[arm] = int(count)
        for arm, value in (snapshot.get("values") or {}).items():
            if arm in self.values:
                self.values[arm] = float(value)
        self.total = int(snapshot.get("total", sum(self.counts.values())))


class LinUCB:
    """Contextual bandit (disjoint LinUCB).

    Each arm keeps a ridge-regression model over a small context vector (e.g.
    ``[difficulty, log-length, evidence, risk]``). :meth:`select` picks the arm
    with the highest upper-confidence estimate ``θ·x + α·√(xᵀA⁻¹x)`` — it
    *explores* an arm whose payoff is uncertain in the current context and
    *exploits* one it is confident about, so routing adapts to the request, not
    just to a global average. Pure-Python ``dim×dim`` linear algebra keeps it
    dependency-free; ``dim`` is small by construction.
    """

    def __init__(self, arms: list[str], *, dim: int, alpha: float = 1.0) -> None:
        if not arms:
            raise ValueError("bandit requires at least one arm")
        if dim < 1:
            raise ValueError("LinUCB requires dim >= 1")
        self.arms = list(arms)
        self.dim = dim
        self.alpha = alpha
        self.A: dict[str, list[list[float]]] = {arm: _identity(dim) for arm in arms}
        self.b: dict[str, list[float]] = {arm: [0.0] * dim for arm in arms}
        self.counts: dict[str, int] = {arm: 0 for arm in arms}

    def _ucb(self, arm: str, x: list[float]) -> float:
        a_inv = _inverse(self.A[arm])
        theta = _matvec(a_inv, self.b[arm])
        mean = _dot(theta, x)
        var = _dot(x, _matvec(a_inv, x))
        return mean + self.alpha * math.sqrt(max(0.0, var))

    def select(self, context: list[float], *, explore: bool = True) -> str:
        x = self._fit(context)
        if not explore:
            # Exploit only: drop the exploration bonus (α=0).
            return max(self.arms, key=lambda arm: _dot(_matvec(_inverse(self.A[arm]), self.b[arm]), x))
        return max(self.arms, key=lambda arm: self._ucb(arm, x))

    def update(self, arm: str, context: list[float], reward: float) -> None:
        if arm not in self.A:
            raise ValueError(f"unknown arm {arm!r}")
        x = self._fit(context)
        for i in range(self.dim):
            for j in range(self.dim):
                self.A[arm][i][j] += x[i] * x[j]
            self.b[arm][i] += reward * x[i]
        self.counts[arm] += 1

    def _fit(self, context: list[float]) -> list[float]:
        x = list(context[: self.dim])
        if len(x) < self.dim:
            x += [0.0] * (self.dim - len(x))
        return x

    def snapshot(self) -> dict[str, Any]:
        return {"dim": self.dim, "alpha": self.alpha, "counts": dict(self.counts),
                "A": {a: m for a, m in self.A.items()}, "b": dict(self.b)}

    def load(self, snapshot: dict[str, Any]) -> None:
        for arm in self.arms:
            if arm in (snapshot.get("A") or {}):
                self.A[arm] = [list(row) for row in snapshot["A"][arm]]
            if arm in (snapshot.get("b") or {}):
                self.b[arm] = list(snapshot["b"][arm])
            if arm in (snapshot.get("counts") or {}):
                self.counts[arm] = int(snapshot["counts"][arm])


# -- tiny dependency-free linear algebra for LinUCB (small dim) --------------


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in m]


def _inverse(m: list[list[float]]) -> list[list[float]]:
    """Gauss–Jordan inverse of a small square matrix (with a ridge fallback)."""
    n = len(m)
    aug = [list(m[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[col][col] += 1e-6  # ridge: keep it invertible
            pivot = col
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col], strict=False)]
    return [row[n:] for row in aug]


# ---------------------------------------------------------------------------
# Registry-backed router provider (1.8)
# ---------------------------------------------------------------------------

from typing import Literal  # noqa: E402 - kept beside the router it annotates

from ..providers.base import ModelProvider  # noqa: E402 - avoids a load-order cycle
from ..providers.capabilities import capability_check, requirements_for  # noqa: E402

RouteStrategy = Literal["cheapest", "fastest", "least_busy"]

_TIER_SPEED = {"fast": 0, "default": 1, "strong": 2}


class RoutingDecision(BaseModel):
    """The record of one router pick — stamped on the trace as a routing decision."""

    model: str
    provider: str = ""
    strategy: str = "cheapest"
    reason: str = ""
    est_cost_usd: float = 0.0
    candidates: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)  # model -> why (incapable / over_budget)
    budget_usd: float | None = None
    downgraded: bool = False
    entry_index: int = -1  # the chosen entry's index (set by Router.pick)


class Router(ModelProvider):
    """A registry-backed router: pick the cheapest / fastest / least-busy *capable*
    model per request, inside your own process and audit boundary.

    Entries are ``(provider, model)`` pairs exactly like
    :class:`~vincio.providers.base.FailoverChain`, so a router nests cleanly
    inside ``CircuitBreaker`` / ``KeyPool`` / ``FailoverChain``. Before a pick,
    every candidate is run through the capability guard against the
    :class:`~vincio.providers.registry.ModelRegistry`: a model that cannot serve
    the request (missing vision, tools, structured output, reasoning, or a wide
    enough context) is skipped, not silently chosen. With ``budget_usd`` set the
    router **downgrades** to the cheapest capable model that fits the per-request
    cap. Each pick is returned as a :class:`RoutingDecision` and emitted as a
    ``model.routed`` event when an event bus is supplied.
    """

    name = "router"

    def __init__(
        self,
        entries: list[tuple[ModelProvider, str]],
        *,
        strategy: RouteStrategy = "cheapest",
        registry: Any | None = None,
        price_table: Any | None = None,
        budget_usd: float | None = None,
        guard_capabilities: bool = True,
        events: Any | None = None,
    ) -> None:
        if not entries:
            raise ValueError("Router requires at least one (provider, model) entry")
        self.entries = entries
        self.strategy = strategy
        self._registry = registry
        self._price_table = price_table
        self.budget_usd = budget_usd
        self.guard_capabilities = guard_capabilities
        self._events = events
        self.last_decision: RoutingDecision | None = None
        self._inflight = [0 for _ in entries]
        self._latency_ms = [0.0 for _ in entries]  # EWMA of observed latency

    @classmethod
    def from_models(
        cls, provider: ModelProvider, models: list[str], **kwargs: Any
    ) -> Router:
        """Build a router over one provider that serves several models."""
        if not models:
            raise ValueError("Router.from_models requires at least one model")
        return cls([(provider, m) for m in models], **kwargs)

    def _reg(self) -> Any:
        if self._registry is None:
            from ..providers.registry import default_model_registry

            self._registry = default_model_registry()
        return self._registry

    def _prices(self) -> Any:
        if self._price_table is None:
            from ..observability.costs import default_price_table

            self._price_table = default_price_table()
        return self._price_table

    def _estimate_cost(self, model: str, request: ModelRequest, input_tokens: int) -> float:
        out = request.max_output_tokens or 512
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=out)
        return self._prices().cost(model, usage)

    @staticmethod
    def _input_tokens(request: ModelRequest) -> int:
        from ..core.tokens import count_tokens

        text = "\n".join(m.text for m in request.messages)
        return count_tokens(text)

    def _rank_key(self, index: int, model: str, est_cost: float) -> tuple[float, ...]:
        if self.strategy == "fastest":
            profile = self._reg().resolve(model)
            tier = _TIER_SPEED.get(profile.tier, 1) if profile is not None else 1
            return (self._latency_ms[index] or float(tier), est_cost, index)
        if self.strategy == "least_busy":
            return (self._inflight[index], est_cost, index)
        return (est_cost, index)  # cheapest

    def pick(self, request: ModelRequest, *, budget_usd: float | None = None) -> RoutingDecision:
        """Choose a capable model for *request* without dispatching it."""
        budget = budget_usd if budget_usd is not None else self.budget_usd
        if budget is None:
            meta_budget = request.metadata.get("max_cost_usd")
            budget = float(meta_budget) if meta_budget is not None else None
        input_tokens = self._input_tokens(request)
        needs = requirements_for(request, input_tokens=input_tokens)
        registry = self._reg()

        skipped: dict[str, str] = {}
        capable: list[tuple[int, str, float]] = []  # (index, model, est_cost)
        for index, (_, model) in enumerate(self.entries):
            if self.guard_capabilities:
                verdict = capability_check(needs, registry.guard_capabilities(model), model=model)
                if not verdict.ok:
                    skipped[model] = verdict.reason
                    continue
            capable.append((index, model, self._estimate_cost(model, request, input_tokens)))

        if not capable:
            from ..core.errors import CapabilityMismatchError

            raise CapabilityMismatchError(
                f"no capable model for request needs {needs.summary()}; "
                f"skipped {skipped}",
                missing=needs.summary(),
                provider=self.name,
            )

        ranked = sorted(capable, key=lambda c: self._rank_key(c[0], c[1], c[2]))
        downgraded = False
        chosen = ranked[0]
        if budget is not None:
            within = [c for c in ranked if c[2] <= budget]
            if within:
                if within[0] != chosen:
                    downgraded = True
                chosen = within[0]
            else:  # nothing fits — fall back to the cheapest capable, flagged
                cheapest = min(capable, key=lambda c: c[2])
                downgraded = True
                chosen = cheapest
                for _idx, model, cost in capable:
                    if cost > budget and model not in skipped:
                        skipped.setdefault(model, f"over_budget (${cost:.6f} > ${budget:.6f})")

        index, model, est_cost = chosen
        decision = RoutingDecision(
            model=model,
            provider=self.entries[index][0].name,
            strategy=self.strategy,
            reason=(
                f"{self.strategy} of {len(capable)} capable model(s)"
                + (" (budget downgrade)" if downgraded else "")
            ),
            est_cost_usd=round(est_cost, 8),
            candidates=[m for _, m in self.entries],
            skipped=skipped,
            budget_usd=budget,
            downgraded=downgraded,
            entry_index=index,
        )
        return decision

    def _dispatch(self, decision: RoutingDecision) -> tuple[int, ModelProvider]:
        # Fast path: pick() recorded the chosen entry index on the decision.
        if 0 <= decision.entry_index < len(self.entries):
            return decision.entry_index, self.entries[decision.entry_index][0]
        for index, (provider, model) in enumerate(self.entries):
            if model == decision.model and provider.name == decision.provider:
                return index, provider
        # Fall back to first matching model id (provider name unchanged is rare).
        for index, (provider, model) in enumerate(self.entries):
            if model == decision.model:
                return index, provider
        return 0, self.entries[0][0]

    def _emit(self, decision: RoutingDecision) -> None:
        self.last_decision = decision
        if self._events is not None:
            self._events.emit("model.routed", decision.model_dump())

    async def generate(self, request: ModelRequest) -> ModelResponse:
        decision = self.pick(request)
        self._emit(decision)
        index, provider = self._dispatch(decision)
        attempt = request.model_copy(update={"model": decision.model})
        self._inflight[index] += 1
        started = time.monotonic()
        try:
            response = await provider.generate(attempt)
        finally:
            self._inflight[index] -= 1
            self._observe_latency(index, started)
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        decision = self.pick(request)
        self._emit(decision)
        index, provider = self._dispatch(decision)
        attempt = request.model_copy(update={"model": decision.model})
        self._inflight[index] += 1
        started = time.monotonic()
        try:
            async for event in provider.stream(attempt):
                yield event
        finally:
            self._inflight[index] -= 1
            self._observe_latency(index, started)

    def _observe_latency(self, index: int, started: float) -> None:
        elapsed = (time.monotonic() - started) * 1000
        prior = self._latency_ms[index]
        self._latency_ms[index] = elapsed if prior == 0.0 else 0.7 * prior + 0.3 * elapsed

    def capabilities(self, model: str) -> Any:
        return self.entries[0][0].capabilities(model)

    async def list_models(self) -> Any:
        from ..providers.base import _merge_model_lists

        seen: set[int] = set()
        lists = []
        for provider, _ in self.entries:
            if id(provider) not in seen:
                seen.add(id(provider))
                lists.append(await provider.list_models())
        return _merge_model_lists(lists)

    async def aclose(self) -> None:
        seen: set[int] = set()
        for provider, _ in self.entries:
            if id(provider) not in seen:
                seen.add(id(provider))
                await provider.aclose()


class BanditDecision(BaseModel):
    """The record of one guarded-bandit route — stamped on the trace."""

    model: str
    provider: str = ""
    arm: str = ""
    explored: bool = False
    risk: str = "low"
    reward: float | None = None
    cumulative_regret: float = 0.0
    frozen: bool = False
    rolled_back: bool = False
    reason: str = ""


class GuardedBanditRouter(ModelProvider):
    """A live routing bandit with a safety floor, regret tracking, and auto-rollback.

    Arms are ``(provider, model)`` entries; the configured bandit
    (``epsilon_greedy`` / ``ucb1`` / contextual ``linucb``) learns which model
    pays off and routes live traffic to it — but never blindly:

    - **safety floor** — on safety- or high-risk-tagged traffic the router
      *exploits* the best-known arm and never explores, so an experiment can
      never run on a request that can't afford it;
    - **persisted state** — arm statistics persist to the shared store, so the
      learner is restart-safe and aggregatable across workers;
    - **regret + auto-freeze/rollback** — per-pull regret accumulates; once it
      exceeds ``regret_budget`` the router *freezes* (exploit only), and if the
      learned best arm falls below the safe arm by a margin it *rolls back* and
      pins the safe model — turning the dead primitives into a real, safe online
      learner.

    Implements :class:`~vincio.providers.base.ModelProvider`, so it nests cleanly
    inside ``CircuitBreaker`` / ``KeyPool`` / ``FailoverChain``.
    """

    name = "guarded_bandit"

    def __init__(
        self,
        entries: list[tuple[ModelProvider, str]],
        *,
        bandit: str = "epsilon_greedy",
        safe_model: str | None = None,
        reward_fn: Any | None = None,
        context_fn: Any | None = None,
        epsilon: float = 0.1,
        alpha: float = 1.0,
        context_dim: int = 4,
        seed: int | None = None,
        regret_budget: float = 2.0,
        rollback_margin: float = 0.2,
        store: Any | None = None,
        app_name: str = "",
        events: Any | None = None,
    ) -> None:
        if not entries:
            raise ValueError("GuardedBanditRouter requires at least one (provider, model) entry")
        self.entries = entries
        self.arms = [m for _, m in entries]
        self.safe_model = safe_model or self.arms[0]
        self.reward_fn = reward_fn or (lambda resp: response_confidence(resp))
        self.context_fn = context_fn
        self.context_dim = context_dim
        self.regret_budget = regret_budget
        self.rollback_margin = rollback_margin
        self.kind = bandit
        if bandit == "ucb1":
            self.bandit: Any = UCB1Bandit(self.arms)
        elif bandit == "linucb":
            self.bandit = LinUCB(self.arms, dim=context_dim, alpha=alpha)
        else:
            self.bandit = EpsilonGreedyBandit(self.arms, epsilon=epsilon, seed=seed)
        self.cumulative_regret = 0.0
        self.frozen = False
        self.rolled_back = False
        self._store = store
        self.app_name = app_name
        self._events = events
        self._state_id = f"{app_name}:guarded_bandit"
        self.last_decision: BanditDecision | None = None
        self.load_state()

    # -- selection -----------------------------------------------------------

    @staticmethod
    def _risk(request: ModelRequest) -> str:
        meta = request.metadata or {}
        if meta.get("safety_critical") or meta.get("risk") == "high":
            return "high"
        tags = meta.get("tags") or []
        if any(t in ("safety", "high_risk") for t in tags):
            return "high"
        return str(meta.get("risk", "low"))

    def _context(self, request: ModelRequest) -> list[float]:
        if self.context_fn is not None:
            return list(self.context_fn(request))
        text = "\n".join(m.text for m in request.messages)
        evidence = int((request.metadata or {}).get("evidence_count", 0))
        difficulty = estimate_difficulty(text, evidence_count=evidence)
        length = min(1.0, len(text) / 4000.0)
        risk = 1.0 if self._risk(request) == "high" else 0.0
        return [difficulty, length, min(1.0, evidence / 40.0), risk]

    def select(self, request: ModelRequest) -> tuple[str, bool]:
        """Choose an arm; returns ``(model, explored)``. Honours the safety floor."""
        risk = self._risk(request)
        # Safety floor / frozen / rolled-back: exploit only, never explore.
        if self.rolled_back:
            return self.safe_model, False
        exploit_only = risk == "high" or self.frozen
        if isinstance(self.bandit, LinUCB):
            arm = self.bandit.select(self._context(request), explore=not exploit_only)
            return arm, not exploit_only
        if exploit_only:
            return max(self.arms, key=lambda a: self.bandit.values[a]), False
        before = dict(self.bandit.counts)
        arm = self.bandit.select()
        # An untried arm or a different-from-greedy pick counts as exploration.
        greedy = max(self.arms, key=lambda a: self.bandit.values[a])
        explored = arm != greedy or before.get(arm, 0) == 0
        return arm, explored

    def _value_estimates(self, request: ModelRequest) -> dict[str, float]:
        if isinstance(self.bandit, LinUCB):
            x = self.bandit._fit(self._context(request))
            return {
                a: _dot(_matvec(_inverse(self.bandit.A[a]), self.bandit.b[a]), x) for a in self.arms
            }
        return dict(self.bandit.values)

    def record(self, arm: str, reward: float, *, request: ModelRequest) -> BanditDecision:
        """Update the bandit with an observed reward; track regret + guards."""
        values_before = self._value_estimates(request)
        best_before = max(values_before.values()) if values_before else 0.0
        if isinstance(self.bandit, LinUCB):
            self.bandit.update(arm, self._context(request), reward)
        else:
            self.bandit.update(arm, reward)
        # Instantaneous regret vs the best arm's pre-update estimate.
        regret = max(0.0, best_before - reward)
        self.cumulative_regret += regret
        if self.cumulative_regret > self.regret_budget and not self.frozen:
            self.frozen = True
        values_after = self._value_estimates(request)
        best_arm = max(values_after, key=values_after.__getitem__) if values_after else self.safe_model
        safe_value = values_after.get(self.safe_model, 0.0)
        if best_arm != self.safe_model and values_after.get(best_arm, 0.0) + self.rollback_margin < safe_value:
            self.rolled_back = True
        decision = BanditDecision(
            model=arm, arm=arm, reward=round(reward, 6),
            cumulative_regret=round(self.cumulative_regret, 6),
            frozen=self.frozen, rolled_back=self.rolled_back,
            risk=self._risk(request),
        )
        self.last_decision = decision
        self.save_state()
        if self._events is not None:
            self._events.emit("bandit.updated", decision.model_dump())
        return decision

    # -- ModelProvider ------------------------------------------------------

    def _provider_for(self, model: str) -> tuple[int, ModelProvider]:
        for i, (provider, m) in enumerate(self.entries):
            if m == model:
                return i, provider
        return 0, self.entries[0][0]

    async def generate(self, request: ModelRequest) -> ModelResponse:
        arm, explored = self.select(request)
        _, provider = self._provider_for(arm)
        attempt = request.model_copy(update={"model": arm})
        response = await provider.generate(attempt)
        decision = self.record(arm, float(self.reward_fn(response)), request=request)
        decision.explored = explored
        decision.provider = provider.name
        decision.reason = (
            f"{self.kind} {'explored' if explored else 'exploited'} {arm}"
            + (" [safety-floor]" if self._risk(request) == "high" else "")
        )
        self.last_decision = decision
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        arm, explored = self.select(request)
        _, provider = self._provider_for(arm)
        attempt = request.model_copy(update={"model": arm})
        async for event in provider.stream(attempt):
            yield event
        # Streaming reward is not observable mid-flight; credit a clean finish.
        self.record(arm, 1.0, request=request)

    def capabilities(self, model: str) -> Any:
        return self.entries[0][0].capabilities(model)

    async def list_models(self) -> Any:
        from ..providers.base import _merge_model_lists

        seen: set[int] = set()
        lists = []
        for provider, _ in self.entries:
            if id(provider) not in seen:
                seen.add(id(provider))
                lists.append(await provider.list_models())
        return _merge_model_lists(lists)

    async def aclose(self) -> None:
        seen: set[int] = set()
        for provider, _ in self.entries:
            if id(provider) not in seen:
                seen.add(id(provider))
                await provider.aclose()

    # -- persisted state -----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self._state_id,
            "app_id": self.app_name,
            "kind": self.kind,
            "bandit": self.bandit.snapshot(),
            "cumulative_regret": self.cumulative_regret,
            "frozen": self.frozen,
            "rolled_back": self.rolled_back,
        }

    def save_state(self) -> None:
        if self._store is None:
            return
        from ..core.utils import utcnow

        self._store.save("bandit_state", {**self.snapshot(), "updated_at": utcnow().isoformat()})

    def load_state(self) -> None:
        if self._store is None:
            return
        row = self._store.get("bandit_state", self._state_id)
        if row is None:
            return
        if row.get("bandit"):
            self.bandit.load(row["bandit"])
        self.cumulative_regret = float(row.get("cumulative_regret", 0.0))
        self.frozen = bool(row.get("frozen", False))
        self.rolled_back = bool(row.get("rolled_back", False))
