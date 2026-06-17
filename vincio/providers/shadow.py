"""Shadow & canary providers for live model qualification (1.8).

Two provider-layer rollout primitives that qualify a candidate model on real
traffic without a hosted control plane:

* :class:`ShadowProvider` wraps ``(primary, candidate)``, returns the **primary**
  response to the user, and asynchronously **dual-dispatches** the same request
  to the candidate, recording both for an offline diff. The user never waits on,
  or is affected by, the candidate.
* :class:`CanaryRouter` ramps a configurable **percentage** of live traffic onto
  a candidate, scores both arms online, and **auto-rolls-back** to the last
  known-good model (and, optionally, the last known-good prompt-registry head)
  the moment the candidate's quality regresses past a threshold.

Both implement :class:`~vincio.providers.base.ModelProvider`, so they nest
cleanly inside ``CircuitBreaker`` / ``KeyPool`` / ``FailoverChain``. The
canary-driven prompt/policy *promotion* that needs a new serving surface is
reserved for 2.0; this is the observe-and-revert provider-layer form.
``@experimental`` on the frozen 1.0 API.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel

from ..core.types import ModelEvent, ModelRequest, ModelResponse
from ..stability import experimental
from .base import ModelProvider

__all__ = [
    "ShadowObservation",
    "ShadowProvider",
    "CanaryState",
    "CanaryRouter",
]


def _last_user_text(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if getattr(message.role, "value", message.role) in ("user", "developer"):
            return message.text
    return request.messages[-1].text if request.messages else ""


class ShadowObservation(BaseModel):
    """One primary/candidate pair captured by a :class:`ShadowProvider`."""

    input: str = ""
    primary_model: str = ""
    candidate_model: str = ""
    primary_text: str = ""
    candidate_text: str | None = None
    primary_cost_usd: float = 0.0
    candidate_cost_usd: float | None = None
    primary_latency_ms: int = 0
    candidate_latency_ms: int | None = None
    output_similarity: float | None = None
    candidate_error: str | None = None


@experimental(since="1.8")
class ShadowProvider(ModelProvider):
    """Return the primary's answer; dual-dispatch the candidate for offline diff.

    ``block=False`` (default) fires the candidate as a background task so it never
    adds latency to the user's request; call :meth:`drain` (or :meth:`aclose`) to
    await any in-flight shadow calls before reading :attr:`observations`. Set
    ``block=True`` for a deterministic, fully-awaited shadow (tests, batch diff).
    A candidate failure is recorded, never raised — the user is never affected.
    """

    name = "shadow"

    def __init__(
        self,
        primary: ModelProvider,
        candidate: ModelProvider,
        *,
        candidate_model: str | None = None,
        block: bool = False,
        price_table: Any | None = None,
        recorder: Callable[[ShadowObservation], None] | None = None,
        events: Any | None = None,
        max_observations: int = 1000,
    ) -> None:
        self.primary = primary
        self.candidate = candidate
        self.candidate_model = candidate_model
        self.block = block
        self._price_table = price_table
        self._recorder = recorder
        self._events = events
        self.observations: deque[ShadowObservation] = deque(maxlen=max(1, max_observations))
        self._pending: set[asyncio.Task[Any]] = set()

    def _prices(self) -> Any:
        if self._price_table is None:
            from ..observability.costs import default_price_table

            self._price_table = default_price_table()
        return self._price_table

    def _cost(self, response: ModelResponse) -> float:
        if response.cost_usd:
            return response.cost_usd
        try:
            return self._prices().cost(response.model, response.usage)
        except Exception:  # noqa: BLE001 - cost is best-effort for a shadow diff
            return 0.0

    async def _run_candidate(
        self, obs: ShadowObservation, request: ModelRequest, primary_text: str
    ) -> None:
        started = time.monotonic()
        try:
            response = await self.candidate.generate(request)
            obs.candidate_text = response.text
            obs.candidate_cost_usd = round(self._cost(response), 8)
            obs.candidate_latency_ms = int((time.monotonic() - started) * 1000)
            obs.output_similarity = (
                1.0 if primary_text == response.text
                else round(SequenceMatcher(None, primary_text, response.text).ratio(), 4)
            )
        except Exception as exc:  # noqa: BLE001 - candidate failure must not surface
            obs.candidate_error = f"{type(exc).__name__}: {exc}"
        if self._recorder is not None:
            self._recorder(obs)
        if self._events is not None:
            self._events.emit("model.shadow", obs.model_dump())

    def _candidate_request(self, request: ModelRequest) -> ModelRequest:
        if self.candidate_model:
            return request.model_copy(update={"model": self.candidate_model})
        return request

    def _record_primary(self, request: ModelRequest, response: ModelResponse, latency_ms: int) -> ShadowObservation:
        obs = ShadowObservation(
            input=_last_user_text(request),
            primary_model=request.model,
            candidate_model=self.candidate_model or request.model,
            primary_text=response.text,
            primary_cost_usd=round(self._cost(response), 8),
            primary_latency_ms=latency_ms,
        )
        self.observations.append(obs)
        return obs

    def _dispatch_candidate(self, obs: ShadowObservation, request: ModelRequest, primary_text: str) -> None:
        coro = self._run_candidate(obs, self._candidate_request(request), primary_text)
        if self.block:
            self._pending.add(asyncio.ensure_future(coro))
        else:
            task = asyncio.create_task(coro)
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        started = time.monotonic()
        response = await self.primary.generate(request)
        latency_ms = int((time.monotonic() - started) * 1000)
        obs = self._record_primary(request, response, latency_ms)
        if self.block:
            await self._run_candidate(obs, self._candidate_request(request), response.text)
        else:
            self._dispatch_candidate(obs, request, response.text)
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        started = time.monotonic()
        final: ModelResponse | None = None
        async for event in self.primary.stream(request):
            if event.type == "done" and event.response is not None:
                final = event.response
            yield event
        if final is not None:
            latency_ms = int((time.monotonic() - started) * 1000)
            obs = self._record_primary(request, final, latency_ms)
            if self.block:
                await self._run_candidate(obs, self._candidate_request(request), final.text)
            else:
                self._dispatch_candidate(obs, request, final.text)

    async def drain(self) -> None:
        """Await every in-flight shadow dispatch (so observations are complete)."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def diff(self) -> dict[str, Any]:
        """Aggregate the captured observations into a primary-vs-candidate diff."""
        obs = list(self.observations)
        paired = [o for o in obs if o.candidate_text is not None]
        errors = [o for o in obs if o.candidate_error is not None]
        sims = [o.output_similarity for o in paired if o.output_similarity is not None]
        primary_cost = sum(o.primary_cost_usd for o in obs)
        candidate_cost = sum(o.candidate_cost_usd or 0.0 for o in paired)
        return {
            "observations": len(obs),
            "paired": len(paired),
            "candidate_errors": len(errors),
            "candidate_error_rate": round(len(errors) / len(obs), 4) if obs else 0.0,
            "mean_output_similarity": round(sum(sims) / len(sims), 4) if sims else None,
            "primary_cost_usd": round(primary_cost, 8),
            "candidate_cost_usd": round(candidate_cost, 8),
            "cost_ratio": round(candidate_cost / primary_cost, 4) if primary_cost > 0 else None,
        }

    def capabilities(self, model: str) -> Any:
        return self.primary.capabilities(model)

    async def aclose(self) -> None:
        await self.drain()
        await self.primary.aclose()
        await self.candidate.aclose()


class CanaryState(BaseModel):
    """A canary's live state — what it has routed and what it has measured."""

    percent: float = 0.0
    rolled_back: bool = False
    calls: int = 0
    primary_n: int = 0
    candidate_n: int = 0
    primary_mean: float = 0.0
    candidate_mean: float = 0.0
    rollback_reason: str = ""


@experimental(since="1.8")
class CanaryRouter(ModelProvider):
    """Ramp a percentage of live traffic onto a candidate, with auto-rollback.

    A deterministic accumulator routes ~``percent``% of calls to the candidate
    and the rest to the primary, scoring both arms online (``score_fn`` defaults
    to the cascade confidence signal). Once both arms have at least
    ``min_samples`` observations, a candidate mean below the primary mean by more
    than ``regression_threshold`` triggers an **auto-rollback**: all traffic
    reverts to the primary, ``on_rollback`` fires, and — when a
    :class:`~vincio.prompts.registry.PromptRegistry` and ``prompt_name`` are
    supplied — the prompt is rolled back to its prior head (rollback-as-new-head).
    Implements :class:`ModelProvider`, so it nests inside the reliability stack.
    """

    name = "canary"

    def __init__(
        self,
        primary: ModelProvider,
        candidate: ModelProvider,
        *,
        percent: float = 5.0,
        candidate_model: str | None = None,
        score_fn: Callable[[ModelResponse], float] | None = None,
        min_samples: int = 20,
        window: int = 200,
        regression_threshold: float = 0.05,
        on_rollback: Callable[[CanaryState], None] | None = None,
        prompt_registry: Any | None = None,
        prompt_name: str | None = None,
        events: Any | None = None,
    ) -> None:
        self.primary = primary
        self.candidate = candidate
        self.candidate_model = candidate_model
        self.percent = max(0.0, min(100.0, percent))
        self._score_fn = score_fn
        self.min_samples = max(1, min_samples)
        self.regression_threshold = regression_threshold
        self._on_rollback = on_rollback
        self._prompt_registry = prompt_registry
        self._prompt_name = prompt_name
        self._events = events
        self.rolled_back = False
        self.rollback_reason = ""
        self.calls = 0
        self._accumulator = 0.0
        self._primary_scores: deque[float] = deque(maxlen=max(self.min_samples, window))
        self._candidate_scores: deque[float] = deque(maxlen=max(self.min_samples, window))

    def set_percent(self, percent: float) -> None:
        """Ramp the canary share (ignored once rolled back)."""
        if not self.rolled_back:
            self.percent = max(0.0, min(100.0, percent))

    def _score(self, response: ModelResponse) -> float:
        if self._score_fn is not None:
            try:
                return float(self._score_fn(response))
            except Exception:  # noqa: BLE001 - a bad signal must not break the route
                return 0.0
        from ..optimize.routing import response_confidence

        return response_confidence(response)

    def _route_to_candidate(self) -> bool:
        if self.rolled_back or self.percent <= 0.0:
            return False
        self._accumulator += self.percent / 100.0
        if self._accumulator >= 1.0:
            self._accumulator -= 1.0
            return True
        return False

    @staticmethod
    def _mean(scores: deque[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    def _maybe_rollback(self) -> None:
        if self.rolled_back:
            return
        if len(self._primary_scores) < self.min_samples or len(self._candidate_scores) < self.min_samples:
            return
        primary_mean = self._mean(self._primary_scores)
        candidate_mean = self._mean(self._candidate_scores)
        if candidate_mean < primary_mean - self.regression_threshold:
            self.rolled_back = True
            self.percent = 0.0
            self.rollback_reason = (
                f"candidate mean {candidate_mean:.4f} < primary {primary_mean:.4f} "
                f"- {self.regression_threshold} over {len(self._candidate_scores)} samples"
            )
            if self._prompt_registry is not None and self._prompt_name is not None:
                try:
                    self._prompt_registry.rollback(self._prompt_name)
                except Exception:  # noqa: BLE001 - rollback is best-effort
                    pass
            state = self.state()
            if self._events is not None:
                self._events.emit("canary.rollback", state.model_dump())
            if self._on_rollback is not None:
                self._on_rollback(state)

    def _observe(self, response: ModelResponse, *, candidate: bool) -> None:
        score = self._score(response)
        (self._candidate_scores if candidate else self._primary_scores).append(score)
        self._maybe_rollback()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        to_candidate = self._route_to_candidate()
        if to_candidate:
            attempt = (
                request.model_copy(update={"model": self.candidate_model})
                if self.candidate_model else request
            )
            response = await self.candidate.generate(attempt)
            self._observe(response, candidate=True)
        else:
            response = await self.primary.generate(request)
            self._observe(response, candidate=False)
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.calls += 1
        to_candidate = self._route_to_candidate()
        provider = self.candidate if to_candidate else self.primary
        attempt = (
            request.model_copy(update={"model": self.candidate_model})
            if to_candidate and self.candidate_model else request
        )
        final: ModelResponse | None = None
        async for event in provider.stream(attempt):
            if event.type == "done" and event.response is not None:
                final = event.response
            yield event
        if final is not None:
            self._observe(final, candidate=to_candidate)

    def state(self) -> CanaryState:
        return CanaryState(
            percent=self.percent,
            rolled_back=self.rolled_back,
            calls=self.calls,
            primary_n=len(self._primary_scores),
            candidate_n=len(self._candidate_scores),
            primary_mean=round(self._mean(self._primary_scores), 4),
            candidate_mean=round(self._mean(self._candidate_scores), 4),
            rollback_reason=self.rollback_reason,
        )

    def capabilities(self, model: str) -> Any:
        return self.primary.capabilities(model)

    async def aclose(self) -> None:
        await self.primary.aclose()
        await self.candidate.aclose()
