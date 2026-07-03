"""Support classes for :class:`~vincio.core.app.ContextApp`.

Extracted verbatim from ``vincio/core/app.py`` (v7.5 structure line) so the
``vincio/core/_app_*.py`` verb mixins can construct :class:`RunHandle`,
``_SourceConfig``, and ``_AgentHandle`` without importing
:mod:`vincio.core.app` (which imports the mixins). ``vincio.core.app``
re-imports all three, so ``from vincio.core.app import RunHandle`` (and
``vincio.__init__``'s re-export) keep working unchanged.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..providers.base import run_sync
from .types import Budget, RunResult

if TYPE_CHECKING:
    from ..agents.executor import AgentExecutor
    from .app import ContextApp


class RunHandle:
    """Handle to an in-flight run started by :meth:`ContextApp.submit`.

    Wraps the run's task and exposes cooperative cancellation that is identical
    across the streaming and non-streaming paths: :meth:`cancel` propagates a
    ``CancelledError`` into the run's bounded-concurrency groups, and the
    cancelled run is still fully recorded on its trace and audit chain. Await the
    handle (or :meth:`result`) for the :class:`RunResult`.
    """

    def __init__(self, task: asyncio.Future[RunResult]) -> None:
        self._task = task

    def cancel(self) -> bool:
        """Request cooperative cancellation; returns False if already done."""
        return self._task.cancel()

    def cancelled(self) -> bool:
        return self._task.cancelled()

    def done(self) -> bool:
        return self._task.done()

    async def result(self) -> RunResult:
        return await self._task

    def __await__(self):  # type: ignore[no-untyped-def]
        return self._task.__await__()


class _SourceConfig(BaseModel):
    name: str
    path: str | None = None
    loader: str | None = None
    chunking: str = "adaptive"
    retrieval: str = "hybrid"
    document_count: int = 0
    chunk_count: int = 0
    anchor: bool = False  # a task-frame source (always-on compact brief + on-demand detail)
    brief_tokens: int = 400


class _AgentHandle:
    """Returned by app.agent(): sync/async runner over an AgentExecutor."""

    def __init__(self, app: ContextApp, executor: AgentExecutor, max_steps: int) -> None:
        self._app = app
        self._executor = executor
        self._max_steps = max_steps

    async def arun(
        self,
        objective: str,
        *,
        budget: Budget | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
    ):
        budget = budget or self._app.budget.model_copy(update={"max_steps": self._max_steps})
        attribution = {
            k: v
            for k, v in {"tenant_id": tenant_id, "user_id": user_id, "feature": feature}.items()
            if v is not None
        }
        return await self._executor.run(objective, budget=budget, attribution=attribution or None)

    def run(
        self,
        objective: str,
        *,
        budget: Budget | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        feature: str | None = None,
    ):
        return run_sync(
            self.arun(
                objective, budget=budget, tenant_id=tenant_id, user_id=user_id, feature=feature
            )
        )
