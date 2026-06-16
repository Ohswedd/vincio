"""Bounded concurrency primitives for the hot paths.

Every concurrent fan-out in Vincio (retrieval, embedding, tool execution,
eval cases) goes through these helpers so concurrency is bounded, errors are
deterministic, and cancellation propagates: cancelling the caller cancels
every in-flight subtask, and the first failure cancels the rest of the group.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar, cast

__all__ = ["gather_bounded", "map_bounded", "race_with_timeout", "DEFAULT_CONCURRENCY"]

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_CONCURRENCY = 8


async def gather_bounded(
    coros: Iterable[Awaitable[T]],
    *,
    limit: int = DEFAULT_CONCURRENCY,
    return_exceptions: bool = False,
) -> list[T]:
    """Run coroutines concurrently under a semaphore, preserving order.

    Unlike a bare ``asyncio.gather``, the fan-out is bounded to *limit*
    in-flight tasks. Unless ``return_exceptions`` is set, the first failure
    cancels every other task in the group before the error is re-raised, so
    no work leaks past an error or a cancellation.
    """
    coros = list(coros)
    if not coros:
        return []
    semaphore = asyncio.Semaphore(max(1, limit))

    async def bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    tasks = [asyncio.ensure_future(bounded(coro)) for coro in coros]
    try:
        # With return_exceptions the caller opts into receiving exceptions in
        # place of results; the declared list[T] is the success-path contract.
        return cast("list[T]", list(await asyncio.gather(*tasks, return_exceptions=return_exceptions)))
    except BaseException:
        for task in tasks:
            task.cancel()
        # Let cancellations settle so no "exception was never retrieved"
        # warnings or orphan tasks escape the group.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def map_bounded(
    fn: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    *,
    limit: int = DEFAULT_CONCURRENCY,
    return_exceptions: bool = False,
) -> list[R]:
    """Apply an async function to every item with bounded concurrency."""
    return await gather_bounded(
        (fn(item) for item in items), limit=limit, return_exceptions=return_exceptions
    )


async def race_with_timeout(coro: Awaitable[T], timeout_s: float | None) -> T:
    """Await *coro* under an optional deadline; raises ``asyncio.TimeoutError``."""
    if timeout_s is None or timeout_s <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_s)
