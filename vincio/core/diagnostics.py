"""Observable best-effort failure: log it and count it, never swallow it silently.

A best-effort fallback that catches a broad ``Exception`` and continues is sound
policy — a broken embedder, a rejected memory write, or a store that does not
support a delete must never break a run. But a fallback that swallows the
exception *silently* (a bare ``pass`` behind a ``# pragma: no cover``, with no log
and no metric) hides a real bug inside itself: the symptom is a quietly degraded
result with nothing to trace it to.

This module makes such a fallback **observable** in one call. :func:`note_suppressed`
records the suppressed failure on a dedicated diagnostics log channel
(:data:`SUPPRESSED_LOGGER_NAME`) — capturing the active exception's traceback — and
increments a process-wide counter keyed by a stable label, so an operator can both
*watch* the failures (enable the channel at ``DEBUG``) and *scrape their rate*
(:func:`suppressed_failure_counts`) without the fallback ever breaking the run it
guards. It is the runtime half of the hardening line's "observable failure"
contract; the static half is the lint :mod:`vincio._observable_failure`, which
forbids a broad ``except`` that neither re-raises nor records its failure (here or
through a logger) unless it carries a justifying ``# noqa: BLE001``.

Use it inside a broad ``except`` whose body must continue rather than re-raise::

    try:
        manifest = mark_synthetic_content(result.raw_text, ...)
    except Exception:
        note_suppressed("runtime.content_marking")
        # ... fall back: the run continues without the manifest ...

The counter and the channel are an observability surface, not a stable API
contract: the *fact* that a suppressed failure is observable is guaranteed, but the
exact log wording and label spellings may change between releases.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter

__all__ = [
    "SUPPRESSED_LOGGER_NAME",
    "note_suppressed",
    "suppressed_failure_counts",
    "reset_suppressed_failures",
]

# The dedicated diagnostics channel every suppressed best-effort failure logs to.
# An operator turns on `logging.getLogger("vincio.suppressed").setLevel(DEBUG)` to
# see them all in one place; the per-call ``label`` identifies the subsystem and
# operation. Kept off the noisy default path: the standard level is DEBUG, so a
# production logger (WARNING root) stays quiet unless the operator opts in.
SUPPRESSED_LOGGER_NAME = "vincio.suppressed"

_logger = logging.getLogger(SUPPRESSED_LOGGER_NAME)
_lock = threading.Lock()
_counts: Counter[str] = Counter()


def note_suppressed(
    label: str,
    *,
    level: int = logging.DEBUG,
    detail: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Record a suppressed best-effort failure: log it (with traceback) and count it.

    Call this from inside the ``except`` block of a best-effort fallback whose body
    must continue rather than re-raise. ``label`` is a stable dotted key
    (``"<subsystem>.<operation>"``, e.g. ``"runtime.content_marking"``) the counter
    aggregates by; keep it constant for a given site so its rate is meaningful. The
    active exception's traceback is captured automatically (``exc_info``), so no
    exception argument is needed — but the call is only meaningful inside a live
    ``except``. ``level`` raises the log level for an *unexpected* failure (the
    default ``DEBUG`` keeps an *expected* fallback quiet in production); ``detail``
    adds a short human note to the message; ``logger`` overrides the dedicated
    diagnostics channel when a subsystem already has its own.

    Never raises — observability must not break the run it observes. The log call is
    lazy (it does no traceback formatting unless the channel is enabled at
    ``level``), so an unobserved fallback pays only the counter increment.
    """
    with _lock:
        _counts[label] += 1
    target = logger if logger is not None else _logger
    if detail is None:
        target.log(level, "suppressed best-effort failure [%s]", label, exc_info=True)
    else:
        target.log(level, "suppressed best-effort failure [%s]: %s", label, detail, exc_info=True)


def suppressed_failure_counts() -> dict[str, int]:
    """A snapshot of the suppressed-failure counts, keyed by label.

    The metric an operator scrapes to see how often each best-effort fallback fired,
    independent of whether the log channel is enabled. Returns a copy, so mutating
    the result never disturbs the live counters.
    """
    with _lock:
        return dict(_counts)


def reset_suppressed_failures() -> None:
    """Clear the suppressed-failure counters.

    A test and process-boundary helper; production code only reads the counts. The
    counters are process-global, so a test that asserts on them should reset first.
    """
    with _lock:
        _counts.clear()
