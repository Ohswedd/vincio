"""The ergonomic 'ad-hoc' front door — task-shaped constructors over ``ContextApp``.

The platform is feature-complete, but its power is broad: a :class:`~vincio.core.app.ContextApp`
carries a couple hundred methods, and the five jobs a newcomer actually has —
grounded **RAG Q&A**, a **tool-using agent**, **structured extraction**, an
**eval**, and a **multi-step flow** — each take a fistful of string-keyed builder
calls. This namespace is the missing *top layer* (not a new capability): a small,
discoverable set of one-line, task-shaped constructors.

* :func:`rag` — a grounded-RAG question answerer (:class:`RagTask`).
* :func:`tool_agent` — an approval-gated tool-using agent (:class:`ToolAgent`).
* :func:`extractor` — a typed structured extractor from a schema (:class:`Extractor`).
* :func:`evaluation` — an offline evaluation (:class:`Evaluation`).
* :func:`chat` — a re-presentation of :meth:`~vincio.core.app.ContextApp.assistant`.
* :class:`Flow` — one fluent, immutable pipeline (retrieve → ground → call →
  validate → evaluate), the Vincio answer to LCEL.

Every constructor is a **purely-compositional facade** in the proven
:class:`~vincio.assistant.Assistant` / :class:`~vincio.settlement.CrossOrgEngagement`
/ :class:`~vincio.data.DataEngagement` mold: it configures a ``ContextApp`` with
sane governed defaults using the *same* public builder calls a caller would make
by hand, so the one-liner **lowers to the exact same governed ``ContextApp.run``
packet** — retrieval, grounding, validation, rails, budgets, tracing, and the
audit chain all apply unchanged. The common case is one expression; ``facade.app``
is the escape hatch to every deep method for the complex case (nothing shadowed,
nothing unreachable).

These symbols are :func:`~vincio.experimental` until their shape settles. They are
also re-exported at the top level (``from vincio import rag, Flow``); the concrete
facade types (:class:`RagTask`, :class:`Extractor`, :class:`ToolAgent`,
:class:`Evaluation`) live here in ``vincio.tasks``.
"""

from __future__ import annotations

from ._facades import (
    Evaluation,
    Extractor,
    RagTask,
    ToolAgent,
    chat,
    evaluation,
    extractor,
    rag,
    tool_agent,
)
from ._flow import Flow

__all__ = [
    "rag",
    "extractor",
    "tool_agent",
    "evaluation",
    "chat",
    "Flow",
    "RagTask",
    "Extractor",
    "ToolAgent",
    "Evaluation",
]
