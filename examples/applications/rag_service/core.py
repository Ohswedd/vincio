"""Grounded document-QA core — pure Vincio, no web framework.

This module is the brain of the microservice. It deliberately imports *nothing*
from FastAPI so it can be imported, tested, and run fully offline with just the
Vincio library (``.venv/bin/python -c "import core"``). The HTTP shell lives in
``main.py`` and only ever calls the plain functions defined here.

Offline-first contract
-----------------------
A bare ``ContextApp()`` defaults to OpenAI and fails without an API key, so this
file uses the standard ``_provider()`` helper: a deterministic in-process mock by
default, a real provider only when ``VINCIO_PROVIDER`` is set in the environment.
No network and no keys are required to import this module or call ``answer()``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from vincio import ContextApp
from vincio.providers import MockProvider, build_provider

# Where the bundled sample knowledge base is written, next to this file so the
# service is self-contained and never touches the user's wider project tree.
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# The bundled corpus. In a real deployment you would point ``add_source`` at a
# mounted volume, an object store, or a database connector instead — the rest of
# the pipeline (chunking, hybrid retrieval, grounding policy, evaluation) is
# identical. Kept tiny here so the demo stays readable and deterministic.
_SAMPLE_DOCS: dict[str, str] = {
    "refund_policy.md": (
        "# Refund Policy\n\n"
        "Customers on the Pro plan may request a full refund within 30 days of "
        "purchase with no processing fee. Basic plan customers have a 14-day "
        "refund window and a $5 processing fee.\n\n"
        "## Refund windows\n\n"
        "| Plan  | Window  | Fee |\n"
        "|-------|---------|-----|\n"
        "| Pro   | 30 days | $0  |\n"
        "| Basic | 14 days | $5  |\n"
    ),
    "billing.md": (
        "# Billing\n\n"
        "Subscriptions renew automatically on the anniversary of the signup "
        "date. The subscription renews unless it is cancelled at least 60 days "
        "before the renewal date. Late payments accrue 1.5% monthly interest.\n"
    ),
    "support.md": (
        "# Support\n\n"
        "Pro plan customers receive 24/7 priority support with a one-hour "
        "first-response target. Basic plan customers receive email support with "
        "a next-business-day response target.\n"
    ),
}


def _provider():
    """Offline mock by default; a real provider when VINCIO_PROVIDER is set."""
    name = os.environ.get("VINCIO_PROVIDER", "mock")
    if name == "mock":
        return MockProvider(), "mock-1"
    return build_provider(name), os.environ.get("VINCIO_MODEL", "gpt-4o-mini")


def _write_sample_knowledge(directory: Path) -> Path:
    """Materialise the bundled corpus on disk (idempotent)."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, body in _SAMPLE_DOCS.items():
        path = directory / filename
        # Only write when missing/changed so we don't churn the file on every
        # import while still keeping an edited corpus authoritative.
        if not path.exists() or path.read_text(encoding="utf-8") != body:
            path.write_text(body, encoding="utf-8")
    return directory


# The app is expensive-ish to build (it indexes the corpus), so build it once
# and reuse it across requests. ``build_app()`` is also exported for tests.
_APP: ContextApp | None = None


def build_app() -> ContextApp:
    """Configure a grounded document-QA ``ContextApp`` over the sample corpus.

    Steps, in order:
      1. write the bundled knowledge base to ``./knowledge``;
      2. wire the provider via ``_provider()`` (mock offline, real when set);
      3. attach the corpus as a hybrid-retrieval source;
      4. forbid answering from anything but retrieved evidence; and
      5. attach a groundedness evaluator so every answer is scored.
    """
    knowledge_dir = _write_sample_knowledge(KNOWLEDGE_DIR)

    provider, model = _provider()
    app = ContextApp(
        name="rag_service",
        provider=provider,
        model=model,
    )
    app.configure(
        role="grounded customer-support knowledge assistant",
        objective="Answer customer questions using only the provided knowledge base.",
        rules=[
            "Answer strictly from the retrieved documents.",
            "If the documents do not contain the answer, say so plainly.",
            "Cite the evidence that supports the answer.",
        ],
    )
    # Hybrid = keyword + vector retrieval, a sensible default for prose + tables.
    app.add_source("knowledge", path=str(knowledge_dir), retrieval="hybrid")
    # Grounding guardrail: the model may only use retrieved evidence.
    app.set_policy("answer_only_from_sources", True)
    # Score how well each answer is supported by the cited evidence.
    app.add_evaluator("groundedness")
    return app


def get_app() -> ContextApp:
    """Return the process-wide app, building it lazily on first use."""
    global _APP
    if _APP is None:
        _APP = build_app()
    return _APP


def answer(question: str) -> dict[str, Any]:
    """Answer a question against the knowledge base.

    Returns a plain JSON-able dict so the HTTP layer can serialise it directly:

        {"answer": str, "citations": list[str], "cost_usd": float,
         "trace_id": str, "groundedness": float | None}

    Raises ``ValueError`` on empty input so the HTTP layer can map it to a 400.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    result = get_app().run(question.strip())

    # ``result.eval_scores`` is a mapping of evaluator name -> score; expose the
    # groundedness number when present so callers can threshold on it.
    eval_scores = getattr(result, "eval_scores", None) or {}
    groundedness = eval_scores.get("groundedness")

    return {
        "answer": str(result.output),
        "citations": list(result.citations),
        "cost_usd": result.cost_usd,
        "trace_id": result.trace_id,
        "groundedness": groundedness,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import json

    print(json.dumps(answer("what is the refund window?"), indent=2))
