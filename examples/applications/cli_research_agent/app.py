"""A bounded local research agent — a single-file command-line application.

Point this CLI at a small folder of your own markdown notes and ask a question.
The app retrieves the relevant passages, answers *only* from that evidence, and
prints the answer together with the exact citations that back it and the cost of
the run. It is grounded RAG with guardrails, packaged as a Unix-style command.

No framework. No web server. Just `python app.py "your question"`.

Runs fully offline by default on the bundled deterministic mock provider — no
API keys, no network. Set VINCIO_PROVIDER (and the matching API key) to answer
with a real model instead; see the README.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vincio import ContextApp
from vincio.providers import MockProvider, build_provider


# --- Provider selection ----------------------------------------------------
# Offline mock by default; a real provider when VINCIO_PROVIDER is set. This is
# the one helper every Vincio example shares, so the same code runs with zero
# setup *and* against a real model by flipping a single environment variable.
def _provider():
    """Offline mock by default; a real provider when VINCIO_PROVIDER is set."""
    name = os.environ.get("VINCIO_PROVIDER", "mock")
    if name == "mock":
        return MockProvider(), "mock-1"
    return build_provider(name), os.environ.get("VINCIO_MODEL", "gpt-4o-mini")


# Notes live next to this script so the CLI is self-contained and the corpus is
# easy to inspect and extend — drop more .md files in ./notes and they are
# indexed on the next run.
NOTES_DIR = Path(__file__).resolve().parent / "notes"

# A tiny, self-contained knowledge base written on first run if absent. The mix
# of policy windows, fees, and SLA terms gives the retriever something concrete
# to ground (and cite) against.
_SAMPLE_NOTES: dict[str, str] = {
    "refund_policy.md": (
        "# Refund Policy\n\n"
        "Customers on the Pro plan may request a refund within 30 days of "
        "purchase, with no processing fee.\n\n"
        "Basic plan refunds must be requested within 14 days and incur a $5 "
        "processing fee. Refunds are issued by the billing team within five "
        "business days of approval.\n"
    ),
    "subscription_terms.md": (
        "# Subscription Terms\n\n"
        "The subscription renews automatically unless it is terminated at least "
        "60 days before the renewal date. The initial term is 24 months.\n\n"
        "Late payments accrue interest at 1.5% per month on the outstanding "
        "balance.\n"
    ),
    "support_sla.md": (
        "# Support SLA\n\n"
        "The service guarantees 99.9% monthly uptime. For each full hour of "
        "downtime beyond that threshold, customers receive a 10% service credit "
        "on the affected month.\n\n"
        "Priority support tickets are acknowledged within one business hour.\n"
    ),
}


def ensure_notes(directory: Path = NOTES_DIR) -> Path:
    """Write the sample notes corpus if it does not already exist."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, body in _SAMPLE_NOTES.items():
        note = directory / filename
        if not note.exists():
            note.write_text(body, encoding="utf-8")
    return directory


def build_app(notes_dir: Path) -> ContextApp:
    """A grounded research agent over the local notes folder.

    `answer_only_from_sources` forbids the model from drawing on anything but
    retrieved evidence, and the `groundedness` evaluator scores how well the
    answer is actually supported by that evidence — the difference between "the
    model said so" and "the notes say so".
    """
    provider, model = _provider()
    app = ContextApp(
        name="cli_research_agent",
        provider=provider,
        model=model,
    )
    app.configure(
        role="local research assistant",
        objective="Answer questions strictly from the provided notes, with citations.",
        rules=[
            "Only use facts found in the retrieved notes.",
            "If the notes do not contain the answer, say so plainly.",
        ],
    )
    # Index the notes folder: adaptive chunking + hybrid (keyword + vector)
    # retrieval are sensible defaults for mixed prose and small tables.
    app.add_source("notes", path=str(notes_dir), chunking="adaptive", retrieval="hybrid")
    app.set_policy("answer_only_from_sources", True)
    app.add_evaluator("groundedness")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli_research_agent",
        description="Ask a question and get a cited answer grounded in your local notes.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What is the refund window for the Pro plan?",
        help="The question to research (defaults to a sample question).",
    )
    parser.add_argument(
        "--notes",
        default=str(NOTES_DIR),
        help="Directory of markdown notes to research over (default: ./notes).",
    )
    args = parser.parse_args(argv)

    notes_dir = ensure_notes(Path(args.notes))
    app = build_app(notes_dir)

    result = app.run(args.question)

    print(f"Q: {args.question}\n")
    print(f"A: {result.output}\n")

    citations = result.citations or []
    if citations:
        print(f"Citations ({len(citations)}):")
        for citation in citations:
            print(f"  - {citation}")
    else:
        print("Citations: (none)")

    grounded = (result.eval_scores or {}).get("groundedness")
    if grounded is not None:
        print(f"\nGroundedness: {grounded}")
    print(f"Cost: ${result.cost_usd:.6f}")
    print(f"Trace: {result.trace_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
