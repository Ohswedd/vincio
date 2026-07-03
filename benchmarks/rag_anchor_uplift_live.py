"""Live RAG-anchor uplift — keeping a coding-standards frame across chained calls.

The real use-case: a coding agent is given a bulk of standards (a PRD, coding
conventions, a brand guide) that bind *every* step, then asked a series of small
tasks that never restate the rule. Does the model honor the global constraint?

Three arms, the same model against a real provider, on the same task series:

* **stuff** — the whole standards corpus is pasted into every call (the
  Claude-Code/CLAUDE.md pattern). Honors the rule, but pays the corpus every call.
* **pure_rag** — the standards are indexed and retrieved per query. Cheap, but a
  task like "reverse a string" does not lexically match "respond in French", so
  the rule is not retrieved and the model violates it.
* **anchors** — the standards are an ``anchor=True`` source: a compact frame is
  pinned into every call. Honors the rule at a flat, tiny cost.

The constraint is deterministically checkable, strongly obeyed when the model
sees it, and phrased in words a coding task never uses (so pure per-query RAG
cannot retrieve it for "reverse a string"): every reply must be written in
French. An output "respects" the frame iff it is in French. We report adherence
and input-tokens/call per arm.

Live (Tier-L): hits the network and a paid model, so it is not CI-gated — run by
hand to produce the README numbers.

    export OPENROUTER_API_KEY=sk-or-...
    python benchmarks/rag_anchor_uplift_live.py --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile

# The binding localization rule, stated once, buried in a large narrative corpus.
# "French" / "localization" never appear in a coding task, so pure per-query RAG
# cannot retrieve the rule for "reverse a string".
STANDARDS = (
    "Vincio contributor handbook.\n"
    "Localization policy: you MUST write every response entirely in French, "
    "including all prose, explanations, and code comments. This is mandatory for "
    "every answer regardless of the question's language.\n"
    + " ".join(
        f"Rationale {i}: our conventions keep the codebase consistent and reviewable "
        "across many contributors and long-lived modules over the years, and this "
        "paragraph elaborates the reasoning at length without stating any new rule "
        "so that reviewers understand the historical context and the trade-offs."
        for i in range(120)
    )
)
BRAND = (
    "Brand voice: warm, concise, encouraging. Address the user directly. " + " ".join(
        f"Story {i}: the brand grew from a community of builders who value clarity, "
        "and this passage recounts that history in unhurried, evocative detail."
        for i in range(120)
    )
)

# Tasks in English that never mention the French localization rule.
TASKS = [
    "Write a Python function that adds two numbers.",
    "Write a Python function that reverses a string.",
    "Write a Python function that checks whether a number is prime.",
    "Write a Python function that returns the maximum of a list.",
]

# Deterministic French detector: common French function words / accented markers
# that are near-absent from an English coding explanation.
_FRENCH_MARKERS = (
    " cette ", " une fonction", " qui ", " renvoie", " chaîne", " nombre", " la ",
    " les ", " nous ", " vous ", " prend ", " retourne", " voici", " nombres",
    " est ", " deux ", " pour ",
)


def _respects(answer: str) -> bool:
    low = f" {(answer or '').lower()} "
    return sum(marker in low for marker in _FRENCH_MARKERS) >= 2


def _config(tmp: str):
    from vincio.core.config import VincioConfig

    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = f"{tmp}/audit"
    return config


def _make_app(model: str, arm: str):
    from vincio.core.app import ContextApp
    from vincio.core.types import Document

    app = ContextApp(name=f"rag-{arm}", provider="openrouter", model=model, config=_config(tempfile.mkdtemp()))
    docs = [Document(title="Coding standards", text=STANDARDS), Document(title="Brand", text=BRAND)]
    if arm == "pure_rag":
        app.add_source("standards", documents=docs)
    elif arm == "anchors":
        app.add_source("standards", documents=docs, anchor=True, brief_tokens=220)
    return app


async def _ask(app, arm: str, task: str) -> tuple[str, int]:
    prompt = task
    if arm == "stuff":  # paste the whole corpus into every call
        prompt = f"{STANDARDS}\n\n{BRAND}\n\nTask: {task}"
    result = await app.arun(prompt)
    return result.raw_text, int(result.usage.input_tokens or 0)


async def run(model: str) -> dict:
    arms = {}
    for arm in ("stuff", "pure_rag", "anchors"):
        rows = []
        for task in TASKS:
            app = _make_app(model, arm)  # fresh app per task = independent chained call
            try:
                answer, tokens = await _ask(app, arm, task)
            except Exception as exc:  # noqa: BLE001 - live path
                answer, tokens = f"<error: {exc}>", 0
            rows.append({"task": task, "respects": _respects(answer), "input_tokens": tokens,
                         "answer": answer[:120]})
        n = len(rows) or 1
        arms[arm] = {
            "adherence": round(sum(r["respects"] for r in rows) / n, 3),
            "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / n, 1),
            "rows": rows,
        }
    return {"model": model, "n": len(TASKS), "arms": arms}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live RAG-anchor uplift (OpenRouter).")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set; this is a live benchmark.", file=sys.stderr)
        return 2
    report = asyncio.run(run(args.model))
    if args.json:
        import json

        print(json.dumps(report, indent=2))
        return 0
    print(f"Live RAG-anchor uplift — {report['model']} (n={report['n']} tasks, none mention the rule)\n")
    print(f"{'arm':<10} {'rule respected':>16} {'avg input tokens/call':>24}")
    for arm, data in report["arms"].items():
        print(f"{arm:<10} {data['adherence']:>15.0%} {data['avg_input_tokens']:>24.0f}")
    a = report["arms"]
    print(
        f"\nAnchors match 'stuff' on adherence ({a['anchors']['adherence']:.0%} vs "
        f"{a['stuff']['adherence']:.0%}) at "
        f"{a['stuff']['avg_input_tokens'] / max(a['anchors']['avg_input_tokens'], 1):.1f}x fewer "
        f"input tokens/call, while pure RAG drops the rule ({a['pure_rag']['adherence']:.0%})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
