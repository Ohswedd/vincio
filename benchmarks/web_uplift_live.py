"""Live web-search uplift — the same model, direct vs through Vincio's web plane.

Measures the actual product value proposition: asked about facts that changed
*after* a model's training cutoff, does routing the model through Vincio's
governed web search let it answer correctly where the bare model answers from
stale memory?

For each freshness question it runs the model two ways against a real provider:

* **direct** — a plain `ContextApp` call: the model answers from training alone.
* **vincio** — `app.use_web_search()` enabled: the model searches the open web
  (DuckDuckGo) and answers from what it reads, with governed, offline-verifiable
  evidence.

Each answer is scored ``fresh`` iff it contains the current fact. The gap
between the two accuracies is the measured uplift.

This is a **Live (Tier-L)** measurement: it hits the network and a paid model,
so it is *not* gated in CI — it is run by hand to produce the numbers reported
in the README. Run it:

    export OPENROUTER_API_KEY=sk-or-...
    python benchmarks/web_uplift_live.py --model openai/gpt-5.2-mini

Gold facts are current as of the harness date noted in ``--as-of`` and should be
refreshed when re-running much later.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

# Freshness questions with gold current as of 2026-07 (verified from the live web
# at authoring time). Each gold token must appear in the answer to score fresh.
QUESTIONS = [
    {
        "q": "What is the latest stable minor version of Python? Answer with just major.minor.",
        "gold": ["3.14"],
        "note": "Python 3.14 series is current (3.14.x); 3.13 was the prior line.",
    },
    {
        "q": "What is the current (non-LTS 'Current') major version line of Node.js as of 2026? "
             "Answer with the major version number only.",
        "gold": ["26"],
        "note": "Node.js 26 is the Current line; 24 is Active LTS.",
    },
    {
        "q": "Which Node.js major version is the Active LTS line in mid-2026? "
             "Answer with the major version number only.",
        "gold": ["24"],
        "note": "Node.js 24 is Active LTS.",
    },
]


def _contains(text: str, gold: str) -> bool:
    """Match *gold* in *text*, word-boundaried for a bare number so "26" does not
    match the year "2026"; substring for versions like "3.14"."""
    low, g = text.lower(), gold.lower()
    if g.isalnum():
        return re.search(rf"(?<!\d){re.escape(g)}(?!\d)", low) is not None
    return g in low


def _score(answer: str, gold: list[str]) -> bool:
    text = answer or ""
    return all(_contains(text, g) for g in gold)


def _make_app(model: str, *, web: bool):
    from vincio.core.app import ContextApp

    app = ContextApp(name="uplift-web" if web else "uplift-direct", provider="openrouter", model=model)
    if web:
        app.use_web_search(preset="research", today="2026-07-03")
    return app


async def _answer(app, question: str) -> str:
    result = await app.arun(question)
    return result.raw_text


async def run(model: str, as_of: str) -> dict:
    direct = _make_app(model, web=False)
    vincio = _make_app(model, web=True)
    rows = []
    for item in QUESTIONS:
        try:
            d = await _answer(direct, item["q"])
        except Exception as exc:  # noqa: BLE001 - live path, report and continue
            d = f"<error: {exc}>"
        try:
            v = await _answer(vincio, item["q"])
        except Exception as exc:  # noqa: BLE001 - live path, report and continue
            v = f"<error: {exc}>"
        rows.append(
            {
                "q": item["q"],
                "gold": item["gold"],
                "direct": d,
                "direct_fresh": _score(d, item["gold"]),
                "vincio": v,
                "vincio_fresh": _score(v, item["gold"]),
            }
        )
    n = len(rows) or 1
    direct_acc = sum(r["direct_fresh"] for r in rows) / n
    vincio_acc = sum(r["vincio_fresh"] for r in rows) / n
    return {
        "model": model,
        "as_of": as_of,
        "n": len(rows),
        "direct_accuracy": round(direct_acc, 3),
        "vincio_accuracy": round(vincio_acc, 3),
        "uplift": round(vincio_acc - direct_acc, 3),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live web-search uplift (OpenRouter).")
    parser.add_argument("--model", default="openai/gpt-5.2-mini", help="OpenRouter model id")
    parser.add_argument("--as-of", default="2026-07-03", help="date the gold facts are current for")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set; this is a live benchmark.", file=sys.stderr)
        return 2

    report = asyncio.run(run(args.model, args.as_of))
    if args.json:
        import json

        print(json.dumps(report, indent=2))
        return 0
    print(f"Live web-search uplift — {report['model']} (as of {report['as_of']})\n")
    for row in report["rows"]:
        print(f"Q: {row['q']}")
        print(f"   gold: {row['gold']}")
        print(f"   direct  [{'FRESH' if row['direct_fresh'] else 'stale'}]: {row['direct'][:160]}")
        print(f"   +vincio [{'FRESH' if row['vincio_fresh'] else 'stale'}]: {row['vincio'][:160]}\n")
    print(
        f"direct accuracy {report['direct_accuracy']:.0%}  →  "
        f"with Vincio web search {report['vincio_accuracy']:.0%}  "
        f"(uplift +{report['uplift']:.0%}, n={report['n']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
