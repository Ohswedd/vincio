"""Live universal-reasoning uplift across native and non-native models.

Runs the same model direct and through ``UniversalReasoningEngine`` on a small,
reviewable mix of arithmetic, logic, contradiction and freshness questions.
It reports exact-answer accuracy, live-source verification, refusal to overclaim,
tokens and cost. This is Tier-L evidence: it uses OpenRouter and the real web,
is never run in CI, and must not be presented as a general benchmark result.

Usage::

    export OPENROUTER_API_KEY=sk-or-...
    python benchmarks/reasoning_uplift_live.py \
      --models meta-llama/llama-3.1-8b-instruct,openai/gpt-5.2-mini --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

CASES: list[dict[str, Any]] = [
    {
        "id": "arithmetic",
        "prompt": "Calculate 17 * 23, verify the equality, and answer with the integer.",
        "gold": ["391"],
        "live": False,
    },
    {
        "id": "logic",
        "prompt": (
            "All red keys open door A. No brass key opens door A. Key K is red and brass. "
            "Are the premises mutually consistent? Answer yes or no and identify the contradiction."
        ),
        "gold": ["no", "contradict"],
        "leading_verdict": "no",
        "live": False,
    },
    {
        "id": "multi_step",
        "prompt": (
            "A service handles 240 requests/minute. Traffic rises 25%, then two identical workers "
            "split it evenly. Calculate and verify the requests/minute per worker."
        ),
        "gold": ["150"],
        "live": False,
    },
    {
        "id": "freshness",
        "prompt": (
            "Search the web for the latest stable Python major.minor line as of July 2026. "
            "Fact-check it against a primary source and answer with the version."
        ),
        "gold": ["3.14"],
        "live": True,
    },
]


def _matches(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(
        (
            re.search(rf"(?<!\d){re.escape(term.lower())}(?!\d)", lowered)
            if term.isdigit() or term.lower() in {"yes", "no"}
            else term.lower() in lowered
        )
        for term in terms
    )


def _score_case(text: str, case: dict[str, Any]) -> bool:
    if expected := case.get("leading_verdict"):
        verdict = re.search(r"\b(?:yes|no)\b", text, re.IGNORECASE)
        if verdict is None or verdict.group(0).lower() != expected:
            return False
    return _matches(text, case["gold"])


def _overclaims(text: str) -> bool:
    lowered = text.lower()
    unavailable = any(token in lowered for token in ("could not verify", "unable to verify", "insufficient evidence"))
    asserts_live = any(token in lowered for token in ("latest", "current", "as of"))
    return asserts_live and not unavailable and not re.search(r"https?://|\[[^]]+\]", text)


def _app(model: str, *, reasoning: bool):
    from vincio import ContextApp, UniversalReasoningPolicy

    app = ContextApp(name=f"reasoning-live-{'engine' if reasoning else 'direct'}", provider="openrouter", model=model)
    if reasoning:
        app.use_web_search(preset="research", today="2026-07-07")
        app.use_reasoning_engine(
            policy=UniversalReasoningPolicy(max_passes=4, web="auto")
        )
    return app


async def _one(app: Any, case: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await app.arun(case["prompt"])
        receipt = result.metadata.get("universal_reasoning", {})
        return {
            "answer": result.raw_text,
            "correct": _score_case(result.raw_text, case),
            "status": result.status.value,
            "error": result.error,
            "web_verified": bool(receipt.get("web_verified")) if case["live"] else None,
            "overclaim": _overclaims(result.raw_text) if case["live"] else False,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "reasoning_tokens": result.usage.reasoning_tokens,
            "cost_usd": result.cost_usd,
            "receipt": receipt,
        }
    except Exception as exc:  # noqa: BLE001 - live harness reports provider/web failures
        return {"answer": f"<error: {exc}>", "correct": False, "error": type(exc).__name__}


async def run(models: list[str]) -> dict[str, Any]:
    model_reports = []
    for model in models:
        direct = _app(model, reasoning=False)
        engine = _app(model, reasoning=True)
        rows = []
        for case in CASES:
            baseline = await _one(direct, case)
            reasoned = await _one(engine, case)
            rows.append({"id": case["id"], "gold": case["gold"], "direct": baseline, "reasoned": reasoned})
        count = len(rows) or 1
        direct_accuracy = sum(row["direct"]["correct"] for row in rows) / count
        reasoned_accuracy = sum(row["reasoned"]["correct"] for row in rows) / count
        model_reports.append(
            {
                "model": model,
                "cases": rows,
                "direct_accuracy": round(direct_accuracy, 4),
                "reasoned_accuracy": round(reasoned_accuracy, 4),
                "accuracy_uplift": round(reasoned_accuracy - direct_accuracy, 4),
                "live_evidence_verified_rate": round(
                    sum(row["reasoned"].get("web_verified") is True for row in rows if row["id"] == "freshness"),
                    4,
                ),
                "safe_refusal_rate": round(
                    sum(
                        row["reasoned"].get("status") == "failed"
                        and not row["reasoned"].get("answer")
                        for row in rows
                        if row["id"] == "freshness"
                    ),
                    4,
                ),
                "overclaim_rate": round(
                    sum(row["reasoned"].get("overclaim") is True for row in rows) / count, 4
                ),
                "direct_tokens": sum(
                    row["direct"].get("input_tokens", 0) + row["direct"].get("output_tokens", 0)
                    for row in rows
                ),
                "reasoned_tokens": sum(
                    row["reasoned"].get("input_tokens", 0) + row["reasoned"].get("output_tokens", 0)
                    for row in rows
                ),
                "direct_cost_usd": round(sum(row["direct"].get("cost_usd", 0.0) for row in rows), 8),
                "reasoned_cost_usd": round(sum(row["reasoned"].get("cost_usd", 0.0) for row in rows), 8),
                "local_cost_accounting_available": any(
                    row[arm].get("cost_usd", 0.0) > 0
                    for row in rows
                    for arm in ("direct", "reasoned")
                ),
            }
        )
    return {
        "tier": "live",
        "provider": "openrouter",
        "as_of": "2026-07-07",
        "cases": len(CASES),
        "models": model_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live universal-reasoning uplift (OpenRouter).")
    parser.add_argument(
        "--models",
        default="meta-llama/llama-3.1-8b-instruct,openai/gpt-5.2-mini",
        help="comma-separated OpenRouter model ids (include non-native and native reasoning models)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set; this is a paid live benchmark.", file=sys.stderr)
        return 2
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    report = asyncio.run(run(models))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Universal reasoning live uplift (Tier-L)\n")
        for model in report["models"]:
            print(
                f"{model['model']}: {model['direct_accuracy']:.0%} direct -> "
                f"{model['reasoned_accuracy']:.0%} reasoned "
                f"({model['accuracy_uplift']:+.0%}, ${model['reasoned_cost_usd']:.4f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
