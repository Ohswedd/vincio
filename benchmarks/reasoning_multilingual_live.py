"""Live model-native multilingual routing smoke benchmark.

This Tier-L harness checks that Vincio follows the selected model's language
coverage instead of a finite local language list. It intentionally evaluates
routing receipts rather than language fluency: depth, task kind, web intent,
no-web intent, semantic-route success, and fully accounted token/cost overhead.

Usage::

    export OPENROUTER_API_KEY=sk-or-...
    python benchmarks/reasoning_multilingual_live.py \
      --models meta-llama/llama-3.1-8b-instruct --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

CASES: list[dict[str, Any]] = [
    {
        "id": "spanish_decision",
        "language": "es",
        "prompt": "Compara ambas opciones, analiza los riesgos y recomienda la mejor.",
        "depth": {"standard", "deep"},
        "kinds": {"decision_analysis", "multi_step"},
        "search": {"not_needed"},
    },
    {
        "id": "japanese_web_prohibited",
        "language": "ja",
        "prompt": "ウェブを使わずに、現在のCEOを教えてください。",
        "depth": {"standard", "deep"},
        "kinds": {"factual_verification", "live_factual"},
        "search": {"user_declined"},
    },
    {
        "id": "arabic_math",
        "language": "ar",
        "prompt": "احسب 17 * 23 وتحقق من النتيجة.",
        "depth": {"standard", "deep"},
        "kinds": {"mathematical"},
        "search": {"not_needed"},
    },
    {
        "id": "swahili_planning",
        "language": "sw",
        "prompt": "Panga hatua za uhamishaji, tambua hatari, kisha pendekeza mpango bora.",
        "depth": {"standard", "deep"},
        "kinds": {"planning", "multi_step", "decision_analysis"},
        "search": {"not_needed"},
    },
    {
        "id": "chinese_current_web",
        "language": "zh",
        "prompt": "请在网上核实当前稳定版，并引用最新来源。",
        "depth": {"standard", "deep"},
        "kinds": {"factual_verification", "live_factual"},
        "search": {"search"},
    },
]


def _app(model: str):
    from vincio import ContextApp, UniversalReasoningPolicy

    app = ContextApp(
        name="reasoning-multilingual-live",
        provider="openrouter",
        model=model,
    )
    app.use_reasoning_engine(
        policy=UniversalReasoningPolicy(
            max_passes=2,
            semantic_routing="always",
            web="auto",
        )
    )
    return app


async def _one(app: Any, case: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await app.arun(case["prompt"])
        receipt = result.metadata.get("universal_reasoning", {})
        kinds = set(receipt.get("task_kinds", []))
        correct = bool(
            receipt.get("semantic_routing_succeeded")
            and receipt.get("detected_language", "").split("-", 1)[0] == case["language"]
            and receipt.get("depth") in case["depth"]
            and kinds.intersection(case["kinds"])
            and receipt.get("search_decision") in case["search"]
        )
        return {
            "id": case["id"],
            "correct": correct,
            "status": result.status.value,
            "language": receipt.get("detected_language"),
            "depth": receipt.get("depth"),
            "task_kinds": receipt.get("task_kinds", []),
            "search_decision": receipt.get("search_decision"),
            "semantic_routing_tokens": receipt.get("semantic_routing_tokens", 0),
            "total_tokens": result.usage.total_tokens,
            "cost_usd": result.cost_usd,
            "error": result.error,
        }
    except Exception as exc:  # noqa: BLE001 - live harness reports failures
        return {"id": case["id"], "correct": False, "error": type(exc).__name__}


async def run(models: list[str]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for model in models:
        app = _app(model)
        rows = [await _one(app, case) for case in CASES]
        reports.append(
            {
                "model": model,
                "routing_accuracy": sum(row["correct"] for row in rows) / len(rows),
                "semantic_success_rate": sum(
                    bool(row.get("semantic_routing_tokens")) for row in rows
                )
                / len(rows),
                "tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
                "cost_usd": round(sum(float(row.get("cost_usd", 0.0)) for row in rows), 8),
                "cases": rows,
            }
        )
    return {
        "tier": "live",
        "provider": "openrouter",
        "capability": "universal_reasoning_multilingual_routing",
        "cases": len(CASES),
        "models": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="meta-llama/llama-3.1-8b-instruct",
        help="comma-separated OpenRouter model ids",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 2
    report = asyncio.run(run([item.strip() for item in args.models.split(",") if item.strip()]))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
