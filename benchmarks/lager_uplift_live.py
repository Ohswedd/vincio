"""Live LAGER uplift — reasoning-driven retrieval vs classic RAG, same model.

The corpus is the honest multi-hop fixture from the offline ``lager`` bench: a
long incident report whose root-cause paragraph shares zero content words with
the question, two pages of runbook/procedure prose holding the deeper causal
chain, and forty distractor documents saturated with the question's own words.

Three arms, the same model against a real provider, the same questions:

* **floor** — no retrieval at all (what the bare model knows: nothing, these
  are private facts).
* **classic** — the in-repo default pipeline: recursive 400-token chunks,
  hybrid top-k retrieval through ``app.add_source`` + ``app.run``.
* **lager** — ``app.use_lager()``: Evidence Objects, the typed graph, and the
  lazy loop feed the model the verified minimum.

Each question has a deterministic gold check (a substring the correct answer
must contain). We report answer correctness and input tokens per call.

Live (Tier-L): hits the network and a paid model — never CI-gated; run by hand
to produce the README numbers.

    export OPENROUTER_API_KEY=sk-or-...
    python benchmarks/lager_uplift_live.py --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vinciobench import _lager_corpus  # noqa: E402 - the shared honest fixture

# (question, gold substrings — the answer is correct iff ANY appears)
QUESTIONS = [
    ("why did the checkout outage happen",
     ["payments gateway", "certificate"]),
    ("who owns certificate management",
     ["platform team"]),
    ("why did the payments gateway reject connections",
     ["certificate", "expired"]),
    ("why had the rotation script been disabled",
     ["migration", "frozen", "froze"]),
]


def _config():
    from vincio.core.config import VincioConfig

    tmp = tempfile.mkdtemp()
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = f"{tmp}/audit"
    return config


def _make_app(model: str, arm: str, docs, *, embedder: str | None = None):
    from vincio.core.app import ContextApp

    app = ContextApp(name=f"lager-{arm}", provider="openrouter", model=model,
                     config=_config())
    if arm in ("classic", "lager"):
        app.add_source("corpus", documents=docs)
    if arm == "lager":
        # *embedder* (e.g. "auto") wires the dense signal into the lazy loop —
        # it tightens the two deliberate lexical residuals (recalls a topic
        # paraphrase of the cause, rejects an off-topic same-document decoy)
        # while the pure-stdlib default stays byte-identical. "auto" needs a
        # genuinely semantic local model (fastembed); "local" is the hash path.
        app.use_lager(embedder=embedder)
    return app


def _correct(answer: str, gold: list[str]) -> bool:
    lowered = (answer or "").lower()
    return any(g.lower() in lowered for g in gold)


def run(model: str, *, embedder: str | None = None) -> dict:
    docs, _hard, _easy, _bridge = _lager_corpus()
    arms: dict[str, dict] = {}
    for arm in ("floor", "classic", "lager"):
        rows = []
        for question, gold in QUESTIONS:
            app = _make_app(model, arm, docs, embedder=embedder)  # fresh app per question
            try:
                result = app.run(
                    question if arm != "floor"
                    else f"{question} (answer from your own knowledge)"
                )
                answer = result.raw_text
                tokens = int(result.usage.input_tokens or 0)
            except Exception as exc:  # noqa: BLE001 - live path
                answer, tokens = f"<error: {exc}>", 0
            rows.append({"q": question, "correct": _correct(answer, gold),
                         "input_tokens": tokens, "answer": (answer or "")[:120]})
        count = len(rows) or 1
        arms[arm] = {
            "accuracy": round(sum(r["correct"] for r in rows) / count, 3),
            "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / count, 1),
            "rows": rows,
        }
    return {"model": model, "embedder": embedder or "off (lexical)",
            "n": len(QUESTIONS), "arms": arms}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live LAGER uplift (OpenRouter).")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--embedder", default=None,
                        help="wire a dense signal into the lager arm's lazy loop "
                             "(e.g. 'auto' for a semantic local model); the "
                             "default is the pure-stdlib lexical path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set; this is a live benchmark.", file=sys.stderr)
        return 2
    report = run(args.model, embedder=args.embedder)
    if args.json:
        import json

        print(json.dumps(report, indent=2))
        return 0
    print(f"Live LAGER uplift — {report['model']} · dense signal: {report['embedder']} "
          f"(n={report['n']} multi-hop questions)\n")
    print(f"{'arm':<10} {'accuracy':>10} {'avg input tokens/call':>24}")
    for arm, data in report["arms"].items():
        print(f"{arm:<10} {data['accuracy']:>9.0%} {data['avg_input_tokens']:>24.0f}")
    classic, lager = report["arms"]["classic"], report["arms"]["lager"]
    ratio = classic["avg_input_tokens"] / max(lager["avg_input_tokens"], 1)
    print(
        f"\nLAGER accuracy {lager['accuracy']:.0%} vs classic {classic['accuracy']:.0%} "
        f"at {ratio:.1f}x fewer input tokens/call "
        f"(floor without retrieval: {report['arms']['floor']['accuracy']:.0%})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
