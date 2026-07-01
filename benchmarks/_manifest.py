"""The benchmark provenance manifest — one machine-readable source of truth for
Vincio's **three-track** benchmark platform and *how real* every number is.

The platform answers three questions, each as its own track, and each supporting a
**live** run and an offline **mockup**:

  1. **model**   — how good is a *model* on the standard public benchmarks?
  2. **uplift**  — how much does routing a model *through Vincio* change its scores?
  3. **feature** — how good is a Vincio *feature* vs the same feature in a competitor?

Every number carries a **provenance tier** — **L** Live (the real thing ran end to
end: a live model, or a real competitor library), **R** Recorded (a hash-pinned
replay), **S** Static/Mockup (offline, reproducible, gates CI). A separate internal
**VincioBench** gate keeps the library's own mechanisms honest and CI-gates the
deterministic core of all three tracks.

Run ``python benchmarks/_manifest.py`` to regenerate ``benchmarks/manifest.json``.
Each track's catalog is folded in **live** from its registry, so the committed
manifest never drifts. ``tests/test_benchmark_manifest.py`` gates that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
MANIFEST_PATH = HERE / "manifest.json"


PROVENANCE_TIERS: dict[str, dict[str, Any]] = {
    "S": {
        "name": "Static / Mockup",
        "live": False,
        "reproducible": True,
        "gates_ci": True,
        "meaning": (
            "An offline, reproducible run — a fabricated fixture (model track), a "
            "recorded two-arm illustration (uplift), or a competitor-absent/baseline "
            "comparison (feature). Byte-identical across machines; gates CI. Model "
            "scores saturate by design — a mechanism check, not a real-world claim."
        ),
    },
    "R": {
        "name": "Recorded",
        "live": False,
        "reproducible": True,
        "gates_ci": True,
        "meaning": "A hash-pinned slice of the real thing, replayed against recorded outputs.",
    },
    "L": {
        "name": "Live",
        "live": True,
        "reproducible": False,
        "gates_ci": False,
        "meaning": (
            "The real thing ran end to end — a live model + provider (model / uplift), "
            "or the actual competitor library executed on this machine (feature). "
            "Reported, never gated; only exists from a real key or a real install."
        ),
    },
}


def _model_catalog() -> dict[str, Any]:
    # A fresh registry with only the shipped built-ins, so the committed manifest
    # reflects what ships and is immune to any runtime `register_*` calls.
    from vincio.evals.suite import NICHES, BenchmarkRegistry

    registry = BenchmarkRegistry(with_builtins=True)
    return {
        "total": len(registry.ids()),
        "niches": {
            key: {
                "label": NICHES.get(key, key),
                "benchmarks": [
                    {"id": s.id, "title": s.title, "primary_metric": s.primary_metric,
                     "long_context": s.long_context, "has_loader": s.loader is not None}
                    for s in specs
                ],
            }
            for key, specs in registry.niches().items()
        },
    }


def _uplift_catalog() -> dict[str, Any]:
    from vincio.evals.suite import UpliftRegistry

    reg = UpliftRegistry(with_builtins=True)
    return {
        "total": len(reg.ids()),
        "benchmarks": [
            {"id": b.id, "title": b.title, "capability": b.capability, "primary_metric": b.primary_metric}
            for b in reg.all()
        ],
    }


def _feature_catalog() -> dict[str, Any]:
    from vincio.evals.suite import FeatureRegistry

    reg = FeatureRegistry(with_builtins=True)
    return {
        "total": len(reg.ids()),
        "capabilities": sorted(reg.by_capability()),
        "contests": [
            {"id": c.id, "title": c.title, "capability": c.capability,
             "primary_metric": c.primary_metric, "higher_is_better": c.higher_is_better}
            for c in reg.all()
        ],
    }


def build_tracks() -> dict[str, Any]:
    return {
        "model": {
            "title": "Model",
            "question": "how good is a model on the standard public benchmarks?",
            "compares": "a model vs the benchmark's verifiable gold",
            "location": "vincio/evals/suite (engine, adapters, builtin)",
            "cli": "vincio bench model <benchmark|niche|all> [--tier static|recorded|live]",
            "live_driver": "python benchmarks/eval_live.py --provider anthropic --model claude-opus-4-8",
            "tiers": ["S", "R", "L"],
            "gates_ci": True,
            "custom": "register_benchmark(BenchmarkSpec(...))",
            "catalog": _model_catalog(),
        },
        "uplift": {
            "title": "Uplift",
            "question": "how much does routing a model through Vincio change its scores?",
            "compares": "the same model, Vincio-routed vs direct — per-benchmark delta",
            "location": "vincio/evals/suite/uplift.py",
            "cli": "vincio bench uplift <benchmark|all> [--tier static|live]",
            "live_driver": "UpliftSuite().run(..., tier='live', direct=<app>, vincio=<app>)",
            "tiers": ["S", "L"],
            "gates_ci": True,
            "custom": "register_uplift_benchmark(UpliftBenchmark(...))",
            "catalog": _uplift_catalog(),
        },
        "feature": {
            "title": "Feature",
            "question": "how does a Vincio feature compare to the same feature in a competitor library?",
            "compares": "a Vincio feature vs a real competitor library (and a naive baseline)",
            "location": "vincio/evals/suite/feature_bench.py",
            "cli": "vincio bench feature <contest|capability|all>",
            "live_driver": "vincio bench feature   (runs live vs whatever competitors are installed)",
            "tiers": ["S", "L"],
            "gates_ci": True,
            "custom": "register_feature_contest(FeatureContest(...))",
            "catalog": _feature_catalog(),
        },
    }


def build_manifest() -> dict[str, Any]:
    return {
        "_comment": (
            "The benchmark provenance manifest — machine-readable source of truth for "
            "Vincio's three-track benchmark platform. Regenerate with "
            "`python benchmarks/_manifest.py`; gated by tests/test_benchmark_manifest.py. "
            "See benchmarks/PROVENANCE.md for the human-readable map."
        ),
        "schema_version": "2.0",
        "provenance_tiers": PROVENANCE_TIERS,
        "tracks": build_tracks(),
        "internal_gate": {
            "title": "VincioBench",
            "location": "benchmarks/vinciobench.py",
            "role": (
                "The internal mechanism / regression gate — not one of the three public "
                "tracks. It proves each Vincio mechanism still works against a naive baseline "
                "(offline, saturating) and CI-gates the deterministic core of all three tracks "
                "via the `families.bench_tracks.*` budgets."
            ),
            "run": "python benchmarks/vinciobench.py && python benchmarks/check_budgets.py",
        },
    }


def write(path: Path = MANIFEST_PATH) -> Path:
    path.write_text(json.dumps(build_manifest(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    written = write()
    m = json.loads(written.read_text(encoding="utf-8"))
    tracks = m["tracks"]
    print(
        f"wrote {written} — 3 tracks "
        f"(model={tracks['model']['catalog']['total']} benchmarks, "
        f"uplift={tracks['uplift']['catalog']['total']}, "
        f"feature={tracks['feature']['catalog']['total']} contests), "
        f"{len(PROVENANCE_TIERS)} provenance tiers"
    )
