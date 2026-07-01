"""The benchmark provenance manifest is fresh, well-formed, and honest.

``benchmarks/manifest.json`` is the machine-readable source of truth for Vincio's
three-track benchmark platform and how real every number is. These tests gate that
it stays in lock-step with the three live registries and that its provenance tiers
line up with ``ProvenanceTier`` — so the honesty contract is enforced, not just
documented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import _manifest  # noqa: E402  (benchmarks/ is not a package; add to path first)

COMMITTED = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
_TRACKS = ("model", "uplift", "feature")


def test_manifest_is_fresh() -> None:
    """The committed manifest.json equals a fresh rebuild — no drift from code."""
    assert _manifest.build_manifest() == COMMITTED, (
        "benchmarks/manifest.json is stale — run `python benchmarks/_manifest.py`"
    )


def test_provenance_tiers_match_provenance_tier() -> None:
    from vincio.evals.suite import ProvenanceTier

    codes = {t.code for t in ProvenanceTier}
    tiers = COMMITTED["provenance_tiers"]
    assert set(tiers) == codes
    for code, cfg in tiers.items():
        assert {"name", "live", "reproducible", "gates_ci", "meaning"} <= set(cfg)
        if cfg["gates_ci"]:
            assert cfg["reproducible"], f"tier {code} gates CI but is not reproducible"
        if cfg["live"]:
            assert not cfg["gates_ci"], f"live tier {code} must not gate CI"


def test_three_tracks_present_and_reference_valid_tiers() -> None:
    tracks = COMMITTED["tracks"]
    assert tuple(tracks) == _TRACKS
    valid = set(COMMITTED["provenance_tiers"])
    for name, track in tracks.items():
        assert set(track["tiers"]) <= valid, f"track {name} references an unknown tier"
        assert track["question"] and track["compares"] and track["cli"]
        assert track["catalog"]["total"] >= 1


def test_model_catalog_matches_shipped_builtins() -> None:
    from vincio.evals.suite import BenchmarkRegistry

    shipped = set(BenchmarkRegistry(with_builtins=True).ids())
    cat = COMMITTED["tracks"]["model"]["catalog"]
    ids = {b["id"] for n in cat["niches"].values() for b in n["benchmarks"]}
    assert cat["total"] == len(shipped)
    assert ids == shipped


def test_uplift_catalog_matches_shipped_builtins() -> None:
    from vincio.evals.suite import UpliftRegistry

    shipped = set(UpliftRegistry(with_builtins=True).ids())
    cat = COMMITTED["tracks"]["uplift"]["catalog"]
    assert cat["total"] == len(shipped)
    assert {b["id"] for b in cat["benchmarks"]} == shipped


def test_feature_catalog_matches_shipped_builtins() -> None:
    from vincio.evals.suite import FeatureRegistry

    shipped = set(FeatureRegistry(with_builtins=True).ids())
    cat = COMMITTED["tracks"]["feature"]["catalog"]
    assert cat["total"] == len(shipped)
    assert {c["id"] for c in cat["contests"]} == shipped


def test_all_three_tracks_support_live_and_mockup() -> None:
    """Each track supports Live and an offline mockup — the user's requirement."""
    for name, track in COMMITTED["tracks"].items():
        assert "L" in track["tiers"], f"track {name} has no Live tier"
        assert "S" in track["tiers"], f"track {name} has no offline mockup tier"


def test_internal_gate_is_documented_separately() -> None:
    gate = COMMITTED["internal_gate"]
    assert gate["title"] == "VincioBench"
    assert "bench_tracks" in gate["role"]  # it gates the three tracks' deterministic core


@pytest.mark.parametrize("track", _TRACKS)
def test_each_track_declares_custom_extension(track: str) -> None:
    assert COMMITTED["tracks"][track]["custom"]
