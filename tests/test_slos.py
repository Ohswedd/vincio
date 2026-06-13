"""The published SLOs must be backed by CI budgets at least as strict."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
SLOS = json.loads((BENCH / "slos.json").read_text())["slos"]
BUDGETS = json.loads((BENCH / "budgets.json").read_text())["budgets"]


def test_slos_well_formed():
    ids = [s["id"] for s in SLOS]
    assert len(ids) == len(set(ids)), "duplicate SLO ids"
    for slo in SLOS:
        assert slo["direction"] in ("gte", "lte", "eq")
        assert slo["statement"].strip()
        assert slo["rationale"].strip()
        assert slo["category"] in ("performance", "cost", "quality", "reliability", "security")


@pytest.mark.parametrize("slo", SLOS, ids=lambda s: s["id"])
def test_each_slo_is_enforced_by_a_budget_at_least_as_strict(slo):
    key = slo["enforced_by"]
    assert key in BUDGETS, f"SLO {slo['id']} references unknown budget key {key!r}"
    budget = BUDGETS[key]
    direction = slo["direction"]
    threshold = slo["threshold"]

    if direction == "eq":
        assert budget.get("eq") == threshold
    elif direction == "gte":
        # A higher floor in the budget is at least as strict as the public promise.
        assert "gte" in budget, f"{key} should gate a gte floor"
        assert budget["gte"] >= threshold, (
            f"{slo['id']}: budget floor {budget['gte']} weaker than published {threshold}"
        )
    elif direction == "lte":
        # A lower ceiling in the budget is at least as strict.
        assert "lte" in budget, f"{key} should gate an lte ceiling"
        assert budget["lte"] <= threshold, (
            f"{slo['id']}: budget ceiling {budget['lte']} weaker than published {threshold}"
        )


def test_slo_metric_matches_enforced_key():
    # The metric an SLO names and the budget that enforces it must be the same path.
    for slo in SLOS:
        assert slo["metric"] == slo["enforced_by"]
