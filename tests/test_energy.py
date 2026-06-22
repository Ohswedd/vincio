"""Energy & carbon accounting.

The energy analogue of the dollar cost report: a deterministic, offline estimate
of a run's energy (watt-hours) and carbon (grams CO₂e), accrued on the
cost-report surface and budgeted the way the cost report budgets dollars.
"""

from __future__ import annotations

import pytest

from vincio import (
    ContextApp,
    EnergyEstimate,
    EnergyIntensityTable,
    EnergyProfile,
    EnergyReport,
    VincioConfig,
)
from vincio.core.errors import EnergyBudgetError
from vincio.core.types import TokenUsage
from vincio.observability.energy import (
    DEFAULT_CARBON_INTENSITY,
    WORLD_AVERAGE_CARBON_INTENSITY,
    default_energy_table,
)
from vincio.observability.finops import CostLedger
from vincio.providers.mock import MockProvider


@pytest.fixture()
def config(tmp_path):
    cfg = VincioConfig()
    cfg.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    cfg.observability.exporter = "memory"
    cfg.security.audit_dir = str(tmp_path / "audit")
    return cfg


def make_app(config, **kwargs):
    return ContextApp(name="energy_test", provider=MockProvider(), config=config, **kwargs)


# --------------------------------------------------------------------------- #
# The estimation model
# --------------------------------------------------------------------------- #


def test_estimate_is_deterministic_and_decomposes():
    table = default_energy_table()
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    a = table.estimate("gpt-5.2-mini", usage, region="eu")
    b = table.estimate("gpt-5.2-mini", usage, region="eu")
    assert a.model_dump() == b.model_dump()  # mechanical, reproducible
    # Energy is the sum of the prefill and decode terms.
    assert a.energy_wh == pytest.approx(a.input_wh + a.output_wh)
    assert a.energy_wh > 0.0
    # Carbon is energy (kWh) × the regional grid intensity.
    assert a.co2e_grams == pytest.approx(a.energy_wh / 1000.0 * a.carbon_intensity_g_per_kwh)
    assert a.co2e_kg == pytest.approx(a.co2e_grams / 1000.0)


def test_decode_dominates_prefill_energy():
    table = default_energy_table()
    # An equal token count costs far more energy as output (autoregressive decode)
    # than as input (parallel prefill).
    prefill = table.estimate("gpt-5.2", TokenUsage(input_tokens=1000, output_tokens=0))
    decode = table.estimate("gpt-5.2", TokenUsage(input_tokens=0, output_tokens=1000))
    assert decode.energy_wh > prefill.energy_wh * 5


def test_stronger_tier_costs_more_energy_per_token():
    table = default_energy_table()
    usage = TokenUsage(input_tokens=500, output_tokens=500)
    fast = table.estimate("gpt-5.2-nano", usage)  # fast tier
    default = table.estimate("gpt-5.2-mini", usage)  # default tier
    strong = table.estimate("gpt-5.2", usage)  # strong tier
    assert fast.energy_wh < default.energy_wh < strong.energy_wh


def test_unknown_model_falls_back_to_default_tier_not_zero():
    table = default_energy_table()
    est = table.estimate("some-unlisted-model-9000", TokenUsage(input_tokens=100, output_tokens=100))
    assert est.energy_wh > 0.0
    assert est.energy_wh == table.estimate(
        "gpt-5.2-mini", TokenUsage(input_tokens=100, output_tokens=100)
    ).energy_wh  # both resolve to the default-tier reference


def test_region_intensity_resolution():
    table = default_energy_table()
    # Exact country token wins over its coarser jurisdiction (fr=56, not eu=240).
    assert table.intensity_for("fr") == ("fr", DEFAULT_CARBON_INTENSITY["fr"])
    assert table.intensity_for("eu") == ("eu", DEFAULT_CARBON_INTENSITY["eu"])
    # An AWS/GCP region string resolves via its jurisdiction.
    assert table.intensity_for("eu-west-1") == ("eu", DEFAULT_CARBON_INTENSITY["eu"])
    assert table.intensity_for("us-east-1") == ("us", DEFAULT_CARBON_INTENSITY["us"])
    # Unknown / unset region falls back to the world average.
    region, g = table.intensity_for("antarctica")
    assert g == WORLD_AVERAGE_CARBON_INTENSITY
    assert table.intensity_for(None)[1] == WORLD_AVERAGE_CARBON_INTENSITY


def test_cleaner_grid_lowers_carbon_for_same_energy():
    table = default_energy_table()
    usage = TokenUsage(input_tokens=400, output_tokens=400)
    fr = table.estimate("gpt-5.2", usage, region="fr")  # low-carbon grid
    india = table.estimate("gpt-5.2", usage, region="in")  # high-carbon grid
    assert fr.energy_wh == pytest.approx(india.energy_wh)  # same compute
    assert fr.co2e_grams < india.co2e_grams  # cleaner grid, less carbon


def test_overrides_for_model_and_region():
    table = default_energy_table()
    table.set("custom-model", EnergyProfile(wh_per_input_mtok=1.0, wh_per_output_mtok=2.0))
    est = table.estimate("custom-model", TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
    # 1 Mtok in + 1 Mtok out at the override, times PUE.
    assert est.input_wh == pytest.approx(1.0 * table.pue)
    assert est.output_wh == pytest.approx(2.0 * table.pue)
    table.set_region_intensity("on_prem", 12.0)  # operator measured a clean grid
    assert table.intensity_for("on_prem") == ("on_prem", 12.0)


def test_pue_scales_energy():
    base = EnergyIntensityTable(pue=1.0)
    overhead = EnergyIntensityTable(pue=1.5)
    usage = TokenUsage(input_tokens=1000, output_tokens=1000)
    assert overhead.estimate("gpt-5.2", usage).energy_wh == pytest.approx(
        base.estimate("gpt-5.2", usage).energy_wh * 1.5
    )


def test_estimate_accumulates():
    total = EnergyEstimate()
    table = default_energy_table()
    one = table.estimate("gpt-5.2", TokenUsage(input_tokens=10, output_tokens=10))
    total.add(one)
    total.add(one)
    assert total.energy_wh == pytest.approx(one.energy_wh * 2)
    assert total.co2e_grams == pytest.approx(one.co2e_grams * 2)


# --------------------------------------------------------------------------- #
# Opt-in: off by default, on the cost-report surface once enabled
# --------------------------------------------------------------------------- #


def test_off_by_default(config):
    app = make_app(config)
    assert app.energy_accounting_enabled is False
    result = app.run("summarize the quarterly report please")
    assert result.energy_wh == 0.0
    assert result.co2e_grams == 0.0


def test_enabled_accrues_on_run_and_tracker(config):
    app = make_app(config)
    app.use_energy_accounting(region="eu")
    result = app.run("summarize the quarterly report please")
    assert result.status.value == "succeeded"
    assert result.energy_wh > 0.0
    assert result.co2e_grams > 0.0
    summary = app.cost_tracker.summary()
    assert summary["energy_wh"] == pytest.approx(round(result.energy_wh, 6))
    assert summary["co2e_grams"] == pytest.approx(round(result.co2e_grams, 6))
    # Carbon reflects the declared region (eu), pinned over the mock's on_prem.
    assert result.co2e_grams == pytest.approx(
        result.energy_wh / 1000.0 * DEFAULT_CARBON_INTENSITY["eu"], rel=1e-6
    )


def test_energy_report_rolls_up_by_dimension(config):
    app = make_app(config)
    app.use_energy_accounting()
    app.run("first question about the contract terms", )
    app.run("second question about the renewal window")
    report = app.energy_report(by="model")
    assert isinstance(report, EnergyReport)
    assert report.total_energy_wh > 0.0
    assert report.total_co2e_grams > 0.0
    assert report.total_co2e_kg == pytest.approx(report.total_co2e_grams / 1000.0)
    assert sum(r.calls for r in report.rows) >= 2
    # The energy report and the cost report draw from the same attributed events.
    assert report.total_energy_wh == pytest.approx(
        round(app.cost_tracker.energy_wh, 6), rel=1e-6
    )


def test_region_override_pins_grid(config):
    fr_app = make_app(config)
    fr_app.use_energy_accounting(region="fr")
    fr = fr_app.run("summarize please for the report")
    us_app = make_app(config)
    us_app.use_energy_accounting(region="us")
    us = us_app.run("summarize please for the report")
    assert fr.energy_wh == pytest.approx(us.energy_wh)  # same model, same work
    assert fr.co2e_grams < us.co2e_grams  # France's grid is cleaner


def test_custom_intensity_merge(config):
    app = make_app(config)
    app.use_energy_accounting(region="on_prem", carbon_intensity={"on_prem": 10.0}, pue=1.0)
    result = app.run("summarize the report please")
    assert result.co2e_grams == pytest.approx(result.energy_wh / 1000.0 * 10.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Budgeted like a dollar: refused on breach, on the audit chain
# --------------------------------------------------------------------------- #


def test_set_energy_budget_requires_a_ceiling(config):
    app = make_app(config)
    with pytest.raises(EnergyBudgetError):
        app.set_energy_budget(scope="global")


def test_set_energy_budget_enables_accounting(config):
    app = make_app(config)
    assert app.energy_accounting_enabled is False
    app.set_energy_budget(limit_wh=1000.0)
    assert app.energy_accounting_enabled is True


def test_carbon_budget_refuses_over_envelope(config):
    app = make_app(config)
    # One run accrues ~0.03 gCO₂e at the on_prem grid; cap at 0.02 so the second
    # run is refused once the period total reaches the ceiling.
    app.set_energy_budget(scope="global", limit_co2e_grams=0.02, period="total")
    statuses = [app.run(f"request {i} please summarize the content").status.value for i in range(4)]
    assert statuses[0] == "succeeded"
    assert "denied" in statuses[1:]


def test_energy_budget_refuses_on_energy_too(config):
    app = make_app(config)
    app.set_energy_budget(scope="global", limit_wh=0.04, period="total")
    statuses = [app.run(f"request {i} please summarize").status.value for i in range(4)]
    assert statuses[0] == "succeeded"
    assert "denied" in statuses[1:]


def test_refusal_is_audited_and_chain_verifies(config):
    app = make_app(config)
    app.set_energy_budget(scope="global", limit_co2e_grams=0.02, period="total")
    for i in range(3):
        app.run(f"request {i} please summarize the content thoroughly")
    actions = [e.action for e in app.audit.entries]
    assert "energy_budget" in actions
    assert app.audit.verify_chain()


def test_per_run_estimate_on_audit_chain(config):
    app = make_app(config)
    app.use_energy_accounting(region="eu")
    result = app.run("summarize the contract please")
    run_entries = [
        e
        for e in app.audit.entries
        if e.action == "run" and "energy_wh" in (e.details or {})
    ]
    assert run_entries
    assert run_entries[-1].details["energy_wh"] == pytest.approx(round(result.energy_wh, 6))
    assert run_entries[-1].details["co2e_grams"] == pytest.approx(round(result.co2e_grams, 6))
    assert app.audit.verify_chain()


def test_budget_scoped_to_tenant(config):
    app = make_app(config)
    app.set_energy_budget(scope="tenant", id="acme", limit_co2e_grams=0.02, period="total")
    # acme exhausts its envelope...
    a1 = app.run("acme question please summarize", tenant_id="acme")
    a2 = app.run("acme second question please summarize", tenant_id="acme")
    assert a1.status.value == "succeeded"
    assert a2.status.value == "denied"
    # ...but another tenant is unaffected.
    other = app.run("globex question please summarize", tenant_id="globex")
    assert other.status.value == "succeeded"


# --------------------------------------------------------------------------- #
# Ledger roll-up
# --------------------------------------------------------------------------- #


def test_ledger_totals_energy_and_carbon():
    ledger = CostLedger()
    table = default_energy_table()
    est = table.estimate("gpt-5.2", TokenUsage(input_tokens=100, output_tokens=100), region="eu")
    ledger.record_model_call(
        model="gpt-5.2",
        usage=TokenUsage(input_tokens=100, output_tokens=100),
        cost_usd=0.001,
        tenant_id="acme",
        energy_wh=est.energy_wh,
        co2e_grams=est.co2e_grams,
    )
    assert ledger.total_energy(tenant_id="acme") == pytest.approx(round(est.energy_wh, 6))
    assert ledger.total_co2e(tenant_id="acme") == pytest.approx(round(est.co2e_grams, 6))
    assert ledger.total_energy(tenant_id="other") == 0.0
    report = ledger.energy_report("tenant")
    assert report.rows[0].key == "acme"
    assert report.rows[0].energy_wh == pytest.approx(round(est.energy_wh, 6))


def test_cost_report_carries_energy(config):
    app = make_app(config)
    app.use_energy_accounting()
    app.run("summarize the document please thoroughly")
    cost = app.cost_report(by="model")
    assert cost.rows
    assert cost.rows[0].energy_wh > 0.0
    assert cost.rows[0].co2e_grams > 0.0


def test_batch_path_accrues_energy(config):
    app = make_app(config)
    app.use_energy_accounting(region="eu")
    results = app.batch(["first batch item please", "second batch item please"])
    assert all(r.energy_wh > 0.0 for r in results)
    assert app.cost_tracker.energy_wh > 0.0
