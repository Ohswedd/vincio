"""Energy & carbon accounting — the sustainability analogue of the cost report.

The cost report already makes a run's dollar spend an auditable number held by a
budget SLO. This example adds the rung beside it: a per-run **energy** (watt-hours)
and **carbon** (grams CO₂e) estimate — the disclosure sustainability-reporting
regimes are beginning to demand — accrued on the *same* cost-report surface, never
a new plane.

Five steps, all offline and deterministic:

  1. Enable: opt in to energy accounting; pick the deployment region whose grid
     intensity carbon is estimated against.
  2. Per-run estimate: every run reports energy and carbon on the result, accrued
     mechanically from its token accounting against a per-model (by-tier) intensity
     and the regional grid factor — no external service.
  3. On the cost surface: the estimate rolls up by model/tenant/feature on the same
     attributed events the cost report uses, via `app.energy_report(...)`.
  4. Budgeted like a dollar: an energy/carbon budget refuses a run that would exceed
     its sustainability envelope — the energy analogue of a hard cost cap.
  5. Auditable: the per-run estimate and every refusal land on the hash-chained,
     verifiable audit log, so the sustainability number is mechanical, not a claim.

Everything here is opt-in and additive; without `use_energy_accounting`/
`set_energy_budget` a run behaves exactly as before (energy fields stay 0.0).
"""

from __future__ import annotations

import asyncio

from vincio import ContextApp, VincioConfig
from vincio.providers import MockProvider


def _app(region: str) -> ContextApp:
    config = VincioConfig()
    config.observability.exporter = "memory"
    app = ContextApp(name="energy_demo", provider=MockProvider(), config=config)
    app.use_energy_accounting(region=region)
    return app


async def main() -> None:
    print("Energy & carbon accounting — energy is the new dollar\n")

    # 1. Enable energy accounting for an EU deployment.
    print("1. Enable — opt in and declare the deployment region")
    app = _app(region="eu")
    print("   energy accounting on; carbon estimated against the EU grid (~240 gCO₂e/kWh)")

    # 2. Every run carries a deterministic per-run estimate.
    print("\n2. Per-run estimate — energy and carbon on the result, like cost")
    result = await app.arun("Summarize our quarterly sustainability disclosure.")
    print(
        f"   run: ${result.cost_usd:.6f}  |  {result.energy_wh:.4f} Wh  |  "
        f"{result.co2e_grams:.4f} gCO₂e"
    )
    await app.arun("Draft the executive summary for the board.")

    # 3. The estimate rolls up on the same cost-report surface.
    print("\n3. On the cost surface — roll up by model on the same attributed events")
    app.energy_report(by="model").print_summary()
    print(f"   (the dollar cost report still works: {app.cost_report(by='model').total_usd:.6f} total)")

    # 4. The same compute on a cleaner grid emits less carbon.
    print("\n4. Region matters — the same run on France's grid emits far less carbon")
    fr_app = _app(region="fr")
    fr = await fr_app.arun("Summarize our quarterly sustainability disclosure.")
    print(
        f"   EU:     {result.energy_wh:.4f} Wh -> {result.co2e_grams:.4f} gCO₂e\n"
        f"   France: {fr.energy_wh:.4f} Wh -> {fr.co2e_grams:.4f} gCO₂e  "
        "(same compute, cleaner grid)"
    )

    # 5. Budgeted like a dollar: a carbon envelope refuses the over-budget run.
    print("\n5. Budgeted like a dollar — a carbon cap refuses the over-budget run")
    budgeted = _app(region="us")
    one = await budgeted.arun("probe the per-run carbon")
    budgeted.set_energy_budget(scope="global", limit_co2e_grams=one.co2e_grams * 1.5, period="total")
    statuses = []
    for i in range(4):
        r = await budgeted.arun(f"generate report section {i}")
        statuses.append(r.status.value)
    print(f"   run statuses under a {one.co2e_grams * 1.5:.4f} gCO₂e/total cap: {statuses}")
    print("   the envelope is enforced like a hard cost cap — over-budget runs are refused")

    # 6. Auditable & offline — the estimate and the refusal are on the chain.
    print("\n6. Auditable — the estimate and every refusal are on the verifiable chain")
    refusals = [e for e in budgeted.audit.entries if e.action == "energy_budget"]
    estimates = [
        e for e in budgeted.audit.entries if e.action == "run" and "co2e_grams" in (e.details or {})
    ]
    print(
        f"   {len(estimates)} per-run estimates and {len(refusals)} refusals on the chain; "
        f"chain verifies = {budgeted.audit.verify_chain()}"
    )

    print("\nA run's footprint, measured and budgeted the way its cost already is.")


if __name__ == "__main__":
    asyncio.run(main())
