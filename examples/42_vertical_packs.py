"""Vertical packs: a regulated domain configured in one line.

A *vertical* pack is a full-stack starting point — prompt + schema + policies +
deterministic rails + domain metrics + scoped memory + a data-residency posture
+ a golden eval set — for a high-stakes use case. ``use_pack`` wires it all
through the public ``ContextApp`` API, so you can layer your own settings on top.

Built-in verticals: ``healthcare`` (PHI), ``ediscovery`` (legal),
``kyc`` (financial KYC/AML), ``customer_support``, and ``code_review``.
"""

from _shared import example_provider, json_responder

from vincio import ContextApp, available_packs, load_pack

# A KYC assessment the deterministic provider will return offline.
assessment = {
    "risk_rating": "high",
    "sanctions_hit": True,
    "pep": False,
    "sar_recommended": True,
    "rationale": "Screening returned a confirmed OFAC match against the beneficial owner.",
}
provider, model = example_provider(json_responder(assessment))

app = ContextApp(name="kyc_desk", provider=provider, model=model).use_pack("kyc")
# Standalone demo: relax source-grounding (in production attach your case files
# with app.add_source(...) and keep grounding + citations on).
app.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)


if __name__ == "__main__":
    verticals = ["healthcare", "ediscovery", "kyc", "customer_support", "code_review"]
    print("available packs:", available_packs())
    for name in verticals:
        pack = load_pack(name)
        print(
            f"- {name:16s} schema={pack.output_schema_name:18s} "
            f"rails={[r['name'] for r in pack.rails]} "
            f"residency={pack.residency or '—'} golden={len(pack.eval_cases)}"
        )

    print("\nApplied 'kyc' vertical:")
    print("  metrics  :", app.evaluators)
    print("  memory   :", app.memory is not None)
    # Residency is fail-closed: an unresolvable provider region is refused. The
    # offline mock resolves to on-prem, so this runs; a live deployment pins a
    # region-bearing endpoint or declares set_residency(provider_regions={...}).
    print("  residency:", app.residency.allowed_regions, "(fail-closed:",
          app.residency.deny_on_unknown, ")")

    result = app.run("Screen this customer against sanctions and adverse media.")
    output = result.output
    output = output.model_dump() if hasattr(output, "model_dump") else output
    print("\n  risk_rating  :", output["risk_rating"])
    print("  sanctions_hit:", output["sanctions_hit"])
    print("  SAR advised  :", output["sar_recommended"])

    # Each vertical ships a golden eval set you can gate quality on from day one.
    print("\n  golden eval set (kyc):", [c.id for c in load_pack("kyc").eval_cases])
