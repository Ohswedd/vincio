"""Multi-agent crew: roles, delegation, and a shared blackboard."""

from _shared import example_provider

from vincio import ContextApp

provider, model = example_provider(
    default_responder=lambda request: (
        "Findings: refunds spiked 14% in Q3, driven by the Basic plan's $5 fee."
        if "as researcher" in "\n".join(m.text for m in request.messages)
        else "Report: Q3 refunds rose 14%; recommend waiving the Basic plan fee."
    )
)

app = ContextApp(name="crew_demo", provider=provider, model=model)

# A sequential crew: the writer sees everything the researcher posted on the
# shared blackboard. Each member runs under its share of the crew budget, so
# the team is guaranteed to terminate.
crew = app.crew(
    name="refund_team",
    members=[
        {"name": "researcher", "goal": "gather the relevant numbers", "keywords": ["find", "data"]},
        {"name": "writer", "goal": "draft a crisp recommendation"},
    ],
    process="sequential",
)

if __name__ == "__main__":
    result = crew.run("Explain the Q3 refund trend and recommend a fix")
    print("status:", result.status)
    print("output:", result.output)
    for report in result.reports:
        print(f"- {report.role}: {str(report.answer)[:80]} ({report.termination_reason})")
    print("blackboard keys:", list(result.blackboard["entries"]))
    print("crew metrics:", result.metrics())

    # Hierarchical: a manager delegates to the best-matching member (LLM-planned
    # when a real provider is configured, deterministic keyword routing offline).
    triage = app.crew(
        name="support_triage",
        members=[
            {"name": "billing", "description": "invoices, refunds, payments", "keywords": ["invoice", "refund"]},
            {"name": "legal", "description": "contracts and clauses", "keywords": ["contract", "clause"]},
        ],
        process="hierarchical",
    )
    routed = triage.run("Is invoice INV-77 refundable under the contract?")
    print("\ndelegations:", [(d.to_agent, d.reason) for d in routed.delegations])
    print("routed output:", routed.output)
