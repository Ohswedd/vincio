"""Agents & tools — plain Python functions as *governed* actions.

Vincio gives you one runtime where every tool call is permission-checked,
schema-validated, optionally sandboxed and (for writes) approval-gated BEFORE it
runs. On top of that runtime sit interchangeable planners (ReAct / HTN), in-place
plan repair, cost-aware model selection, the budgeted deep-research agent, and a
grounded computer-use action plane. All deterministic and fully offline.
"""

from __future__ import annotations

import asyncio

from _shared import citing_responder, example_provider
from pydantic import BaseModel

from vincio import ActionPolicy, ContextApp, UIAction, build_web_checkout
from vincio.agents import HTNDomain
from vincio.core.types import Document
from vincio.tools.sandbox import SandboxedPython, SubprocessIsolation


def section_tool_registry() -> None:
    # A plain function becomes a typed tool: its input schema is DERIVED from the
    # type hints, and its contract (required scopes, side effects, approval) is
    # declared at registration. The permission check is deterministic and runs
    # *before* execution — best practice is to mark every write approval_required.
    app = ContextApp(name="ops", provider=example_provider()[0], model="mock-1")

    def fleet_status(region: str) -> dict:
        """Read fleet health for a region (no special scope needed)."""
        return {"region": region, "healthy": 41, "degraded": 2}

    def drain_node(node_id: str, force: bool = False) -> dict:
        """Cordon and drain a node (a write action)."""
        return {"node_id": node_id, "drained": True}

    app.add_tool(fleet_status, side_effects="read")  # allowed for the default principal
    app.add_tool(drain_node, permissions=["ops:write"], side_effects="write", approval_required=True)

    spec = app.tool_registry.get("drain_node").spec
    print("1. tool registry: drain_node schema fields",
          sorted(spec.input_schema.get("properties", {})),
          f"| requires {spec.permissions} approval={spec.approval_required}")


def section_approval_gated_writes() -> None:
    # A ReAct agent interleaves think -> tool -> observe. Its default principal
    # holds no write scope, so the runtime BLOCKS the side-effecting call and
    # records it as 'denied' — the agent still reasons to a decision, but the
    # irreversible action never fires. This is the human-in-the-loop seam.
    class RefundDecision(BaseModel):
        decision: str
        reason: str
        requires_human: bool

    provider, model = example_provider(script=[
        {"tool_call": {"name": "billing_lookup", "arguments": {"invoice_id": "INV-123"}}},
        '{"decision": "eligible", "reason": "Paid within the 30-day window.", "requires_human": true}',
    ])
    app = ContextApp(name="refunds", output_schema=RefundDecision, provider=provider, model=model)

    def billing_lookup(invoice_id: str) -> dict:
        """Look up a billing record."""
        return {"invoice_id": invoice_id, "amount": 49.0, "status": "paid", "age_days": 12}

    def refund_create(invoice_id: str, amount: float) -> dict:
        """Issue a refund (irreversible write)."""
        return {"refunded": amount, "invoice_id": invoice_id}

    app.add_tool(billing_lookup, permissions=["billing:read"])
    app.add_tool(refund_create, permissions=["billing:write"], side_effects="write", approval_required=True)

    state = app.agent(planner="react", max_steps=6).run(
        "Customer asks for a refund on INV-123. Decide eligibility; do not issue the refund.")
    print("2. approval-gated ReAct:", state.final_answer,
          "| verdicts", [(r.tool_name, r.status) for r in state.tool_results])


async def section_sandbox() -> None:
    # Generated or untrusted code runs in a separate process under `python -I`
    # with a wall-clock timeout, output caps, a scrubbed env and POSIX resource
    # limits. The default subprocess backend is tool-grade isolation, NOT a
    # boundary against a hostile kernel — swap a container/microVM/WASM backend
    # for adversarial workloads (box.isolation.real is False here to say so).
    box = SandboxedPython(timeout_s=5.0, max_cpu_seconds=2, max_memory_bytes=256 * 1024 * 1024,
                          max_open_files=32, isolation=SubprocessIsolation())
    result = await box.run("print(sum(i * i for i in range(1000)))")
    print(f"3. sandbox: stdout={result.stdout.strip()} exit={result.exit_code} "
          f"real_boundary={box.isolation.real}")


def section_hierarchical_repair() -> None:
    # An HTN domain decomposes a goal into a sub-goal tree. One leaf binds a flaky
    # tool with a declared fallback; when the primary raises, the planner RE-BINDS
    # in place and finishes — recorded as a trajectory repair, not a silent retry
    # or a full restart. Use HTN when the decomposition is known and auditable.
    app = ContextApp(name="triage", provider=example_provider()[0], model="mock-1")

    def enrich_primary(alert: str = "") -> dict:
        """Primary enrichment service (currently failing)."""
        raise RuntimeError("enrichment upstream 503")

    def enrich_backup(alert: str = "") -> dict:
        """Backup enrichment service."""
        return {"alert": alert, "owner": "db-team", "severity": "high"}

    app.add_tool(enrich_primary)
    app.add_tool(enrich_backup)
    domain = (
        HTNDomain()
        .method("root", ["assess", "respond"])
        .method("assess", ["classify", "enrich"], ordering="parallel")
        .operator("classify", step_type="think", instruction="classify the severity")
        .operator("enrich", step_type="tool", tool_name="enrich_primary",
                  instruction="enrich the alert", fallbacks=["enrich_backup"])  # declared fallback
        .operator("respond", step_type="finalize", instruction="recommend a response")
    )
    state = app.agent(tools=["enrich_primary", "enrich_backup"], planner="hierarchical",
                      domain=domain, max_steps=10).run("Triage the database CPU alert")
    print(f"4. HTN + repair: {state.termination_reason} | repairs",
          [f"{r.from_binding}->{r.to_binding}" for r in state.repairs])


def section_cost_aware() -> None:
    # Given a cheapest->strongest model ladder, the agent reads pricing/capability
    # from the model registry and spends the cheap model on easy steps, escalating
    # only when needed — cost-aware action selection instead of a fixed model.
    app = ContextApp(name="summarize", provider=example_provider()[0], model="mock-1")
    state = app.agent(cost_aware_models=["gpt-5.2-mini", "gpt-5.2"]).run("Summarize the incident report")
    picks = [s["model"] for s in state.working_memory.get("_selections", [])]
    print("5. cost-aware model picks:", picks or "(no model-bearing steps this run)")


def section_research() -> None:
    # app.research runs the full search -> read -> reflect -> verify -> synthesize
    # loop over the app's sources under an explicit breadth/depth/source budget and
    # returns a cited, eval-scored report — every claim cites an evidence id.
    provider, model = example_provider(
        default_responder=citing_responder("The Pro plan refund window is 30 days. [{ref}]"))
    app = ContextApp(name="researcher", provider=provider, model=model)
    app.add_source("kb", documents=[Document(id="refunds", title="Refund Policy",
                   text="The refund window for the Pro plan is 30 days from purchase. "
                        "Enterprise customers get a 60-day window.")])
    report = app.research("What is the refund window for the Pro plan?")
    print(f"6. research: {report.answer[:70]!r} | {len(report.sources)} sources, "
          f"citation coverage {report.metrics['citation_coverage']:.0%}, "
          f"grounding {report.metrics['grounding']:.0%}")


async def section_computer_use() -> None:
    # The action plane drives a screen SAFELY: it perceives typed UI elements (not
    # pixels), grounds an intent to a stable role+name selector, PRE-GATES the
    # action against an ActionPolicy, acts, then post-verifies. A destructive or
    # out-of-scope action is gated like a write tool. Offline against an in-process
    # WebArena-shaped app; a real browser sits behind vincio[computer-use].
    app = ContextApp(name="operator", provider=example_provider()[0])
    spec, task = build_web_checkout()
    address, checkout, place, delete = (
        "role=textbox[name='Address']", "role=button[name='Checkout']",
        "role=button[name='Place order']", "role=button[name='Delete account']")

    env = app.computer_use(screen=spec, policy=ActionPolicy(allow_urls=["https://shop.test"]),
                           approve=lambda action, decision: "Place order" in action.selector)

    def policy(state):  # map the perceived screen to the next grounded action
        s = state.state
        if s["screen"] == "cart" and not s["fields"].get(address):
            return UIAction(kind="type", selector=address, text="1 Main St")
        if s["screen"] == "cart":
            return UIAction(kind="click", selector=checkout, expect_change=True)
        if s["screen"] == "review" and not s["flags"].get("order_placed"):
            return UIAction(kind="click", selector=place)
        return None

    run = env.run(policy, task)
    guard = app.computer_use(screen=build_web_checkout()[0],
                             policy=ActionPolicy(allow_urls=["https://shop.test"]))
    blocked = await guard.act(UIAction(kind="click", selector=delete))  # unapproved destructive
    print(f"7. computer-use: success={run.success} safe={run.safe} steps={run.steps_taken} | "
          f"'Delete account' gated={blocked.gated} performed={blocked.performed} | "
          f"audit chain intact={app.audit.verify_chain()}")


async def main() -> None:
    section_tool_registry()
    section_approval_gated_writes()
    await section_sandbox()
    section_hierarchical_repair()
    section_cost_aware()
    section_research()
    await section_computer_use()


if __name__ == "__main__":
    asyncio.run(main())
