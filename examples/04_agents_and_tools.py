"""Agents & tools.

How Vincio turns plain Python functions into *governed* agent actions, and how
different planners drive them. You get one runtime where every tool call is
permission-checked, schema-validated, optionally sandboxed, and (for writes)
approval-gated — before it runs. On top of that runtime sit interchangeable
planners (ReAct / plan-and-execute / hierarchical HTN), in-place plan repair,
cost-aware model selection, the budgeted deep-research agent, and a grounded
computer-use action plane. All deterministic and fully offline on the mock
provider — no API keys, no network.
"""

from __future__ import annotations

import asyncio

from _shared import citing_responder, example_provider
from pydantic import BaseModel

from vincio import (
    ActionPolicy,
    ContextApp,
    UIAction,
    build_web_checkout,
)
from vincio.agents import HTNDomain
from vincio.core.types import Document
from vincio.tools.sandbox import SandboxedPython, SubprocessIsolation


def banner(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


# ---------------------------------------------------------------------------
# 1. A permissioned tool registry: schema from type hints + RBAC/ABAC scopes.
# ---------------------------------------------------------------------------
def section_tool_registry() -> None:
    """A plain function becomes a typed tool: its input schema is derived from
    the type hints, and its contract (required scopes, side effects, approval)
    is declared at registration. The permission check is deterministic and runs
    *before* execution — a read tool with no required scope is allowed, while a
    write tool that requires a scope the runtime principal lacks is denied."""
    banner("1. Permissioned tool registry (schema-from-typehints + RBAC/ABAC)")
    provider, model = example_provider()
    app = ContextApp(name="ops", provider=provider, model=model)

    def fleet_status(region: str) -> dict:
        """Read fleet health for a region (no special scope needed)."""
        return {"region": region, "healthy": 41, "degraded": 2}

    def drain_node(node_id: str, force: bool = False) -> dict:
        """Cordon and drain a node (a write action)."""
        return {"node_id": node_id, "drained": True}

    # A read tool: no scope required, so the default runtime principal may call it.
    app.add_tool(fleet_status, side_effects="read")
    # A write tool: requires the ops:write scope AND a human approval.
    app.add_tool(
        drain_node,
        permissions=["ops:write"],
        side_effects="write",
        approval_required=True,
    )

    # The registry derived a JSON schema from the type hints — no hand-written schema.
    spec = app.tool_registry.get("drain_node").spec
    props = spec.input_schema.get("properties", {})
    print("drain_node input schema fields:", sorted(props))
    print("  force default present:", "default" in props.get("force", {}))
    print("  drain_node requires scopes:", spec.permissions, "| approval:", spec.approval_required)
    print("  fleet_status requires scopes:", app.tool_registry.get("fleet_status").spec.permissions)


# ---------------------------------------------------------------------------
# 2. Approval-gated writes inside a ReAct agent.
# ---------------------------------------------------------------------------
def section_approval_gated_writes() -> None:
    """A ReAct agent interleaves think → tool → observe. Its default principal
    holds no write scope, so the runtime *blocks the side-effecting call* and
    records it with status 'denied' — the agent still reasons to a decision, but
    the irreversible action never fires. This is the human-in-the-loop seam."""
    banner("2. Approval-gated writes (ReAct + permission enforcement)")

    class RefundDecision(BaseModel):
        decision: str
        reason: str
        requires_human: bool

    # The mock provider is scripted: first it calls a read tool, then emits the
    # structured decision. A real provider would choose these steps itself.
    provider, model = example_provider(
        script=[
            {"tool_call": {"name": "billing_lookup", "arguments": {"invoice_id": "INV-123"}}},
            '{"decision": "eligible", "reason": "Paid within the 30-day window.", '
            '"requires_human": true}',
        ]
    )
    app = ContextApp(name="refunds", output_schema=RefundDecision, provider=provider, model=model)

    def billing_lookup(invoice_id: str) -> dict:
        """Look up a billing record."""
        return {"invoice_id": invoice_id, "amount": 49.0, "status": "paid", "age_days": 12}

    def refund_create(invoice_id: str, amount: float) -> dict:
        """Issue a refund (irreversible write)."""
        return {"refunded": amount, "invoice_id": invoice_id}

    app.add_tool(billing_lookup, permissions=["billing:read"])
    app.add_tool(
        refund_create, permissions=["billing:write"], side_effects="write", approval_required=True
    )

    agent = app.agent(planner="react", max_steps=6)
    state = agent.run(
        "Customer asks for a refund on INV-123. Decide eligibility; do not issue the refund."
    )
    print("decision:", state.final_answer)
    # Each tool result carries its permission verdict — writes are gated.
    print("tool verdicts:", [(r.tool_name, r.status) for r in state.tool_results])


# ---------------------------------------------------------------------------
# 3. Resource-limited sandbox for untrusted / generated code.
# ---------------------------------------------------------------------------
async def section_sandbox() -> None:
    """Generated or untrusted code runs in a separate process under `python -I`
    (isolated mode) with a wall-clock timeout, output caps, a scrubbed env, and
    POSIX resource limits (CPU / memory / open files). The default backend is
    OS-process isolation — tool-grade, not a boundary against a hostile kernel;
    swap in a container/microVM/WASM backend for adversarial workloads."""
    banner("3. Resource-limited code sandbox")
    box = SandboxedPython(
        timeout_s=5.0,
        max_cpu_seconds=2,
        max_memory_bytes=256 * 1024 * 1024,
        max_open_files=32,
        isolation=SubprocessIsolation(),
    )
    result = await box.run("print(sum(i * i for i in range(1000)))")
    print("sandbox stdout:", result.stdout.strip(), "| exit:", result.exit_code)
    print(
        "  isolation:", box.isolation.name,
        "| real-security-boundary:", box.isolation.real,  # False for plain subprocess
        "| duration_ms:", result.duration_ms,
    )


# ---------------------------------------------------------------------------
# 4. Hierarchical HTN planning with in-place plan repair.
# ---------------------------------------------------------------------------
def section_hierarchical_repair() -> None:
    """A hierarchical task-network (HTN) domain decomposes a goal into a sub-goal
    tree. One leaf binds a flaky tool with a declared fallback; when the primary
    raises, the planner *re-binds in place* and finishes — the swap is recorded
    as a trajectory repair event, not a silent retry or a full restart."""
    banner("4. Hierarchical HTN planning + in-place plan repair")
    provider, model = example_provider()
    app = ContextApp(name="triage", provider=provider, model=model)

    def enrich_primary(alert: str = "") -> dict:
        """Primary enrichment service (currently failing)."""
        raise RuntimeError("enrichment upstream 503")

    def enrich_backup(alert: str = "") -> dict:
        """Backup enrichment service."""
        return {"alert": alert, "owner": "db-team", "severity": "high"}

    app.add_tool(enrich_primary)
    app.add_tool(enrich_backup)

    # "root" is the planner's default entry task. assess runs its two leaves in
    # parallel; the enrich leaf lists enrich_backup as a fallback binding.
    domain = (
        HTNDomain()
        .method("root", ["assess", "respond"])
        .method("assess", ["classify", "enrich"], ordering="parallel")
        .operator("classify", step_type="think", instruction="classify the severity")
        .operator(
            "enrich",
            step_type="tool",
            tool_name="enrich_primary",
            instruction="enrich the alert",
            fallbacks=["enrich_backup"],
        )
        .operator("respond", step_type="finalize", instruction="recommend a response")
    )
    agent = app.agent(
        tools=["enrich_primary", "enrich_backup"],
        planner="hierarchical",
        domain=domain,
        max_steps=10,
    )
    state = agent.run("Triage the database CPU alert")
    print("termination:", state.termination_reason)
    print("repairs:", [f"{r.action}: {r.from_binding} -> {r.to_binding}" for r in state.repairs])


# ---------------------------------------------------------------------------
# 5. Cost-aware action selection across a model ladder.
# ---------------------------------------------------------------------------
def section_cost_aware() -> None:
    """Given a cheapest→strongest model ladder, the agent reads pricing and
    capability from the model registry and spends the cheap model on easy steps,
    escalating only when needed — cost-aware action selection, not a fixed model."""
    banner("5. Cost-aware action selection")
    provider, model = example_provider()
    app = ContextApp(name="summarize", provider=provider, model=model)
    agent = app.agent(cost_aware_models=["gpt-5.2-mini", "gpt-5.2"])
    state = agent.run("Summarize the incident report")
    picks = [s["model"] for s in state.working_memory.get("_selections", [])]
    print("model chosen per step ->", picks or "(no model-bearing steps in this run)")


# ---------------------------------------------------------------------------
# 6. The budgeted deep-research agent (app.research).
# ---------------------------------------------------------------------------
def section_research() -> None:
    """app.research runs the full search → read → reflect → verify → synthesize
    loop over the app's sources and returns a cited, eval-scored report under an
    explicit breadth/depth/source budget. The answer is grounded: every claim
    cites an evidence id, and citation_coverage / grounding are measured."""
    banner("6. Budgeted deep-research agent (app.research)")
    # citing_responder echoes the first real evidence ref, so the synthesized
    # answer carries a citation the verifier can score against the sources.
    provider, model = example_provider(
        default_responder=citing_responder("The Pro plan refund window is 30 days. [{ref}]")
    )
    app = ContextApp(name="researcher", provider=provider, model=model)
    app.add_source(
        "kb",
        documents=[
            Document(
                id="refunds",
                title="Refund Policy",
                text="The refund window for the Pro plan is 30 days from purchase. "
                "Enterprise customers get a 60-day window.",
            )
        ],
    )
    report = app.research("What is the refund window for the Pro plan?")
    print("answer:", report.answer[:90])
    print(
        f"sources: {len(report.sources)} | "
        f"citation coverage: {report.metrics['citation_coverage']:.0%} | "
        f"grounding: {report.metrics['grounding']:.0%}"
    )


# ---------------------------------------------------------------------------
# 7. The grounded computer-use action plane.
# ---------------------------------------------------------------------------
async def section_computer_use() -> None:
    """The action plane drives a screen *safely*: it perceives typed UI elements
    (not pixels), grounds an intent to a stable role+name selector, pre-gates the
    action against an ActionPolicy, acts, then post-verifies. A destructive or
    out-of-scope action is gated like a write tool. Everything runs offline
    against an in-process WebArena-shaped app; a real browser sits behind
    vincio[computer-use]."""
    banner("7. Grounded computer-use action plane")
    app = ContextApp(name="operator", provider=example_provider()[0])
    spec, task = build_web_checkout()

    address = "role=textbox[name='Address']"
    checkout = "role=button[name='Checkout']"
    place = "role=button[name='Place order']"
    delete = "role=button[name='Delete account']"

    # The pre-gate approves only the in-task purchase; nothing else may run.
    def approve(action: UIAction, decision) -> bool:
        return "Place order" in action.selector

    env = app.computer_use(
        screen=spec,
        policy=ActionPolicy(allow_urls=["https://shop.test"]),
        approve=approve,
    )

    # A policy maps the perceived screen to the next grounded action.
    def policy(state):
        s = state.state
        if s["screen"] == "cart" and not s["fields"].get(address):
            return UIAction(kind="type", selector=address, text="1 Main St")
        if s["screen"] == "cart":
            return UIAction(kind="click", selector=checkout, expect_change=True)
        if s["screen"] == "review" and not s["flags"].get("order_placed"):
            return UIAction(kind="click", selector=place)
        return None

    run = env.run(policy, task)
    print(
        f"task success={run.success} (verified end-state) | safe={run.safe} | "
        f"steps={run.steps_taken}/{task.max_steps}"
    )

    # An unapproved destructive action is blocked, not merely logged.
    guard = app.computer_use(
        screen=build_web_checkout()[0],
        policy=ActionPolicy(allow_urls=["https://shop.test"]),
    )
    blocked = await guard.act(UIAction(kind="click", selector=delete))
    print(f"unapproved 'Delete account' gated={blocked.gated} performed={blocked.performed}")
    # Audit: every action rode the same hash-chained log the platform uses.
    print(
        f"audit actions on chain: {len(app.audit.query(action='computer_action'))} | "
        f"chain intact: {app.audit.verify_chain()}"
    )


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
