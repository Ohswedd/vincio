"""Verified reasoning & neuro-symbolic certificates.

The platform grades outputs with judges, oracles, and a governance verifier — but
those signals are *probabilistic*. For the classes of question where it is possible,
an answer can instead carry a **checkable certificate** a deterministic verifier
confirms independently of the model. This example shows the three planes of that
discipline, all offline and deterministic (no model call):

  1. **Proof-carrying answers** — `app.verify_reasoning(...)` runs deterministic
     kernels (arithmetic, units, temporal, schema, constraints, citation entailment)
     and *refuses to emit* a refuted answer; a `regenerate` callback drives the
     bounded self-correction loop to repair it.
  2. **Runtime verification & shielding** — a `BehaviorSpec` states a property over an
     agent's trajectory, and a `Shield` wired into the tool runtime blocks a
     policy-violating action (an unapproved write) *before* it executes.
  3. **Verified tool use & synthesized programs** — a `ToolContract` enforces
     pre/post-conditions against the actual call, and `app.synthesize_program(...)`
     emits a proof-carrying transform whose properties are checked before it runs.

This is a library capability inside your process — never a hosted prover service.
"""

from __future__ import annotations

import asyncio

from vincio import (
    BehaviorSpec,
    ContextApp,
    EventPattern,
    ProgramOp,
    ProgramProperty,
    ProgramSpec,
    ToolContract,
)
from vincio.core.errors import ToolContractError
from vincio.core.types import EvidenceItem, ToolCall
from vincio.providers import MockProvider


async def main() -> None:
    app = ContextApp(name="verified-solver", provider=MockProvider(default_text="ok"))

    # 1. Proof-carrying answers. A wrong arithmetic claim is refuted and refused.
    refuted = app.verify_reasoning("The order total is 2 + 2 = 5 items.")
    print(f"1. Refuted answer holds={refuted.holds}, refused={refuted.refused}")
    print("   " + refuted.certificate.render().replace("\n", "\n   "))

    #    Drive the bounded self-correction loop with a deterministic repair.
    repaired = app.verify_reasoning("2 + 2 = 5", regenerate=lambda ans, critique: "2 + 2 = 4")
    print(f"   self-corrected: holds={repaired.holds} after {repaired.attempts} attempt(s)")

    #    Ground a citation check: an uncited numeric claim is refuted.
    evidence = [EvidenceItem(source_id="POLICY", text="The refund window is 30 days.")]
    cited = app.verify_reasoning("The refund window is 30 days.", evidence=evidence)
    contradicted = app.verify_reasoning("The refund window is 90 days.", evidence=evidence)
    print(f"   citation: supported holds={cited.holds}, contradicted holds={contradicted.holds}")

    # 2. Runtime shielding. A behavior spec forbids an unapproved write tool; the shield
    #    blocks the action before it executes.
    def delete_account(account_id: str) -> dict:
        return {"deleted": account_id}

    app.add_tool(delete_account, side_effects="write")
    app.shield(
        BehaviorSpec(
            name="approval-before-write",
            forbid=[EventPattern(kind="tool_call",
                                 where={"side_effects": "write", "approved": False})],
        ),
        use=True,
    )
    blocked = await app.tool_runtime.execute(
        ToolCall(tool_name="delete_account", arguments={"account_id": "acct-7"}))
    approved = await app.tool_runtime.execute(
        ToolCall(tool_name="delete_account", arguments={"account_id": "acct-7"}), approved=True)
    print(f"2. Shield: unapproved write status={blocked.status!r}; approved status={approved.status!r}")

    # 3a. Verified tool use — a contract on behaviour, not just schema.
    def charge(amount: float) -> dict:
        return {"amount": amount}

    app.add_tool(
        charge, side_effects="write",
        contract=ToolContract().requires_that("amount > 0", lambda a: a["amount"] > 0),
    )
    try:
        await app.tool_runtime.execute(
            ToolCall(tool_name="charge", arguments={"amount": -10}), approved=True)
    except ToolContractError as exc:
        print(f"3. Tool contract refused an out-of-contract call: {exc}")

    # 3b. A proof-carrying synthesized program — properties proven before it runs.
    program = app.synthesize_program(
        ProgramSpec(
            name="line-total",
            ops=[ProgramOp(op="derive", field="total", expr="price * quantity")],
            properties=[
                ProgramProperty(kind="row_count", relation="preserved"),
                ProgramProperty(kind="field_nonnegative", field="total"),
            ],
        ),
        examples=[{"price": 3.0, "quantity": 2}, {"price": 5.0, "quantity": 4}],
    )
    rows = program.run([{"price": 2.0, "quantity": 10}])
    print(f"   synthesized program holds={program.holds}; run -> {rows}")

    print(f"\nAudit chain: {len(app.audit.entries)} entries, verifies={app.audit.verify_chain()}")


if __name__ == "__main__":
    asyncio.run(main())
