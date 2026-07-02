"""The packet compile receipt — prove *why* a packet was compiled, text-light.

Vincio already treats the context packet as the governed boundary and traces the
stages that produce it. This tour is the compact operational artifact a reviewer
wants when a run is surprising: a **compile receipt** — a fingerprint-heavy,
text-light manifest of the compile decision (what was included and why, what was
excluded and why, which conflicts were resolved and by which rule, the budget and
privacy posture) that carries no raw prompt or evidence text, so it is safe to
attach to a pull request or an incident.

Sections:
  1. Run an app and read the receipt linked from its trace.
  2. Inspect the decision: included, excluded/superseded, conflicts, budget.
  3. Verify it re-derives from its own bytes and carries no raw text.
  4. Diff a changed compile against a baseline — an explicit divergence.

Runs fully offline on the bundled mock provider.
"""

from __future__ import annotations

import json

from _shared import example_provider

from vincio import ContextApp
from vincio.context import CompileReceipt
from vincio.core.types import EvidenceItem


def _evidence(window: str) -> list[EvidenceItem]:
    """A current, authoritative refund policy plus an older, lower-authority,
    contradictory one and an irrelevant note."""
    return [
        EvidenceItem(
            id="D1",
            source_id="refunds.md",
            text=f"Refunds are allowed within {window} of purchase.",
            authority=0.9,
            relevance=0.95,
            page=2,
        ),
        EvidenceItem(
            id="D9",
            source_id="refunds_old.md",
            text="Refunds are allowed within 14 days of purchase.",
            authority=0.4,
            relevance=0.9,
        ),
        EvidenceItem(
            id="D3",
            source_id="misc.md",
            text="Bananas are rich in potassium.",
            authority=0.5,
            relevance=0.01,
        ),
    ]


def main() -> None:
    provider, model = example_provider()
    app = ContextApp(name="compile_receipt", provider=provider, model=model)

    # 1. Run the app. The current policy (30 days) contradicts an older one
    #    (14 days); the compiler keeps the authoritative source and supersedes
    #    the contradictory one. Every run links a receipt from its trace.
    app.pending_evidence = _evidence("30 days")
    result = app.run("What is the refund window?")

    receipt_dict = result.metadata["compile_receipt"]
    receipt = CompileReceipt.model_validate(receipt_dict)
    print("1. Receipt linked from run", result.run_id)
    print("   receipt_hash:", receipt.receipt_hash)
    print("   trace_id:", receipt.trace_id, " packet_id:", receipt.packet_id)
    print("   render:", receipt.render.provider, receipt.render.model)

    # 2. The compile decision.
    print("\n2. Compile decision")
    print("   included:", [(it.id, it.reason, round(it.score or 0, 3)) for it in receipt.included])
    print("   excluded:", [(it.id, it.reason, it.superseded_by) for it in receipt.excluded])
    print("   conflicts:", [(c.winner, c.loser, c.rule) for c in receipt.conflicts])
    print(
        "   budget:",
        receipt.budget.used_tokens,
        "/",
        receipt.budget.max_input_tokens,
        "tokens",
    )
    print("   privacy:", receipt.privacy.privacy_scope, "omitted_raw_text=", receipt.privacy.omitted_raw_text)

    # 3. Self-verifying and text-light.
    blob = json.dumps(receipt.to_export())
    no_raw_text = all(s not in blob for s in ("Refunds are allowed", "Bananas", "potassium"))
    print("\n3. verify():", receipt.verify(), " carries no raw text:", no_raw_text)

    # 4. Diff a changed compile against this baseline — the source now says 45
    #    days, so the receipt diverges explicitly rather than silently.
    app.pending_evidence = _evidence("45 days")
    changed = CompileReceipt.model_validate(
        app.run("What is the refund window?").metadata["compile_receipt"]
    )
    print("\n4. Divergence vs baseline")
    if changed.receipt_hash == receipt.receipt_hash:
        print("   compile decision unchanged")
    else:
        divergence = changed.diverges_from(receipt)
        print("   used_tokens_delta:", divergence["used_tokens_delta"])
        print("   score_changes:", divergence["score_changes"])
        print("   render_changed:", divergence["render_changed"])

    print("\nDone — the receipt explains the compile without exposing any text.")


if __name__ == "__main__":
    main()
