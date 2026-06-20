"""Cookbook recipe — contract redlining.

Review a contract clause-by-clause with the ``legal`` pack, propose safer
language, and emit a tracked-change **redline** (markdown here; DOCX with
``pip install "vincio[gen-docx]"``). The review is grounded in the contract text;
the redline is a deterministic diff of original vs. revised.
"""

from _shared import example_provider, json_responder

from vincio import ContextApp, generate_redline

ORIGINAL = (
    "The Agreement renews automatically for successive 24-month terms unless either "
    "party gives notice 90 days before renewal. Liability is unlimited."
)
# What the reviewer recommends (offline, the mock returns this verbatim).
REVIEW = {
    "clause": "auto-renewal and liability",
    "risk_level": "high",
    "rationale": "A 90-day notice window plus unlimited liability is one-sided.",
    "recommendation": "Shorten the renewal term, widen the notice window, and cap liability.",
}
REVISED = (
    "The Agreement renews for successive 12-month terms unless either party gives "
    "notice 30 days before renewal. Liability is capped at fees paid in the prior 12 months."
)

provider, model = example_provider(json_responder(REVIEW))
app = ContextApp(name="contract_redline", provider=provider, model=model).use_pack("legal")
app.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)


if __name__ == "__main__":
    result = app.run(f"Review this clause and flag risk:\n{ORIGINAL}")
    review = result.output
    review = review.model_dump() if hasattr(review, "model_dump") else review
    print("risk:", review["risk_level"], "—", review["recommendation"])

    redline = generate_redline(ORIGINAL, REVISED, format="markdown", title="MSA — renewal & liability")
    content = redline.content
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    print("\nredline (", redline.format, "):", sep="")
    print(content[:400])
