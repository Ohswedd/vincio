"""Structured-extraction service — pure Vincio logic, no web framework.

This module is the heart of the service and deliberately knows nothing about
HTTP. It builds a :class:`~vincio.ContextApp` whose output contract is the
:class:`Invoice` Pydantic model, turns bounded self-correction on (so a reply
that comes back mis-shaped is repaired rather than thrown away), and exposes a
single plain function :func:`extract` that takes free text and returns a
validated invoice as a JSON-able ``dict``.

Runs fully offline on the deterministic mock provider — no API keys, no
network. Set ``VINCIO_PROVIDER`` (and the matching key) to extract with a real
model instead; nothing else changes.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from vincio import ContextApp
from vincio.providers import MockProvider, build_provider


def _provider():
    """Offline mock by default; a real provider when VINCIO_PROVIDER is set."""
    name = os.environ.get("VINCIO_PROVIDER", "mock")
    if name == "mock":
        return MockProvider(), "mock-1"
    return build_provider(name), os.environ.get("VINCIO_MODEL", "gpt-4o-mini")


class Invoice(BaseModel):
    """The structured shape every extraction must conform to.

    This Pydantic model *is* the contract: its JSON schema rides the provider's
    structured-output path, and the model's reply is parsed and validated back
    into an ``Invoice`` instance before it ever leaves the service.
    """

    vendor: str = Field(description="The name of the company that issued the invoice.")
    total: float = Field(description="The invoice grand total as a number.")
    currency: str = Field(description="The three-letter currency code, e.g. USD.")
    line_items: list[str] = Field(
        default_factory=list,
        description="Short descriptions of the goods or services billed.",
    )


def build_app() -> ContextApp:
    """Construct the extraction app: Invoice output contract + repair loop.

    ``output_schema=Invoice`` makes the app parse and validate every reply into
    an ``Invoice``. ``enable_self_correction`` adds a bounded validate ->
    critique -> repair loop: if the first reply is mis-shaped in a way the
    structural parser cannot fix alone, the app re-prompts with the specific
    errors. The repair is structure-only (it never invents facts) and hard-
    capped by ``max_cycles`` / ``max_cost_usd`` so it cannot spin forever.
    """
    provider, model = _provider()
    app = ContextApp(
        name="invoice_extractor",
        provider=provider,
        model=model,
        output_schema=Invoice,
    )
    app.configure(
        role="invoice_extraction_engine",
        objective="Extract a structured invoice from raw document text.",
        rules=[
            "Only use facts present in the supplied text; never invent values.",
            "Return the total as a number and the currency as a 3-letter code.",
        ],
    )
    app.enable_self_correction(max_cycles=2, max_cost_usd=0.05)
    return app


def extract(text: str) -> dict:
    """Extract a structured invoice from ``text`` and return it as a dict.

    Raises ``ValueError`` on empty input. Any provider/validation failure that
    survives the self-correction loop surfaces the run's error message as a
    ``RuntimeError`` so the HTTP shell can translate it into a clean response.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    app = build_app()
    result = app.run(text)

    invoice = result.output
    if not isinstance(invoice, Invoice):
        # The contract was not satisfied even after repair — surface why.
        detail = getattr(result, "error", None) or f"status={result.status.value}"
        raise RuntimeError(f"extraction failed: {detail}")

    return invoice.model_dump()


if __name__ == "__main__":
    import json

    sample = "Invoice from Acme Corp, total 1200.50 USD for widgets and gadgets"
    print(json.dumps(extract(sample), indent=2))
