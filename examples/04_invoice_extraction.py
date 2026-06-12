"""Invoice field extraction with extraction-F1 evaluation."""

from _shared import example_provider, json_responder
from pydantic import BaseModel

from vincio import ContextApp, Dataset
from vincio.evals import EvalCase


class InvoiceFields(BaseModel):
    invoice_id: str
    date: str
    amount: float
    vendor: str


provider, model = example_provider(
    json_responder({"invoice_id": "INV-1001", "date": "2026-01-15", "amount": 250.0, "vendor": "Acme Corp"})
)

app = ContextApp(name="invoice_extraction", output_schema=InvoiceFields, provider=provider, model=model)
app.configure(
    role="invoice_field_extraction_engine",
    objective="Extract structured fields from invoice text",
    rules=["Extract values exactly as written; normalize dates to YYYY-MM-DD."],
)

if __name__ == "__main__":
    result = app.run("Invoice INV-1001 dated 2026-01-15 for $250.00 from Acme Corp")
    print("extracted:", result.output.model_dump())

    dataset = Dataset(
        name="invoices",
        cases=[
            EvalCase(
                id="inv1",
                input="Invoice INV-1001 dated 2026-01-15 for $250.00 from Acme Corp",
                expected={"invoice_id": "INV-1001", "date": "2026-01-15", "amount": 250.0, "vendor": "Acme Corp"},
            )
        ],
    )
    report = app.evaluate(dataset, metrics=["extraction_f1", "schema_validity", "cost"])
    report.print_summary()
