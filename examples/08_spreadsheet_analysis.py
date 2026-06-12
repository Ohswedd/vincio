"""Spreadsheet/CSV analysis: table-aware chunking with schema inference."""

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider

from vincio import ContextApp

csv_dir = Path(tempfile.mkdtemp()) / "data"
csv_dir.mkdir(parents=True)
(csv_dir / "revenue.csv").write_text(
    "quarter,region,revenue,growth\n"
    "Q1,EMEA,120000,0.04\nQ1,AMER,200000,0.07\n"
    "Q2,EMEA,131000,0.09\nQ2,AMER,210000,0.05\n",
)

provider, model = example_provider(
    citing_responder("EMEA grew fastest in Q2 at 9% quarter-over-quarter. [{ref}]")
)

app = ContextApp(name="spreadsheet_analysis", provider=provider, model=model)
app.add_source("data", path=str(csv_dir), chunking="table_aware", retrieval="hybrid")

if __name__ == "__main__":
    from vincio.documents import load_document

    document = load_document(csv_dir / "revenue.csv")
    print("inferred schema:", document.tables[0]["inferred_schema"])
    print("quality:", document.tables[0]["quality"])

    result = app.run("Which region grew fastest in Q2?")
    print("answer:", result.output)
