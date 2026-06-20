"""Cookbook recipe — data-room Q&A (due diligence).

Answer diligence questions over a virtual data room with citations back to the
exact documents. Sources are grounded and answers must cite, so a reviewer can
trace every claim to a file — the heart of M&A / investment diligence.
"""

from _shared import citing_responder, example_provider

from vincio import ContextApp
from vincio.core.types import Document

DATA_ROOM = [
    Document(id="fin-2023", title="FY2023 financials",
             text="FY2023 revenue was $42.0M, up 28% year over year. Gross margin was 71%."),
    Document(id="cap-table", title="Cap table",
             text="The company has raised $30M across seed, Series A, and Series B. "
                  "Founders retain 48% fully diluted."),
    Document(id="contracts", title="Top customer contracts",
             text="The top customer accounts for 9% of ARR on a 3-year contract with auto-renewal."),
]

provider, model = example_provider(
    citing_responder("FY2023 revenue was $42.0M, up 28% YoY, at 71% gross margin. [{ref}]")
)
app = ContextApp(name="data_room", provider=provider, model=model)
app.add_source("room", documents=DATA_ROOM, retrieval="hybrid")
app.set_policy("answer_only_from_sources", True).set_policy("require_citations", True)


if __name__ == "__main__":
    questions = [
        "What was FY2023 revenue and growth?",
        "How much has the company raised in total?",
        "What is the customer concentration risk?",
    ]
    for q in questions:
        result = app.run(q)
        print("Q:", q)
        print("A:", result.output if isinstance(result.output, str) else result.raw_text)
        print("   citations:", result.citations or "—")
        print()
