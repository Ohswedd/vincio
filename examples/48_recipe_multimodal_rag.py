"""Cookbook recipe — multimodal RAG over slides & PDFs.

Text, a table, and a slide image are all first-class candidates in **one scored,
budgeted context packet**. The context compiler scores and orders them together
with modality-aware token cost, so an answer can draw on a chart slide and a
metrics table alongside prose — and cite each.
"""

from _shared import example_provider

from vincio import ContextApp
from vincio.core.types import EvidenceItem, ImageRef

# A mixed-modality evidence set, as if extracted from a slide deck + a PDF.
EVIDENCE = [
    EvidenceItem(
        id="deck:slide3", source_id="Q3_deck", modality="text",
        text="Q3 net revenue retention reached 121%, the highest in eight quarters.",
        relevance=0.9, authority=0.8,
    ),
    EvidenceItem(
        id="deck:slide7", source_id="Q3_deck", modality="image",
        image=ImageRef(url="https://example.com/slides/nrr_chart.png", detail="high"),
        text="Slide 7: NRR trend chart, Q4 last year through Q3 this year.",
        relevance=0.7, authority=0.7,
    ),
    EvidenceItem(
        id="pdf:p12", source_id="Q3_report", modality="table",
        table={"columns": ["segment", "nrr"], "rows": [["Enterprise", "129%"], ["SMB", "104%"]]},
        text="Table 4: NRR by segment.",
        relevance=0.85, authority=0.9,
    ),
]

# Cite a real evidence id so the citation validates against the packet.
provider, model = example_provider(
    lambda request: "Q3 NRR was 121%, led by Enterprise at 129%. [pdf:p12]"
)
app = ContextApp(name="multimodal_rag", provider=provider, model=model)
# Inject the mixed-modality evidence for this run (in production these come from
# slide/PDF loaders via app.add_source).
app.pending_evidence = EVIDENCE
app.set_policy("require_citations", True)


if __name__ == "__main__":
    result = app.run("What was Q3 net revenue retention, and which segment led?")
    print("answer  :", result.output if isinstance(result.output, str) else result.raw_text)
    print("citations:", result.citations or "—")
    # The packet scored all three modalities together.
    kept = {(e.modality) for e in result.evidence}
    print("modalities in packet:", sorted(kept))
