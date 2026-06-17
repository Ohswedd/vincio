"""2.0 multimodal-native Context Packet: image and table evidence are
first-class candidates the compiler scores, budgets, orders, and cites
alongside text; slim packets materialize cross-process from a content-addressed
store; the evidence ledger carries entailment links."""

from __future__ import annotations

from vincio.context.compression import distill_evidence_ledger, link_entailments
from vincio.context.evidence_store import (
    BlobEvidenceStore,
    InMemoryEvidenceStore,
    content_hash,
)
from vincio.context.ir import ContextIR
from vincio.context.packet import ContextPacket
from vincio.context.scoring import ContextCandidate, ContextScorer
from vincio.core.types import EvidenceItem, ImageRef, Objective
from vincio.storage.base import FileBlobStore

# -- typed modality on evidence --------------------------------------------


def test_evidence_modalities_and_scorable_text():
    text_ev = EvidenceItem(source_id="d1", modality="text", text="Refunds within 30 days.")
    image_ev = EvidenceItem(
        source_id="d2",
        modality="image",
        source_type="image",
        image=ImageRef(path="/x/chart.png", detail="high", metadata={"caption": "Q3 revenue chart"}),
    )
    table_ev = EvidenceItem(
        source_id="d3",
        modality="table",
        table={"columns": ["q", "rev"], "rows": [["Q1", 10], ["Q2", 12]], "markdown": "| q | rev |"},
    )
    assert text_ev.scorable_text == "Refunds within 30 days."
    assert image_ev.scorable_text == "Q3 revenue chart"
    assert "rev" in table_ev.scorable_text
    # Modality-aware token cost: high-detail image is the calibrated budget.
    assert image_ev.estimated_token_cost() == 765
    assert table_ev.estimated_token_cost() > 0


def test_scorer_modality_token_cost():
    scorer = ContextScorer(max_token_cost=1000)
    img = ContextCandidate(
        id="c", type="evidence", content="caption", modality="image",
        image=ImageRef(path="/x.png", detail="low"),
    )
    # The image competes on its real footprint (85), not the 1-token caption.
    assert scorer.modality_token_cost(img) == 85
    assert 0.0 < scorer.normalized_token_cost(img) <= 1.0


# -- multimodal candidates from the compiler -------------------------------


async def test_compiler_selects_image_and_table_evidence():
    from vincio.context.compiler import ContextCompiler

    compiler = ContextCompiler()
    evidence = [
        EvidenceItem(source_id="d1", text="The annual fee is $99.", relevance=0.9),
        EvidenceItem(
            source_id="d2", modality="image", source_type="image", relevance=0.8,
            image=ImageRef(path="/pricing.png", metadata={"caption": "annual fee pricing table image"}),
        ),
        EvidenceItem(
            source_id="d3", modality="table", relevance=0.7,
            table={"columns": ["plan", "fee"], "rows": [["pro", 99]], "markdown": "annual fee pro 99"},
        ),
    ]
    candidates = compiler._collect(evidence=evidence, memory=[], tool_results=[])
    modalities = {c.modality for c in candidates}
    assert {"text", "image", "table"} <= modalities
    image_candidate = next(c for c in candidates if c.modality == "image")
    assert image_candidate.image is not None
    assert image_candidate.token_cost > 0


# -- slim packet materialization (cross-process) ---------------------------


def _ir_with_evidence() -> ContextIR:
    return ContextIR(
        objective=Objective("test"),
        evidence=[
            EvidenceItem(id="e1", source_id="d1", text="Bordeaux is in France."),
            EvidenceItem(
                id="e2", source_id="d2", modality="image",
                image=ImageRef(path="/m.png", metadata={"caption": "a map of France"}),
            ),
        ],
    )


def test_slim_packet_carries_modality_and_materializes_from_ir():
    ir = _ir_with_evidence()
    packet = ContextPacket.from_ir(ir, slim=True)
    assert packet.slim is True
    image_entry = next(e for e in packet.evidence_items if e["id"] == "e2")
    assert image_entry["modality"] == "image"
    assert image_entry["image"]["path"] == "/m.png"
    assert "text" not in image_entry and "text_hash" in image_entry
    # In-process materialization fills text from the held IR.
    packet.materialize()
    assert packet.slim is False
    text_entry = next(e for e in packet.evidence_items if e["id"] == "e1")
    assert text_entry["text"] == "Bordeaux is in France."


def test_slim_packet_materializes_cross_process_from_store():
    ir = _ir_with_evidence()
    store = InMemoryEvidenceStore()
    packet = ContextPacket.from_ir(ir, slim=True, evidence_store=store)
    assert len(store) == 2  # both scorable texts persisted under their hashes
    # Simulate cross-process: drop the in-memory IR link, round-trip through JSON.
    shipped = ContextPacket.model_validate_json(packet.model_dump_json())
    assert shipped._ir is None
    assert shipped.slim is True
    shipped.materialize(store=store)
    assert shipped.slim is False
    text_entry = next(e for e in shipped.evidence_items if e["id"] == "e1")
    assert text_entry["text"] == "Bordeaux is in France."


def test_blob_evidence_store_roundtrip(tmp_path):
    store = BlobEvidenceStore(FileBlobStore(tmp_path))
    digest = store.put("some evidence text")
    assert digest == content_hash("some evidence text")
    assert store.get(digest) == "some evidence text"
    assert store.get("deadbeef") is None


# -- evidence-ledger entailment links --------------------------------------


async def test_ledger_supports_links_via_entailment():
    evidence = [
        EvidenceItem(source_id="A", text="The refund window is 30 days for all plans."),
        EvidenceItem(source_id="B", text="Refunds are available within a 30 day window."),
    ]
    ledger = await distill_evidence_ledger(evidence, "what is the refund window")
    # Cross-source claims that agree corroborate each other.
    assert any(entry["supports"] for entry in ledger)


def test_link_entailments_detects_contradiction():
    ledger = [
        {"id": "E1", "source": "A", "claim": "The refund window is 30 days."},
        {"id": "E2", "source": "B", "claim": "The refund window is 14 days."},
    ]
    linked = link_entailments(ledger)
    # Topically similar but disagree on the salient number → contradiction.
    assert "E2" in linked[0]["contradicts"]
    assert "E1" in linked[1]["contradicts"]
