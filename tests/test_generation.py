"""Tests for the 1.9 generation engine: documents & media flow OUT."""

from __future__ import annotations

import warnings

import pytest

from vincio.core.errors import (
    BudgetExceededError,
    CitationValidationError,
    DocumentContractError,
    GenerationError,
)
from vincio.core.types import Budget, EvidenceItem, TrustLevel
from vincio.generation import (
    CitationContract,
    CitedReportBuilder,
    DocumentBuilder,
    DocumentContract,
    ImageGenRequest,
    MockImageProvider,
    MockSpeechProvider,
    Slot,
    SpeechRequest,
    TableSpec,
    fill_text_template,
    generate_redline,
    markdown_to_model,
)
from vincio.generation.media import image_cost, meter_media_cost
from vincio.governance.transparency import (
    embed_provenance,
    mark_synthetic_content,
    verify_manifest,
    write_sidecar_manifest,
)
from vincio.security.audit import AuditLog

pytestmark = pytest.mark.filterwarnings("ignore::vincio.VincioExperimentalWarning")

SAMPLE_MD = """# Quarterly Memo

## Summary

Revenue grew 30% [E1]. Operating costs fell [E2].

## Risks

- Customer churn
- FX exposure

| Metric | Q1 | Q2 |
| --- | --- | --- |
| Revenue | 10 | 13 |
"""


# -- DocumentBuilder ---------------------------------------------------------


class TestDocumentBuilder:
    def test_markdown_roundtrip(self):
        art = DocumentBuilder().build(SAMPLE_MD, format="markdown")
        assert art.format == "markdown" and art.media_type == "text/markdown"
        assert "Quarterly Memo" in art.text
        assert "| Metric |" in art.text
        assert art.sha256()

    def test_html_render_has_table_and_headings(self):
        art = DocumentBuilder().build(SAMPLE_MD, format="html")
        assert "<h1>" in art.text and "<table>" in art.text and "<th>Metric</th>" in art.text

    def test_mapping_input(self):
        data = {
            "title": "Report",
            "sections": [
                {"heading": "Intro", "body": "Hello world."},
                {"heading": "Data", "table": [{"k": "a", "v": "1"}]},
            ],
        }
        art = DocumentBuilder().build(data, format="markdown")
        assert "# Report" in art.text and "## Intro" in art.text and "| k | v |" in art.text

    def test_runresult_carries_evidence_ids(self):
        from vincio.core.types import RunResult

        result = RunResult(
            raw_text="# Answer\n\nThe sky is blue [E1].",
            evidence=[EvidenceItem(id="E1", source_id="D1", text="sky is blue")],
        )
        art = DocumentBuilder().build(result, format="markdown")
        assert art.source_evidence_ids == ["E1"]

    def test_contract_pass(self):
        contract = DocumentContract(
            required_sections=["Summary", "Risks"],
            table_specs=[TableSpec(required_columns=["Metric", "Q1"], min_rows=1)],
            min_words=5,
        )
        art = DocumentBuilder().build(SAMPLE_MD, format="markdown", contract=contract)
        assert "contract_repairs" in art.metadata

    def test_contract_missing_section_raises(self):
        with pytest.raises(DocumentContractError) as exc:
            DocumentBuilder().build(SAMPLE_MD, format="markdown",
                                    contract=DocumentContract(required_sections=["Appendix"]))
        assert "Appendix" in str(exc.value)
        assert exc.value.violations

    def test_contract_citations_per_section(self):
        # Summary cites [E1]/[E2]; Risks section has no citation → violation.
        with pytest.raises(DocumentContractError):
            DocumentBuilder().build(SAMPLE_MD, format="markdown",
                                    contract=DocumentContract(citations_per_section=True))

    def test_contract_length_bound(self):
        with pytest.raises(DocumentContractError):
            DocumentBuilder().build("# T\n\nshort", format="markdown",
                                    contract=DocumentContract(min_words=100))

    def test_formatting_repair_derives_title(self):
        model = markdown_to_model("## Only A Subheading\n\nbody text here")
        model.title = ""
        from vincio.generation.contracts import repair_formatting

        actions = repair_formatting(model, DocumentContract())
        assert model.title and any("title" in a for a in actions)

    def test_audit_event_recorded(self):
        audit = AuditLog(directory=None)
        DocumentBuilder(audit_log=audit).build(SAMPLE_MD, format="markdown")
        events = [e for e in audit.entries if e.action == "document_generate"]
        assert events and events[0].details["format"] == "markdown"
        assert events[0].details["content_sha256"]

    def test_binary_format_text_property_raises(self):
        art = DocumentBuilder().build("# T\n\nbody", format="markdown")
        art.format = "docx"  # type: ignore[assignment]
        with pytest.raises(GenerationError):
            _ = art.text

    def test_docx_render(self):
        pytest.importorskip("docx")
        art = DocumentBuilder().build(SAMPLE_MD, format="docx")
        assert art.content[:2] == b"PK" and len(art.content) > 0

    def test_pdf_render(self):
        pytest.importorskip("reportlab")
        art = DocumentBuilder().build(SAMPLE_MD, format="pdf")
        assert art.content.startswith(b"%PDF")

    def test_pptx_render(self):
        pytest.importorskip("pptx")
        art = DocumentBuilder().build(SAMPLE_MD, format="pptx")
        assert art.content[:2] == b"PK"


# -- Redline -----------------------------------------------------------------


class TestRedline:
    def test_markdown_redline_marks_changes(self):
        art = generate_redline("The cat sat.", "The dog sat down.", format="markdown")
        assert "~~" in art.text and "**" in art.text

    def test_docx_redline(self):
        pytest.importorskip("docx")
        art = generate_redline("a b c", "a x c", format="docx")
        assert art.format == "docx" and art.content[:2] == b"PK"


class TestRenderingSafety:
    def test_ragged_and_columnless_table_parity(self):
        from vincio.documents.parsers import TableData
        from vincio.generation.model import DocBlock, DocumentModel
        from vincio.generation.render import render

        model = DocumentModel(title="T")
        # No columns, ragged rows — every renderer must synthesize headers and
        # keep all cells (no truncation, no dropped table).
        model.blocks.append(DocBlock(kind="table", table=TableData(rows=[["a", "b", "c"], ["1"]])))
        md = render(model, "markdown").text
        assert "| col1 | col2 | col3 |" in md and "| a | b | c |" in md and "| 1 |  |  |" in md
        html = render(model, "html").text
        assert "<th>col1</th>" in html and "<td>a</td>" in html

    def test_markdown_image_breakout_neutralized(self):
        from vincio.generation.model import DocumentModel
        from vincio.generation.render import render

        model = DocumentModel()
        model.image("x.png) <img src=x onerror=alert(1)>", alt="a")
        md = render(model, "markdown").text
        assert "<img" not in md  # the raw tag cannot break out

    def test_html_image_scheme_allowlist(self):
        from vincio.generation.model import DocumentModel
        from vincio.generation.render import render

        bad = DocumentModel()
        bad.image("javascript:alert(1)")
        assert "javascript:" not in render(bad, "html").text
        good = DocumentModel()
        good.image("https://example.com/c.png", alt="chart")
        assert '<img src="https://example.com/c.png"' in render(good, "html").text


# -- CitedReportBuilder ------------------------------------------------------


@pytest.fixture()
def cite_evidence():
    return [
        EvidenceItem(id="E1", source_id="D1", text="Revenue grew 30% year over year.",
                     page=4, trust_level=TrustLevel.UNTRUSTED_DOCUMENT),
        EvidenceItem(id="E2", source_id="D2", text="Operating costs declined materially.",
                     trust_level=TrustLevel.USER),
    ]


class TestCitedReport:
    async def test_resolution_and_footnotes(self, cite_evidence):
        builder = CitedReportBuilder()
        report = await builder.build_report(
            "Revenue grew 30% [E1]. Costs fell [E2].", cite_evidence, title="Memo"
        )
        assert [c.number for c in report.citations] == [1, 2]
        assert report.citations[0].page == 4
        rendered = report.render("markdown").text
        assert "[1]" in rendered and "Notes" in rendered and "Sources" in rendered
        assert not report.unresolved_markers

    async def test_unresolved_markers_surface(self, cite_evidence):
        report = await CitedReportBuilder().build_report(
            "A claim [E1]. A bad ref [E9].", cite_evidence
        )
        assert report.unresolved_markers == ["E9"]

    async def test_coverage_metric(self, cite_evidence):
        report = await CitedReportBuilder().build_report(
            "Revenue grew 30% [E1]. The weather is nice today and pleasant.", cite_evidence
        )
        # One of two verifiable claims is cited.
        assert 0.0 < report.coverage.coverage < 1.0

    async def test_entailment_check(self, cite_evidence):
        report = await CitedReportBuilder().build_report(
            "Revenue grew 30% [E1].", cite_evidence,
            contract=CitationContract(require_entailment=True, min_entailment_rate=0.5),
        )
        assert report.coverage.entailment_rate == 1.0

    async def test_contract_rejects_unresolved(self, cite_evidence):
        with pytest.raises(CitationValidationError):
            await CitedReportBuilder().build_report(
                "Claim [E9].", cite_evidence, contract=CitationContract(min_coverage=0.0)
            )

    async def test_contract_rejects_low_coverage(self, cite_evidence):
        with pytest.raises(CitationValidationError):
            await CitedReportBuilder().build_report(
                "Revenue grew 30% over the prior year period. "
                "Costs increased 5% in the same window.",
                cite_evidence,
                contract=CitationContract(min_coverage=1.0),
            )

    async def test_custom_entailment_backend(self, cite_evidence):
        calls = []

        def always_false(claim, evidence):
            calls.append(claim)
            return False

        with pytest.raises(CitationValidationError):
            await CitedReportBuilder(entailment=always_false).build_report(
                "Revenue grew 30% [E1].", cite_evidence,
                contract=CitationContract(require_entailment=True),
            )
        assert calls


# -- Templates / forms-fill --------------------------------------------------


class TestTemplates:
    def test_type_coercion(self):
        out = fill_text_template(
            "Fee {{fee}}, count {{n}}, active {{ok}}",
            {"fee": 1234.5, "n": 3, "ok": True},
            slots=[Slot(name="fee", type="currency"), Slot(name="n", type="number"),
                   Slot(name="ok", type="bool")],
        )
        assert "$1,234.50" in out and "count 3" in out and "active Yes" in out

    def test_must_cite_enforced(self):
        with pytest.raises(GenerationError):
            fill_text_template("Finding: {{f}}", {"f": "no citation here"},
                               slots=[Slot(name="f", must_cite=True)])
        ok = fill_text_template("Finding: {{f}}", {"f": "supported [E1]"},
                                slots=[Slot(name="f", must_cite=True)], evidence_ids=["E1"])
        assert "[E1]" in ok

    def test_unknown_slot_strict(self):
        with pytest.raises(GenerationError):
            fill_text_template("{{missing}}", {}, strict=True)
        assert fill_text_template("{{missing}}", {}, strict=False) == "{{missing}}"

    def test_must_cite_checks_raw_value_keeps_formatting(self):
        # A citation supplied alongside a typed value is detected on the raw value
        # while the rendered output keeps its type formatting.
        out = fill_text_template(
            "Amount: {{amt}}", {"amt": "4200 [E1]"},
            slots=[Slot(name="amt", type="currency", must_cite=True)], evidence_ids=["E1"],
        )
        assert "[E1]" in out


# -- Image generation --------------------------------------------------------


class TestImageProvider:
    async def test_mock_generates_valid_png(self):
        provider = MockImageProvider()
        resp = await provider.generate_image(ImageGenRequest(prompt="a sunset", n=2))
        assert len(resp.images) == 2
        for image in resp.images:
            assert image.data.startswith(b"\x89PNG")
            assert image.manifest is not None
            assert verify_manifest(image.manifest, image.data)
            assert image.manifest.media_type == "image/png"

    async def test_deterministic_by_prompt(self):
        # The rendered pixels are reproducible for a prompt; the stamped bytes
        # carry a timestamped provenance manifest, so determinism is asserted on
        # the rendered image, not the (intentionally unique) credential.
        provider = MockImageProvider()
        a = provider._render(ImageGenRequest(prompt="cat", seed=1))[0]
        b = provider._render(ImageGenRequest(prompt="cat", seed=1))[0]
        c = provider._render(ImageGenRequest(prompt="dog", seed=1))[0]
        assert a == b and a != c
        # And different prompts always yield different stamped outputs too.
        out_cat = (await provider.generate_image(ImageGenRequest(prompt="cat", seed=1))).images[0].data
        out_dog = (await provider.generate_image(ImageGenRequest(prompt="dog", seed=1))).images[0].data
        assert out_cat != out_dog

    async def test_edit_marks_composite(self):
        from vincio.core.types import ImageRef

        provider = MockImageProvider()
        resp = await provider.edit_image(ImageRef(path="x.png"), ImageGenRequest(prompt="add a hat"))
        assert "composite" in resp.images[0].manifest.digital_source_type

    def test_cost_table(self):
        assert image_cost("gpt-image-1", n=2, quality="high") > 0
        assert image_cost("unknown-model", n=1, quality="auto") == 0.0

    async def test_budget_metering_raises(self):
        provider = MockImageProvider()
        resp = await provider.generate_image(ImageGenRequest(prompt="x"))
        from vincio.core.types import BudgetUsage

        usage = BudgetUsage()
        # Force a non-zero cost and a tiny budget.
        resp.cost_usd = 5.0
        with pytest.raises(BudgetExceededError):
            meter_media_cost(resp.cost_usd, budget=Budget(max_cost_usd=1.0), usage=usage)


# -- Speech synthesis --------------------------------------------------------


class TestSpeechProvider:
    async def test_mock_generates_valid_wav(self):
        provider = MockSpeechProvider()
        resp = await provider.synthesize_speech(SpeechRequest(text="hello world " * 4))
        assert resp.audio.data.startswith(b"RIFF") and b"WAVE" in resp.audio.data[:16]
        assert resp.audio.media_type == "audio/wav"
        assert resp.audio.manifest is not None
        assert verify_manifest(resp.audio.manifest, resp.audio.data)
        assert resp.characters == len("hello world " * 4)


# -- Media-aware provenance --------------------------------------------------


class TestMediaProvenance:
    def test_text_binding_backward_compatible(self):
        manifest = mark_synthetic_content("hello", model_id="m")
        assert manifest.media_type == "text/plain"
        assert verify_manifest(manifest, "hello")
        assert not verify_manifest(manifest, "world")

    def test_bytes_binding(self):
        data = b"\x00\x01\x02raw-bytes"
        manifest = mark_synthetic_content(data, media_type="image/png")
        assert verify_manifest(manifest, data)
        assert not verify_manifest(manifest, data + b"x")

    def test_png_embed_and_sidecar(self, tmp_path):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00IEND\xaeB`\x82"
        manifest = mark_synthetic_content(png, media_type="image/png")
        stamped = embed_provenance(png, manifest)
        assert stamped.startswith(b"\x89PNG") and b"c2pa.manifest" in stamped
        path = tmp_path / "img.png"
        path.write_bytes(stamped)
        sidecar = write_sidecar_manifest(path, manifest)
        assert sidecar.exists() and sidecar.name.endswith(".c2pa.json")

    def test_watermark_hook_runs(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00IEND\xaeB`\x82"
        manifest = mark_synthetic_content(png, media_type="image/png")
        called = {}

        def hook(data: bytes) -> bytes:
            called["yes"] = True
            return data

        embed_provenance(png, manifest, watermark_hook=hook)
        assert called.get("yes")

    def test_watermark_hook_corrupting_png_raises(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00IEND\xaeB`\x82"
        manifest = mark_synthetic_content(png, media_type="image/png")
        with pytest.raises(GenerationError):
            embed_provenance(png, manifest, watermark_hook=lambda d: b"not a png")
        with pytest.raises(GenerationError):
            embed_provenance(png, manifest, watermark_hook=lambda d: b"")

    def test_embedded_manifest_carries_no_stale_hash(self):
        # Embedding changes the bytes, so the embedded credential must not claim a
        # content hash; the returned/sidecar manifest binds the final bytes.
        import json
        import re

        from vincio.generation.media import attach_media_provenance

        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00IEND\xaeB`\x82"
        stamped, manifest = attach_media_provenance(png, media_type="image/png", model="m")
        assert verify_manifest(manifest, stamped)  # returned manifest binds final bytes
        blob = stamped[stamped.find(b"{"): stamped.rfind(b"}") + 1]
        embedded = json.loads(blob)
        assert embedded["content_binding"]["hash"] is None
        assert re.search(rb"c2pa.manifest", stamped)


# -- Audio input wiring (chat providers) -------------------------------------


class TestAudioInput:
    def _audio_request(self, tmp_path):
        from vincio.core.types import AudioRef, ContentPart, Message, ModelRequest

        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        return ModelRequest(
            model="x",
            messages=[Message(role="user", content=[
                ContentPart(type="text", text="Transcribe"),
                ContentPart(type="audio", audio=AudioRef(path=str(wav), media_type="audio/wav")),
            ])],
        )

    def test_openai_renders_input_audio(self, tmp_path):
        from vincio.providers.openai import OpenAIProvider

        request = self._audio_request(tmp_path)
        rendered = OpenAIProvider(api_key="x")._render_messages(request.messages)
        parts = rendered[0]["content"]
        audio_parts = [p for p in parts if p.get("type") == "input_audio"]
        assert audio_parts and audio_parts[0]["input_audio"]["format"] == "wav"
        assert audio_parts[0]["input_audio"]["data"]

    def test_google_renders_inline_audio(self, tmp_path):
        from vincio.providers.google import GoogleProvider

        request = self._audio_request(tmp_path)
        _, contents = GoogleProvider(api_key="x")._render(request)
        parts = contents[0]["parts"]
        inline = [p for p in parts if "inlineData" in p]
        assert inline and inline[0]["inlineData"]["mimeType"] == "audio/wav"


# -- Eval metrics ------------------------------------------------------------


class TestCitationMetrics:
    def test_citation_coverage_and_entailment(self):
        from vincio.evals.datasets import EvalCase
        from vincio.evals.metrics import METRICS, RunOutput

        evidence = [EvidenceItem(id="E1", source_id="D1", text="Revenue grew 30% last year.")]
        run = RunOutput(
            raw_text="Revenue grew 30% [E1]. The market is large and growing fast.",
            evidence=evidence,
        )
        case = EvalCase(id="t", input="q")
        coverage = METRICS["citation_coverage"](case, run)
        entail = METRICS["claim_entailment"](case, run)
        assert 0.0 < coverage.value < 1.0
        assert entail.value == 1.0


# -- App-level wrappers ------------------------------------------------------


class TestAppGeneration:
    @pytest.fixture()
    def app(self, offline_config):
        from vincio import ContextApp
        from vincio.providers import MockProvider

        return ContextApp(name="gen", provider=MockProvider(), model="mock-1", config=offline_config)

    def test_build_document(self, app):
        art = app.build_document(SAMPLE_MD, format="markdown")
        assert "Quarterly Memo" in art.text
        assert any(e.action == "document_generate" for e in app.audit.entries)

    def test_cited_report_sync(self, app):
        evidence = [EvidenceItem(id="E1", source_id="D1", text="The sky is blue.")]
        art = app.cited_report("The sky is blue [E1].", evidence, format="markdown")
        assert "[1]" in art.text

    def test_generate_image(self, app):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = app.generate_image("a logo", provider=MockImageProvider())
        assert resp.images and resp.images[0].data.startswith(b"\x89PNG")
        assert any(e.action == "image_generate" for e in app.audit.entries)

    def test_synthesize_speech(self, app):
        resp = app.synthesize_speech("hello", provider=MockSpeechProvider())
        assert resp.audio.data.startswith(b"RIFF")
        assert any(e.action == "speech_synthesize" for e in app.audit.entries)

    def test_media_budget_accumulates_across_calls(self, app, monkeypatch):
        import vincio.generation.media as media_mod

        monkeypatch.setitem(
            media_mod.IMAGE_PRICES, "mock-image",
            media_mod.ImagePrice(low=0.03, medium=0.03, high=0.03),
        )
        app.budget = Budget(max_cost_usd=0.05)
        app.generate_image("a", provider=MockImageProvider())  # 0.03 cumulative
        with pytest.raises(BudgetExceededError):
            app.generate_image("b", provider=MockImageProvider())  # 0.06 > 0.05
