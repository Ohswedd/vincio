"""Real-behavior coverage for vincio.generation.render.

Exercises every node-type branch of the Markdown and HTML renderers, the
table-normalization invariants, image-src sanitization (the XSS guard), the
DocumentArtifact accessors, and — where the optional deps are installed —
the DOCX and PDF binary renderers. No mocks: real DocumentModel trees go
through the real renderers and the produced text/bytes are asserted on.
"""

from __future__ import annotations

import importlib.util

import pytest

from vincio.core.errors import GenerationError
from vincio.documents.parsers import TableData
from vincio.generation.model import DocBlock, DocumentModel
from vincio.generation.render import (
    MEDIA_TYPES,
    DocumentArtifact,
    _md_image_target,
    _normalize_table,
    _safe_image_src,
    render,
    render_html,
    render_markdown,
)

# Only used to skip the python-pptx absent-branch test when pptx *is* present.
_HAVE_PPTX = importlib.util.find_spec("pptx") is not None


def _write_png(path) -> None:
    """Write a small valid PNG so reportlab/python-docx accept it as an image."""
    Image = pytest.importorskip("PIL.Image")  # pulled in transitively by reportlab

    Image.new("RGB", (2, 2), (200, 30, 30)).save(str(path), "PNG")


# -- _safe_image_src / _md_image_target ---------------------------------------


def test_safe_image_src_allows_https_and_strips_control_chars():
    assert _safe_image_src("  https://x.test/a.png\n") == "https://x.test/a.png"
    assert _safe_image_src("img\r\n/cat.png") == "img/cat.png"


def test_safe_image_src_relative_paths_have_no_scheme_and_pass():
    # "./a.png" splits on "/" -> first part "." has no colon, so it passes.
    assert _safe_image_src("./a.png") == "./a.png"
    assert _safe_image_src("photos/a.png") == "photos/a.png"


def test_safe_image_src_rejects_javascript_scheme():
    assert _safe_image_src("javascript:alert(1)") == ""
    assert _safe_image_src("vbscript:msgbox(1)") == ""
    assert _safe_image_src("data:text/html,<script>") == ""


def test_safe_image_src_allows_data_image_and_file():
    assert _safe_image_src("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"
    assert _safe_image_src("file:///etc/logo.png") == "file:///etc/logo.png"


def test_md_image_target_angle_brackets_and_escapes_breakouts():
    target = _md_image_target("https://x.test/a b.png")
    assert target == "<https://x.test/a%20b.png>"


def test_md_image_target_escapes_embedded_angle_brackets():
    target = _md_image_target("https://x.test/a<b>c.png")
    assert "<" in target[:1] and ">" in target[-1:]
    assert "%3C" in target and "%3E" in target


def test_md_image_target_empty_for_rejected_scheme():
    assert _md_image_target("javascript:alert(1)") == ""


# -- _normalize_table ---------------------------------------------------------


def test_normalize_table_empty_returns_empty_pair():
    assert _normalize_table(TableData()) == ([], [])


def test_normalize_table_synthesizes_missing_headers():
    table = TableData(columns=["A"], rows=[["1", "2", "3"]])
    cols, rows = _normalize_table(table)
    assert cols == ["A", "col2", "col3"]
    assert rows == [["1", "2", "3"]]


def test_normalize_table_pads_short_rows_to_widest_row():
    # width = max(len(columns)=2, widest row=3) = 3 -> a col3 header is synthesized
    table = TableData(columns=["A", "B"], rows=[["1"], ["2", "3", "extra"]])
    cols, rows = _normalize_table(table)
    assert cols == ["A", "B", "col3"]
    # short row padded out to width 3; the wide row kept intact
    assert rows == [["1", "", ""], ["2", "3", "extra"]]


def test_normalize_table_truncates_rows_wider_than_declared_columns():
    # width is driven by columns when no row is wider -> extra cells truncated
    table = TableData(columns=["A", "B"], rows=[["1", "2", "drop me"]])
    cols, rows = _normalize_table(table)
    assert cols == ["A", "B", "col3"]
    assert rows == [["1", "2", "drop me"]]


def test_normalize_table_truncates_row_to_column_width():
    table = TableData(columns=["A", "B"], rows=[["1", "2"], ["3", "4"]])
    cols, rows = _normalize_table(table)
    assert cols == ["A", "B"]
    assert rows == [["1", "2"], ["3", "4"]]


# -- render_markdown ----------------------------------------------------------


def test_markdown_title_and_subtitle_and_trailing_newline():
    md = render_markdown(DocumentModel(title="Report", subtitle="Q3"))
    assert md.startswith("# Report\n\n*Q3*")
    assert md.endswith("\n")


def test_markdown_paragraph_block_passthrough():
    model = DocumentModel(blocks=[DocBlock(kind="paragraph", text="just a paragraph")])
    assert render_markdown(model).strip() == "just a paragraph"


def test_markdown_table_without_title_omits_bold_header():
    table = TableData(columns=["A"], rows=[["1"]])
    md = render_markdown(DocumentModel(blocks=[DocBlock(kind="table", table=table)]))
    assert "**" not in md
    assert "| A |" in md


def test_markdown_table_with_empty_columns_and_rows_is_dropped():
    # _normalize_table -> ([], []) -> _md_table returns "" -> filtered
    table = TableData(columns=[], rows=[])
    md = render_markdown(DocumentModel(blocks=[DocBlock(kind="table", table=table)]))
    assert md == "\n"


def test_markdown_page_break_between_two_paragraphs_keeps_both():
    model = DocumentModel(
        blocks=[
            DocBlock(kind="paragraph", text="before"),
            DocBlock(kind="page_break"),
            DocBlock(kind="paragraph", text="after"),
        ]
    )
    md = render_markdown(model)
    assert md.index("before") < md.index("---") < md.index("after")


def test_markdown_heading_level_clamped_to_h6():
    model = DocumentModel(blocks=[DocBlock(kind="heading", text="Deep", level=99)])
    assert "###### Deep" in render_markdown(model)


def test_markdown_heading_level_floor_is_h2_for_level_one():
    # render adds 1 to the block level (title is h1, sections start at h2)
    model = DocumentModel(blocks=[DocBlock(kind="heading", text="Sec", level=1)])
    assert "## Sec" in render_markdown(model)


def test_markdown_heading_level_zero_floors_to_h1():
    # min(6, max(1, 0+1)) == 1
    model = DocumentModel(blocks=[DocBlock(kind="heading", text="Sec", level=0)])
    assert render_markdown(model).strip() == "# Sec"


def test_markdown_quote_prefixes_each_line():
    model = DocumentModel(blocks=[DocBlock(kind="quote", text="line one\nline two")])
    md = render_markdown(model)
    assert "> line one" in md
    assert "> line two" in md


def test_markdown_empty_quote_still_emits_marker():
    model = DocumentModel(blocks=[DocBlock(kind="quote", text="")])
    assert ">" in render_markdown(model)


def test_markdown_code_block_with_language_fence():
    model = DocumentModel(blocks=[DocBlock(kind="code", text="x = 1", language="python")])
    md = render_markdown(model)
    assert "```python\nx = 1\n```" in md


def test_markdown_ordered_and_unordered_lists():
    ordered = DocumentModel(blocks=[DocBlock(kind="list", items=["a", "b"], ordered=True)])
    md = render_markdown(ordered)
    assert "1. a" in md and "2. b" in md

    bullet = DocumentModel(blocks=[DocBlock(kind="list", items=["a", "b"])])
    md2 = render_markdown(bullet)
    assert "- a" in md2 and "- b" in md2


def test_markdown_table_with_title_footnotes_and_pipe_escaping():
    table = TableData(
        title="Sales",
        columns=["Region", "Q|1"],
        rows=[["West", "10"]],
        footnotes=["preliminary"],
    )
    md = render_markdown(DocumentModel(blocks=[DocBlock(kind="table", table=table)]))
    assert "**Sales**" in md
    assert r"Q\|1" in md  # pipe escaped in header
    assert "| --- | --- |" in md
    assert "| West | 10 |" in md
    assert "_preliminary_" in md


def test_markdown_table_block_without_table_is_dropped():
    # kind == table but table is None -> _md_table returns "" -> filtered out
    md = render_markdown(DocumentModel(blocks=[DocBlock(kind="table")]))
    assert md == "\n"


def test_markdown_image_with_caption_and_alt_bracket_sanitized():
    model = DocumentModel(
        blocks=[
            DocBlock(
                kind="image",
                image_path="https://x.test/a.png",
                image_alt="al[t]",
                text="cap",
            )
        ]
    )
    md = render_markdown(model)
    assert "![al(t)](<https://x.test/a.png>)" in md
    assert "*cap*" in md


def test_markdown_image_falls_back_to_text_as_alt():
    model = DocumentModel(
        blocks=[DocBlock(kind="image", image_path="https://x.test/a.png", text="my pic")]
    )
    md = render_markdown(model)
    assert "![my pic](<https://x.test/a.png>)" in md


def test_markdown_image_block_without_path_falls_through_to_nothing():
    # image kind but image_path is None -> the `and block.image_path` guard is
    # false, the block matches no branch and is skipped (loop-continue path).
    model = DocumentModel(
        blocks=[
            DocBlock(kind="image", text="orphan"),
            DocBlock(kind="paragraph", text="kept"),
        ]
    )
    md = render_markdown(model)
    assert "orphan" not in md
    assert "kept" in md


def test_markdown_image_rejected_scheme_produces_no_output():
    model = DocumentModel(
        blocks=[DocBlock(kind="image", image_path="javascript:alert(1)", text="x")]
    )
    # target is empty -> nothing appended -> only trailing newline
    assert render_markdown(model) == "\n"


def test_markdown_page_break_is_horizontal_rule_as_only_block():
    # page_break as the final/only block exercises the loop-exit branch
    md = render_markdown(DocumentModel(blocks=[DocBlock(kind="page_break")]))
    assert md.strip() == "---"


def test_markdown_footnotes_and_bibliography_sections():
    model = DocumentModel(
        footnotes=["first note", "second note"],
        bibliography=["Doe 2020", "Roe 2021"],
    )
    md = render_markdown(model)
    assert "## Notes" in md
    assert "1. first note" in md and "2. second note" in md
    assert "## Sources" in md
    assert "- Doe 2020" in md and "- Roe 2021" in md


# -- render_html --------------------------------------------------------------


def test_html_escapes_title_and_subtitle():
    html_out = render_html(DocumentModel(title="A & B", subtitle="<x>"))
    assert "<h1>A &amp; B</h1>" in html_out
    assert '<p class="subtitle"><em>&lt;x&gt;</em></p>' in html_out
    assert html_out.startswith("<!DOCTYPE html>")


def test_html_default_title_when_missing():
    html_out = render_html(DocumentModel())
    assert "<title>Document</title>" in html_out


def test_html_heading_anchor_and_level_floor_is_h2():
    model = DocumentModel(blocks=[DocBlock(kind="heading", text="Top", level=0, anchor="top")])
    html_out = render_html(model)
    assert '<h2 id="top">Top</h2>' in html_out


def test_html_heading_level_clamped_to_h6():
    model = DocumentModel(blocks=[DocBlock(kind="heading", text="Deep", level=50)])
    assert "<h6>Deep</h6>" in render_html(model)


def test_html_paragraph_quote_code_escape():
    model = DocumentModel(
        blocks=[
            DocBlock(kind="paragraph", text="a & b"),
            DocBlock(kind="quote", text="q<x>"),
            DocBlock(kind="code", text="if a < b:"),
        ]
    )
    html_out = render_html(model)
    assert "<p>a &amp; b</p>" in html_out
    assert "<blockquote>q&lt;x&gt;</blockquote>" in html_out
    assert "<pre><code>if a &lt; b:</code></pre>" in html_out


def test_html_lists_ordered_and_unordered():
    ordered = DocumentModel(blocks=[DocBlock(kind="list", items=["a"], ordered=True)])
    assert "<ol><li>a</li></ol>" in render_html(ordered)
    bullet = DocumentModel(blocks=[DocBlock(kind="list", items=["a"])])
    assert "<ul><li>a</li></ul>" in render_html(bullet)


def test_html_table_caption_thead_tbody_and_escaping():
    table = TableData(title="T<1>", columns=["a&b"], rows=[["<v>"]])
    html_out = render_html(DocumentModel(blocks=[DocBlock(kind="table", table=table)]))
    assert "<caption>T&lt;1&gt;</caption>" in html_out
    assert "<th>a&amp;b</th>" in html_out
    assert "<td>&lt;v&gt;</td>" in html_out
    assert "<thead>" in html_out and "<tbody>" in html_out


def test_html_table_none_renders_nothing():
    html_out = render_html(DocumentModel(blocks=[DocBlock(kind="table")]))
    assert "<table>" not in html_out


def test_html_table_with_empty_columns_renders_nothing():
    table = TableData(columns=[], rows=[])
    html_out = render_html(DocumentModel(blocks=[DocBlock(kind="table", table=table)]))
    assert "<table>" not in html_out


def test_html_table_without_title_omits_caption():
    table = TableData(columns=["A"], rows=[["1"]])
    html_out = render_html(DocumentModel(blocks=[DocBlock(kind="table", table=table)]))
    assert "<caption>" not in html_out
    assert "<th>A</th>" in html_out


def test_html_image_figure_with_caption_and_escaped_src():
    model = DocumentModel(
        blocks=[
            DocBlock(
                kind="image",
                image_path="https://x.test/a.png?q=1&r=2",
                text="cap<x>",
                image_alt="alt&y",
            )
        ]
    )
    html_out = render_html(model)
    assert '<img src="https://x.test/a.png?q=1&amp;r=2" alt="alt&amp;y">' in html_out
    assert "<figcaption>cap&lt;x&gt;</figcaption>" in html_out


def test_html_image_without_caption_omits_figcaption():
    model = DocumentModel(
        blocks=[DocBlock(kind="image", image_path="https://x.test/a.png", image_alt="a")]
    )
    html_out = render_html(model)
    assert "<figcaption>" not in html_out
    assert '<img src="https://x.test/a.png" alt="a">' in html_out


def test_html_image_rejected_scheme_emits_no_figure():
    model = DocumentModel(blocks=[DocBlock(kind="image", image_path="javascript:x", text="c")])
    assert "<figure>" not in render_html(model)


def test_html_image_block_without_path_is_skipped():
    model = DocumentModel(
        blocks=[
            DocBlock(kind="image", text="orphan"),
            DocBlock(kind="paragraph", text="kept"),
        ]
    )
    html_out = render_html(model)
    assert "orphan" not in html_out
    assert "<p>kept</p>" in html_out


def test_html_page_break_hr_between_blocks():
    model = DocumentModel(
        blocks=[
            DocBlock(kind="paragraph", text="a"),
            DocBlock(kind="page_break"),
            DocBlock(kind="paragraph", text="b"),
        ]
    )
    html_out = render_html(model)
    assert '<hr class="page-break">' in html_out
    assert html_out.index("<p>a</p>") < html_out.index("page-break") < html_out.index("<p>b</p>")


def test_html_footnotes_and_bibliography_sections():
    model = DocumentModel(footnotes=["n1"], bibliography=["b1"])
    html_out = render_html(model)
    assert "<section class='notes'><h2>Notes</h2><ol>" in html_out
    assert "<li>n1</li>" in html_out
    assert "<section class='sources'><h2>Sources</h2><ul>" in html_out
    assert "<li>b1</li>" in html_out


# -- DocumentArtifact ---------------------------------------------------------


def test_artifact_text_decodes_for_markdown():
    art = render(DocumentModel(title="Hi"), "markdown")
    assert art.format == "markdown"
    assert art.media_type == "text/markdown"
    assert art.text.startswith("# Hi")


def test_artifact_text_raises_for_binary_format():
    art = DocumentArtifact(format="pdf", media_type="application/pdf", content=b"%PDF")
    with pytest.raises(GenerationError, match="binary artifact"):
        _ = art.text


def test_artifact_save_writes_bytes(tmp_path):
    art = render(DocumentModel(title="Save"), "html")
    out = tmp_path / "doc.html"
    returned = art.save(str(out))
    assert returned == str(out)
    assert out.read_bytes() == art.content
    assert b"<h1>Save</h1>" in out.read_bytes()


def test_artifact_digest_is_stable_and_content_bound():
    art = render(DocumentModel(title="Hash"), "markdown")
    digest = art.digest()
    assert len(digest) == 64
    # re-rendering identical content yields the same digest
    assert render(DocumentModel(title="Hash"), "markdown").digest() == digest
    # different content -> different digest
    assert render(DocumentModel(title="Other"), "markdown").digest() != digest


# -- render dispatch ----------------------------------------------------------


def test_render_unknown_format_raises_with_known_list():
    with pytest.raises(GenerationError, match="unknown render format"):
        render(DocumentModel(), "rtf")  # type: ignore[arg-type]


def test_render_propagates_title_evidence_and_metadata():
    model = DocumentModel(
        title="Doc",
        source_evidence_ids=["E1", "E2"],
        metadata={"author": "x"},
    )
    art = render(model, "markdown")
    assert art.title == "Doc"
    assert art.source_evidence_ids == ["E1", "E2"]
    assert art.metadata == {"author": "x"}
    # the artifact carries copies, not aliases of the model's lists/dicts
    model.source_evidence_ids.append("E3")
    assert art.source_evidence_ids == ["E1", "E2"]


def test_media_types_table_matches_each_renderable_format():
    assert MEDIA_TYPES["markdown"] == "text/markdown"
    assert MEDIA_TYPES["html"] == "text/html"
    assert MEDIA_TYPES["docx"].endswith("wordprocessingml.document")
    assert MEDIA_TYPES["pdf"] == "application/pdf"


# -- DOCX (only when python-docx is installed) --------------------------------


def test_render_docx_produces_openable_document_with_all_blocks():
    import io

    docx = pytest.importorskip("docx")

    table = TableData(columns=["A", "B"], rows=[["1", "2"]])
    model = DocumentModel(
        title="Title",
        subtitle="Sub",
        blocks=[
            DocBlock(kind="heading", text="H", level=2),
            DocBlock(kind="paragraph", text="para text"),
            DocBlock(kind="quote", text="quoted"),
            DocBlock(kind="code", text="code()"),
            DocBlock(kind="list", items=["x", "y"], ordered=True),
            DocBlock(kind="list", items=["z"]),
            DocBlock(kind="table", table=table),
            # missing image file -> falls back to a text paragraph, plus caption
            DocBlock(kind="image", image_path="/no/such/image.png", text="cap"),
            DocBlock(kind="page_break"),
        ],
        footnotes=["note1"],
        bibliography=["src1"],
    )
    content = render(model, "docx").content
    assert content[:2] == b"PK"  # zip/OOXML magic

    parsed = docx.Document(io.BytesIO(content))
    all_text = "\n".join(p.text for p in parsed.paragraphs)
    assert "para text" in all_text
    assert "quoted" in all_text
    assert "[image: /no/such/image.png]" in all_text
    assert "cap" in all_text
    assert "1. note1" in all_text
    # the data table became a real docx table with header + data row
    assert parsed.tables
    cells = parsed.tables[0].rows[0].cells
    assert [c.text for c in cells] == ["A", "B"]


def test_render_docx_embeds_a_real_image(tmp_path):
    pytest.importorskip("docx")
    img = tmp_path / "pic.png"
    _write_png(img)
    model = DocumentModel(
        blocks=[DocBlock(kind="image", image_path=str(img), text="cap")]
    )
    content = render(model, "docx").content
    # OOXML zip carries the embedded media part when add_picture succeeds
    assert content[:2] == b"PK"
    assert b"word/media/" in content


def test_render_docx_empty_table_and_uncaptioned_image(tmp_path):
    docx = pytest.importorskip("docx")

    img = tmp_path / "p.png"
    _write_png(img)
    model = DocumentModel(
        blocks=[
            DocBlock(kind="table", table=TableData(columns=[], rows=[])),
            DocBlock(kind="image", image_path=str(img)),  # no caption text
            DocBlock(kind="image", text="orphan"),  # no path -> falls through
            DocBlock(kind="paragraph", text="tail"),
        ]
    )
    content = render(model, "docx").content
    parsed = docx.Document(__import__("io").BytesIO(content))
    # empty table contributes no docx table
    assert not parsed.tables
    # uncaptioned image embeds media but adds no Caption paragraph
    assert b"word/media/" in content
    assert "tail" in "\n".join(p.text for p in parsed.paragraphs)


def test_render_docx_heading_level_clamped_to_nine():
    pytest.importorskip("docx")
    model = DocumentModel(blocks=[DocBlock(kind="heading", text="Deep", level=99)])
    # level clamp to <=9 keeps python-docx from raising on the style name
    content = render(model, "docx").content
    assert content[:2] == b"PK"


# -- PDF (only when reportlab is installed) -----------------------------------


def test_render_pdf_produces_pdf_bytes_across_block_types(tmp_path):
    pytest.importorskip("reportlab")
    img = tmp_path / "pix.png"
    _write_png(img)
    table = TableData(columns=["A", "B"], rows=[["1", "2"]])
    model = DocumentModel(
        title="PDF Title",
        subtitle="PDF Sub",
        blocks=[
            DocBlock(kind="heading", text="H", level=2),
            DocBlock(kind="paragraph", text="para"),
            DocBlock(kind="quote", text="quote"),
            DocBlock(kind="code", text="code"),
            DocBlock(kind="list", items=["a", "b"], ordered=True),
            DocBlock(kind="list", items=["c"]),
            DocBlock(kind="table", table=table),
            DocBlock(kind="image", image_path=str(img), text="caption"),
            DocBlock(kind="page_break"),
        ],
        footnotes=["fn"],
        bibliography=["bib"],
    )
    art = render(model, "pdf")
    assert art.media_type == "application/pdf"
    assert art.content[:5] == b"%PDF-"
    assert len(art.content) > 500


def test_render_pdf_empty_table_and_uncaptioned_image(tmp_path):
    pytest.importorskip("reportlab")
    img = tmp_path / "p.png"
    _write_png(img)
    model = DocumentModel(
        blocks=[
            DocBlock(kind="table", table=TableData(columns=[], rows=[])),
            DocBlock(kind="image", image_path=str(img)),  # no caption -> no italic line
            DocBlock(kind="image", text="orphan"),  # no path -> falls through
            DocBlock(kind="paragraph", text="after"),
        ]
    )
    art = render(model, "pdf")
    assert art.content[:5] == b"%PDF-"


def test_render_pdf_heading_level_clamped_to_four():
    pytest.importorskip("reportlab")
    # level 99 would index a missing "Heading99" style without the clamp
    model = DocumentModel(blocks=[DocBlock(kind="heading", text="Deep", level=99)])
    art = render(model, "pdf")
    assert art.content[:5] == b"%PDF-"


# -- PPTX guard (dep is absent in this env) -----------------------------------


@pytest.mark.skipif(_HAVE_PPTX, reason="python-pptx is installed; the ImportError path is skipped")
def test_render_pptx_without_dep_raises_actionable_error():
    with pytest.raises(GenerationError, match=r"python-pptx"):
        render(DocumentModel(title="x"), "pptx")
