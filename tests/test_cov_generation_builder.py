"""Real-behavior coverage tests for vincio.generation.builder.

Targets the uncovered branches: Markdown code-fence/table/list-switch/quote
parsing, ``mapping_to_model`` section + bare-mapping + table-coercion branches,
``DocumentBuilder.to_model`` source dispatch (DocumentModel/RunResult/pydantic/
dict/list/str/unsupported), the build footnotes/bibliography/evidence merge and
contract-failure path, audit emission, and the word-level redline diff.
"""

from __future__ import annotations

import pytest

from vincio.core.errors import DocumentContractError, GenerationError
from vincio.core.types import EvidenceItem, RunResult
from vincio.documents.parsers import TableData
from vincio.generation.builder import (
    DocumentBuilder,
    generate_redline,
    mapping_to_model,
    markdown_to_model,
)
from vincio.generation.contracts import DocumentContract, TableSpec
from vincio.generation.model import DocBlock, DocumentModel
from vincio.security.audit import AuditLog

# -- markdown_to_model -------------------------------------------------------


def test_markdown_code_fence_captures_body_and_language():
    model = markdown_to_model("```python\nx = 1\ny = 2\n```")
    code = [b for b in model.blocks if b.kind == "code"]
    assert len(code) == 1
    assert code[0].language == "python"
    assert code[0].text == "x = 1\ny = 2"


def test_markdown_unterminated_code_fence_consumes_rest():
    # No closing fence: the parser runs index off the end and still emits a block.
    model = markdown_to_model("```\nline one\nline two")
    code = [b for b in model.blocks if b.kind == "code"]
    assert len(code) == 1
    assert code[0].text == "line one\nline two"
    assert code[0].language == ""


def test_markdown_table_parsed_until_non_row():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n\nafter"
    model = markdown_to_model(md)
    tables = [b for b in model.blocks if b.kind == "table"]
    assert len(tables) == 1
    assert tables[0].table.columns == ["A", "B"]
    assert tables[0].table.rows == [["1", "2"], ["3", "4"]]
    # The trailing paragraph after the table stops the row scan (line 88 break).
    paras = [b for b in model.blocks if b.kind == "paragraph"]
    assert paras[-1].text == "after"


def test_markdown_table_header_only_has_empty_rows():
    # Header + separator but no data rows: the row scan stops immediately.
    model = markdown_to_model("| A | B |\n| --- | --- |\n\nbody")
    table = next(b for b in model.blocks if b.kind == "table").table
    assert table.columns == ["A", "B"]
    assert table.rows == []


def test_markdown_table_at_eof_no_trailing_lines():
    # Header + separator are the LAST lines: the row-scan loop never iterates
    # (cursor == len(lines)).
    model = markdown_to_model("| A | B |\n| --- | --- |")
    table = next(b for b in model.blocks if b.kind == "table").table
    assert table.columns == ["A", "B"]
    assert table.rows == []


def test_markdown_pipe_row_without_separator_is_not_a_table():
    # A pipe row not followed by a separator must NOT be treated as a table.
    model = markdown_to_model("| just | text |\nmore text")
    assert all(b.kind != "table" for b in model.blocks)


def test_markdown_switching_list_type_flushes_first_list():
    md = "- a\n- b\n1. one\n2. two"
    model = markdown_to_model(md)
    lists = [b for b in model.blocks if b.kind == "list"]
    assert len(lists) == 2
    assert lists[0].ordered is False
    assert lists[0].items == ["a", "b"]
    assert lists[1].ordered is True
    assert lists[1].items == ["one", "two"]


def test_markdown_blockquote_becomes_quote_block():
    model = markdown_to_model("> quoted wisdom")
    quotes = [b for b in model.blocks if b.kind == "quote"]
    assert len(quotes) == 1
    assert quotes[0].text == "quoted wisdom"


def test_markdown_leading_h1_becomes_title_lower_levels_demoted():
    model = markdown_to_model("# Real Title\n\n## Section")
    assert model.title == "Real Title"
    headings = [b for b in model.blocks if b.kind == "heading"]
    # The H2 is demoted by one level (max(1, 2-1) == 1).
    assert headings[0].text == "Section"
    assert headings[0].level == 1


def test_markdown_h1_not_title_when_title_already_set():
    # An explicit title means the H1 stays a heading instead of being consumed.
    model = markdown_to_model("# Heading One", title="Preset")
    assert model.title == "Preset"
    headings = [b for b in model.blocks if b.kind == "heading"]
    assert headings == [headings[0]]
    assert headings[0].text == "Heading One"


def test_markdown_paragraph_flushes_list_before_text():
    model = markdown_to_model("- item\nplain paragraph")
    kinds = [b.kind for b in model.blocks]
    assert kinds == ["list", "paragraph"]


# -- mapping_to_model --------------------------------------------------------


def test_mapping_sections_full_shape():
    data = {
        "title": "Report",
        "subtitle": "Q3",
        "sections": [
            {
                "heading": "Intro",
                "level": 2,
                "anchor": "intro",
                "paragraphs": ["p one", "p two"],
                "items": ["x", "y"],
                "table": {"columns": ["c"], "rows": [["v"]]},
            }
        ],
    }
    model = mapping_to_model(data)
    assert model.title == "Report"
    assert model.subtitle == "Q3"
    heading = next(b for b in model.blocks if b.kind == "heading")
    assert heading.text == "Intro"
    assert heading.level == 2
    assert heading.anchor == "intro"
    paras = [b.text for b in model.blocks if b.kind == "paragraph"]
    assert paras == ["p one", "p two"]
    lists = [b for b in model.blocks if b.kind == "list"]
    assert lists[0].items == ["x", "y"]
    tables = [b for b in model.blocks if b.kind == "table"]
    assert tables[0].table.columns == ["c"]


def test_mapping_section_body_fallback_and_bullets_alias():
    data = {"sections": [{"title": "T", "body": "single body", "bullets": ["b1"]}]}
    model = mapping_to_model(data)
    paras = [b.text for b in model.blocks if b.kind == "paragraph"]
    assert paras == ["single body"]
    lists = [b for b in model.blocks if b.kind == "list"]
    assert lists[0].items == ["b1"]


def test_mapping_non_dict_section_entry_rendered_as_paragraph():
    model = mapping_to_model({"sections": ["plain string entry", 42]})
    paras = [b.text for b in model.blocks if b.kind == "paragraph"]
    assert paras == ["plain string entry", "42"]


def test_mapping_section_text_key_used_when_no_body():
    model = mapping_to_model({"sections": [{"heading": "H", "text": "via text"}]})
    paras = [b.text for b in model.blocks if b.kind == "paragraph"]
    assert paras == ["via text"]


def test_mapping_without_sections_renders_definition_section():
    data = {"title": "Doc", "revenue_total": "100", "regions": ["NA", "EU"]}
    model = mapping_to_model(data)
    headings = {b.text for b in model.blocks if b.kind == "heading"}
    assert "Revenue Total" in headings  # key titleized with underscores replaced
    assert "Regions" in headings
    paras = [b.text for b in model.blocks if b.kind == "paragraph"]
    assert "100" in paras
    lists = [b for b in model.blocks if b.kind == "list"]
    assert lists[0].items == ["NA", "EU"]


def test_mapping_definition_section_table_value():
    data = {"grid": {"columns": ["a", "b"], "rows": [["1", "2"]]}}
    model = mapping_to_model(data)
    tables = [b for b in model.blocks if b.kind == "table"]
    assert tables[0].table.columns == ["a", "b"]
    assert tables[0].table.rows == [["1", "2"]]


def test_table_from_value_list_of_dicts_unions_keys():
    # list-of-dicts path inside a section table.
    data = {"sections": [{"table": [{"a": 1, "b": 2}, {"a": 3, "c": 4}]}]}
    model = mapping_to_model(data)
    table = next(b for b in model.blocks if b.kind == "table").table
    assert table.columns == ["a", "b", "c"]
    assert table.rows == [["1", "2", ""], ["3", "", "4"]]


def test_table_from_value_existing_tabledata_passes_through():
    td = TableData(columns=["k"], rows=[["v"]])
    data = {"sections": [{"table": td}]}
    model = mapping_to_model(data)
    out = next(b for b in model.blocks if b.kind == "table").table
    assert out is td


def test_table_from_value_returns_none_for_unrecognized():
    # A bare string is not coercible -> no table block emitted for the section.
    data = {"sections": [{"heading": "H", "table": "not a table"}]}
    model = mapping_to_model(data)
    assert all(b.kind != "table" for b in model.blocks)


# -- DocumentBuilder.to_model dispatch ---------------------------------------


def test_to_model_passes_through_documentmodel_and_sets_title():
    src = DocumentModel()
    out = DocumentBuilder().to_model(src, title="Given")
    assert out is src
    assert out.title == "Given"


def test_to_model_documentmodel_keeps_existing_title():
    src = DocumentModel(title="Original")
    out = DocumentBuilder().to_model(src, title="Ignored")
    assert out.title == "Original"


def test_to_model_runresult_uses_structured_output_and_evidence():
    result = RunResult(
        output={"sections": [{"heading": "Sec", "body": "b"}]},
        evidence=[
            EvidenceItem(id="E1", source_id="s1", text="t1"),
            EvidenceItem(id="E2", source_id="s2", text="t2"),
        ],
    )
    model = DocumentBuilder().to_model(result)
    assert model.source_evidence_ids == ["E1", "E2"]
    headings = [b.text for b in model.blocks if b.kind == "heading"]
    assert headings == ["Sec"]


def test_to_model_runresult_falls_back_to_raw_text_when_output_empty():
    # Empty structured output ({}) is falsy -> raw_text is used instead.
    result = RunResult(output={}, raw_text="# Title\n\nbody")
    model = DocumentBuilder().to_model(result)
    assert model.title == "Title"
    assert any(b.text == "body" for b in model.blocks)


def test_to_model_pydantic_model_dump_path():
    contract = DocumentContract(min_words=5)  # any pydantic model with model_dump
    model = DocumentBuilder().to_model(contract)
    # model_dump produces a bare mapping -> definition section rendering.
    headings = {b.text for b in model.blocks if b.kind == "heading"}
    assert "Min Words" in headings


def test_to_model_plain_dict_dispatch():
    model = DocumentBuilder().to_model({"sections": [{"heading": "D", "body": "y"}]})
    headings = [b.text for b in model.blocks if b.kind == "heading"]
    assert headings == ["D"]


def test_to_model_list_wrapped_as_sections():
    model = DocumentBuilder().to_model([{"heading": "A", "body": "x"}])
    headings = [b.text for b in model.blocks if b.kind == "heading"]
    assert headings == ["A"]


def test_to_model_unsupported_type_raises():
    with pytest.raises(GenerationError, match="cannot build a document from int"):
        DocumentBuilder().to_model(123)


# -- DocumentBuilder.build ---------------------------------------------------


def test_build_merges_footnotes_bibliography_and_dedupes_evidence():
    model = DocumentModel(title="T", source_evidence_ids=["E1"])
    model.paragraph("body text here with several words")
    art = DocumentBuilder().build(
        model,
        format="markdown",
        footnotes=["fn1"],
        bibliography=["bib1"],
        evidence_ids=["E1", "E2"],
    )
    # Footnotes/bibliography land in rendered markdown; evidence dedupes E1.
    assert model.footnotes == ["fn1"]
    assert model.bibliography == ["bib1"]
    assert model.source_evidence_ids == ["E1", "E2"]
    assert "fn1" in art.text
    assert "bib1" in art.text


def test_build_contract_failure_raises_with_violations():
    with pytest.raises(DocumentContractError, match="does not satisfy its contract") as exc:
        DocumentBuilder().build(
            "# Doc\n\nshort",
            format="markdown",
            contract=DocumentContract(required_sections=["Methodology"]),
        )
    assert any("Methodology" in v for v in exc.value.violations)


def test_build_contract_success_records_repairs_in_metadata():
    art = DocumentBuilder().build(
        "# Memo\n\nbody one [E1].",
        format="markdown",
        contract=DocumentContract(require_title=True),
    )
    assert "contract_repairs" in art.metadata
    assert isinstance(art.metadata["contract_repairs"], list)


def test_build_emits_audit_entry_with_computed_details():
    audit = AuditLog()
    DocumentBuilder(audit_log=audit, tenant_id="acme").build(
        DocumentModel(title="Memo", blocks=[DocBlock(kind="paragraph", text="one two three")]),
        format="markdown",
    )
    assert len(audit.entries) == 1
    entry = audit.entries[0]
    assert entry.action == "document_generate"
    assert entry.tenant_id == "acme"
    assert entry.resource == "Memo"
    assert entry.details["blocks"] == 1
    assert entry.details["words"] == 3
    assert entry.details["format"] == "markdown"
    assert len(entry.details["content_sha256"]) == 64


def test_build_no_audit_log_is_noop():
    # audit_log is None -> _audit returns early without raising.
    art = DocumentBuilder().build("# T\n\nbody", format="markdown")
    assert art.format == "markdown"


def test_build_contract_table_spec_violation():
    contract = DocumentContract(
        table_specs=[TableSpec(match="*", required_columns=["X"], min_rows=1)],
    )
    with pytest.raises(DocumentContractError):
        DocumentBuilder().build("# Doc\n\nno tables here", format="markdown", contract=contract)


# -- redline -----------------------------------------------------------------


def test_redline_markdown_marks_insertions_and_deletions():
    art = generate_redline("the cat sat", "the dog sat", format="markdown")
    body = art.text
    assert "~~cat~~" in body  # deleted word struck
    assert "**dog**" in body  # inserted word bolded
    assert "the" in body  # equal tokens kept


def test_redline_pure_insertion_only_bolds():
    # Appending a trailing token is a clean insertion (no replace, no delete).
    art = generate_redline("alpha beta ", "alpha beta gamma", format="markdown")
    body = art.text
    assert "**gamma**" in body
    assert "~~" not in body


def test_redline_pure_deletion_only_strikes():
    art = generate_redline("alpha beta", "alpha", format="html")
    body = art.text
    # HTML renders the marked-up paragraph; both tokens appear.
    assert "beta" in body
    assert "alpha" in body


def test_redline_ops_pure_delete_emits_no_insert():
    from vincio.generation.builder import _redline_ops

    ops = _redline_ops("a b c", "a c")
    assert ops == [("equal", "a "), ("delete", "b "), ("equal", "c")]
    # A pure-delete tag must NOT also emit an insert op.
    assert not any(op == "insert" for op, _ in ops)


def test_redline_ops_replace_emits_both_delete_and_insert():
    from vincio.generation.builder import _redline_ops

    ops = _redline_ops("a old c", "a new c")
    kinds = [op for op, _ in ops]
    # A replace expands into a delete followed by an insert.
    assert "delete" in kinds
    assert "insert" in kinds
    assert kinds.index("delete") < kinds.index("insert")


def test_redline_docx_produces_zip_artifact():
    pytest.importorskip("docx")
    art = generate_redline("the cat sat", "the dog sat", format="docx", title="Diff")
    assert art.format == "docx"
    assert art.title == "Diff"
    # A .docx is a ZIP container — verify the PK magic bytes and non-empty body.
    assert art.content[:2] == b"PK"
    assert len(art.content) > 0


def test_redline_identical_text_has_no_marks():
    art = generate_redline("same words here", "same words here", format="markdown")
    body = art.text
    assert "**" not in body
    assert "~~" not in body
    assert "same words here" in body
