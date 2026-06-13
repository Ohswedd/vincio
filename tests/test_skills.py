"""Agent Skills (1.1): SKILL.md parsing, progressive disclosure, sandboxed scripts."""

from __future__ import annotations

import pytest

from vincio import ContextApp
from vincio.providers import MockProvider
from vincio.skills import SkillError, SkillLibrary, load_skill, load_skills, parse_skill_md

SKILL_MD = """---
name: pdf-extract
description: Extract text and tables from PDF invoices. Use when a PDF invoice is provided.
keywords: [pdf, invoice, extract]
license: Apache-2.0
---

# PDF extraction

1. Open the PDF.
2. Extract the line items.
"""


def _write_skill(directory, *, body: str = SKILL_MD, script: bool = False):
    d = directory / "pdf-extract"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    if script:
        (d / "scripts").mkdir()
        (d / "scripts" / "extract.py").write_text("print('extracted')\n", encoding="utf-8")
    return d


def test_parse_skill_md_splits_frontmatter_and_body():
    meta, body = parse_skill_md(SKILL_MD)
    assert meta["name"] == "pdf-extract"
    assert meta["license"] == "Apache-2.0"
    assert body.startswith("# PDF extraction")


def test_parse_skill_md_requires_frontmatter():
    with pytest.raises(SkillError):
        parse_skill_md("no frontmatter here")


def test_load_skill_from_directory(tmp_path):
    skill = load_skill(_write_skill(tmp_path))
    assert skill.name == "pdf-extract"
    assert "Apache-2.0" == skill.metadata["license"]
    assert skill.keywords == ["pdf", "invoice", "extract"]
    assert "Extract" in skill.description


def test_load_skill_discovers_scripts(tmp_path):
    skill = load_skill(_write_skill(tmp_path, script=True))
    assert [s.name for s in skill.scripts] == ["extract"]
    assert skill.scripts[0].language == "python"


def test_load_skill_requires_description(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: x\n---\nbody", encoding="utf-8")
    with pytest.raises(SkillError):
        load_skill(d)


def test_load_skills_directory(tmp_path):
    _write_skill(tmp_path)
    skills = load_skills(tmp_path)
    assert len(skills) == 1


def test_match_score_is_relevance_ordered(tmp_path):
    skill = load_skill(_write_skill(tmp_path))
    assert skill.match_score("extract this pdf invoice") > 0
    assert skill.match_score("what is the weather today") == 0.0


def test_progressive_disclosure(tmp_path):
    lib = SkillLibrary()
    lib.add(load_skill(_write_skill(tmp_path)))
    # Index is always present; the body only when relevant.
    relevant = lib.evidence_for("please extract the pdf invoice line items")
    assert [e.metadata["kind"] for e in relevant] == ["skill_index", "skill"]
    irrelevant = lib.evidence_for("what is the capital of France")
    assert [e.metadata["kind"] for e in irrelevant] == ["skill_index"]
    # Skill body carries provenance.
    body = relevant[1]
    assert body.metadata["origin"] == "skill:pdf-extract"


def test_add_skill_to_app_injects_evidence(tmp_path):
    app = ContextApp(name="t", provider=MockProvider(default_text="ok"), model="mock-1")
    app.add_skill(_write_skill(tmp_path))
    assert len(app.skill_library) == 1
    # The run completes and the skill index is available as evidence.
    result = app.run("extract the pdf invoice")
    assert result.output == "ok"


def test_add_skill_registers_scripts_as_tools(tmp_path):
    app = ContextApp(name="t", provider=MockProvider(default_text="ok"), model="mock-1")
    app.add_skill(_write_skill(tmp_path, script=True), register_scripts=True)
    assert "pdf-extract.extract" in app.enabled_tools
    spec = app.tool_registry.get("pdf-extract.extract").spec
    assert spec.side_effects == "external"  # sandboxed + external-policy governed


@pytest.mark.asyncio
async def test_skill_script_runs_in_sandbox(tmp_path):
    from vincio.core.types import ToolCall

    app = ContextApp(name="t", provider=MockProvider(), model="mock-1")
    app.add_skill(_write_skill(tmp_path, script=True), register_scripts=True)
    result = await app.tool_runtime.execute(
        ToolCall(tool_name="pdf-extract.extract", arguments={})
    )
    assert result.status == "ok"
    assert "extracted" in result.output["stdout"]
