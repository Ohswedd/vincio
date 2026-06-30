#!/usr/bin/env python3
"""Generate the Vincio GitHub Wiki from the in-repo ``docs/`` tree.

The wiki is a *mirror* of ``docs/`` — the documentation lives in the repository
(reviewed, versioned, CI-gated) and this script renders it into the flat page
namespace a GitHub wiki uses. It is run by ``.github/workflows/wiki-sync.yml`` on
every push to ``main`` that touches ``docs/``, so the wiki is always current; it
can also be run locally for the initial population.

What it does:

* maps each ``docs/<section>/<name>.md`` to a flat, well-titled wiki page
  (``Memory.md`` → page "Memory"), de-duplicating across sections;
* rewrites intra-doc relative links to wiki page slugs, and links to other repo
  files (examples, source, root docs) to absolute ``blob/main`` URLs, so no link
  dangles in the flattened namespace;
* strips each page's leading H1 (the wiki renders the page title from its name)
  and prepends a short "synced from source" banner so nobody edits the wiki by
  hand;
* generates ``Home.md`` (landing page + index), ``_Sidebar.md`` (grouped nav),
  and ``_Footer.md``.

Unmapped/new docs still sync: a sensible title is derived from the filename, so
adding a doc never silently drops it from the wiki.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

DOCS_DIR = "docs"
DEFAULT_REPO = "Ohswedd/vincio"

# Section ordering and human labels for the sidebar / home index.
SECTION_ORDER = ["", "concepts", "guides", "reference", "comparisons", "security"]
SECTION_LABELS = {
    "": "Overview",
    "concepts": "Concepts",
    "guides": "Guides",
    "reference": "Reference",
    "comparisons": "Comparisons",
    "security": "Security",
}

# Curated titles (repo-relative path -> wiki title). Titles use plain ASCII
# (no ":" or "&") because a wiki page title is derived from its file name; the
# sidebar supplies prettier display text where needed.
TITLES: dict[str, str] = {
    "docs/README.md": "Documentation Index",
    "docs/getting-started.md": "Getting Started",
    "docs/learning-path.md": "Learning Path",
    # concepts
    "docs/concepts/context-packets.md": "Context Packets",
    "docs/concepts/prompt-compiler.md": "Prompt Compiler",
    "docs/concepts/memory.md": "Memory",
    "docs/concepts/retrieval.md": "Retrieval",
    "docs/concepts/agents.md": "Agents and Workflows",
    "docs/concepts/evals.md": "Evaluation",
    "docs/concepts/observability.md": "Observability",
    "docs/concepts/ergonomic-surface.md": "Ergonomic Surface",
    "docs/concepts/tabular-evidence.md": "Tabular Evidence",
    "docs/concepts/dataset-profiling.md": "Dataset Profiling",
    "docs/concepts/governed-text-to-query.md": "Governed Text-to-Query",
    "docs/concepts/data-analysis-agent.md": "Data Analysis Agent",
    "docs/concepts/charts-and-cited-artifacts.md": "Charts and Cited Artifacts",
    "docs/concepts/streaming-and-out-of-core.md": "Streaming and Out-of-Core",
    "docs/concepts/semantic-layer-and-governed-metrics.md": "Semantic Layer and Governed Metrics",
    "docs/concepts/realtime-streaming-analytics.md": "Real-Time Streaming Analytics",
    "docs/concepts/data-engagement.md": "Data Engagement",
    "docs/concepts/federated-data-engagement.md": "Federated Data Engagement",
    # guides
    "docs/guides/build-rag-app.md": "Build a RAG App",
    "docs/guides/assistant.md": "Build an Assistant",
    "docs/guides/connectors.md": "Connect Data Sources",
    "docs/guides/structured-output.md": "Structured Output",
    "docs/guides/reliability-guardrails.md": "Reliability and Guardrails",
    "docs/guides/add-tools.md": "Add Tools",
    "docs/guides/orchestrate-agents.md": "Orchestrate Agents",
    "docs/guides/run-evals.md": "Run Evals",
    "docs/guides/test-llm-apps.md": "Test LLM Apps",
    "docs/guides/agentic-eval.md": "Agentic Evaluation",
    "docs/guides/optimize-context.md": "Optimize Context",
    "docs/guides/close-the-loop.md": "Close the Loop",
    "docs/guides/reasoning.md": "Reasoning Control",
    "docs/guides/verified-reasoning.md": "Verified Reasoning",
    "docs/guides/analyze-data.md": "Analyze Data",
    "docs/guides/generate-documents.md": "Generate Documents and Media",
    "docs/guides/realtime.md": "Realtime Voice and Streaming",
    "docs/guides/performance.md": "Performance and Streaming",
    "docs/guides/cost-and-reliability.md": "Cost and Reliability",
    "docs/guides/edge.md": "Edge and WASM Runtime",
    "docs/guides/computer-use.md": "Computer Use",
    "docs/guides/video.md": "Video Understanding and Generation",
    "docs/guides/mcp.md": "MCP Client and Server",
    "docs/guides/a2a.md": "A2A Agent-to-Agent",
    "docs/guides/agent-fabric.md": "Agent Fabric",
    "docs/guides/agent-skills.md": "Agent Skills",
    "docs/guides/agent-identity.md": "Agent Identity and Delegation",
    "docs/guides/negotiation.md": "Agent Negotiation and Contracting",
    "docs/guides/choreography.md": "Cross-Org Choreography",
    "docs/guides/settlement.md": "Settlement and Metering",
    "docs/guides/governance.md": "Governance",
    "docs/guides/governance-verification.md": "Governance Verification",
    "docs/guides/differential-privacy.md": "Differential Privacy",
    "docs/guides/assurance.md": "Assurance and Certification",
    "docs/guides/skill-acquisition.md": "Skill Acquisition",
    "docs/guides/vertical-packs.md": "Vertical Packs",
    "docs/guides/plugins.md": "Plugins",
    "docs/guides/integrations.md": "Integrations",
    "docs/guides/cookbook.md": "Cookbook",
    "docs/guides/migrate-from-langchain.md": "Migrate from LangChain",
    "docs/guides/migrate-from-llamaindex.md": "Migrate from LlamaIndex",
    "docs/guides/migrate-from-ragas.md": "Migrate from Ragas",
    "docs/guides/migrate-from-mem0.md": "Migrate from Mem0",
    # reference
    "docs/reference/api.md": "API Reference",
    "docs/reference/api-generated.md": "API Reference (Generated)",
    "docs/reference/capability-map.md": "Capability Map",
    "docs/reference/cli.md": "CLI Reference",
    "docs/reference/config.md": "Configuration",
    "docs/reference/errors.md": "Error Reference",
    "docs/reference/typing.md": "Typing",
    "docs/reference/stability.md": "API Stability and Deprecation",
    "docs/reference/slo.md": "Performance and Quality SLOs",
    # comparisons
    "docs/comparisons/langchain.md": "Vincio vs LangChain",
    "docs/comparisons/llamaindex.md": "Vincio vs LlamaIndex",
    "docs/comparisons/ragatouille.md": "Vincio vs RAGatouille",
    "docs/comparisons/mem0.md": "Vincio vs Mem0",
    "docs/comparisons/crewai.md": "Vincio vs CrewAI",
    "docs/comparisons/openai-agents-sdk.md": "Vincio vs OpenAI Agents SDK",
    "docs/comparisons/dspy.md": "Vincio vs DSPy",
    "docs/comparisons/pydantic-ai.md": "Vincio vs Pydantic AI",
    "docs/comparisons/guardrails.md": "Vincio vs Guardrails AI",
    "docs/comparisons/nemo-guardrails.md": "Vincio vs NeMo Guardrails",
    "docs/comparisons/ragas.md": "Vincio vs Ragas",
    "docs/comparisons/deepeval.md": "Vincio vs DeepEval",
    "docs/comparisons/langsmith-langfuse.md": "Vincio vs LangSmith and Langfuse",
    "docs/comparisons/litellm.md": "Vincio vs LiteLLM",
    # security
    "docs/security/threat-model.md": "Threat Model",
}

# Within-section ordering for the index/sidebar (by repo path). Anything not
# listed is appended alphabetically by title, so new docs still appear.
GUIDE_ORDER = [
    "docs/guides/build-rag-app.md",
    "docs/guides/assistant.md",
    "docs/guides/connectors.md",
    "docs/guides/structured-output.md",
    "docs/guides/reliability-guardrails.md",
    "docs/guides/add-tools.md",
    "docs/guides/orchestrate-agents.md",
    "docs/guides/run-evals.md",
    "docs/guides/test-llm-apps.md",
    "docs/guides/agentic-eval.md",
    "docs/guides/optimize-context.md",
    "docs/guides/close-the-loop.md",
    "docs/guides/reasoning.md",
    "docs/guides/verified-reasoning.md",
    "docs/guides/analyze-data.md",
    "docs/guides/generate-documents.md",
    "docs/guides/realtime.md",
    "docs/guides/performance.md",
    "docs/guides/cost-and-reliability.md",
    "docs/guides/edge.md",
    "docs/guides/computer-use.md",
    "docs/guides/video.md",
    "docs/guides/mcp.md",
    "docs/guides/a2a.md",
    "docs/guides/agent-fabric.md",
    "docs/guides/agent-skills.md",
    "docs/guides/agent-identity.md",
    "docs/guides/negotiation.md",
    "docs/guides/choreography.md",
    "docs/guides/settlement.md",
    "docs/guides/governance.md",
    "docs/guides/governance-verification.md",
    "docs/guides/differential-privacy.md",
    "docs/guides/assurance.md",
    "docs/guides/skill-acquisition.md",
    "docs/guides/vertical-packs.md",
    "docs/guides/plugins.md",
    "docs/guides/integrations.md",
    "docs/guides/cookbook.md",
    "docs/guides/migrate-from-langchain.md",
    "docs/guides/migrate-from-llamaindex.md",
    "docs/guides/migrate-from-ragas.md",
    "docs/guides/migrate-from-mem0.md",
]
CONCEPT_ORDER = [
    "docs/concepts/context-packets.md",
    "docs/concepts/prompt-compiler.md",
    "docs/concepts/retrieval.md",
    "docs/concepts/memory.md",
    "docs/concepts/agents.md",
    "docs/concepts/evals.md",
    "docs/concepts/observability.md",
    "docs/concepts/ergonomic-surface.md",
    "docs/concepts/tabular-evidence.md",
    "docs/concepts/dataset-profiling.md",
    "docs/concepts/governed-text-to-query.md",
    "docs/concepts/data-analysis-agent.md",
    "docs/concepts/charts-and-cited-artifacts.md",
    "docs/concepts/streaming-and-out-of-core.md",
    "docs/concepts/semantic-layer-and-governed-metrics.md",
    "docs/concepts/realtime-streaming-analytics.md",
    "docs/concepts/data-engagement.md",
    "docs/concepts/federated-data-engagement.md",
]
REFERENCE_ORDER = [
    "docs/reference/api.md",
    "docs/reference/api-generated.md",
    "docs/reference/capability-map.md",
    "docs/reference/cli.md",
    "docs/reference/config.md",
    "docs/reference/errors.md",
    "docs/reference/typing.md",
    "docs/reference/stability.md",
    "docs/reference/slo.md",
]

# Acronyms to preserve when auto-titling an unmapped file name.
_ACRONYMS = {
    "rag": "RAG", "api": "API", "cli": "CLI", "mcp": "MCP", "a2a": "A2A",
    "slo": "SLO", "llm": "LLM", "ai": "AI", "qa": "QA", "sql": "SQL",
    "pii": "PII", "sso": "SSO", "dspy": "DSPy", "crewai": "CrewAI",
}


def _slug(title: str) -> str:
    """Wiki page slug == file stem; the wiki shows it as the title with hyphens
    rendered as spaces."""
    cleaned = re.sub(r"[^0-9A-Za-z\- ]+", "", title)
    return re.sub(r"\s+", "-", cleaned.strip())


def _auto_title(path: str) -> str:
    stem = Path(path).stem
    words = []
    for part in stem.replace("_", "-").split("-"):
        words.append(_ACRONYMS.get(part.lower(), part.capitalize()))
    return " ".join(words)


def _discover(root: Path) -> list[str]:
    docs = root / DOCS_DIR
    return sorted(str(p.relative_to(root)).replace(os.sep, "/") for p in docs.rglob("*.md"))


def _section_of(relpath: str) -> str:
    parts = relpath.split("/")
    return parts[1] if len(parts) > 2 else ""


def _title_of(relpath: str) -> str:
    return TITLES.get(relpath) or _auto_title(relpath)


def _build_page_map(paths: list[str]) -> dict[str, str]:
    """repo-relative doc path -> wiki slug, de-duplicated."""
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for path in paths:
        slug = _slug(_title_of(path))
        if slug in used:  # disambiguate with the section
            slug = _slug(f"{SECTION_LABELS.get(_section_of(path), '')} {_title_of(path)}")
        used.add(slug)
        mapping[path] = slug
    return mapping


def _rewrite_links(body: str, src_relpath: str, page_map: dict[str, str], repo: str, branch: str) -> str:
    blob = f"https://github.com/{repo}/blob/{branch}"
    src_dir = os.path.dirname(src_relpath)
    link_re = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")

    def repl(m: re.Match[str]) -> str:
        bang, text, target = m.group(1), m.group(2), m.group(3).strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "<")):
            return m.group(0)
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        if not target:  # pure anchor
            return f"{bang}[{text}]({anchor})"
        resolved = os.path.normpath(os.path.join(src_dir, target)).replace(os.sep, "/")
        if bang == "!":  # image -> raw URL
            raw = f"https://raw.githubusercontent.com/{repo}/{branch}"
            return f"![{text}]({raw}/{resolved})"
        if resolved in page_map:
            return f"[{text}]({page_map[resolved]}{anchor})"
        return f"[{text}]({blob}/{resolved}{anchor})"

    return link_re.sub(repl, body)


def _strip_leading_h1(body: str) -> tuple[str, str]:
    lines = body.splitlines()
    title = ""
    out = []
    skipped = False
    for line in lines:
        if not skipped and line.startswith("# "):
            title = line[2:].strip()
            skipped = True
            continue
        if not skipped and line.strip() == "":
            continue  # leading blank lines before the H1
        out.append(line)
    return "\n".join(out).lstrip("\n"), title


def _render_page(src_relpath: str, root: Path, page_map: dict[str, str], repo: str, branch: str) -> str:
    raw = (root / src_relpath).read_text(encoding="utf-8")
    body, h1 = _strip_leading_h1(raw)
    body = _rewrite_links(body, src_relpath, page_map, repo, branch)
    blob = f"https://github.com/{repo}/blob/{branch}/{src_relpath}"
    heading = h1 or _title_of(src_relpath)
    banner = (
        f"> **{heading}** &nbsp;·&nbsp; "
        f"_Synced from [`{src_relpath}`]({blob}). Edit the source in the repo — "
        f"changes here are overwritten on the next sync._\n"
    )
    return f"{banner}\n{body}\n"


def _ordered(paths: list[str], section: str) -> list[str]:
    order = {"concepts": CONCEPT_ORDER, "guides": GUIDE_ORDER, "reference": REFERENCE_ORDER}.get(section, [])
    in_section = [p for p in paths if _section_of(p) == section]
    ranked = sorted(
        in_section,
        key=lambda p: (order.index(p) if p in order else len(order), _title_of(p).lower()),
    )
    return ranked


def _render_sidebar(paths: list[str], page_map: dict[str, str]) -> str:
    lines = ["### Vincio", "", "[Home](Home)", ""]
    for section in SECTION_ORDER:
        section_paths = _ordered(paths, section)
        if not section_paths:
            continue
        lines.append(f"**{SECTION_LABELS[section]}**")
        lines.append("")
        for path in section_paths:
            lines.append(f"- [{_title_of(path)}]({page_map[path]})")
        lines.append("")
    lines += ["---", "[PyPI](https://pypi.org/project/vincio/) · "
              "[Repo](https://github.com/Ohswedd/vincio) · "
              "[Discussions](https://github.com/Ohswedd/vincio/discussions)"]
    return "\n".join(lines) + "\n"


def _render_footer(repo: str) -> str:
    return (
        f"---\n"
        f"_This wiki mirrors [`docs/`](https://github.com/{repo}/tree/main/docs) and is "
        f"regenerated automatically — open a PR against the docs to change a page._ · "
        f"[Apache-2.0](https://github.com/{repo}/blob/main/LICENSE)\n"
    )


def _render_home(paths: list[str], page_map: dict[str, str], repo: str) -> str:
    out: list[str] = []
    out.append("# Vincio Wiki\n")
    out.append(
        "**Vincio** is the context engineering platform for AI applications. It compiles "
        "prompts, memory, retrieval, tools, schemas, and policies into optimized, validated, "
        "observable, provider-neutral **context packets** — then evaluates every output.\n"
    )
    out.append(
        "> This wiki is an always-current mirror of the repository's "
        f"[`docs/`](https://github.com/{repo}/tree/main/docs). It is generated on every change "
        "to the docs, so it never drifts from the shipped library. To edit a page, edit its "
        "source in the repo.\n"
    )
    out.append("## Start here\n")
    out.append(
        "```bash\n"
        "pip install vincio          # core — runs fully offline on a deterministic mock provider\n"
        'pip install "vincio[all]"   # every optional integration\n'
        "```\n"
    )
    out.append(
        "- New to Vincio? Read **[Getting Started](Getting-Started)**.\n"
        "- Want the big picture? See the **[Concepts](#concepts)** below.\n"
        "- Building something specific? Jump to a **[Guide](#guides)**.\n"
        "- Looking up an API or CLI command? Go to the **[Reference](#reference)**.\n"
    )
    for section in SECTION_ORDER:
        if section == "":
            continue
        section_paths = _ordered(paths, section)
        if not section_paths:
            continue
        anchor = SECTION_LABELS[section]
        out.append(f"## {anchor}\n")
        for path in section_paths:
            out.append(f"- **[{_title_of(path)}]({page_map[path]})**")
        out.append("")
    out.append(
        "## Beyond the docs\n"
        f"- [Changelog](https://github.com/{repo}/blob/main/CHANGELOG.md) — "
        "what changed in each release\n"
        f"- [Roadmap](https://github.com/{repo}/blob/main/ROADMAP.md) — shipped and planned\n"
        f"- [Examples]({f'https://github.com/{repo}/tree/main/examples'}) — 16 runnable, offline examples\n"
        f"- [Discussions](https://github.com/{repo}/discussions) — questions, ideas, and show-and-tell\n"
        f"- [Security policy](https://github.com/{repo}/blob/main/SECURITY.md)\n"
    )
    return "\n".join(out) + "\n"


def build(root: Path, out: Path, repo: str, branch: str) -> int:
    paths = _discover(root)
    if not paths:
        raise SystemExit(f"no docs found under {root / DOCS_DIR}")
    page_map = _build_page_map(paths)
    out.mkdir(parents=True, exist_ok=True)

    for path in paths:
        (out / f"{page_map[path]}.md").write_text(
            _render_page(path, root, page_map, repo, branch), encoding="utf-8"
        )
    (out / "Home.md").write_text(_render_home(paths, page_map, repo), encoding="utf-8")
    (out / "_Sidebar.md").write_text(_render_sidebar(paths, page_map), encoding="utf-8")
    (out / "_Footer.md").write_text(_render_footer(repo), encoding="utf-8")
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Vincio wiki from docs/.")
    parser.add_argument("--out", default="wiki_build", help="output directory for wiki pages")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    parser.add_argument("--branch", default="main")
    parser.add_argument("--root", default=".", help="repository root")
    args = parser.parse_args()

    count = build(Path(args.root), Path(args.out), args.repo, args.branch)
    print(f"Generated {count} doc pages + Home/_Sidebar/_Footer into {args.out}/")


if __name__ == "__main__":
    main()
