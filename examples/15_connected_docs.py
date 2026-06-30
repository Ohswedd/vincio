"""The connected docs & capability map: the documentation as one verifiable graph.

Vincio's docs are ~80 leaf pages — a concept, a guide, a reference entry, and a
runnable example per subsystem. 5.4 adds the **connective tissue**: a single source
of truth (`vincio._docmap`, the module behind `vincio docs`) that binds every public
`app.*` verb to the concept that explains it, the guide that applies it, the example
that demonstrates it, and the reference that specifies it — grouped by the six
capability facades — and renders the capability map, the staged learning path, a
Related cross-link block on every concept and guide, and `llms.txt` from it.

This program walks that graph, fully offline (no provider, no network):

  * the **capability map** — every `app.*` verb placed under one of six facades and
    bound to its concept / guide / example / reference;
  * **Related cross-links** — the single-sourced block that lets a reader traverse
    laterally instead of returning to the index;
  * the **learning path** — a staged getting-started → grow-into-depth spine;
  * the **docs-graph check** — link integrity, capability-map coverage, navigation
    reachability, no orphans, and `llms.txt` freshness, the `docs_conformance`
    VincioBench family runs; and
  * the gate **bites** — a synthetic broken link and an unmapped verb are caught, and
    `llms.txt` is regenerated from `vincio.__all__`.

Everything is deterministic and dependency-free — the same checks `vincio docs check`
and the `docs_conformance` benchmark run.
"""

from __future__ import annotations

from vincio import _docmap


def banner(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def section_capability_map() -> None:
    banner("1. the capability map — every app.* verb has a documented home")
    verbs = _docmap.app_verbs()
    print(f"   {len(verbs)} public app.* methods across {len(_docmap.FACETS)} capability facades:")
    index = {v: _docmap.topic_for_verb(v) for v in verbs}
    for key, _title, _blurb in _docmap.FACETS:
        n = sum(1 for v in verbs if index[v] and index[v].facet == key)
        print(f"     - app.{key:<13} {n:>3} verbs")
    # Show one topic fully bound: concept + guide + example + reference.
    topic = next(t for t in _docmap.TOPICS if t.key == "retrieval")
    print(f"\n   e.g. the '{topic.title}' topic (facet: {topic.facet}) binds:")
    print(f"     verbs:    {', '.join('app.' + v for v in topic.verbs)}")
    print(f"     concept:  {topic.concept}")
    print(f"     guide(s): {', '.join(topic.guides)}")
    print(f"     example:  {', '.join(topic.examples)}")
    print("   (rendered to docs/reference/capability-map.md by `vincio docs map`)")


def section_related_cross_links() -> None:
    banner("2. Related cross-links — traverse laterally, not back to the index")
    page = "docs/concepts/retrieval.md"
    block = _docmap.render_related_block(page)
    # Print the human-readable bullets of the single-sourced block.
    for line in block.splitlines():
        if line.startswith("- ") or line.startswith("## "):
            print(f"   {line}")
    print("   (one of these blocks lands on every concept and guide, single-sourced)")


def section_learning_path() -> None:
    banner("3. the learning path — a staged getting-started → depth spine")
    for line in _docmap.render_learning_path().splitlines():
        if line.startswith("## "):
            print(f"   {line[3:]}")
    print("   (rendered to docs/learning-path.md)")


def section_docs_graph_check() -> None:
    banner("4. the docs-graph check — what `vincio docs check` and the bench gate")
    for check in _docmap.docs_graph_report():
        flag = "PASS" if check.ok else "FAIL"
        print(f"   [{flag}] {check.name}")
        for problem in check.problems[:5]:
            print(f"          - {problem}")


def section_gate_bites() -> None:
    banner("5. the gate bites — a broken link and an unmapped verb are caught")
    broken = _docmap.MarkdownLink(source="docs/README.md", text="x", target="../nope/missing.md")
    print(f"   synthetic broken link detected:   {_docmap._resolve_link(broken) is not None}")
    print(f"   unmapped verb has no topic:       {_docmap.topic_for_verb('not_a_real_verb') is None}")
    print(f"   every real verb is mapped:        {_docmap.uncovered_verbs() == []}")


def section_llms_txt() -> None:
    banner("6. llms.txt is regenerated from vincio.__all__ and current")
    import vincio

    rendered = _docmap.render_llms_txt()
    print(f"   public symbols in vincio.__all__: {len(vincio.__all__)}")
    print(f"   llms.txt lines (generated):       {rendered.count(chr(10))}")
    print(f"   committed llms.txt is current:    {_docmap.llms_txt_current().ok}")


def main() -> None:
    section_capability_map()
    section_related_cross_links()
    section_learning_path()
    section_docs_graph_check()
    section_gate_bites()
    section_llms_txt()
    print(
        "\nDone — the docs are one connected, verifiable graph: every verb documented, "
        "every link resolving, every concept reachable, offline."
    )


if __name__ == "__main__":
    main()
