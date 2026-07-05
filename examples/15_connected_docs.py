"""Connected docs & the capability map — the documentation as one verifiable graph.

``vincio._docmap`` (the module behind the ``vincio docs`` CLI) is the single source
of truth binding every public ``app.*`` verb to the concept / guide / example /
reference that document it, grouped by the six capability facades. It renders the
capability map, the Related cross-links, the learning path, and ``llms.txt`` — and
a graph check (link integrity, coverage, orphans) gates CI. This is the tour a
*contributor* reads to understand how ``vincio docs`` works; it runs fully offline
with no provider and no network.
"""

from __future__ import annotations

from vincio import _docmap


def main() -> None:
    # 1. The capability map: every app.* verb has exactly one documented home under
    #    one of six facades, bound to a concept/guide/example/reference. This is what
    #    guarantees a new public verb can't ship undocumented — the gate (section 4)
    #    fails if any verb is unmapped.
    verbs = _docmap.app_verbs()
    index = {v: _docmap.topic_for_verb(v) for v in verbs}
    print(f"1. Capability map — {len(verbs)} app.* verbs across {len(_docmap.FACETS)} facades")
    for key, _title, _blurb in _docmap.FACETS:
        n = sum(1 for v in verbs if index[v] and index[v].facet == key)
        print(f"   app.{key:<13} {n:>3} verbs")
    topic = next(t for t in _docmap.TOPICS if t.key == "retrieval")
    print(f"   e.g. '{topic.title}' binds: concept={topic.concept}, "
          f"guide(s)={topic.guides}, example={topic.examples}")

    # 2. Related cross-links: one single-sourced block lands on every concept and
    #    guide so a reader traverses laterally instead of returning to the index.
    #    Single-sourcing it from the graph means links can't rot independently.
    print("\n2. Related cross-links (docs/concepts/retrieval.md)")
    block = _docmap.render_related_block("docs/concepts/retrieval.md")
    for line in block.splitlines():
        if line.startswith(("- ", "## ")):
            print(f"   {line}")

    # 3. The learning path: a staged getting-started → grow-into-depth spine,
    #    rendered to docs/learning-path.md from the same graph.
    print("\n3. Learning path stages")
    stages = [ln[3:] for ln in _docmap.render_learning_path().splitlines() if ln.startswith("## ")]
    print(f"   {' → '.join(stages)}")

    # 4. The docs-graph check — exactly what `vincio docs check` and the
    #    docs_conformance bench run: link integrity, map coverage, reachability,
    #    no orphans, llms.txt freshness. Each check pinpoints its own problems.
    print("\n4. Docs-graph check (the CI gate)")
    for check in _docmap.docs_graph_report():
        print(f"   [{'PASS' if check.ok else 'FAIL'}] {check.name}"
              + (f" — {check.problems[:3]}" if check.problems else ""))

    # 5. The gate bites: a synthetic broken link and an unmapped verb are both
    #    caught, and llms.txt is regenerated from vincio.__all__ (so it can't drift
    #    from the real public surface — the committed copy is checked for freshness).
    import vincio

    broken = _docmap.MarkdownLink(source="docs/README.md", text="x", target="../nope/missing.md")
    print("\n5. The gate bites")
    print(f"   broken link caught={_docmap._resolve_link(broken) is not None}; "
          f"unmapped verb has no topic={_docmap.topic_for_verb('not_a_real_verb') is None}; "
          f"every real verb mapped={_docmap.uncovered_verbs() == []}")
    print(f"   llms.txt: {len(vincio.__all__)} public symbols, "
          f"committed copy current={_docmap.llms_txt_current().ok}")

    print("\nDone — the docs are one connected, verifiable graph: every verb "
          "documented, every link resolving, every concept reachable, offline.")


if __name__ == "__main__":
    main()
