# Agent Skills

[Agent Skills](https://www.anthropic.com/news/skills) package procedural
knowledge as a `SKILL.md` file (YAML frontmatter + Markdown body, optional
bundled scripts), donated to the Agentic AI Foundation. Vincio loads them as
**budgeted, scored, cited** context, not a privileged side channel, with
**progressive disclosure**.

## SKILL.md format

```markdown
---
name: pdf-invoice
description: Extract totals and line items from PDF invoices. Use for invoice/PDF tasks.
keywords: [pdf, invoice, extract]
license: Apache-2.0
---

# Extracting PDF invoices

1. Locate the invoice header (vendor, date, invoice number).
2. Read the line-item table; sum the amounts.
3. Reconcile the sum against the stated total.
```

`name` and `description` are required (the description drives relevance and the
always-on index). A conventional `scripts/` subdirectory is auto-discovered.

## Load a skill

```python
from vincio import ContextApp

app = ContextApp(name="assistant")
app.add_skill("skills/pdf-invoice")            # a directory or a SKILL.md path
app.add_skill("skills/pdf-invoice", register_scripts=True)  # + bundled scripts
```

You can also load directly:

```python
from vincio.skills import load_skill, load_skills

skill = load_skill("skills/pdf-invoice")
skills = load_skills("skills/")                # every subdir with a SKILL.md
```

## Progressive disclosure

Skills cost context only when used:

1. **Level 1, always disclosed.** A one-line index (name + description) per
   skill, so the model knows which skills exist. Cheap; always in budget.
2. **Level 2, disclosed on relevance.** A skill's full instructions enter the
   packet *only* when the task matches it above a threshold. The context
   compiler then scores, budgets, and cites the body like any other evidence,
   so an unused skill costs only its index line.

```python
# Off-topic task → index only; relevant task → index + the matching body.
app.skill_library.evidence_for("translate to French")   # [skill_index]
app.skill_library.evidence_for("extract the pdf total") # [skill_index, skill]
```

Skill evidence carries `metadata["origin"] = "skill:<name>"`, so a skill that
shapes an answer is traceable.

## Bundled scripts as sandboxed tools

With `register_scripts=True`, each bundled script becomes a tool that runs in
the resource-limited subprocess sandbox (timeout, output caps, scrubbed env,
POSIX `setrlimit`) through the permissioned, audited tool runtime, namespaced
`"<skill>.<script>"`, `side_effects="external"`. Pass `permissions=[...]` to
`register_skill_scripts` to additionally gate them behind an RBAC scope.

See [`examples/10_interop_and_protocols.py`](../../examples/10_interop_and_protocols.py).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: add tools](add-tools.md)
- [Example: 04_agents_and_tools.py](../../examples/04_agents_and_tools.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
