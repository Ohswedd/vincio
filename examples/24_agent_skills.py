"""Agent Skills — portable SKILL.md procedural knowledge with budgeting (1.1).

Load Anthropic-style ``SKILL.md`` skills. Vincio injects them through the
context compiler with *progressive disclosure*: a one-line summary is always
available, and a skill's full instructions are included only when the task is
relevant — so an unused skill costs only its index line. Skills are scored,
budgeted, and cited like any other context (not a privileged side channel).

Runs fully offline. No API keys needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from _shared import example_provider

from vincio import ContextApp

SKILL_MD = """---
name: pdf-invoice
description: Extract totals and line items from PDF invoices. Use for invoice/PDF tasks.
keywords: [pdf, invoice, extract, total]
license: Apache-2.0
---

# Extracting PDF invoices

1. Locate the invoice header (vendor, date, invoice number).
2. Read the line-item table; sum the amounts.
3. Reconcile the sum against the stated total.
"""


def write_skill(root: Path) -> Path:
    skill_dir = root / "pdf-invoice"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (skill_dir / "scripts" / "checksum.py").write_text("print('rows: 3, total: 120.00')\n")
    return skill_dir


def main() -> None:
    provider, model = example_provider()
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = write_skill(Path(tmp))

        app = ContextApp(name="skills_demo", provider=provider, model=model)
        # register_scripts=True exposes bundled scripts as sandboxed tools.
        app.add_skill(str(skill_dir), register_scripts=True)
        print("loaded skills:", [s.name for s in app.skill_library.skills])
        print("bundled script tool:", "pdf-invoice.checksum" in app.enabled_tools)

        # Progressive disclosure: the body discloses only when the task matches.
        relevant = app.skill_library.evidence_for("extract the total from this pdf invoice")
        print("relevant task discloses:", [e.metadata["kind"] for e in relevant])
        off_topic = app.skill_library.evidence_for("what is the capital of France")
        print("off-topic task discloses:", [e.metadata["kind"] for e in off_topic])

        result = app.run("Extract the total from invoice INV-204 (PDF).")
        print("run output:", str(result.output)[:50])
        print("trace:", result.trace_id, f"cost: ${result.cost_usd:.6f}")


if __name__ == "__main__":
    main()
