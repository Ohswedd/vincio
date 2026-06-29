"""Every Colab notebook must be valid and run end to end, fully offline.

The executable half of the "Colab-ready notebooks" deliverable: each notebook is
valid Jupyter JSON, opens with a Colab badge and a single ``pip install`` cell,
its code cells carry no stale outputs, and — the real guarantee — its Python code
cells (excluding shell cells like ``!pip install``) run top to bottom with no
network and no API keys, in an isolated subprocess on the bundled mock provider.
So a notebook can never drift from the working API.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "examples" / "notebooks").glob("*.ipynb"))


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _is_shell_cell(cell: dict) -> bool:
    return any(line.lstrip().startswith(("!", "%")) for line in _source(cell).splitlines())


def test_notebooks_present():
    assert len(NOTEBOOKS) >= 5, "expected at least five Colab notebooks"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_well_formed(nb):
    doc = json.loads(nb.read_text(encoding="utf-8"))
    assert doc.get("nbformat") == 4, f"{nb.name}: not nbformat 4"
    cells = doc["cells"]
    assert cells and cells[0]["cell_type"] == "markdown", f"{nb.name}: first cell must be markdown"
    assert "colab.research.google.com" in _source(cells[0]), f"{nb.name}: missing Open-in-Colab badge"

    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert code_cells, f"{nb.name}: no code cells"
    assert any("pip install" in _source(c) for c in code_cells), f"{nb.name}: missing a pip-install cell"
    for c in code_cells:
        assert c.get("execution_count") is None, f"{nb.name}: a code cell carries a stale execution_count"
        assert c.get("outputs") == [], f"{nb.name}: a code cell carries stale outputs"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs_offline(nb):
    doc = json.loads(nb.read_text(encoding="utf-8"))
    code = "\n\n".join(
        _source(c)
        for c in doc["cells"]
        if c["cell_type"] == "code" and not _is_shell_cell(c)
    )
    # Sanity: the assembled program parses before we try to run it.
    compile(code, nb.name, "exec")

    env = dict(os.environ)
    env.pop("VINCIO_PROVIDER", None)  # force the offline mock default
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(nb.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, f"{nb.name} failed to run offline:\n{proc.stderr}"
