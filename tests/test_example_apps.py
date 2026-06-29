"""Every real-world example application must run end to end, fully offline.

The executable half of the "real-world backend examples" deliverable: the docs
and READMEs make claims, these tests prove the code behind them runs with no
network and no API keys, on the bundled mock provider. Each app's dependency-free
``core.py`` is exercised in an isolated subprocess (so the four same-named
``core`` modules never collide and stray cwd writes stay contained); the FastAPI
``main.py`` shell is additionally exercised with a test client when FastAPI is
installed (CI installs only ``.[dev]``, so that part skips there).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APPS = ROOT / "examples" / "applications"

# (directory, snippet that imports core and prints a JSON result, required keys)
CORE_CASES = [
    (
        "rag_service",
        "import core, json; print(json.dumps(core.answer('what is the refund window?')))",
        {"answer", "citations", "cost_usd", "trace_id"},
    ),
    (
        "support_triage_api",
        "import core, json; print(json.dumps(core.triage('I was double charged', 'u1')))",
        {"category", "priority", "summary"},
    ),
    (
        "extraction_service",
        "import core, json; "
        "print(json.dumps(core.extract('Invoice from Acme Corp, total 1200.50 USD for widgets and gadgets')))",
        {"vendor", "total", "currency", "line_items"},
    ),
]


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("VINCIO_PROVIDER", None)  # force the offline mock default
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(app_dir: Path, snippet: str) -> subprocess.CompletedProcess:
    # `python -c` puts the cwd ('') on sys.path[0], so `import core` resolves the
    # app's own module without polluting this interpreter's import state.
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(app_dir),
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_application_dirs_present():
    for name in ("rag_service", "support_triage_api", "extraction_service", "cli_research_agent"):
        assert (APPS / name).is_dir(), f"missing example application: {name}"


@pytest.mark.parametrize("name, snippet, required", CORE_CASES, ids=[c[0] for c in CORE_CASES])
def test_app_core_runs_offline(name, snippet, required):
    proc = _run(APPS / name, snippet)
    assert proc.returncode == 0, f"{name}/core.py failed offline:\n{proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert required <= set(payload), f"{name} result missing keys: {required - set(payload)}"


@pytest.mark.parametrize("name, snippet, required", CORE_CASES, ids=[c[0] for c in CORE_CASES])
def test_app_fastapi_shell(name, snippet, required):
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")
    endpoint = {
        "rag_service": ("/ask", {"question": "what is the refund window?"}),
        "support_triage_api": ("/triage", {"ticket": "I was double charged", "user_id": "u1"}),
        "extraction_service": ("/extract", {"text": "Invoice from Acme Corp, total 5 USD for widgets"}),
    }[name]
    path, body = endpoint
    snippet = (
        "import json\n"
        "from starlette.testclient import TestClient\n"
        "import main\n"
        "c = TestClient(main.app)\n"
        "assert c.get('/health').json()['status'] == 'ok'\n"
        f"r = c.post({path!r}, json={body!r})\n"
        "assert r.status_code == 200, (r.status_code, r.text)\n"
        "print(json.dumps(r.json()))\n"
    )
    proc = _run(APPS / name, snippet)
    assert proc.returncode == 0, f"{name}/main.py (FastAPI) failed offline:\n{proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert required <= set(payload), f"{name} HTTP response missing keys: {required - set(payload)}"


def test_cli_research_agent_runs_offline():
    app = APPS / "cli_research_agent" / "app.py"
    assert app.is_file()
    proc = subprocess.run(
        [sys.executable, str(app), "what is the refund window?"],
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"cli_research_agent failed offline:\n{proc.stderr}"
    assert proc.stdout.strip(), "cli_research_agent produced no output"
