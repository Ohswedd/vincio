"""Every shipped example must run end to end, fully offline.

This is the executable half of the "tested example for every subsystem"
guarantee: the docs make claims, these tests prove the code behind them runs.
Examples default to the deterministic MockProvider, so no API keys or network
are needed. Each example is executed exactly as ``python examples/NN_*.py``
would run it — as the ``__main__`` module, so Pydantic forward references
resolve identically — but in-process and sandboxed in a temp working directory.
"""

from __future__ import annotations

import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(p for p in EXAMPLES_DIR.glob("[0-9]*.py"))


def test_examples_present():
    # Guards against an example being dropped without notice. The suite covers
    # the whole platform end-to-end, one solid example per area (see
    # examples/README.md).
    assert len(EXAMPLE_FILES) == 20


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
def test_example_runs_offline(path, tmp_path, monkeypatch):
    # Force offline mode and isolate any stray cwd writes.
    monkeypatch.delenv("VINCIO_PROVIDER", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))  # for `import _shared`

    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    module = types.ModuleType("__main__")
    module.__dict__.update({"__file__": str(path), "__name__": "__main__"})

    saved_main = sys.modules.get("__main__")
    sys.modules["__main__"] = module
    try:
        with redirect_stdout(io.StringIO()):
            exec(code, module.__dict__)  # noqa: S102 - first-party example code
    finally:
        if saved_main is not None:
            sys.modules["__main__"] = saved_main
        else:  # pragma: no cover
            sys.modules.pop("__main__", None)
        sys.modules.pop("_shared", None)  # rebind cleanly per example
