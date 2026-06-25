# Testing

The suite runs **fully offline** — no API keys, no network — on the deterministic
`MockProvider`. It must stay green *and* stay real: coverage is measured, not
assumed.

```bash
pip install -e ".[dev]"
pytest -q                                              # the offline suite
coverage run -m pytest -q && coverage report -m        # honest line + branch coverage
ruff check vincio/ tests/
mypy vincio
```

## What a good Vincio test looks like

Tests assert **specific, meaningful behavior** — the opposite of AI-slop coverage:

- **Test real code, not a mock of it.** Use `MockProvider` (a deterministic model
  double) for model calls; build real engines, stores, and types. The suite uses
  **no** `unittest.mock` / `@patch` — if you reach for one, you are probably
  testing the test.
- **Assert an exact outcome**: a value, a state transition, a computed number, a
  raised error *with its message* (`pytest.raises(SomeError, match=...)`). A test
  whose only assertion is `is not None`, a lone `isinstance(...)`, or a constant
  you just wrote is not a test.
- **Cover the branches that bite**: error paths, empty / boundary / conflicting /
  oversized inputs, every arm of a conditional. Branch coverage is the metric that
  catches the bugs line coverage misses.
- **Be deterministic and independent**: no ordering dependencies, no wall-clock or
  RNG flakiness, no shared mutable state between tests.

Coverage-focused suites live in `tests/test_cov_<module>.py`, one per module,
written against the module's measured line/branch gaps.

## Measuring coverage — the one caveat

Vincio registers a `pytest11` plugin (`vincio.testing.plugin`, in `pyproject.toml`),
which imports library modules at pytest startup — **before** `pytest-cov`'s tracer
attaches. As a result `pytest --cov` *under-reports*: every class/field-definition
line that ran during plugin import is falsely counted as "missing" (this once made
the suite look like 61% line coverage when it was really 83%). Always measure with
the tracer started first:

```bash
coverage run -m pytest -q && coverage report -m
```

`[tool.coverage.report] fail_under` in `pyproject.toml` is the floor CI enforces.
A handful of lines are intentionally **not** covered offline — optional-dependency
provider/store adapters, real media generation, live MCP/HTTP sockets — and the
per-module `tests/test_cov_*.py` files note these explicitly rather than faking them.
