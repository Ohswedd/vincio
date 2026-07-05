# Testing Vincio

Vincio's test suite is a **product feature, not an afterthought**: it runs fully offline — no API
keys, no network, no flakiness — and it must stay green *and* stay real. The same discipline the
library sells (deterministic, measured, gated) is the discipline its own tests are held to.

```bash
pip install -e ".[dev]"
pytest -q                                              # the full offline suite
coverage run -m pytest -q && coverage report -m        # honest line + branch coverage (see the caveat)
ruff check vincio/ tests/                              # lint
mypy vincio                                            # types (CI gate; must stay clean)
```

## How the suite runs offline

Every test builds **real engines, stores, and types** — only the *model call* is doubled. That double
is `MockProvider`, a deterministic `ModelProvider` that emits schema-valid output, so retrieval,
validation, evals, tracing, and cost all execute for real against it. Nothing is `@patch`-ed; if you
reach for `unittest.mock`, you are probably testing the test.

Determinism is engineered four ways, and each has a place:

| Technique | What it does | Reach for it when |
|---|---|---|
| `MockProvider(responder=fn)` | a callable returns the model's reply from the request | you want the "model" to branch on the prompt (grounded QA, chat) |
| `MockProvider(script=[…])` | a fixed list of replies / tool-calls, consumed in order | you're exercising a bounded tool loop or a multi-step agent |
| recorded `httpx` transport | replays real provider/web bytes with no socket | you're testing a provider adapter or the web plane end to end |
| the auto-responder (no args) | synthesizes schema-valid output from the `output_schema` | you only care that the pipeline runs, not the exact text |

The examples and tests select a provider through one helper, `examples/_shared.example_provider()`:
it returns the mock offline and a real provider when `VINCIO_PROVIDER` (+ the matching key) is set, so
the *same* code path is what ships.

## What a good Vincio test looks like

Tests assert **specific, meaningful behavior** — the opposite of coverage-for-its-own-sake:

- **Assert an exact outcome** — a value, a state transition, a computed number, or a raised error
  *with its message* (`pytest.raises(VincioError, match=...)`). A test whose only assertion is
  `is not None`, a lone `isinstance(...)`, or a constant you just wrote is not a test.
- **Cover the branches that bite** — error paths, empty / boundary / conflicting / oversized inputs,
  every arm of a conditional. Branch coverage is the metric that catches the bugs line coverage misses.
- **Be deterministic and independent** — no ordering dependencies, no wall-clock or RNG flakiness, no
  shared mutable state between tests. A test that fails only when run second is a bug in the test.
- **Test the contract, not the implementation** — assert on the public behavior (`result.citations`,
  a `RunResult` field, an emitted event), so a refactor that preserves behavior keeps the test green.

Coverage-focused suites live in `tests/test_cov_<module>.py`, one per module, written against that
module's measured line/branch gaps — not against a guess.

## The assertion helpers

`vincio.testing` ships domain assertions so eval-shaped tests read as intent, not plumbing:

```python
from vincio.testing import assert_grounded, assert_metric, assert_safe

result = app.run("What is the refund window?")
assert_grounded(result)                    # every claim is backed by a citation
assert_metric(result, "groundedness", ">= 0.8")
assert_safe(result)                        # no PII / injection leaked to the output
```

`assert_backend_conformance(store)` runs a store (metadata or vector) through the canonical async
contract so a new adapter is proven equivalent, not assumed. And the byte-identical-lowering harness
(`run_signature`, `selection_signature`) proves a `vincio.tasks` one-liner compiles the *same* packet
as its verbose `ContextApp` form — the guarantee the ergonomic front door rests on.

## Measuring coverage — the one caveat

Vincio registers a `pytest11` plugin (`vincio.testing.plugin`, in `pyproject.toml`) that imports
library modules at pytest startup — **before** `pytest-cov`'s tracer attaches. As a result
`pytest --cov` *under-reports*: every class/field-definition line that ran during plugin import is
counted as "missing" (this once made the suite look like 61% line coverage when it was really 83%).
Always start the tracer first:

```bash
coverage run -m pytest -q && coverage report -m
```

`[tool.coverage.report] fail_under` in `pyproject.toml` is the floor CI enforces. A handful of lines
are intentionally **not** covered offline — optional-dependency provider/store adapters, real media
generation, live MCP/HTTP sockets — and the per-module `tests/test_cov_*.py` files note these
explicitly rather than faking them.

## The gate suite — more than pytest

A change is green only when the whole gate set passes; these are the same gates CI runs
(`.github/workflows/ci.yml`), and they're what keep the library's promises honest:

```bash
pytest -q                                                   # the offline behavior suite
mypy vincio                                                 # strict typing (a hard CI gate)
ruff check vincio/ tests/                                   # lint
python benchmarks/vinciobench.py && python benchmarks/check_budgets.py   # perf + quality budgets
vincio docs check                                           # docs graph: links, coverage, llms.txt freshness
```

- **VincioBench + budgets** (`benchmarks/budgets.json`) assert every engine still works on a bundled
  synthetic corpus and stays within its performance/quality budget — a regression fails the build.
  Published [SLOs](docs/reference/slo.md) are each held by a budget at least as strict
  (`tests/test_slos.py` enforces that invariant).
- **The docs graph** (`tests/test_docs_graph.py`) fails on any drift: a dead link, an `app.*` verb with
  no page, an orphan, or a stale `llms.txt`. Run `vincio docs map` after any docstring or doc change.
- **The frozen public surface** (`docs/reference/public-surface.txt`, `subpackage-surface.txt`) fails
  the build if a public symbol appears or disappears without an intentional freeze — SemVer enforced
  mechanically.

## Running a subset

```bash
pytest tests/ -q --ignore=tests/test_examples.py       # fast core suite (skip the example tours)
pytest tests/test_retrieval.py -q                      # one module
pytest -q -k "grounded and not slow"                   # by expression
pytest tests/test_examples.py -q                       # run every feature tour offline
```

CI runs the full matrix on Python 3.11 / 3.12 / 3.13 and installs only `.[dev]`, so
optional-dependency code paths (fastapi, vector stores, OCR, …) must import lazily and their tests must
`pytest.importorskip` when the dependency is absent — never fail a machine that didn't opt in.

See [`AGENTS.md`](AGENTS.md) for the codebase layout and the invariants every change must hold.
