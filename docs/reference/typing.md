# Reference: typing

Vincio is fully type-annotated and ships a [PEP 561](https://peps.python.org/pep-0561/)
`py.typed` marker, so your type-checker sees Vincio's inline contract with no
extra stubs to install. All public data contracts are Pydantic models and every
engine is annotated end to end.

## What downstream sees

Install Vincio and your checker (mypy, Pyright, Pylance) resolves Vincio types
directly:

```python
from vincio import ContextApp, RunResult

app = ContextApp(name="demo")
result: RunResult = app.run("hello")   # fully typed, no `Any` leakage
```

The entire package is checked under mypy on every commit (the `Types (mypy)` CI
job), so a change that breaks the public type contract fails CI.

## The strict-typing ladder

Beyond the baseline check, a growing set of modules is held to the much stricter
`mypy --strict` bar — no untyped defs, no implicit `Any`, no unused ignores,
checked generics. New modules ship strict, and the set expands over time without
ever loosening. The strict set is enforced two ways:

- The regular `mypy vincio` job tightens these modules via per-module overrides
  in `pyproject.toml` (`[[tool.mypy.overrides]]`).
- A dedicated `mypy --strict` CI step double-checks the same list.

Currently graduated to `--strict`:

| Module | Surface |
|---|---|
| `vincio.stability` | deprecation / experimental markers, `stability_of` |
| `vincio.core.errors` | the `VincioError` hierarchy |
| `vincio.core.error_catalog` | stable codes, remediation, docs links, i18n |
| `vincio.core.config` | `VincioConfig`, `load_config`, schema |
| `vincio.core.config_migrations` | versioned config migrations |
| `vincio._apiref` | docstring-driven API reference generator |
| `vincio.cli.doctor` | the `vincio doctor` engine |

This is a one-way ratchet: modules move onto the ladder, never off it.

## Conventions

- Public functions and methods are fully annotated; `from __future__ import
  annotations` is used throughout, so annotations are lazy and cheap.
- Optional dependencies are imported lazily inside the functions that need them,
  so importing `vincio` never requires an extra to be installed and type-checks
  without it.
- Error `.code` values and the [error catalog](errors.md) are part of the typed,
  stable contract; message strings are not.
