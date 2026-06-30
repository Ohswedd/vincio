# Contributing to Vincio

Thanks for your interest in improving Vincio! This guide covers the basics.

## Development setup

Vincio targets Python 3.11+. Set up an editable install with the dev extras:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

The full test suite runs **fully offline** (no network or API keys) using the deterministic mock
provider. All of these must be green:

```bash
ruff check vincio/ tests/      # lint
mypy vincio                    # type check
pytest -q                      # tests
```

CI runs the same checks across Python 3.11, 3.12, and 3.13, plus a coverage floor, the VincioBench
budgets, and a package build, so it is worth running them locally first.

## Project conventions

- **Offline-first tests** — everything must pass without network access or API keys. Use
  `MockProvider`, which generates schema-valid structured output.
- **Optional dependencies import lazily** inside functions/constructors with a helpful
  `pip install "vincio[extra]"` error. Core dependencies stay limited to `pydantic`, `httpx`,
  `pyyaml`, and `typing-extensions`.
- **Every public data contract is a Pydantic v2 model**; engines are async-first with sync
  wrappers.
- **Security is deterministic** — never gate a security decision on model output.
- **Every run produces a trace.**
- Update `ROADMAP.md` when adding subsystems or changing release status, and add a `CHANGELOG.md`
  entry under *Unreleased* for user-visible changes.

## Commit messages & PRs

- Keep commits focused and write clear, imperative commit messages.
- Describe the motivation and the change in the PR body; link related issues.

## Reporting bugs & requesting features

Use the [issue templates](https://github.com/Ohswedd/vincio/issues/new/choose). For security
issues, please follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing, you agree that your contributions are licensed under the project's
[Apache 2.0 License](LICENSE).
