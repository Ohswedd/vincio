"""Pytest plugin: the ``vincio_snapshot`` fixture and snapshot refresh flag.

Registered automatically via the ``pytest11`` entry point when vincio is
installed. Snapshots live in ``__snapshots__/<test_file>/<test_name>.json``
next to the test file.
"""

from __future__ import annotations

import pytest

from .snapshots import Snapshot


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("vincio")
    group.addoption(
        "--vincio-update-snapshots",
        action="store_true",
        default=False,
        help="rewrite vincio snapshot files with current values",
    )


@pytest.fixture()
def vincio_snapshot(request: pytest.FixtureRequest) -> Snapshot:
    """Snapshot store scoped to the requesting test."""
    test_file = request.path
    directory = test_file.parent / "__snapshots__" / test_file.stem
    return Snapshot(
        directory,
        request.node.name,
        update=request.config.getoption("--vincio-update-snapshots"),
    )
