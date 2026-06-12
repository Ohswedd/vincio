"""Vincio testing ergonomics: eval assertions, snapshots, pytest plugin.

Unit-test LLM behavior the way you unit-test code::

    from vincio.testing import assert_eval, assert_grounded

    def test_refund_answer():
        result = app.run("What is the refund window?")
        assert_grounded(result, threshold=0.8)
        assert_eval(result, "What is the refund window?",
                    expected="30 days", metrics={"answer_relevance": 0.5})

The pytest plugin (auto-registered on install) adds the ``vincio_snapshot``
fixture for packet/trace snapshot tests and the ``--vincio-update-snapshots``
option for refreshing them.
"""

from .asserts import assert_eval, assert_grounded, assert_metric, assert_safe
from .snapshots import Snapshot, normalize_packet, normalize_trace

__all__ = [
    "assert_eval",
    "assert_grounded",
    "assert_metric",
    "assert_safe",
    "Snapshot",
    "normalize_packet",
    "normalize_trace",
]
