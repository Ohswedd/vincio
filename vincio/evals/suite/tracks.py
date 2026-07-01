"""The three benchmark **tracks** — the top-level shape of Vincio's benchmark
platform. Each track answers a different question, and every track supports the
same provenance tiers (a real *live* run, a pinned *recorded* replay, or an
offline *mockup*), so the honesty contract is uniform across all of them.

============  ==========================================================  =====================
Track         Answers                                                      Compares
============  ==========================================================  =====================
``model``     how good is a *model* on the standard public benchmarks?     model vs the benchmark
``uplift``    how much does routing a model *through Vincio* change it?     Vincio-routed vs direct
``feature``   how good is a Vincio *feature* vs the same feature elsewhere? Vincio vs a competitor
============  ==========================================================  =====================

The provenance tier (:class:`~vincio.evals.suite.tiers.ProvenanceTier`) is
orthogonal and means the same thing in every track: **Live** — the real thing ran
end to end (a live model for ``model``/``uplift``; the competitor library actually
executed for ``feature``); **Static/Mockup** — an offline, reproducible
illustration that gates CI. The middle **Recorded** tier (a hash-pinned real
dataset replayed against recorded outputs) applies to the **model** track, whose
benchmarks have a real-dataset loader; the ``uplift`` and ``feature`` tracks span
only Static and Live, because their built-in comparisons are a fabricated mockup
(uplift) or run the competitor live (feature) with no distinct recorded slice. A
lower tier can never print a higher tier's label — the engines refuse it.
"""

from __future__ import annotations

from enum import StrEnum

from ...core.errors import EvalSuiteError

__all__ = ["BenchmarkTrack"]


class BenchmarkTrack(StrEnum):
    """Which of the three benchmark questions a run answers.

    The string value is the CLI/report token (``"model"`` / ``"uplift"`` /
    ``"feature"``), so a track round-trips through JSON and a command flag
    unchanged.
    """

    MODEL = "model"
    UPLIFT = "uplift"
    FEATURE = "feature"

    @property
    def label(self) -> str:
        """A short human label (``"Model"`` / ``"Uplift"`` / ``"Feature"``)."""
        return self.name.capitalize()

    @property
    def question(self) -> str:
        """The one-line question this track answers."""
        return {
            BenchmarkTrack.MODEL: "how good is a model on the standard public benchmarks?",
            BenchmarkTrack.UPLIFT: "how much does routing a model through Vincio change its scores?",
            BenchmarkTrack.FEATURE: "how does a Vincio feature compare to the same feature elsewhere?",
        }[self]

    @classmethod
    def parse(cls, value: str | BenchmarkTrack) -> BenchmarkTrack:
        """Coerce a track from its token (``"model"``), name, or itself."""
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for track in cls:
            if text in (track.value, track.name.lower()):
                return track
        raise EvalSuiteError(
            f"unknown benchmark track {value!r}; known: {', '.join(t.value for t in cls)}"
        )
