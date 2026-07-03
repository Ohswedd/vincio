"""Tests for the API stability contract (vincio.stability)."""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path

import pytest

import vincio
from vincio.stability import (
    API_VERSION,
    StabilityLevel,
    VincioDeprecationWarning,
    VincioExperimentalWarning,
    _resolve_renamed_kwarg,
    deprecated,
    deprecated_alias,
    experimental,
    public_api,
    stability_of,
)


def test_version_and_api_contract():
    assert vincio.__version__ == "7.6.0"
    # API_VERSION is the frozen public-API contract; it bumps only on a MAJOR
    # release, independent of the package minor/patch level. 5.0 is the second
    # long-term-support major: it re-freezes the surface expanded additively across
    # the 4.x data & analytics plane (4.1–5.0) and declares that plane complete. The
    # 6.x hardening line is an additive, surface-preserving paydown of interior
    # quality debt — 6.0 (dead-symbol removal, two-level __all__ reconciliation, the
    # two missing public exceptions, the surface-consistency gate), 6.1
    # (error-contract conformance: off-contract built-in raises converted to
    # VincioError, the contract frozen and gated), 6.2 (observable failure: silent
    # best-effort swallows made observable and a lint forbidding new ones), 6.3
    # (wire-or-retire: formerly-unhooked capabilities given an app.* verb / an
    # internal caller — or documented as advanced API — and a guard holding them
    # reachable), 6.4 (docstring / behaviour parity: docstrings that advertised
    # behaviour the code no longer performed made true or corrected, stale comments
    # cleared, and the parity re-derived from the live code by a gate), 6.5 (-O
    # robustness: load-bearing asserts that vanish under python -O replaced with
    # explicit guards that raise a VincioError, genuine never-happens invariants
    # marked, and a lint forbidding new unmarked asserts), and 6.6 (audit completion
    # & standing guard: the reachability rubric mechanized so a dead-but-resolvable
    # public symbol fails the build, the pure helpers it surfaced now exercised by
    # tests, the structurally-unexercisable surface declared in a frozen baseline,
    # and the whole hygiene family gated in CI). 7.0 opens the open evaluation plane
    # — one pluggable harness for the standard public model benchmarks, with a
    # provenance tier on every number — delivered **additively**: ten new top-level
    # entry points (`BenchmarkSuite`, `BenchmarkRegistry`, `BenchmarkSpec`,
    # `register_benchmark`, `BenchmarkDataset`, `ProvenanceTier`, `SuiteRun`,
    # `SuiteReport`, `Leaderboard`, `RunStore`) plus the `app.benchmark_suite` verb,
    # behind opt-in extras, with **no existing symbol removed or changed**. 7.1
    # completes that line as fit-and-finish: the benchmark provenance manifest and
    # PROVENANCE map, the Recorded/Live tiers made runnable (self-contained live
    # prompts + the `benchmarks/eval_live.py` SOTA runner), the `vincio eval suite
    # list` catalog command, and the README/docs/asset reconciliation — all additive.
    # 7.2 redesigns the benchmark system into a unified **three-track platform**
    # (`vincio bench model|uplift|feature`): the model plane joins a new **uplift**
    # track (the same model through Vincio vs direct) and a new **feature** track (a
    # Vincio feature vs a competitor library, measured live), sharing the provenance
    # tiers, one reporting/CLI surface, and CI gating (`families.bench_tracks.*`) —
    # additive: new `vincio.evals.suite` symbols + the `vincio bench` command, no
    # existing symbol removed. 7.3 adds the **packet compile receipt** — a compact,
    # text-light `CompileReceipt` (in `vincio.context`, with a `vincio trace receipt`
    # command) proving *why* a packet compiled — additively, no existing symbol
    # changed. 7.4 adds the **DS4 local-inference provider** — a running `ds4-server`
    # (self-hosted DeepSeek V4) as a first-class provider (`Ds4Provider`, the `ds4`
    # preset, four catalog models, the `ModelProfile.self_hosted` flag), flowing
    # through the same registry, reasoning controller, residency, and audit chain as
    # every provider — additively, no new hard dependency, no existing symbol changed.
    # 7.4.1 is a test-only hardening patch: the compile-receipt trust-boundary
    # regression fixture (issue #140) proving `to_export()` never carries raw
    # prompt/evidence text while a changed render identity (`rendered_packet_hash`)
    # still surfaces as an explicit divergence — no public symbol added or changed.
    # 7.5 is the consistency & structure line: canonical names (`build_*` factories,
    # `verifier=`, `as_of=`, `digest()`, `content_hash`) added additively with the
    # old spellings kept as the first *active* deprecation runway (warn since 7.5,
    # removed no earlier than 8.0, `vincio migrate 8.0` shipping the codemod), one
    # shared canonical-JSON implementation with zero byte changes, quadratic hot
    # paths made linear byte-identically, and the ContextApp/compile()/
    # combine_attestations decomposition as pure code motion — no existing symbol
    # removed, nothing broken.
    # 7.6 is the universal web browsing & search plane: the new `vincio.web`
    # subpackage (DuckDuckGo/pluggable search, token-budgeted extraction,
    # pre-egress WebPolicy, offline-verifiable WebEvidence), the built-in
    # progressive-disclosure browsing skill, the `ToolProtocolProvider` that
    # grants native-grade tool use to models without function calling, the
    # `websearch` connector, and the `app.use_web_search()` verb — all additive,
    # subpackage-level symbols only, no existing symbol removed or changed.
    # The surface grows by re-freezing it, never by breaking it, so the API contract
    # generation stays "5.0" while the package advances to 7.6.0.
    assert API_VERSION == "5.0"


def test_package_version_matches_dunder_version():
    """The built package version (pyproject) must match ``vincio.__version__``.

    These two are bumped together every release; the build publishes the
    ``pyproject`` version while the runtime reports ``__version__``, so a
    divergence ships a package whose metadata lies about its contents (and, for a
    stale bump, collides with an already-published file on PyPI). This guard fails
    the build the moment they drift.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["version"] == vincio.__version__


def test_public_api_is_stable_surface():
    names = public_api()
    # The frozen surface must include the headline entry points.
    for required in ("ContextApp", "Workflow", "Rail", "MemoryEngine", "OutputContract"):
        assert required in names
    # public_api() mirrors __all__ exactly.
    assert set(names) == set(vincio.__all__)
    # Every advertised name is actually importable from the package.
    for name in names:
        assert hasattr(vincio, name), name


def test_public_surface_is_frozen():
    """The live ``__all__`` must match the committed 5.0 LTS frozen surface.

    This is the mechanical re-freeze: any addition, removal, or rename of a
    public symbol must be a deliberate edit to ``docs/reference/public-surface.txt``
    (regenerate with ``python -m vincio._apiref --freeze``). A SemVer-significant
    surface change cannot land silently.
    """
    from vincio._apiref import load_frozen_surface, render_frozen_surface

    frozen = load_frozen_surface()
    live = sorted(vincio.__all__)
    assert frozen == live, (
        "public surface drifted from the frozen manifest; if intentional, "
        "regenerate with `python -m vincio._apiref --freeze` and review the diff. "
        f"added={sorted(set(live) - set(frozen))} removed={sorted(set(frozen) - set(live))}"
    )
    # The manifest renders deterministically from the live surface.
    surface_path = Path(__file__).resolve().parent.parent / "docs" / "reference" / "public-surface.txt"
    assert surface_path.read_text(encoding="utf-8") == render_frozen_surface()


def test_deprecated_function_warns_and_forwards():
    @deprecated(since="1.1", removed_in="2.0", alternative="new_fn")
    def old_fn(x):
        return x * 2

    with pytest.warns(VincioDeprecationWarning, match="removed in 2.0"):
        assert old_fn(3) == 6

    record = stability_of(old_fn)
    assert record["level"] is StabilityLevel.DEPRECATED
    assert record["removed_in"] == "2.0"
    assert record["alternative"] == "new_fn"
    assert "[DEPRECATED]" in (old_fn.__doc__ or "")


def test_deprecated_class_warns_on_instantiation():
    @deprecated(since="1.1", removed_in="2.0")
    class OldThing:
        def __init__(self, v):
            self.v = v

    with pytest.warns(VincioDeprecationWarning):
        obj = OldThing(7)
    assert obj.v == 7
    assert stability_of(OldThing)["level"] is StabilityLevel.DEPRECATED


def test_experimental_warns_once():
    @experimental(since="1.0", note="shape may change")
    def beta_fn():
        return 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beta_fn()
        beta_fn()
    exp = [w for w in caught if issubclass(w.category, VincioExperimentalWarning)]
    assert len(exp) == 1  # one-time per symbol
    assert stability_of(beta_fn)["level"] is StabilityLevel.EXPERIMENTAL


def test_deprecated_alias_forwards_to_target():
    def new_name(a, b):
        return a + b

    old_name = deprecated_alias(new_name, old_name="old_name", since="1.1", removed_in="2.0")
    with pytest.warns(VincioDeprecationWarning, match="old_name"):
        assert old_name(2, 3) == 5
    assert old_name.__name__ == "old_name"


def test_unmarked_symbol_is_stable_by_default():
    def plain():
        return 1

    assert stability_of(plain)["level"] is StabilityLevel.STABLE


def test_deprecation_warning_can_be_escalated_to_error():
    @deprecated(since="1.1", removed_in="2.0")
    def doomed():
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        with pytest.raises(VincioDeprecationWarning):
            doomed()


# ---------------------------------------------------------------------------
# _resolve_renamed_kwarg — the keyword-rename runway helper (7.5)
# ---------------------------------------------------------------------------


def test_resolve_renamed_kwarg_new_only_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        resolved = _resolve_renamed_kwarg(
            "value",
            None,
            new_name="verifier",
            old_name="verify_with",
            owner="fn",
            since="7.5",
            removed_in="8.0",
            error=ValueError,
        )
    assert resolved == "value"


def test_resolve_renamed_kwarg_old_only_warns_and_forwards():
    with pytest.warns(VincioDeprecationWarning, match="since Vincio 7.5.*removed in 8.0"):
        resolved = _resolve_renamed_kwarg(
            None,
            "legacy",
            new_name="verifier",
            old_name="verify_with",
            owner="fn",
            since="7.5",
            removed_in="8.0",
            error=ValueError,
        )
    assert resolved == "legacy"


def test_resolve_renamed_kwarg_both_raises_the_given_error():
    with pytest.raises(ValueError, match="both verifier= and its deprecated alias"):
        _resolve_renamed_kwarg(
            "new",
            "old",
            new_name="verifier",
            old_name="verify_with",
            owner="fn",
            since="7.5",
            removed_in="8.0",
            error=ValueError,
        )


def test_resolve_renamed_kwarg_neither_returns_none():
    with warnings.catch_warnings():
        warnings.simplefilter("error", VincioDeprecationWarning)
        assert (
            _resolve_renamed_kwarg(
                None,
                None,
                new_name="as_of",
                old_name="at",
                owner="fn",
                since="7.5",
                removed_in="8.0",
                error=ValueError,
            )
            is None
        )
