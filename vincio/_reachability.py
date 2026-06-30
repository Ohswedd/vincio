"""Public-symbol reachability: no dead public surface can silently return.

:mod:`vincio._surface` proves every name in a public ``__all__`` *resolves* to a
live attribute. That catches a renamed or deleted export, but not a symbol that
resolves yet is **referenced nowhere** — the kind of dead surface the hardening
line's 6.0 phase removed by a one-time *manual* repo-wide reference check
(``ashapley_values``, ``truncate_text``, ``race_with_timeout`` and the rest:
each in an ``__all__``, each used by no internal caller, test, example, or
benchmark). That manual check was never mechanized, so a new dead-but-resolvable
public symbol could ship and read as supported API while doing nothing — the
exact debt 6.0 paid down, free to return.

6.6 mechanizes the rubric. This module is the *static* half: the lint behind
``tests/test_reachability.py`` and the ``hygiene`` VincioBench family, built on
exactly the scan idiom :mod:`vincio._observable_failure` and
:mod:`vincio._assert_robustness` use. It enumerates the public surface — the
frozen top-level ``vincio.__all__`` plus every public first-level subpackage's
own ``__all__`` (the deep-import surface :mod:`vincio._surface` classifies) — and
asks, of each symbol, the audit's question: **is it used anywhere in the code
corpus** (``vincio/`` + ``tests/`` + ``examples/`` + ``benchmarks/``)? A *use* is
a real load — an ``ast.Name`` load, an attribute access ``obj.Symbol``, or an
``import ... as alias`` whose alias is then loaded. A bare re-export
(``from .sub import Symbol`` in an ``__init__``) and an ``__all__`` string entry
are *not* uses: they are the declaration, not a reference to it. A symbol with no
use is **unreferenced**.

Most unreferenced public symbols are not dead — they are legitimately
unexercisable by an offline corpus: an abstract base or :class:`typing.Protocol`
a *user* implements, a concrete provider/backend whose real path needs an
optional dependency or a live endpoint, or production wiring that binds a socket
or webhook. Those are declared, with their reason, in the frozen reachability
baseline (:data:`REACHABILITY_BASELINE`, rendered to
``docs/reference/reachability.txt`` for review). The gate then holds two
invariants, both folded into the ``hygiene`` family:

* **clean** (always-on, no allowlist bypass) — every unreferenced public symbol
  is classified in the baseline. A *new* dead-but-resolvable public symbol —
  one that is exported, resolves, and is used nowhere — fails the build the
  moment it lands, unless it is genuinely a base/optional-dep/wiring symbol and
  is deliberately declared. The honest fix for a pure, offline-runnable helper is
  to *exercise* it (a test references it, so it drops out), not to baseline it.
* **frozen** — the baseline is exactly the live unreferenced set: no missing
  entry (caught by *clean*) and no **stale** one (a baselined symbol that has
  since gained a reference must be removed, keeping the declaration minimal and
  honest). The rendered manifest matches ``docs/reference/reachability.txt``
  byte-for-byte, so every baseline edit is a reviewed diff.

The detector is proven to *bite* (an injected unreferenced symbol is flagged; a
referenced one is not). Run ``python -m vincio._reachability`` to reproduce the
check offline; ``--freeze`` re-renders the manifest.
"""

from __future__ import annotations

import ast
import functools
import os.path

from . import _surface

__all__ = [
    "CORPUS_ROOTS",
    "REACHABILITY_BASELINE",
    "public_surface",
    "referenced_symbols_in_source",
    "referenced_symbols",
    "unreferenced_public_symbols",
    "reachability_problems",
    "home_subpackage",
    "render_manifest",
    "load_manifest",
]

_PACKAGE_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

# The code corpus the reference check reads. A public symbol used in any of these
# is reachable; the four roots are exactly the ones the 6.0 manual audit grepped.
# (``docs/`` is intentionally excluded — a name in prose is not a code reference,
# and the auto-generated API pages list every public symbol by construction, so
# they would make the check vacuous.)
CORPUS_ROOTS: tuple[str, ...] = ("vincio", "tests", "examples", "benchmarks")

# The reviewed declaration of the legitimately-unexercised public surface: a
# symbol that resolves and is a real capability, but that an offline code corpus
# cannot reference, with the structural reason it cannot. Categories:
#
# * ``BASE``   — an abstract base / ``typing.Protocol`` a *user* implements; there
#                is nothing for the library's own corpus to instantiate.
# * ``OPTDEP`` — a concrete provider/backend whose real path needs an optional
#                dependency (a provider SDK) or a live endpoint, so it is never
#                constructed in the offline, dependency-free test corpus.
# * ``WIRING`` — production wiring that binds a socket, a Redis instance, or an
#                outbound webhook, not runnable in the offline corpus.
#
# A pure, offline-runnable helper has no entry here: it is *exercised* by a test
# instead (see ``tests/test_public_surface_reachability.py``), which is the
# honest evidence it is a live capability. Keep this sorted by symbol.
REACHABILITY_BASELINE: dict[str, str] = {
    "BlobStore": "BASE",
    "DocumentAI": "BASE",
    "ElevenLabsSpeechProvider": "OPTDEP",
    "GoogleImageProvider": "OPTDEP",
    "GoogleSpeechProvider": "OPTDEP",
    "GoogleVideoProvider": "OPTDEP",
    "HTTPImageProvider": "OPTDEP",
    "HTTPVideoProvider": "OPTDEP",
    "IdempotencyStore": "BASE",
    "OCREngine": "BASE",
    "OpenAIImageProvider": "OPTDEP",
    "OpenAISpeechProvider": "OPTDEP",
    "OpenAIVideoProvider": "OPTDEP",
    "PagerDutyAlertSink": "WIRING",
    "RedisGraphCoordinator": "WIRING",
    "RuntimeBackend": "BASE",
    "SlackAlertSink": "WIRING",
    "TesseractOCR": "OPTDEP",
    "VideoAnalyzer": "BASE",
    "VisionModelOCR": "OPTDEP",
    "WhisperTranscriber": "OPTDEP",
    "serve_viewer": "WIRING",
}

_VALID_CATEGORIES = frozenset({"BASE", "OPTDEP", "WIRING"})


def public_surface() -> dict[str, list[str]]:
    """Map each public symbol to the ``__all__``s that export it, sorted.

    The surface is the frozen top-level ``vincio.__all__`` (recorded under the
    level ``"vincio"``) plus every public first-level subpackage's own ``__all__``
    (recorded under ``"vincio.<subpackage>"``) — the same two-level surface
    :mod:`vincio._surface` classifies. ``__version__`` is not a symbol and is
    excluded. A symbol exported at several levels (e.g. a top-level re-export of a
    subpackage symbol) lists all of them.
    """
    import importlib

    import vincio

    out: dict[str, list[str]] = {}
    for name in vincio.__all__:
        if name == "__version__":
            continue
        out.setdefault(name, []).append("vincio")
    for sub in _surface.public_subpackages():
        module = importlib.import_module(f"vincio.{sub}")
        for name in getattr(module, "__all__", None) or []:
            out.setdefault(name, []).append(f"vincio.{sub}")
    return {name: sorted(set(levels)) for name, levels in out.items()}


def home_subpackage(levels: list[str]) -> str:
    """The owning subpackage for a symbol exported at ``levels``.

    The most specific level — the first ``vincio.<sub>`` if the symbol is
    subpackage-public, else the bare top-level ``"vincio"`` (a symbol that is
    *only* top-level). Used as the manifest's first column so a baseline row reads
    ``<home> <category> <symbol>``.
    """
    subs = sorted(level for level in levels if level != "vincio")
    return subs[0] if subs else "vincio"


def _corpus_files() -> list[str]:
    """Every ``.py`` file under the corpus roots, sorted (private modules included).

    A reference from a *private* internal caller still makes a public symbol
    reachable, so — unlike the public-module lints — the reference scan reads the
    whole tree, not just the public surface.
    """
    out: list[str] = []
    for root in CORPUS_ROOTS:
        base = os.path.join(_REPO_ROOT, root)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for filename in files:
                if filename.endswith(".py"):
                    out.append(os.path.join(dirpath, filename))
    out.sort()
    return out


def referenced_symbols_in_source(source: str, surface: set[str]) -> set[str]:
    """The public symbols ``source`` *uses* (pure, injectable).

    A use is an ``ast.Name`` load of a surface symbol, an attribute access
    ``obj.Symbol`` whose attribute is a surface symbol, or a load of a local alias
    bound by ``from ... import Symbol as alias``. A bare ``from ... import Symbol``
    (a re-export) and an ``__all__`` string entry are declarations, not uses, and
    are correctly excluded — the import binds a name without loading it, and an
    ``__all__`` entry is an ``ast.Constant`` string, not a ``Name``.
    """
    tree = ast.parse(source)
    alias: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported in node.names:
                if imported.asname and imported.name in surface:
                    alias[imported.asname] = imported.name
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in alias:
                used.add(alias[node.id])
            elif node.id in surface:
                used.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in surface:
            used.add(node.attr)
    return used


@functools.lru_cache(maxsize=1)
def referenced_symbols() -> frozenset[str]:
    """Every public symbol used at least once across the code corpus.

    Memoized: the corpus is static within a single lint / test / bench run, and
    the scan AST-parses the whole tree, so the several callers (the always-on
    gate, the frozen check, the manifest) share one pass. The result is a
    ``frozenset`` so a caller cannot corrupt the cache.
    """
    surface = set(public_surface())
    used: set[str] = set()
    for path in _corpus_files():
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        try:
            used |= referenced_symbols_in_source(source, surface)
        except SyntaxError:  # pragma: no cover - a corpus file must parse
            continue
        if used >= surface:  # every symbol already accounted for; stop early
            break
    return frozenset(used)


def unreferenced_public_symbols() -> list[str]:
    """Public symbols used nowhere in the corpus, sorted (the candidate dead set)."""
    surface = set(public_surface())
    return sorted(surface - referenced_symbols())


def reachability_problems(baseline: dict[str, str] | None = None) -> list[str]:
    """Every reachability violation against ``baseline`` (empty == conformant).

    Two always-on invariants (mirroring the surface gate's resolvable/frozen
    split): an unreferenced public symbol **missing** from the baseline is
    undeclared dead surface, and a baselined symbol that is **now referenced** is
    a stale entry to remove. A baseline entry with an unknown category, or for a
    symbol no longer on the public surface, is also reported.
    """
    table = REACHABILITY_BASELINE if baseline is None else baseline
    surface = set(public_surface())
    unreferenced = set(surface) - referenced_symbols()
    problems: list[str] = []
    for symbol in sorted(unreferenced - set(table)):
        problems.append(
            f"{symbol!r} is a public symbol referenced nowhere in the corpus and not "
            f"declared in the reachability baseline; exercise it with a test (preferred), "
            f"or declare it BASE/OPTDEP/WIRING in vincio._reachability.REACHABILITY_BASELINE"
        )
    for symbol in sorted(set(table) - unreferenced):
        reason = "is now referenced" if symbol in surface else "is no longer public surface"
        problems.append(
            f"{symbol!r} is in the reachability baseline but {reason}; remove the stale entry"
        )
    for symbol, category in sorted(table.items()):
        if category not in _VALID_CATEGORIES:
            problems.append(
                f"{symbol!r} has invalid reachability category {category!r} "
                f"(expected one of {sorted(_VALID_CATEGORIES)})"
            )
    return problems


# --- Frozen reachability manifest ------------------------------------------

_MANIFEST_FILE = os.path.join(_REPO_ROOT, "docs", "reference", "reachability.txt")

_MANIFEST_HEADER = (
    "# Vincio reachability baseline — the legitimately-unexercised public surface.\n"
    "# Every public symbol referenced nowhere in the code corpus (vincio/ + tests/ +\n"
    "# examples/ + benchmarks/), one per line, sorted by symbol, as:\n"
    "#   <home-subpackage> <category> <symbol>\n"
    "# Category is the structural reason an offline corpus cannot reference it:\n"
    "#   BASE   = an abstract base / Protocol a user implements;\n"
    "#   OPTDEP = a provider/backend whose real path needs an optional dependency;\n"
    "#   WIRING = production wiring that binds a socket / Redis / webhook.\n"
    "# A pure, offline-runnable helper has no entry here — it is exercised by a test\n"
    "# instead. Re-freeze deliberately: edit REACHABILITY_BASELINE in\n"
    "# vincio/_reachability.py, regenerate with `python -m vincio._reachability\n"
    "# --freeze`, and review the diff. Guarded by tests/test_reachability.py.\n"
)


def render_manifest() -> str:
    """Render the frozen reachability manifest from :data:`REACHABILITY_BASELINE`."""
    surface = public_surface()
    lines: list[str] = []
    for symbol, category in sorted(REACHABILITY_BASELINE.items()):
        home = home_subpackage(surface.get(symbol, ["vincio"]))
        lines.append(f"{home} {category} {symbol}")
    return _MANIFEST_HEADER + "\n".join(lines) + "\n"


def load_manifest(path: str | None = None) -> str:
    """Read the committed reachability manifest verbatim."""
    target = path or _MANIFEST_FILE
    with open(target, encoding="utf-8") as handle:
        return handle.read()


def _freeze(path: str | None = None) -> None:  # pragma: no cover - dev tool
    target = path or _MANIFEST_FILE
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(render_manifest())


if __name__ == "__main__":  # pragma: no cover - dev tool
    import sys

    if "--freeze" in sys.argv[1:]:
        _freeze()
        print(f"froze {len(REACHABILITY_BASELINE)} baseline symbols → {_MANIFEST_FILE}")
    else:
        issues = reachability_problems()
        if issues:
            print("reachability violations:")
            print("\n".join(issues))
            sys.exit(1)
        print("reachability conformant")
