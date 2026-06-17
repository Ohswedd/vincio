"""A unified parser registry for document loaders.

Replaces the if/elif suffix chain with a registry so formats register
additively: a third party (or an optional-dep loader) calls
:func:`register_loader` and ``load_document`` picks it up, instead of editing a
dispatch block. Each loader takes a path (plus passthrough kwargs like ``layout``
/ ``ocr_engine``) and returns a :class:`~vincio.core.types.Document`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.types import Document

__all__ = ["LoaderFn", "ParserRegistry", "default_parser_registry", "register_loader"]

LoaderFn = Callable[..., Document]


class ParserRegistry:
    """Maps file suffixes (lowercased, with the dot) to loader callables."""

    def __init__(self) -> None:
        self._loaders: dict[str, LoaderFn] = {}
        self._extras: dict[str, str] = {}

    def register(self, loader: LoaderFn, *suffixes: str, extra: str | None = None) -> None:
        for suffix in suffixes:
            key = suffix.lower()
            self._loaders[key] = loader
            if extra:
                self._extras[key] = extra

    def get(self, suffix: str) -> LoaderFn | None:
        return self._loaders.get(suffix.lower())

    def extra_for(self, suffix: str) -> str | None:
        """The pip extra a suffix's loader needs, for a helpful error message."""
        return self._extras.get(suffix.lower())

    def supports(self, suffix: str) -> bool:
        return suffix.lower() in self._loaders

    def suffixes(self) -> set[str]:
        return set(self._loaders)

    def load(self, path: str | Path, **kwargs: Any) -> Document:
        loader = self.get(Path(path).suffix)
        if loader is None:
            raise KeyError(Path(path).suffix)
        return loader(Path(path), **kwargs)


_DEFAULT = ParserRegistry()


def default_parser_registry() -> ParserRegistry:
    return _DEFAULT


def register_loader(*suffixes: str, extra: str | None = None) -> Callable[[LoaderFn], LoaderFn]:
    """Decorator: register a loader for one or more suffixes on the default
    registry. ``extra`` names the pip extra it needs (surfaced in errors)."""

    def decorate(fn: LoaderFn) -> LoaderFn:
        _DEFAULT.register(fn, *suffixes, extra=extra)
        return fn

    return decorate
