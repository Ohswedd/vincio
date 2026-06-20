"""Compiled-prompt render program.

A :class:`PromptSpec`'s stable prefix — role, objective, rules, safety
policies, definitions, the output contract, and examples — is a pure function
of the spec, the render format, and whether the schema is inlined. It does not
change when the per-call task, evidence, or memory change, yet a naive compiler
rebuilds and re-renders it on every call.

A :class:`PromptProgram` is that stable prefix compiled once: the normalized,
deduped, example-capped stable nodes; the rendered system text; its token
count; and the spec-level lint findings. The compiler keys a small in-process
cache of programs by the spec and the render options, so a warm spec's compile
renders only the volatile suffix and concatenates the cached prefix. The
output is identical to compiling from scratch — the program is a hot-path
accelerator, not a behavioural change.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from ..core.utils import stable_hash
from .ast import PromptNode
from .lint import LintFinding

__all__ = ["PromptProgram", "ProgramCache"]


@dataclass
class PromptProgram:
    """The compiled, reusable stable prefix of a prompt."""

    key: str
    system_text: str
    stable_tokens: int
    stable_nodes: list[PromptNode]
    excluded_examples: int
    spec_lint: list[LintFinding] = field(default_factory=list)


class ProgramCache:
    """Bounded, content-addressed cache of compiled stable prefixes."""

    def __init__(self, *, max_entries: int = 64) -> None:
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, PromptProgram] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(
        *,
        spec_hash: str,
        format: str,
        section_headers: bool,
        max_examples: int,
        include_schema: bool,
        model: str | None,
        compiler_version: str,
    ) -> str:
        """Key over everything that determines the rendered stable prefix."""
        return stable_hash(
            [
                spec_hash,
                format,
                section_headers,
                max_examples,
                include_schema,
                model or "",
                compiler_version,
            ]
        )

    def get(self, key: str) -> PromptProgram | None:
        with self._lock:
            program = self._entries.get(key)
            if program is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return program

    def put(self, program: PromptProgram) -> None:
        with self._lock:
            self._entries[program.key] = program
            self._entries.move_to_end(program.key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
