"""Prompt variant generation and diffing.

Generates candidate PromptSpecs along the experiment dimensions
(format, examples, reasoning mode, rule ordering). The optimization
engine evaluates candidates and promotes winners through eval gates.
"""

from __future__ import annotations

import difflib
import itertools
from typing import Any

from pydantic import BaseModel, Field

from .compiler import CompilerOptions, RenderFormat
from .templates import PromptSpec

__all__ = ["PromptVariant", "generate_variants", "diff_specs", "diff_rendered"]

REASONING_PREAMBLES = {
    "direct": "",
    "plan": "Before answering, briefly plan the steps needed, then execute the plan.",
    "evidence_first": "First list the evidence relevant to the task, then reason only from that evidence to the answer.",
}


class PromptVariant(BaseModel):
    name: str
    spec: PromptSpec
    compiler_options: CompilerOptions
    dimensions: dict[str, Any] = Field(default_factory=dict)


def _with_reasoning(spec: PromptSpec, mode: str) -> PromptSpec:
    preamble = REASONING_PREAMBLES.get(mode, "")
    update: dict[str, Any] = {"reasoning_mode": mode}
    if preamble:
        update["rules"] = [*spec.rules, preamble]
    return spec.model_copy(update=update)


def generate_variants(
    spec: PromptSpec,
    *,
    formats: list[RenderFormat] | None = None,
    example_counts: list[int] | None = None,
    reasoning_modes: list[str] | None = None,
    rule_orderings: list[str] | None = None,
    max_variants: int = 24,
) -> list[PromptVariant]:
    """Cartesian product over experiment dimensions, capped at *max_variants*."""
    formats = formats or ["markdown", "xml"]
    example_counts = example_counts or sorted({0, min(4, len(spec.examples)), len(spec.examples)})
    reasoning_modes = reasoning_modes or ["direct", "plan"]
    rule_orderings = rule_orderings or ["original", "hard_first"]

    variants: list[PromptVariant] = []
    for fmt, n_examples, reasoning, ordering in itertools.product(
        formats, example_counts, reasoning_modes, rule_orderings
    ):
        candidate = _with_reasoning(spec, reasoning)
        if ordering == "hard_first":
            candidate = candidate.model_copy(update={"rules": sorted(candidate.rules, key=len)})
        candidate = candidate.model_copy(update={"examples": spec.examples[:n_examples]})
        name = f"{spec.name}:{fmt}-ex{n_examples}-{reasoning}-{ordering}"
        variants.append(
            PromptVariant(
                name=name,
                spec=candidate,
                compiler_options=CompilerOptions(format=fmt, max_examples=max(n_examples, 1)),
                dimensions={
                    "format": fmt,
                    "examples": n_examples,
                    "reasoning": reasoning,
                    "rule_ordering": ordering,
                },
            )
        )
        if len(variants) >= max_variants:
            break
    return variants


def diff_specs(a: PromptSpec, b: PromptSpec) -> dict[str, Any]:
    """Field-level diff between two prompt specs."""
    changes: dict[str, Any] = {}
    dump_a, dump_b = a.model_dump(mode="json"), b.model_dump(mode="json")
    for field in sorted(set(dump_a) | set(dump_b)):
        if dump_a.get(field) != dump_b.get(field):
            changes[field] = {"a": dump_a.get(field), "b": dump_b.get(field)}
    return {
        "spec_a": a.name,
        "spec_b": b.name,
        "hash_a": a.spec_hash,
        "hash_b": b.spec_hash,
        "changed_fields": changes,
    }


def diff_rendered(text_a: str, text_b: str, *, context_lines: int = 2) -> str:
    """Unified diff between two rendered prompts (prompt diffing)."""
    return "\n".join(
        difflib.unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            fromfile="prompt_a",
            tofile="prompt_b",
            lineterm="",
            n=context_lines,
        )
    )
