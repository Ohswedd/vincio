"""Prompt compiler.

Pipeline: normalize → dedupe rules → conflict check → stable
prefix layout → example selection → schema render → context block render →
token budget validation → prompt hash.

Output is a :class:`CompiledPrompt`: provider-neutral messages with cache
hints, hashes for versioning, lint findings, and a cacheability score.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.errors import PromptBudgetError
from ..core.tokens import count_tokens
from ..core.types import Example, Message, ModelProfile
from ..core.utils import stable_hash, utcnow
from .ast import ExampleNode, PromptAST, PromptNode
from .lint import LintFinding, lint_ast, lint_spec
from .templates import PromptSpec

__all__ = ["RenderFormat", "CompilerOptions", "CompiledPrompt", "PromptCompiler", "COMPILER_VERSION"]

COMPILER_VERSION = "1.0.0"

RenderFormat = Literal["markdown", "xml", "json", "minimal"]


class CompilerOptions(BaseModel):
    format: RenderFormat = "markdown"
    max_examples: int = 4
    max_prompt_tokens: int | None = None
    fail_on_lint_errors: bool = False
    include_schema_in_prompt: bool = True  # False when provider enforces natively
    section_headers: bool = True


class CompiledPrompt(BaseModel):
    messages: list[Message]
    system_text: str = ""
    user_text: str = ""
    prompt_id: str = ""
    prompt_spec_hash: str = ""
    rendered_hash: str = ""
    compiler_version: str = COMPILER_VERSION
    model_profile: str = ""
    created_at: Any = Field(default_factory=utcnow)
    token_count: int = 0
    stable_prefix_tokens: int = 0
    cacheability: float = 0.0  # StablePrefixTokens / TotalInputTokens
    lint_findings: list[LintFinding] = Field(default_factory=list)
    excluded_examples: int = 0


SECTION_TITLES = {
    "system_role": "Role",
    "objective": "Objective",
    "rule": "Rules",
    "safety_policy": "Safety Policy",
    "definition": "Definitions",
    "output_contract": "Output Contract",
    "example": "Examples",
    "memory_block": "Memory",
    "evidence_block": "Evidence",
    "tool_result_block": "Tool Results",
    "user_task": "Task",
}

XML_TAGS = {
    "system_role": "role",
    "objective": "objective",
    "rule": "rules",
    "safety_policy": "safety_policy",
    "definition": "definitions",
    "output_contract": "output_contract",
    "example": "examples",
    "memory_block": "memory",
    "evidence_block": "evidence",
    "tool_result_block": "tool_results",
    "user_task": "task",
}


def _render_items(items: list[dict[str, Any]], kind: str) -> str:
    lines: list[str] = []
    for item in items:
        ref = item.get("id") or item.get("citation_ref") or ""
        text = item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
        prefix = f"[{ref}] " if ref else ""
        if kind == "tool_result_block":
            tool = item.get("tool_name") or item.get("tool") or "tool"
            lines.append(f"{prefix}{tool}: {text}")
        else:
            lines.append(f"{prefix}{text}")
    return "\n".join(lines)


class PromptCompiler:
    def __init__(self, options: CompilerOptions | None = None, *, cache: Any | None = None) -> None:
        self.options = options or CompilerOptions()
        self.cache = cache  # PromptCompileCache | None
        self.cache_hits = 0

    # -- passes -----------------------------------------------------------------

    def _normalize(self, ast: PromptAST) -> PromptAST:
        """Pass 1: trim whitespace, drop empty nodes."""
        kept: list[PromptNode] = []
        for node in ast.nodes:
            node = node.model_copy(update={"text": (node.text or "").strip()})
            has_payload = bool(node.text) or bool(getattr(node, "items", None)) or (
                getattr(node, "schema_def", None) is not None
            ) or getattr(node, "example", None) is not None
            if has_payload:
                kept.append(node)
        return PromptAST(nodes=kept, metadata=ast.metadata)

    def _dedupe(self, ast: PromptAST) -> PromptAST:
        """Pass 2: drop nodes with identical (kind, normalized text)."""
        seen: set[str] = set()
        kept: list[PromptNode] = []
        for node in ast.nodes:
            key = f"{node.kind}:{' '.join(node.text.lower().split())}" if node.text else node.content_hash
            if key in seen and node.kind in ("rule", "definition", "safety_policy"):
                continue
            seen.add(key)
            kept.append(node)
        return PromptAST(nodes=kept, metadata=ast.metadata)

    def _select_examples(self, ast: PromptAST) -> tuple[PromptAST, int]:
        """Pass 6: keep the highest-quality examples within max_examples."""
        example_nodes = [n for n in ast.nodes if isinstance(n, ExampleNode)]
        if len(example_nodes) <= self.options.max_examples:
            return ast, 0
        ranked = sorted(
            example_nodes,
            key=lambda n: (n.example.quality if n.example else 0.0),
            reverse=True,
        )
        keep = set(id(n) for n in ranked[: self.options.max_examples])
        kept_nodes = [n for n in ast.nodes if not isinstance(n, ExampleNode) or id(n) in keep]
        return PromptAST(nodes=kept_nodes, metadata=ast.metadata), len(example_nodes) - len(keep)

    # -- rendering ----------------------------------------------------------------

    def _render_node_text(self, node: PromptNode, *, include_schema: bool | None = None) -> str:
        if include_schema is None:
            include_schema = self.options.include_schema_in_prompt
        if node.kind in ("memory_block", "evidence_block", "tool_result_block"):
            return _render_items(getattr(node, "items", []), node.kind)
        if node.kind == "output_contract":
            parts = []
            if node.text:
                parts.append(node.text)
            schema_def = getattr(node, "schema_def", None)
            if schema_def is not None and include_schema:
                parts.append(
                    "Return output matching this JSON schema exactly:\n"
                    + json.dumps(schema_def, indent=2, ensure_ascii=False)
                )
            return "\n".join(parts)
        if node.kind == "example":
            example: Example | None = getattr(node, "example", None)
            if example is not None:
                text = f"Input: {example.input}\nOutput: {example.output}"
                if example.explanation:
                    text += f"\nWhy: {example.explanation}"
                return text
        return node.text

    def _render_sections(
        self, nodes: list[PromptNode], format: RenderFormat, *, include_schema: bool | None = None
    ) -> str:
        # Group consecutive nodes by kind to form sections, preserving order.
        sections: list[tuple[str, list[str]]] = []
        for node in nodes:
            text = self._render_node_text(node, include_schema=include_schema)
            if not text:
                continue
            if sections and sections[-1][0] == node.kind:
                sections[-1][1].append(text)
            else:
                sections.append((node.kind, [text]))

        if format == "minimal":
            return "\n\n".join("\n".join(texts) for _, texts in sections)

        if format == "xml":
            rendered = []
            for kind, texts in sections:
                tag = XML_TAGS.get(kind, kind)
                body = "\n".join(f"- {t}" if kind in ("rule", "safety_policy") else t for t in texts)
                rendered.append(f"<{tag}>\n{body}\n</{tag}>")
            return "\n\n".join(rendered)

        if format == "json":
            payload: dict[str, Any] = {}
            for kind, texts in sections:
                key = XML_TAGS.get(kind, kind)
                existing = payload.get(key)
                value: Any = texts if len(texts) > 1 else texts[0]
                if existing is None:
                    payload[key] = value
                else:
                    merged = existing if isinstance(existing, list) else [existing]
                    merged.extend(texts)
                    payload[key] = merged
            return json.dumps(payload, indent=2, ensure_ascii=False)

        # markdown (default)
        rendered = []
        for kind, texts in sections:
            title = SECTION_TITLES.get(kind, kind.replace("_", " ").title())
            if kind in ("rule", "safety_policy"):
                body = "\n".join(f"- {t}" for t in texts)
            elif kind == "example":
                body = "\n\n".join(texts)
            else:
                body = "\n".join(texts)
            if self.options.section_headers:
                rendered.append(f"## {title}\n{body}")
            else:
                rendered.append(body)
        return "\n\n".join(rendered)

    # -- compile ------------------------------------------------------------------

    def compile(
        self,
        spec: PromptSpec,
        *,
        user_task: str = "",
        variables: dict[str, Any] | None = None,
        memory_items: list[dict[str, Any]] | None = None,
        evidence_items: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        model_profile: ModelProfile | None = None,
        provider_enforces_schema: bool = False,
    ) -> CompiledPrompt:
        resolved = spec.substitute(variables)

        cache_key: str | None = None
        if self.cache is not None:
            cache_key = self.cache.key(
                {
                    "spec": resolved.spec_hash,
                    "task": user_task,
                    "memory": memory_items,
                    "evidence": evidence_items,
                    "tools": tool_results,
                    "profile": model_profile.name if model_profile else None,
                    "native_schema": provider_enforces_schema,
                    "options": self.options.model_dump(mode="json"),
                    "version": COMPILER_VERSION,
                }
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return CompiledPrompt.model_validate(cached)

        lint_findings = lint_spec(resolved)

        ast = resolved.build_ast(
            user_task=user_task,
            memory_items=memory_items,
            evidence_items=evidence_items,
            tool_results=tool_results,
        )
        ast = self._normalize(ast)
        ast = self._dedupe(ast)
        ast, excluded_examples = self._select_examples(ast)
        lint_findings.extend(lint_ast(ast))

        if self.options.fail_on_lint_errors:
            errors = [f for f in lint_findings if f.severity == "error"]
            if errors:
                from ..core.errors import PromptLintError

                raise PromptLintError(
                    f"{len(errors)} prompt lint error(s): " + "; ".join(f.code for f in errors),
                    findings=errors,
                )

        # Schema inclusion is resolved per call (never by mutating shared
        # options — compile() must be safe under concurrent use).
        include_schema = self.options.include_schema_in_prompt and not provider_enforces_schema
        ordered = ast.ordered()
        stable_nodes = [n for n in ordered if n.stable]
        volatile_nodes = [n for n in ordered if not n.stable]
        # The user task is delivered as the user message; other volatile
        # context blocks travel with it so the prefix stays stable.
        user_task_nodes = [n for n in volatile_nodes if n.kind == "user_task"]
        context_nodes = [n for n in volatile_nodes if n.kind != "user_task"]

        system_text = self._render_sections(
            stable_nodes, self.options.format, include_schema=include_schema
        )
        context_text = self._render_sections(
            context_nodes, self.options.format, include_schema=include_schema
        )
        task_text = "\n".join(
            self._render_node_text(n, include_schema=include_schema) for n in user_task_nodes
        )
        user_text = "\n\n".join(t for t in (context_text, task_text) if t)

        messages: list[Message] = []
        if system_text:
            messages.append(Message(role="system", content=system_text, cache_hint=True))
        messages.append(Message(role="user", content=user_text or task_text or ""))

        model_name = model_profile.model if model_profile else None
        stable_tokens = count_tokens(system_text, model_name)
        total_tokens = stable_tokens + count_tokens(user_text, model_name)
        max_tokens = self.options.max_prompt_tokens
        if max_tokens is not None and total_tokens > max_tokens:
            raise PromptBudgetError(
                f"compiled prompt uses {total_tokens} tokens, budget is {max_tokens}",
                details={"token_count": total_tokens, "budget": max_tokens},
            )

        spec_hash = resolved.spec_hash
        rendered_hash = stable_hash({"system": system_text, "user": user_text})
        compiled = CompiledPrompt(
            messages=messages,
            system_text=system_text,
            user_text=user_text,
            prompt_id=f"{resolved.name}@{spec_hash[:8]}",
            prompt_spec_hash=spec_hash,
            rendered_hash=rendered_hash,
            model_profile=model_profile.name if model_profile else "",
            token_count=total_tokens,
            stable_prefix_tokens=stable_tokens,
            cacheability=(stable_tokens / total_tokens) if total_tokens else 0.0,
            lint_findings=lint_findings,
            excluded_examples=excluded_examples,
        )
        if self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, compiled.model_dump(mode="json"), spec_hash=spec_hash)
        return compiled
