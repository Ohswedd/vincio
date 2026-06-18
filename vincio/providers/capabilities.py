"""Capability preflight for model substitution.

Before Vincio substitutes one model for another — a :class:`ModelCascade` rung,
a :class:`~vincio.providers.base.FailoverChain` entry, or a
:class:`~vincio.optimize.routing.Router` pick — it intersects what the *request*
needs (vision parts, tool calling, structured output, reasoning, a wide enough
context window) with what the candidate model *can do*, read from the
:class:`~vincio.providers.registry.ModelRegistry`. An incompatible candidate is
skipped or escalated instead of erroring late or silently dropping content.

The guard is deliberately **permissive on unknown models**: a model the registry
has never heard of returns a verdict of ``ok`` (it cannot be judged, so it is not
blocked), exactly like the cost table refuses to invent a price. Only a model
whose capabilities are *known* and *insufficient* is refused.

Everything here is pure data over :class:`~vincio.core.types.ModelCapabilities`;
no network, no provider call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.types import ModelCapabilities, ModelRequest

__all__ = [
    "RequestNeeds",
    "CapabilityVerdict",
    "requirements_for",
    "capability_check",
]


class RequestNeeds(BaseModel):
    """What a :class:`ModelRequest` requires of whatever model serves it."""

    vision: bool = False
    audio: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    reasoning: bool = False
    developer_message: bool = False
    min_context_tokens: int = 0
    min_output_tokens: int = 0

    def summary(self) -> list[str]:
        """The required capabilities as a flat list (for logs / decisions)."""
        out: list[str] = []
        for flag in ("vision", "audio", "tool_calling", "structured_output",
                     "reasoning", "developer_message"):
            if getattr(self, flag):
                out.append(flag)
        if self.min_context_tokens:
            out.append(f"context>={self.min_context_tokens}")
        if self.min_output_tokens:
            out.append(f"output>={self.min_output_tokens}")
        return out


class CapabilityVerdict(BaseModel):
    """The result of checking a request's needs against a model's capabilities."""

    model: str
    ok: bool
    known: bool = True
    missing: list[str] = Field(default_factory=list)
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.ok


def _has_part(request: ModelRequest, kind: str) -> bool:
    for message in request.messages:
        content = message.content
        if isinstance(content, list):
            for part in content:
                if part.type == kind:
                    return True
    return False


def requirements_for(request: ModelRequest, *, input_tokens: int = 0) -> RequestNeeds:
    """Derive the capability requirements a *request* imposes on its model.

    ``input_tokens`` (when known to the caller) becomes a minimum-context-window
    requirement; pass the compiler's estimate. Modalities are read from the
    message content parts, so a request carrying an image part requires vision.

    ``min_output_tokens`` is intentionally left at 0: a request's
    ``max_output_tokens`` is a *ceiling the caller permits*, not output the model
    must produce, so it is not a hard capability requirement (a model with a
    smaller output window simply caps). Callers that genuinely need a wide output
    window can set ``min_output_tokens`` on the returned :class:`RequestNeeds`.
    """
    return RequestNeeds(
        vision=_has_part(request, "image"),
        audio=_has_part(request, "audio"),
        tool_calling=bool(request.tools),
        structured_output=request.output_schema is not None,
        reasoning=request.reasoning_effort is not None or request.thinking_budget_tokens is not None,
        developer_message=any(getattr(m.role, "value", m.role) == "developer" for m in request.messages),
        min_context_tokens=max(0, int(input_tokens)),
    )


def capability_check(
    needs: RequestNeeds,
    capabilities: ModelCapabilities | None,
    *,
    model: str = "",
) -> CapabilityVerdict:
    """Check a request's *needs* against a model's *capabilities*.

    Returns an ``ok`` verdict when the model is unknown (``capabilities is
    None``) — an unknown model is never blocked, only an explicitly incapable
    one. Otherwise every unmet requirement is collected into ``missing``.
    """
    if capabilities is None:
        return CapabilityVerdict(
            model=model, ok=True, known=False, reason="model not in registry — not guarded"
        )

    missing: list[str] = []
    if needs.vision and not (capabilities.vision or "image" in capabilities.input_modalities):
        missing.append("vision")
    if needs.audio and not (capabilities.audio or "audio" in capabilities.input_modalities):
        missing.append("audio")
    if needs.tool_calling and not capabilities.tool_calling:
        missing.append("tool_calling")
    if needs.structured_output and not capabilities.structured_output:
        missing.append("structured_output")
    if needs.reasoning and not capabilities.reasoning:
        missing.append("reasoning")
    if needs.developer_message and not capabilities.supports_developer_message:
        missing.append("developer_message")
    if needs.min_context_tokens and needs.min_context_tokens > capabilities.max_context_tokens:
        missing.append(
            f"context_window ({needs.min_context_tokens} > {capabilities.max_context_tokens})"
        )
    if (
        needs.min_output_tokens
        and capabilities.max_output_tokens
        and needs.min_output_tokens > capabilities.max_output_tokens
    ):
        missing.append(
            f"output_window ({needs.min_output_tokens} > {capabilities.max_output_tokens})"
        )

    if missing:
        return CapabilityVerdict(
            model=model, ok=False, known=True, missing=missing,
            reason="missing capabilities: " + ", ".join(missing),
        )
    return CapabilityVerdict(model=model, ok=True, known=True, reason="capable")
