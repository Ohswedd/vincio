"""Model & system cards generated from the live app configuration.

A **model card** documents one model: its identity, version, capabilities,
limitations, and pricing. A **system card** documents the whole context-
engineering system around it — retrieval, memory, safety filters, and the
human-oversight points — because a regulated buyer cares about the system that
produces an answer, not just the base model.

Both are *views over the running configuration and measured eval evidence*,
not static documents: ``generate_model_card(app)`` reads the app's resolved
provider/model, the price table, the security policy, and (optionally) an
:class:`~vincio.evals.reports.EvalReport`, so the card cannot drift from what
the system actually does. The schema is pluggable (``CardFormat``) because no
single machine-readable format has won — Open Model Card and the EU "AI Cards"
style are both rendered from the same captured facts.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.utils import utcnow
from ..observability.costs import PriceTable, default_price_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.app import ContextApp
    from ..core.config import VincioConfig
    from ..evals.reports import EvalReport

__all__ = [
    "CardFormat",
    "ModelCard",
    "SystemCard",
    "generate_model_card",
    "generate_system_card",
]


class CardFormat(StrEnum):
    """Render target for a card's machine-readable serialization."""

    VINCIO = "vincio"
    """Vincio's native, fully-detailed layout (the default)."""

    OPEN_MODEL_CARD = "open_model_card"
    """Hugging Face / Open Model Card style (`model_details`, `intended_use`…)."""

    AI_CARD = "ai_card"
    """EU "AI Cards" style (`system`, `purpose`, `risk`, `oversight`)."""


# Capability-derived, model-agnostic limitations that every LLM system shares.
# Stated plainly so a card never over-promises; specific eval evidence (when
# attached) qualifies these with measured numbers.
_BASE_LIMITATIONS = [
    "May produce plausible but incorrect statements (hallucination); ground answers in cited evidence.",
    "Quality varies by language; non-English performance is typically lower (see eval slicing).",
    "Knowledge is bounded by the model's training cutoff plus whatever is retrieved at run time.",
    "Not a substitute for professional, legal, medical, or financial advice.",
]


def _config_of(target: ContextApp | VincioConfig) -> VincioConfig:
    cfg = getattr(target, "config", None)
    return cfg if cfg is not None else target  # type: ignore[return-value]


def _price_table_of(target: ContextApp | VincioConfig) -> PriceTable:
    tracker = getattr(target, "cost_tracker", None)
    if tracker is not None and getattr(tracker, "price_table", None) is not None:
        return tracker.price_table
    return default_price_table()


def _eval_evidence(report: EvalReport | None) -> dict[str, float]:
    if report is None:
        return {}
    try:
        summary = report.summary()
    except Exception:  # pragma: no cover - defensive
        return {}
    return {name: round(stats.get("mean", 0.0), 4) for name, stats in summary.items()}


class ModelCard(BaseModel):
    """Machine-readable documentation for a single model."""

    schema_format: CardFormat = CardFormat.VINCIO
    generated_at: datetime = Field(default_factory=utcnow)
    vincio_version: str = ""
    model_id: str
    provider: str
    version: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    pricing: dict[str, float] = Field(default_factory=dict)
    intended_use: str = ""
    out_of_scope: list[str] = Field(default_factory=list)
    evaluation: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self, fmt: CardFormat | None = None) -> dict[str, Any]:
        """Render the card in the requested schema (defaults to its own)."""
        fmt = fmt or self.schema_format
        if fmt == CardFormat.OPEN_MODEL_CARD:
            return {
                "model_details": {
                    "name": self.model_id,
                    "provider": self.provider,
                    "version": self.version,
                    "generated_by": f"vincio {self.vincio_version}".strip(),
                },
                "intended_use": {
                    "primary_uses": self.intended_use,
                    "out_of_scope_uses": self.out_of_scope,
                },
                "capabilities": self.capabilities,
                "limitations": self.limitations,
                "pricing_usd_per_mtok": self.pricing,
                "evaluation": self.evaluation,
            }
        if fmt == CardFormat.AI_CARD:
            return {
                "ai_card": {
                    "model": {"id": self.model_id, "provider": self.provider, "version": self.version},
                    "purpose": self.intended_use,
                    "prohibited_uses": self.out_of_scope,
                    "capabilities": self.capabilities,
                    "known_limitations": self.limitations,
                    "evaluation_evidence": self.evaluation,
                    "transparency": {"generated_from": "live configuration", "tool": "vincio"},
                }
            }
        return self.model_dump(mode="json")

    def to_json(self, fmt: CardFormat | None = None, *, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(fmt), indent=indent, default=str)


class SystemCard(BaseModel):
    """Documentation for the whole system: model + retrieval + memory + safety."""

    schema_format: CardFormat = CardFormat.VINCIO
    generated_at: datetime = Field(default_factory=utcnow)
    vincio_version: str = ""
    name: str = "vincio_app"
    model: ModelCard
    retrieval: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    safety_filters: list[str] = Field(default_factory=list)
    human_oversight: list[str] = Field(default_factory=list)
    data_handling: dict[str, Any] = Field(default_factory=dict)
    governance_controls: list[str] = Field(default_factory=list)
    evaluation: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self, fmt: CardFormat | None = None) -> dict[str, Any]:
        fmt = fmt or self.schema_format
        if fmt == CardFormat.AI_CARD:
            return {
                "ai_system_card": {
                    "system": self.name,
                    "model": self.model.to_dict(CardFormat.AI_CARD),
                    "components": {
                        "retrieval": self.retrieval,
                        "memory": self.memory,
                    },
                    "safety_filters": self.safety_filters,
                    "human_oversight": self.human_oversight,
                    "data_handling": self.data_handling,
                    "governance_controls": self.governance_controls,
                    "evaluation_evidence": self.evaluation,
                }
            }
        if fmt == CardFormat.OPEN_MODEL_CARD:
            data = {
                "system_details": {"name": self.name, "generated_by": f"vincio {self.vincio_version}".strip()},
                "model": self.model.to_dict(CardFormat.OPEN_MODEL_CARD),
                "retrieval": self.retrieval,
                "memory": self.memory,
                "safety": self.safety_filters,
                "human_oversight": self.human_oversight,
                "data_handling": self.data_handling,
                "evaluation": self.evaluation,
            }
            return data
        return self.model_dump(mode="json")

    def to_json(self, fmt: CardFormat | None = None, *, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(fmt), indent=indent, default=str)


def generate_model_card(
    target: ContextApp | VincioConfig,
    *,
    model: str | None = None,
    eval_report: EvalReport | None = None,
    format: CardFormat = CardFormat.VINCIO,
    intended_use: str = "",
) -> ModelCard:
    """Build a :class:`ModelCard` from the live app/config and optional evidence.

    ``target`` may be a :class:`~vincio.core.app.ContextApp` (preferred — its
    resolved provider, model, and price table are read) or a bare
    :class:`~vincio.core.config.VincioConfig`. Pricing comes from the live
    price table; evaluation numbers come from ``eval_report`` when supplied.
    """
    import vincio

    cfg = _config_of(target)
    model_id = model or getattr(target, "model", None) or cfg.provider.model
    provider = cfg.provider.default
    price = _price_table_of(target).lookup(model_id)

    capabilities: dict[str, Any] = {}
    resolved = getattr(target, "resolve_provider", None)
    if callable(resolved):
        try:
            prov = resolved()
            caps = getattr(prov, "capabilities", None)
            caps_obj = caps(model_id) if callable(caps) else caps
            if caps_obj is not None:
                capabilities = caps_obj.model_dump() if hasattr(caps_obj, "model_dump") else dict(caps_obj)
        except Exception:  # pragma: no cover - provider may be unconfigured offline
            capabilities = {}

    out_of_scope = [
        "Fully autonomous decisions without human oversight in high-risk domains.",
        "Processing data outside the regions allowed by the residency policy.",
    ]
    return ModelCard(
        schema_format=format,
        vincio_version=vincio.__version__,
        model_id=model_id,
        provider=provider,
        version=cfg.metadata.get("model_version"),
        capabilities=capabilities,
        limitations=list(_BASE_LIMITATIONS),
        pricing={
            "input_per_mtok": price.input_per_mtok,
            "output_per_mtok": price.output_per_mtok,
            "cached_input_per_mtok": price.cached_input_per_mtok,
        },
        intended_use=intended_use or "Context-engineered generation with cited, budgeted evidence.",
        out_of_scope=out_of_scope,
        evaluation=_eval_evidence(eval_report),
    )


def generate_system_card(
    target: ContextApp | VincioConfig,
    *,
    eval_report: EvalReport | None = None,
    format: CardFormat = CardFormat.VINCIO,
    name: str | None = None,
) -> SystemCard:
    """Build a :class:`SystemCard` describing the whole running system."""
    import vincio

    cfg = _config_of(target)
    model_card = generate_model_card(target, eval_report=eval_report, format=format)

    retrieval = {
        "enabled": getattr(target, "retrieval", None) is not None or bool(cfg.retrieval.embedder),
        "embedder": cfg.retrieval.embedder,
        "reranker": cfg.retrieval.reranker,
        "top_k": cfg.retrieval.top_k,
        "chunking": cfg.retrieval.chunking,
        "query_strategies": list(cfg.retrieval.query_strategies),
    }
    memory = {
        "enabled": cfg.memory.enabled,
        "write_policy": cfg.memory.write_policy,
        "decay_lambda": cfg.memory.decay_lambda,
        "ttl_days": dict(cfg.memory.ttl_days),
    }

    safety_filters: list[str] = []
    if cfg.security.pii_detection:
        safety_filters.append("PII detection & redaction")
    if cfg.security.injection_detection:
        safety_filters.append("prompt-injection detection (trust-tagged)")
    if cfg.policies.safety != "minimal":
        safety_filters.append(f"safety policy = {cfg.policies.safety}")
    if cfg.policies.block_untrusted_instructions:
        safety_filters.append("untrusted-content instruction blocking")
    if cfg.policies.require_citations:
        safety_filters.append("citations required on output")

    human_oversight = [
        "Tool calls with write side effects require approval (idempotency-keyed).",
        "Human-in-the-loop interrupts available on durable graphs and workflows.",
        "Audited, reversible memory edits/deletes and source erasure.",
    ]

    governance_controls = []
    if cfg.security.audit_log:
        governance_controls.append("hash-chained, offline-verifiable audit log")
    governance_controls.append(f"tenant isolation = {cfg.security.tenant_isolation}")
    governance_controls.append(f"privacy policy = {cfg.policies.privacy}")
    if cfg.security.retention_days or cfg.policies.retention_days:
        governance_controls.append("retention policy enforced")

    data_handling = {
        "tenant_isolation": cfg.security.tenant_isolation,
        "audit_log": cfg.security.audit_log,
        "retention_days": cfg.security.retention_days or cfg.policies.retention_days,
        "pii_redaction_in_context": cfg.policies.redact_pii_in_context,
        "answer_only_from_sources": cfg.policies.answer_only_from_sources,
    }

    return SystemCard(
        schema_format=format,
        vincio_version=vincio.__version__,
        name=name or cfg.project,
        model=model_card,
        retrieval=retrieval,
        memory=memory,
        safety_filters=safety_filters,
        human_oversight=human_oversight,
        data_handling=data_handling,
        governance_controls=governance_controls,
        evaluation=_eval_evidence(eval_report),
    )
