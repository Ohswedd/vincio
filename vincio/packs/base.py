"""Domain packs: opt-in prompt + schema + policy + eval bundles.

A :class:`Pack` is a ready-made starting point for a domain: a role / objective
/ rules prompt configuration, a structured output schema, recommended policies
and evaluators, and a small golden eval set so you can measure quality from day
one. Packs ship inside the package (no extra dependencies) and are applied on
demand::

    app = ContextApp(name="helpdesk")
    app.use_pack("support")          # or: load_pack("support").apply(app)

Two tiers ship in the box:

- **Domain packs** (``support`` / ``engineering`` / ``finance`` / ``legal``) — a
  light prompt + schema + policy starting point for a domain.
- **Vertical packs** (``healthcare`` / ``ediscovery`` / ``kyc`` /
  ``customer_support`` / ``code_review``) — a full-stack configuration that also
  preconfigures retrieval, scoped memory, deterministic rails, domain metrics, a
  data-residency posture, and a larger golden eval set for a regulated or
  high-stakes use case.

Applying a pack is deterministic and uses only the public ``ContextApp`` API
(``configure`` / ``set_policy`` / ``add_evaluator`` / ``add_rail`` /
``add_memory`` / ``set_residency``), so you can layer your own configuration on
top afterwards.
"""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import ConfigError
from ..evals.datasets import Dataset, EvalCase
from ..output.schemas import OutputContract, OutputSchema
from ..prompts.templates import PromptSpec

__all__ = ["Pack", "available_packs", "load_pack", "register_pack"]


class Pack(BaseModel):
    """A domain bundle: prompt config + schema + policies + evaluators + evals."""

    name: str
    description: str
    role: str = ""
    objective: str = ""
    rules: list[str] = Field(default_factory=list)
    soft_rules: list[str] = Field(default_factory=list)
    definitions: dict[str, str] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    output_schema_name: str = ""
    policies: dict[str, Any] = Field(default_factory=dict)
    evaluators: list[str] = Field(default_factory=list)
    rails: list[dict[str, Any]] = Field(default_factory=list)
    eval_cases: list[EvalCase] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # -- vertical-pack configuration (all optional / additive) ----------------
    # Retrieval knobs merged into ``app.config.retrieval`` on apply (e.g.
    # ``{"chunking": "sentence_window", "top_k": 10, "query_strategies": [...]}``);
    # ``mode`` is the default retrieval mode a domain source should use and is
    # surfaced via :meth:`retrieval_mode` for ``app.add_source(retrieval=...)``.
    retrieval: dict[str, Any] = Field(default_factory=dict)
    # When set, scoped memory is enabled on apply via ``app.add_memory(**memory)``
    # (e.g. ``{"scope": "user", "strategy": "semantic"}``), so the domain inherits
    # personalization / write-back from day one.
    memory: dict[str, Any] | None = None
    # Allowed provider regions for an in-jurisdiction posture; applied
    # fail-closed via ``app.set_residency(...)`` (an unresolvable region is
    # refused egress). Self-hosted/in-process processing is in jurisdiction by
    # construction, so ``"on_prem"`` is always admitted — which is why the
    # offline mock / local provider (region ``on_prem``) still run.
    residency: list[str] = Field(default_factory=list)
    # GDPR processing purpose this domain operates under (advisory metadata).
    purpose: str = ""

    def prompt_spec(self) -> PromptSpec:
        """Build the pack's prompt spec (handy for inspection or compilation)."""
        return PromptSpec(
            name=self.name,
            role=self.role,
            objective=self.objective,
            rules=list(self.rules),
            soft_rules=list(self.soft_rules),
            definitions=dict(self.definitions),
            output_schema=self.output_schema,
            output_format="json" if self.output_schema else "text",
        )

    def dataset(self) -> Dataset:
        """The pack's golden eval set as a :class:`Dataset`."""
        return Dataset(name=f"{self.name}_pack", cases=list(self.eval_cases))

    def retrieval_mode(self, default: str = "hybrid") -> str:
        """The retrieval mode this pack recommends for ``app.add_source``."""
        return str(self.retrieval.get("mode", default))

    @property
    def is_vertical(self) -> bool:
        """True when the pack preconfigures retrieval, memory, or residency."""
        return bool(self.retrieval or self.memory is not None or self.residency)

    def apply(self, app: Any, *, set_schema: bool = True, merge_rules: bool = False) -> Any:
        """Apply this pack to a :class:`ContextApp` and return the app.

        ``merge_rules=True`` appends the pack rules to any already configured;
        the default replaces them. ``set_schema=False`` skips installing the
        pack's output schema (keep your own).
        """
        config: dict[str, Any] = {}
        if self.role:
            config["role"] = self.role
        if self.objective:
            config["objective"] = self.objective
        if self.rules:
            config["rules"] = ([*app.prompt_spec.rules, *self.rules] if merge_rules else list(self.rules))
        if self.soft_rules:
            config["soft_rules"] = list(self.soft_rules)
        if self.definitions:
            config["definitions"] = dict(self.definitions)
        if config:
            app.configure(**config)
        for name, value in self.policies.items():
            app.set_policy(name, value)
        for evaluator in self.evaluators:
            if evaluator not in app.evaluators:
                app.add_evaluator(evaluator)
        # Idempotent: re-applying a pack (or layering two that share a rail
        # name) must not install the same rail twice — it would be evaluated
        # twice on every generation.
        existing_rails = {getattr(rail, "name", None) for rail in getattr(app.rail_engine, "rails", [])}
        for rail in self.rails:
            if rail.get("name") not in existing_rails:
                app.add_rail(**rail)
                existing_rails.add(rail.get("name"))
        if set_schema and self.output_schema:
            from ..output.repair import Repairer

            schema = OutputSchema.from_json_schema(
                self.output_schema, name=self.output_schema_name or self.name
            )
            app.output_contract = OutputContract.from_schema(
                schema, require_citations=app.policies.require_citations
            )
            app.repairer = Repairer(app.output_contract.repair_policy)
        # -- vertical configuration: retrieval / memory / residency -----------
        # Each goes through the public app API so a pack never reaches past the
        # contract it documents, and re-applying stays idempotent.
        for key, value in self.retrieval.items():
            if key == "mode":  # consumed by add_source, not a config field
                continue
            if hasattr(app.config.retrieval, key):
                setattr(app.config.retrieval, key, value)
        if self.memory is not None and app.memory is None:
            app.add_memory(**self.memory)
        if self.residency:
            # Self-hosted / in-process processing is in jurisdiction by
            # construction, so admit it alongside the declared regions. The
            # posture is fail-*closed* (``deny_on_unknown=True``): a provider
            # whose region cannot be resolved is refused egress. The
            # dependency-free offline path still runs because the deterministic
            # mock (and the local provider) resolve to the known ``on_prem``
            # region; a live deployment must pin a region-bearing endpoint or
            # declare ``provider_regions`` so its region is known too.
            regions = list(dict.fromkeys([*self.residency, "on_prem"]))
            app.set_residency(regions)
        return app


_BUILTIN_MODULES = {
    # Domain packs — a light prompt + schema + policy starting point.
    "support": "vincio.packs.support",
    "engineering": "vincio.packs.engineering",
    "finance": "vincio.packs.finance",
    "legal": "vincio.packs.legal",
    # Vertical packs — full-stack (retrieval + memory + rails + metrics +
    # residency + golden set) for a regulated or high-stakes use case.
    "healthcare": "vincio.packs.healthcare",
    "ediscovery": "vincio.packs.ediscovery",
    "kyc": "vincio.packs.kyc",
    "customer_support": "vincio.packs.customer_support",
    "code_review": "vincio.packs.code_review",
}
_CACHE: dict[str, Pack] = {}


def available_packs() -> list[str]:
    """Names of all packs that can be loaded (built-in + installed plugins + registered)."""
    from ..plugins import ensure_loaded

    ensure_loaded("vincio.packs")  # surface installed plugin packs (loads once)
    return sorted(set(_BUILTIN_MODULES) | set(_CACHE))


def register_pack(pack: Pack) -> Pack:
    """Register a custom pack so ``load_pack(pack.name)`` / ``use_pack`` find it."""
    _CACHE[pack.name] = pack
    return pack


def load_pack(name: str) -> Pack:
    """Load a pack by name (built-in modules import lazily; installed plugin
    packs register via the ``vincio.packs`` entry-point group on first miss)."""
    if name in _CACHE:
        return _CACHE[name]
    if name not in _BUILTIN_MODULES:
        from ..plugins import ensure_loaded

        ensure_loaded("vincio.packs")
        if name in _CACHE:
            return _CACHE[name]
        raise ConfigError(f"unknown pack {name!r}; available: {available_packs()}")
    module = importlib.import_module(_BUILTIN_MODULES[name])
    pack: Pack = module.PACK
    _CACHE[name] = pack
    return pack
