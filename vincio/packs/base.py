"""Domain packs: opt-in prompt + schema + policy + eval bundles.

A :class:`Pack` is a ready-made starting point for a domain (support,
engineering, finance, legal): a role/objective/rules prompt configuration, a
structured output schema, recommended policies and evaluators, and a small
golden eval set so you can measure quality from day one. Packs ship inside the
package (no extra dependencies) and are applied on demand::

    app = ContextApp(name="helpdesk")
    app.use_pack("support")          # or: load_pack("support").apply(app)

Applying a pack is deterministic and uses only the public ``ContextApp`` API
(``configure`` / ``set_policy`` / ``add_evaluator`` / ``add_rail``), so you can
layer your own configuration on top afterwards.
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
        return app


_BUILTIN_MODULES = {
    "support": "vincio.packs.support",
    "engineering": "vincio.packs.engineering",
    "finance": "vincio.packs.finance",
    "legal": "vincio.packs.legal",
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
