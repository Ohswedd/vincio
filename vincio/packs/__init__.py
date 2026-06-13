"""Vincio domain packs.

Opt-in, dependency-free bundles of prompt configuration, output schema,
recommended policies/evaluators, and a golden eval set for a domain. Built-in
packs: ``support``, ``engineering``, ``finance``, ``legal``. Apply one with
``app.use_pack("support")`` or :meth:`Pack.apply`; register your own with
:func:`register_pack`.
"""

from .base import Pack, available_packs, load_pack, register_pack

__all__ = ["Pack", "available_packs", "load_pack", "register_pack"]
