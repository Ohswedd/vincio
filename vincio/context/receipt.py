"""Packet compile receipt — a compact, text-light manifest that proves *why*
this exact context packet was compiled, without exposing raw prompt or evidence
text.

A run's trace shows the compile *stages* and the packet carries provenance, but
neither is a single artifact you can attach to a pull request or an incident and
diff across compile changes. The :class:`CompileReceipt` is that artifact: it is
**fingerprint-heavy and text-light** — ids, content hashes, per-item scores, the
budget and privacy summary, the conflict winners, and a pointer back to the run
trace — and never the underlying text. It is derived purely from a compiled
:class:`~vincio.context.ContextPacket` (plus the run's render context), so the
same inputs always produce the same receipt, and a changed source shows up as an
explicit divergence rather than a silent difference.

The receipt answers, for a surprising production run (a bad answer, stale memory,
a privacy-scope mismatch, budget trimming, or a replay divergence): what was
included and why, what was excluded and why, what conflicts were resolved and by
which rule, and what the budget and privacy posture were — all offline-verifiable
and safe to share.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..core.utils import stable_hash

if TYPE_CHECKING:
    from .compiler import CompiledContext
    from .packet import ContextPacket

__all__ = [
    "ReceiptItem",
    "ConflictSummary",
    "BudgetSummary",
    "PrivacySummary",
    "RenderInfo",
    "CompileReceipt",
]

# Contract version of the receipt schema itself; bumps when the receipt's shape
# or hashing changes. The *compiler* version stamped on each receipt is the
# running package version (see :meth:`CompileReceipt.from_packet`).
RECEIPT_SCHEMA_VERSION = "1.0"

# Exclusion reasons the compiler emits that represent a conflict resolution, and
# the rule each one applied. Any other reason is a plain exclusion.
_CONFLICT_RULES = {
    "conflict_lower_authority": "higher_authority",
    "conflict_stale": "newer",
}


def _content_hash(text: str) -> str:
    """Full SHA-256 of a candidate's scorable text, presented as ``sha256:…``.

    A stable, one-way fingerprint of the source content that lets a reviewer
    confirm two receipts reference the *same* evidence without ever shipping the
    text. Empty text hashes to ``None`` at the call site."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_block(block: Any) -> dict[str, float]:
    """Numeric budget-block figures only, defensively.

    The standard allocator report is a nested dict of numbers, so this is a
    pass-through on the normal path (identical output, so the receipt hash is
    unchanged). A non-standard ``budget_report`` supplied by a direct caller
    degrades to what is numeric instead of crashing the receipt build."""
    if not isinstance(block, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in block.items():
        # ``bool`` is an ``int`` subclass but is not a real budget figure.
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


class ReceiptItem(BaseModel):
    """One included or excluded context item, described by fingerprint and score.

    Carries the item's id, kind, citation locator, content hash, the scoring
    signals that drove the decision, and the reason — but never the item's text.
    """

    id: str
    kind: str  # evidence | memory | tool_result | …
    source_ref: str | None = None  # citation locator (e.g. ``D1:p4``), not text
    source_hash: str | None = None  # ``sha256:…`` of the source content
    score: float | None = None  # total selection utility
    relevance: float | None = None
    authority: float | None = None
    freshness: float | None = None
    reason: str | None = None  # inclusion reason, or the exclusion reason
    superseded_by: str | None = None  # the winner, for a superseded/conflict loss
    token_cost: int | None = None

    def _fingerprint(self) -> dict[str, Any]:
        """The decision-bearing fields, for the receipt hash (order-independent)."""
        return {
            "id": self.id,
            "kind": self.kind,
            # The citation locator is part of the decision: a swapped source
            # document or page is a real provenance change the hash must catch.
            "source_ref": self.source_ref,
            "source_hash": self.source_hash,
            "score": self.score,
            "relevance": self.relevance,
            "authority": self.authority,
            "freshness": self.freshness,
            "reason": self.reason,
            "superseded_by": self.superseded_by,
        }


class ConflictSummary(BaseModel):
    """A resolved or unresolved conflict between two same-topic items.

    ``winner`` / ``loser`` name the kept and dropped items when a rule decided
    it; both are ``None`` for an unresolved conflict where both items were kept
    and the discrepancy was reported to the model. ``rule`` is the deciding rule.

    ``differing_count`` records *how many* value units disagreed — the count only,
    never the disputed values themselves (those are raw evidence fragments, e.g.
    dollar amounts or dates, which the receipt never carries).
    """

    winner: str | None = None
    loser: str | None = None
    rule: str  # higher_authority | newer | unresolved_both_included
    kind: str | None = None  # value_disagreement | polarity_disagreement
    differing_count: int = 0


class BudgetSummary(BaseModel):
    """The input-token budget and how the compile spent it, per block."""

    max_input_tokens: int = 0
    used_tokens: int = 0
    blocks: dict[str, dict[str, float]] = Field(default_factory=dict)


class PrivacySummary(BaseModel):
    """The privacy posture of the compile, without any redacted content.

    ``omitted_raw_text`` is always ``True`` — the receipt structurally cannot
    carry raw prompt or evidence text — so a receipt is safe to attach to a PR,
    an incident note, or a user bug report.
    """

    privacy_scope: str = "open"
    redact_pii_in_context: bool = False
    redacted_count: int = 0  # PII spans redacted from the input context
    scope_excluded_count: int = 0  # items dropped for a privacy-scope mismatch
    omitted_raw_text: bool = True


class RenderInfo(BaseModel):
    """The provider/model the packet rendered to, and the render-identity hashes.

    ``context_ir_hash`` fingerprints the provider-neutral IR; ``rendered_packet_hash``
    fingerprints the exact bytes sent to the model. Populated on a run through the
    pipeline; ``None`` for a bare compiler-only compile.
    """

    provider: str | None = None
    model: str | None = None
    context_ir_hash: str | None = None
    rendered_packet_hash: str | None = None
    prompt_spec_hash: str | None = None


class CompileReceipt(BaseModel):
    """A compact, offline-verifiable manifest of one context-packet compile.

    Built purely from a compiled :class:`~vincio.context.ContextPacket` (plus the
    run's render context) via :meth:`from_packet` / :meth:`from_compiled`. It is
    text-light by construction — it holds ids, hashes, scores, budgets, privacy
    posture, conflict winners, and a pointer back to the trace, never raw text —
    so it can be attached to a PR or incident and diffed across compile changes.

    :attr:`receipt_hash` is a stable digest of the compile *decision* (inputs,
    inclusions/exclusions with their scores, budget, privacy, conflicts, and the
    render identity), excluding the per-run ids. Recompiling identical inputs
    yields the same ``receipt_hash``; a changed source yields a different one,
    which :meth:`diverges_from` reports as an explicit divergence.
    """

    packet_id: str
    trace_id: str | None = None
    run_id: str | None = None
    compiler_version: str
    policy_profile: str | None = None
    input_fingerprint: str  # content hash of the compile inputs (packet spec hash)
    budget: BudgetSummary = Field(default_factory=BudgetSummary)
    included: list[ReceiptItem] = Field(default_factory=list)
    excluded: list[ReceiptItem] = Field(default_factory=list)
    privacy: PrivacySummary = Field(default_factory=PrivacySummary)
    conflicts: list[ConflictSummary] = Field(default_factory=list)
    render: RenderInfo | None = None
    # Set by :meth:`diverges_from` / :meth:`with_divergence` when this receipt is
    # compared against a baseline and the decision changed. ``None`` means the
    # receipt was not compared, or matched its baseline exactly.
    divergence: dict[str, Any] | None = None

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_packet(
        cls,
        packet: ContextPacket,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        render: RenderInfo | None = None,
        redacted_count: int = 0,
        policy_profile: str | None = None,
    ) -> CompileReceipt:
        """Build a receipt from a compiled packet.

        Reads the packet's enriched evidence/memory entries and excluded report to
        reconstruct the included/excluded items with their scores, the budget and
        privacy summary, and the resolved/unresolved conflicts. ``render`` /
        ``run_id`` / ``redacted_count`` are the run-context enrichments the runtime
        supplies; a bare compiler-only compile leaves them at their defaults.
        """
        from .. import __version__

        included: list[ReceiptItem] = []
        for entry in packet.evidence_items:
            included.append(
                ReceiptItem(
                    id=str(entry.get("id")),
                    kind="evidence",
                    source_ref=entry.get("citation_ref") or entry.get("source_id"),
                    source_hash=entry.get("source_hash"),
                    score=entry.get("score"),
                    relevance=entry.get("relevance"),
                    authority=entry.get("authority"),
                    freshness=entry.get("freshness"),
                    reason=entry.get("reason") or "selected",
                    token_cost=entry.get("token_cost"),
                )
            )
        for entry in packet.memory_included:
            included.append(
                ReceiptItem(
                    id=str(entry.get("id")),
                    kind="memory",
                    source_hash=entry.get("source_hash"),
                    score=entry.get("score"),
                    relevance=entry.get("relevance"),
                    authority=entry.get("authority"),
                    freshness=entry.get("freshness"),
                    reason=entry.get("reason") or "selected",
                )
            )

        excluded: list[ReceiptItem] = []
        conflicts: list[ConflictSummary] = []
        scope_excluded = 0
        for entry in packet.excluded_report:
            reason = str(entry.get("reason") or "")
            if reason == "privacy_scope_mismatch":
                scope_excluded += 1
            kind = entry.get("kind")
            if kind is None and reason == "privacy_scope_mismatch":
                kind = "memory"
            excluded.append(
                ReceiptItem(
                    id=str(entry.get("id")),
                    kind=str(kind or "evidence"),
                    source_hash=entry.get("source_hash"),
                    score=entry.get("score"),
                    reason=reason or None,
                    superseded_by=entry.get("superseded_by"),
                    token_cost=entry.get("token_cost"),
                )
            )
            if reason in _CONFLICT_RULES:
                conflicts.append(
                    ConflictSummary(
                        winner=entry.get("superseded_by"),
                        loser=str(entry.get("id")),
                        rule=_CONFLICT_RULES[reason],
                    )
                )
        for entry in packet.conflicts:
            conflicts.append(
                ConflictSummary(
                    winner=None,
                    loser=None,
                    rule="unresolved_both_included",
                    kind=entry.get("kind"),
                    # Only the count of disagreeing value units — never the
                    # disputed values, which are raw evidence fragments.
                    differing_count=len(entry.get("differing") or []),
                )
            )

        policies = packet.policies
        privacy = PrivacySummary(
            privacy_scope=policies.privacy,
            redact_pii_in_context=policies.redact_pii_in_context,
            redacted_count=redacted_count,
            scope_excluded_count=scope_excluded,
            omitted_raw_text=True,
        )
        if policy_profile is None:
            policy_profile = policies.custom.get("profile") or (
                f"{policies.privacy}+{policies.safety}"
            )

        budget = BudgetSummary(
            max_input_tokens=packet.budgets.max_input_tokens,
            used_tokens=packet.token_count,
            blocks={
                str(name): _coerce_block(block)
                for name, block in packet.budget_report.items()
            },
        )

        return cls(
            packet_id=packet.id,
            trace_id=trace_id if trace_id is not None else packet.trace_parent_id,
            run_id=run_id,
            compiler_version=f"vincio/{__version__}",
            policy_profile=policy_profile,
            input_fingerprint="sha256:" + packet.spec_hash,
            budget=budget,
            included=included,
            excluded=excluded,
            privacy=privacy,
            conflicts=conflicts,
            render=render,
        )

    @classmethod
    def from_compiled(
        cls,
        compiled: CompiledContext,
        **kwargs: Any,
    ) -> CompileReceipt:
        """Build a receipt from a :class:`~vincio.context.CompiledContext`.

        Convenience over :meth:`from_packet` — the compiled context holds the same
        packet the runtime persists, so this is the compiler-side entry point when
        no run/render context is available yet.
        """
        return cls.from_packet(compiled.packet, **kwargs)

    # -- verification & divergence --------------------------------------------

    def _hash_payload(self) -> dict[str, Any]:
        """The decision-bearing content the receipt hash covers.

        Excludes the per-run identities (packet/run/trace id) and the divergence
        field so a replay of identical inputs hashes identically; includes the
        compiler version and render identity so a compiler change or a re-render
        against a different provider is a detectable divergence.
        """
        return {
            "compiler_version": self.compiler_version,
            "policy_profile": self.policy_profile,
            "input_fingerprint": self.input_fingerprint,
            "budget": self.budget.model_dump(),
            "included": sorted(
                (it._fingerprint() for it in self.included), key=lambda d: d["id"]
            ),
            "excluded": sorted(
                (it._fingerprint() for it in self.excluded), key=lambda d: (d["id"], d["reason"] or "")
            ),
            "privacy": self.privacy.model_dump(),
            "conflicts": sorted(
                (c.model_dump() for c in self.conflicts),
                key=lambda d: (d.get("loser") or "", d.get("winner") or "", d["rule"]),
            ),
            "render": self.render.model_dump() if self.render is not None else None,
        }

    @property
    def receipt_hash(self) -> str:
        """Stable digest of the compile decision (see the class docstring)."""
        return "sha256:" + stable_hash(self._hash_payload(), length=32)

    def verify(self) -> bool:
        """Re-derive the receipt from its own bytes and check its invariants.

        Confirms the receipt round-trips byte-stably (a serialize→parse cycle
        reproduces ``receipt_hash``), the budget is not overspent, the included
        and excluded id sets are disjoint, and raw text is omitted. A receipt that
        verifies is internally consistent and safe to trust as an audit artifact.
        """
        restored = CompileReceipt.model_validate(self.model_dump(mode="json"))
        if restored.receipt_hash != self.receipt_hash:
            return False
        if self.budget.used_tokens > self.budget.max_input_tokens > 0:
            return False
        included_ids = {it.id for it in self.included}
        excluded_ids = {it.id for it in self.excluded}
        if included_ids & excluded_ids:
            return False
        return self.privacy.omitted_raw_text

    def diverges_from(self, baseline: CompileReceipt) -> dict[str, Any] | None:
        """Return a structured divergence vs a baseline receipt, or ``None``.

        Two receipts of the same inputs share a ``receipt_hash`` and this returns
        ``None``. When they differ, the delta names what changed: which items were
        added or removed from the context, which scores moved, the budget delta,
        and whether the render identity changed — the explicit divergence a replay
        or a changed source must surface rather than hide.
        """
        if self.receipt_hash == baseline.receipt_hash:
            return None
        self_in = {it.id: it for it in self.included}
        base_in = {it.id: it for it in baseline.included}
        self_ex = {it.id for it in self.excluded}
        base_ex = {it.id for it in baseline.excluded}
        # Per-item field changes over the items present in both compiles — so a
        # source that was edited in place (same id, unchanged rounded score) still
        # surfaces via its ``source_hash`` (and any moved score/locator) rather
        # than producing an empty delta that hides the change.
        score_changes = []
        content_changes = []
        for item_id in sorted(set(self_in) & set(base_in)):
            cur, base = self_in[item_id], base_in[item_id]
            if cur.score != base.score:
                score_changes.append(
                    {"id": item_id, "baseline": base.score, "current": cur.score}
                )
            changed: dict[str, dict[str, Any]] = {}
            for field in ("source_ref", "source_hash", "relevance", "authority", "freshness", "reason"):
                before, after = getattr(base, field), getattr(cur, field)
                if before != after:
                    changed[field] = {"baseline": before, "current": after}
            if changed:
                content_changes.append({"id": item_id, "changed": changed})
        return {
            "receipt_hash": {"baseline": baseline.receipt_hash, "current": self.receipt_hash},
            "input_fingerprint_changed": self.input_fingerprint != baseline.input_fingerprint,
            "included_added": sorted(set(self_in) - set(base_in)),
            "included_removed": sorted(set(base_in) - set(self_in)),
            "excluded_added": sorted(self_ex - base_ex),
            "excluded_removed": sorted(base_ex - self_ex),
            "score_changes": score_changes,
            "content_changes": content_changes,
            "used_tokens_delta": self.budget.used_tokens - baseline.budget.used_tokens,
            "render_changed": (
                (self.render.model_dump() if self.render else None)
                != (baseline.render.model_dump() if baseline.render else None)
            ),
        }

    def with_divergence(self, baseline: CompileReceipt) -> CompileReceipt:
        """A copy of this receipt with :attr:`divergence` set against ``baseline``."""
        return self.model_copy(update={"divergence": self.diverges_from(baseline)})

    # -- export ----------------------------------------------------------------

    def to_export(self) -> dict[str, Any]:
        """The receipt as a JSON-safe dict, with its ``receipt_hash`` stamped in.

        This is the safe artifact to persist, attach to a PR/incident, or ship in
        a bug report — it carries no raw prompt or evidence text.
        """
        data = self.model_dump(mode="json")
        data["receipt_hash"] = self.receipt_hash
        return data
