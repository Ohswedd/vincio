"""Context Packet: the universal unit passed to models, tools,
agents, evaluators, and traces.

Zero-copy by construction: *slim* packets reference evidence text by
content hash instead of duplicating it (lazy materialization from the held
Context IR), and :meth:`ContextPacket.iter_json` streams the serialized
packet chunk by chunk so persisting or shipping a large packet never builds
the whole document in memory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from ..core.types import Budget, Objective, PolicySet, UserInput
from ..core.utils import new_id, stable_hash, to_jsonable, utcnow
from .evidence_store import EvidenceStore
from .ir import ContextIR

__all__ = ["ContextPacket"]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_STREAMED_FIELDS = (
    "evidence_items",
    "evidence_ledger",
    "memory_included",
    "memory_excluded",
    "excluded_report",
    "conflicts",
)


class ContextPacket(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ctx"))
    version: int = 1
    objective: Objective
    user_input: UserInput = Field(default_factory=UserInput)
    constraints: list[str] = Field(default_factory=list)
    memory_included: list[dict[str, Any]] = Field(default_factory=list)
    memory_excluded: list[dict[str, Any]] = Field(default_factory=list)
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ledger: list[dict[str, Any]] = Field(default_factory=list)
    tools_allowed: list[str] = Field(default_factory=list)
    tools_denied: list[str] = Field(default_factory=list)
    output_schema_ref: str | None = None
    budgets: Budget = Field(default_factory=Budget)
    policies: PolicySet = Field(default_factory=PolicySet)
    trace_parent_id: str | None = None
    token_count: int = 0
    spec_hash: str = ""
    created_at: Any = Field(default_factory=utcnow)
    excluded_report: list[dict[str, Any]] = Field(default_factory=list)
    budget_report: dict[str, Any] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    slim: bool = False  # evidence entries hold text_hash refs, not text copies

    # In-memory link back to the IR for lazy text materialization (slim
    # packets). Never serialized.
    _ir: ContextIR | None = PrivateAttr(default=None)

    @classmethod
    def from_ir(
        cls,
        ir: ContextIR,
        *,
        excluded_report: list[dict[str, Any]] | None = None,
        budget_report: dict[str, Any] | None = None,
        conflicts: list[dict[str, Any]] | None = None,
        memory_excluded: list[dict[str, Any]] | None = None,
        trace_parent_id: str | None = None,
        token_count: int = 0,
        slim: bool = False,
        evidence_store: EvidenceStore | None = None,
    ) -> ContextPacket:
        evidence_items: list[dict[str, Any]] = []
        for e in ir.evidence:
            entry: dict[str, Any] = {
                "id": e.id,
                "citation_ref": e.citation_ref,
                "source_id": e.source_id,
                "source_type": e.source_type,
                "page": e.page,
                "relevance": e.relevance,
                # multimodal evidence is first-class in the packet, so a
                # downstream renderer/citer knows whether to ship text, an image,
                # or a table — and can cite each uniformly.
                "modality": e.modality,
            }
            if e.modality == "image" and e.image is not None:
                entry["image"] = e.image.model_dump(mode="json")
            if e.modality == "table" and e.table is not None:
                entry["table"] = e.table
            scorable = e.scorable_text
            if slim:
                entry["text_hash"] = _text_hash(scorable)
                entry["token_cost"] = e.token_cost
                # Persist the text under its hash so a packet shipped to another
                # process can materialize() it without the in-memory IR.
                if evidence_store is not None and scorable:
                    evidence_store.put(scorable)
            else:
                entry["text"] = e.text
            evidence_items.append(entry)
        packet = cls(
            objective=ir.objective,
            user_input=ir.input,
            constraints=[c.text for c in ir.constraints],
            memory_included=[
                {"id": m.id, "content": m.content, "scope": m.scope, "confidence": m.confidence}
                for m in ir.memory
            ],
            memory_excluded=memory_excluded or [],
            evidence_items=evidence_items,
            evidence_ledger=list(ir.evidence_ledger),
            tools_allowed=[t.name for t in ir.tool_specs],
            output_schema_ref=ir.output_contract.schema_ref,
            budgets=ir.budgets,
            policies=ir.policies,
            trace_parent_id=trace_parent_id,
            token_count=token_count,
            excluded_report=excluded_report or [],
            budget_report=budget_report or {},
            conflicts=conflicts or [],
            metadata=dict(ir.metadata),
            slim=slim,
        )
        packet._ir = ir
        packet.spec_hash = stable_hash(
            {
                "objective": packet.objective.text,
                "constraints": packet.constraints,
                "evidence": [e.get("id") for e in packet.evidence_items],
                "memory": [m.get("id") for m in packet.memory_included],
                "tools": packet.tools_allowed,
                "schema": packet.output_schema_ref,
            }
        )
        return packet

    # -- lazy materialization (slim packets) ---------------------------------

    def evidence_text(self, item_id: str) -> str | None:
        """The text of an evidence entry, materialized lazily for slim
        packets from the held IR (the text exists exactly once, on the IR's
        evidence items — the packet holds a hash reference)."""
        for entry in self.evidence_items:
            if entry.get("id") == item_id or entry.get("citation_ref") == item_id:
                text = entry.get("text")
                if text is not None:
                    return text
                break
        else:
            return None
        if self._ir is None:
            return None
        for item in self._ir.evidence:
            if item.id == item_id or item.citation_ref == item_id:
                return item.text
        return None

    def materialize(self, store: EvidenceStore | None = None) -> ContextPacket:
        """Fill evidence text in place (slim → full).

        Resolves from the held IR when present (same process), otherwise from a
        content-addressed :class:`EvidenceStore` keyed by each entry's
        ``text_hash`` — so a packet deserialized in another worker still
        materializes without the original IR.
        """
        if not self.slim:
            return self
        by_id = (
            {item.id: item.scorable_text for item in self._ir.evidence}
            if self._ir is not None
            else {}
        )
        resolved_all = True
        for entry in self.evidence_items:
            if "text" in entry:
                continue
            if entry.get("id") in by_id:
                entry["text"] = by_id[entry["id"]]
            elif store is not None and entry.get("text_hash"):
                text = store.get(entry["text_hash"])
                if text is not None:
                    entry["text"] = text
                else:
                    resolved_all = False
            else:
                resolved_all = False
        # Only flip to full when every entry's text was recovered.
        if resolved_all:
            self.slim = False
        return self

    # -- streaming assembly ----------------------------------------------------

    def iter_json(self) -> Iterator[str]:
        """Stream the packet as JSON chunks.

        List-heavy fields are emitted item by item, so serializing a large
        packet never holds more than one item's JSON in memory on top of the
        packet itself. ``"".join(packet.iter_json())`` equals a full dump.
        """
        head = self.model_dump(mode="json", exclude=set(_STREAMED_FIELDS))
        yield "{"
        first = True
        for key, value in head.items():
            prefix = "" if first else ", "
            yield f"{prefix}{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}"
            first = False
        for field_name in _STREAMED_FIELDS:
            prefix = "" if first else ", "
            yield f"{prefix}{json.dumps(field_name)}: ["
            for index, item in enumerate(getattr(self, field_name)):
                item_prefix = "" if index == 0 else ", "
                yield item_prefix + json.dumps(to_jsonable(item), ensure_ascii=False)
            yield "]"
            first = False
        yield "}"

    def approx_size_bytes(self) -> int:
        """Approximate serialized size, computed without building the blob."""
        return sum(len(chunk.encode("utf-8")) for chunk in self.iter_json())
