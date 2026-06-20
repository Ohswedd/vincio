"""Resident-memory footprint estimation for compiled context.

A compiled packet's resident footprint is dominated by the evidence and memory
text it carries — once in the Context IR, and again inline in a full (non-slim)
packet. This module gives a deterministic estimate of that footprint, used both
to enforce a declared per-app ceiling (by slimming the packet and evicting the
lowest-utility evidence) and to surface the figure in the run's cost summary.

The estimate is deterministic and monotonic — more or larger evidence always
estimates larger, slimming and eviction always estimate smaller — which is what
a budget enforcer and a regression gate need. It is an estimate, not a precise
``sys.getsizeof`` walk: introspecting live object graphs is neither
deterministic nor cheap on the hot path.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["estimate_resident_bytes", "ENTRY_OVERHEAD_BYTES"]

# Structural overhead per packet entry — ids, citation refs, scores, and the
# JSON scaffolding around each evidence/memory record. A flat, conservative
# constant keeps the estimate deterministic.
ENTRY_OVERHEAD_BYTES = 256


def _text_bytes(texts: Iterable[str]) -> int:
    return sum(len(text.encode("utf-8")) for text in texts)


def estimate_resident_bytes(
    evidence_texts: list[str], memory_texts: list[str], *, slim: bool
) -> int:
    """Estimated resident bytes of a compiled packet's context.

    A full packet holds each evidence text twice (in the IR and inline in the
    packet); a slim packet holds it once (the packet references it by hash), so
    slimming roughly halves the text footprint. Per-entry structural overhead is
    charged for every evidence and memory record.
    """
    entries = len(evidence_texts) + len(memory_texts)
    evidence_bytes = _text_bytes(evidence_texts)
    memory_bytes = _text_bytes(memory_texts)
    factor = 1 if slim else 2
    return entries * ENTRY_OVERHEAD_BYTES + evidence_bytes * factor + memory_bytes
