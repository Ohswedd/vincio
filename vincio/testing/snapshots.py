"""Snapshot testing for context packets and traces.

Snapshots capture the *structure* of a packet or trace — what was included,
in what shape — while normalizing away volatile fields (ids, timestamps,
durations, hashes), so they fail when behavior changes, not when the clock
does. Stored as pretty-printed JSON next to the tests; refresh with
``pytest --vincio-update-snapshots``.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from ..core.utils import to_jsonable

__all__ = ["Snapshot", "SnapshotMismatch", "normalize_trace", "normalize_packet"]


class SnapshotMismatch(AssertionError):
    pass


_VOLATILE_KEYS = {
    "id", "trace_id", "parent_id", "run_id", "span_id", "packet_id",
    "created_at", "start_time", "end_time", "timestamp", "indexed_at",
    "duration_ms", "latency_ms", "ttft_ms", "cost_usd",
    "spec_hash", "rendered_hash", "request_hash", "text_hash", "hash",
}


def _normalize(value: Any) -> Any:
    """Drop volatile keys recursively; keep structure and stable content."""
    if isinstance(value, dict):
        return {
            key: _normalize(item)
            for key, item in sorted(value.items())
            if key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def normalize_trace(trace: Any) -> dict[str, Any]:
    """Stable view of a trace: span tree shape, types, names, statuses."""
    def span_node(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": node["type"],
            "name": node["name"],
            "status": node["status"],
            "attribute_keys": sorted(node["attributes"].keys()),
            "children": [span_node(child) for child in node["children"]],
        }

    return {
        "app_name": trace.app_name,
        "status": trace.status,
        "attribute_keys": sorted(trace.attributes.keys()),
        "score_names": sorted(trace.scores.keys()),
        "spans": [span_node(node) for node in trace.span_tree()],
    }


def normalize_packet(packet: Any) -> dict[str, Any]:
    """Stable view of a context packet: what was included and excluded."""
    if hasattr(packet, "packet"):  # CompiledContext
        packet = packet.packet
    data = packet.model_dump(mode="json")
    return {
        "objective": _normalize(data.get("objective")),
        "constraints": data.get("constraints", []),
        "evidence": [
            {
                "source_id": item.get("source_id"),
                "section_path": item.get("section_path"),
                "trust_level": item.get("trust_level"),
            }
            for item in data.get("evidence_items", [])
        ],
        "memory_included": len(data.get("memory_included", [])),
        "tools_allowed": data.get("tools_allowed", []),
        "tools_denied": data.get("tools_denied", []),
        "output_schema_ref": data.get("output_schema_ref"),
        "excluded": [
            {"reason": item.get("reason"), "source_id": item.get("source_id")}
            for item in data.get("excluded_report", [])
        ],
    }


class Snapshot:
    """One test's snapshot store (use via the ``vincio_snapshot`` fixture)."""

    def __init__(self, directory: str | Path, test_name: str, *, update: bool = False) -> None:
        self.directory = Path(directory)
        self.test_name = test_name
        self.update = update

    def _path(self, name: str | None) -> Path:
        suffix = f"_{name}" if name else ""
        return self.directory / f"{self.test_name}{suffix}.json"

    def match(self, value: Any, *, name: str | None = None) -> None:
        """Compare a value against the stored snapshot (pydantic models,
        datetimes, sets, etc. are coerced to JSON structure first)."""
        normalized = _normalize(to_jsonable(value))
        rendered = json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True)
        path = self._path(name)
        if self.update or not path.exists():
            # First run records the snapshot; --vincio-update-snapshots rewrites.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
            return
        stored = path.read_text(encoding="utf-8").rstrip("\n")
        if stored != rendered:
            diff = "\n".join(
                difflib.unified_diff(
                    stored.splitlines(), rendered.splitlines(),
                    fromfile=str(path), tofile="current", lineterm="",
                )
            )
            raise SnapshotMismatch(
                f"snapshot mismatch for {path.name} "
                f"(run pytest --vincio-update-snapshots to accept):\n{diff}"
            )

    def match_trace(self, trace: Any, *, name: str | None = None) -> None:
        self.match(normalize_trace(trace), name=name or "trace")

    def match_packet(self, packet: Any, *, name: str | None = None) -> None:
        self.match(normalize_packet(packet), name=name or "packet")
