"""Multimodal-native packet & capability facades.

A tour of the platform's structural core, plus the multimodal-native Context
Packet — all in-process, all offline:

  1. Capability facades: the god-object decomposed into narrow, lazy views
     (app.runs / .knowledge / .governance / .optimization / .serving / .training).
  2. Multimodal-native Context Packet: image and table evidence selected,
     budgeted, ordered, and cited alongside text in one scored packet.
  3. Structured FilterSpec with native pushdown + tenant scope into the engine.
  4. Async-first stores + a typed, versioned event catalog.
  5. Enterprise endpoints (Bedrock / Vertex / Azure) behind a pluggable auth
     strategy, in the same registry as every other provider.
  6. Mandatory egress DLP + a signed, Merkle-checkpointed audit chain.

Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

import asyncio

from _shared import example_provider

from vincio import ContextApp
from vincio.context.evidence_store import InMemoryEvidenceStore
from vincio.context.ir import ContextIR
from vincio.context.packet import ContextPacket
from vincio.core.config import SecurityConfig, VincioConfig
from vincio.core.events import EventBus, RunCompleted
from vincio.core.types import EvidenceItem, ImageRef, Message, ModelRequest, Objective
from vincio.providers import _registry as provider_registry  # noqa: PLC2701 - demo introspection
from vincio.retrieval import FilterSpec, and_, build_filter_spec, eq
from vincio.security.audit import AuditLog, HMACSigner, merkle_proof, verify_merkle_proof
from vincio.security.policy import PolicyEngine


def section(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * 4}")


def main() -> None:
    provider, model = example_provider()
    app = ContextApp(name="breaking_window", provider=provider, model=model)

    # 1) Capability facades — the decomposed, lazily-constructed surface.
    section("1. Capability facades")
    result = app.runs.run("Summarize the breaking window in one line.")
    print("run via app.runs.run ->", (result.output or result.raw_text)[:60])
    print("facade delegates to the same impl:", app.runs.run == app.run)
    print("governance facade exposes:", "model_card" in dir(app.governance))

    # 2) Multimodal-native Context Packet — image + table as first-class evidence.
    section("2. Multimodal-native Context Packet")
    ir = ContextIR(
        objective=Objective("What is the Pro plan annual fee?"),
        evidence=[
            EvidenceItem(id="e1", source_id="d1", text="The Pro plan annual fee is $99."),
            EvidenceItem(
                id="e2", source_id="d2", modality="image", source_type="image",
                image=ImageRef(path="/pricing.png", detail="high",
                               metadata={"caption": "Pro plan pricing chart"}),
            ),
            EvidenceItem(
                id="e3", source_id="d3", modality="table",
                table={"columns": ["plan", "fee"], "rows": [["Pro", 99]], "markdown": "| Pro | 99 |"},
            ),
        ],
    )
    store = InMemoryEvidenceStore()
    packet = ContextPacket.from_ir(ir, slim=True, evidence_store=store)
    modalities = {e["modality"] for e in packet.evidence_items}
    print("packet evidence modalities:", sorted(modalities))
    # Ship the slim packet to "another process" and materialize from the store.
    shipped = ContextPacket.model_validate_json(packet.model_dump_json())
    shipped.materialize(store=store)
    print("cross-process materialized text:",
          next(e["text"] for e in shipped.evidence_items if e["id"] == "e1"))

    # 3) Structured FilterSpec — serializable, pushed down, tenant-scoped.
    section("3. Structured FilterSpec + native pushdown")
    spec = and_(eq("tenant_id", "acme"), eq("kind", "table"))
    print("FilterSpec compiles to Pinecone:", spec.to_pinecone())
    print("FilterSpec compiles to pgvector SQL:", spec.to_sql_where(column="json")[0])
    tenant_scope: FilterSpec = build_filter_spec(tenant_id="acme")
    print("tenant scope is shared-or-mine (round-trips as data):",
          FilterSpec.model_validate_json(tenant_scope.model_dump_json()).is_leaf is False)

    # 4) Typed, versioned event catalog.
    section("4. Typed event catalog")
    bus = EventBus()
    bus.subscribe("run.completed", lambda e: print("  observed:", e.name, e.payload, "v", e.schema_version))
    bus.publish(RunCompleted(run_id="run_123", status="succeeded"))

    # 5) Enterprise endpoints behind a pluggable auth strategy.
    section("5. Enterprise endpoints")
    print("registered providers include:",
          sorted(n for n in provider_registry.names if n in {"bedrock", "vertex", "azure"}))

    # 6) Mandatory egress DLP + signed, Merkle-checkpointed audit chain.
    section("6. Egress DLP + signed audit chain")
    dlp = PolicyEngine(egress_dlp="block")
    # Synthetic api-key assembled at runtime (no scannable literal in source).
    leak = "sk-" + "live-" + "A" * 40
    verdict = dlp.scan_egress(ModelRequest(model="m", messages=[Message(role="user", content=f"key {leak}")]))
    print("egress DLP blocks outbound credential:", not verdict.allowed)

    signer = HMACSigner("demo-key")
    log = AuditLog(directory=None, signer=signer)
    log.record("run", run_id="run_123")
    log.record("output", run_id="run_123")
    print("audit chain verifies with signatures:", log.verify_chain())
    checkpoint = log.checkpoint()
    hashes = [e.entry_hash for e in log.entries]
    print("Merkle checkpoint root signed:", bool(checkpoint.signature))
    print("Merkle inclusion proof verifies:",
          verify_merkle_proof(hashes[0], merkle_proof(hashes, 0), checkpoint.root))

    # An app configured for warn-mode DLP + a signed audit chain end to end.
    cfg = VincioConfig(security=SecurityConfig(audit_log=False, egress_dlp="warn",
                                               audit_signing_key="app-key"))
    governed = ContextApp(name="governed", provider=example_provider()[0], config=cfg)
    print("app audit chain is signed:", governed.audit.signer is not None)

    asyncio.run(app.aclose())


if __name__ == "__main__":
    main()
