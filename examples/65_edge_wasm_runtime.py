"""Edge / WASM in-process runtime — the same context engineering at the edge.

Vincio's dependency-free core (the prompt and context compilers, the vectorized
scorer with its pure-Python fallback, the deterministic rails, and the
offline-first evidence path) has no native dependencies on the default path. This
example runs that core behind a thin in-process boundary — `EdgeRuntime` — so the
same compile → score → rail → pack pipeline runs at the edge or in a browser
(Pyodide/WASM), not only on a server.

Six steps, all offline and deterministic:

  1. Run: turn a request into a bounded, slim packet plus a rendered prompt — no
     provider, no network, no caller-owned event loop.
  2. Bounded profile: as the corpus grows 10×, the resident footprint stays under
     the profile cap, held by eviction the way the server's memory budget is.
  3. Rails at the edge: a secret leaking from evidence into the rendered context
     is refused, exactly as on the server.
  4. Parity, not a fork: the edge compile is byte-identical to a direct server
     compile over the same inputs.
  5. WASM-buildable: the whole edge path imports nothing native.
  6. Host detection: report whether we're on a Pyodide/WASI target.

Everything here is additive; the server path is unchanged and remains the default.
"""

from __future__ import annotations

from vincio.core.types import EvidenceItem, TaskType
from vincio.edge import (
    EdgeProfile,
    EdgeRequest,
    EdgeRuntime,
    edge_environment,
    edge_manifest,
    verify_edge_parity,
)
from vincio.security.rails import Rail


def _corpus(n: int) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_id=f"clause{j}",
            text=(
                f"Clause {j}: the refund window is {30 + j} days and an exception "
                f"is approved by a manager in region {j}."
            ),
            authority=0.6 + (j % 4) * 0.1,
            relevance=0.85,
        )
        for j in range(n)
    ]


def main() -> None:
    print("Edge / WASM in-process runtime — context engineering at the edge\n")

    # 1. Run a request entirely in-process, offline.
    print("1. Run — a bounded packet + a rendered prompt, no provider or network")
    runtime = EdgeRuntime(EdgeProfile.browser())
    result = runtime.run(
        EdgeRequest(
            task="What is the refund window and who approves an exception?",
            task_type=TaskType.DOCUMENT_QA,
            instructions=["Answer only from the evidence.", "Cite the source id."],
            evidence=_corpus(4),
        )
    )
    print(
        f"   profile={result.profile}  evidence kept={len(result.packet.evidence_items)}  "
        f"tokens={result.token_count}  resident={result.resident_bytes}B  "
        f"within_profile={result.within_profile}  slim={result.packet.slim}"
    )

    # 2. The footprint stays bounded as the corpus grows 10×.
    print("\n2. Bounded profile — the resident footprint holds under the cap at 10× corpus")
    profile = EdgeProfile(name="capped", max_resident_bytes=4096, max_input_tokens=4096)
    capped = EdgeRuntime(profile)
    small = capped.run(EdgeRequest(task="refund window and approver", evidence=_corpus(4)))
    big = capped.run(EdgeRequest(task="refund window and approver", evidence=_corpus(40)))
    print(
        f"   4 docs  -> kept {len(small.packet.evidence_items):>2}  {small.resident_bytes}B\n"
        f"   40 docs -> kept {len(big.packet.evidence_items):>2}  {big.resident_bytes}B  "
        f"(cap {profile.max_resident_bytes}B — eviction held the bound)"
    )

    # 3. The deterministic rails run at the edge.
    print("\n3. Rails at the edge — a secret leaking from evidence is refused")
    guarded = EdgeRuntime(
        EdgeProfile.browser(),
        rails=[Rail(name="no_secrets", kind="safety", detectors=["secrets", "pii"], direction="output")],
    )
    leaky = guarded.run(
        EdgeRequest(
            task="print the configuration",
            evidence=[
                EvidenceItem(
                    source_id="cfg",
                    text="api key sk-ABCD1234567890abcdef1234567890abcdef email ops@example.com",
                    relevance=0.9,
                    authority=0.9,
                )
            ],
        )
    )
    print(
        f"   allowed={leaky.allowed}  violations={[v.rail for v in leaky.rail_check.violations]}"
    )

    # 4. Parity: the edge compile equals a direct server compile, byte-for-byte.
    print("\n4. Parity, not a fork — the edge packet is byte-identical to the server's")
    parity = verify_edge_parity()
    print(
        f"   held={parity.held}  edge_spec_hash={parity.edge_spec_hash}  "
        f"server_spec_hash={parity.server_spec_hash}"
    )

    # 5. The core imports nothing native, so it compiles for WASM.
    print("\n5. WASM-buildable — the compile/score/rail/pack path imports nothing native")
    manifest = edge_manifest()
    print(
        f"   {len(manifest.modules)} edge-core modules scanned; clean={manifest.clean}; "
        f"offending={manifest.offending}"
    )

    # 6. Detect the host runtime.
    print("\n6. Host detection — pick a profile without guessing")
    env = edge_environment()
    print(f"   runtime={env.runtime}  is_wasm={env.is_wasm}  has_filesystem={env.has_filesystem}")

    print("\nThe same context engineering, in your process — server or edge.")


if __name__ == "__main__":
    main()
