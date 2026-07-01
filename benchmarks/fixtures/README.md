# benchmarks/fixtures/ — fabricated miniatures, NOT real dataset slices

> **Provenance: fabricated (Tier-S / Mechanism).** Each file here is named after
> a real public benchmark (`gaia.json`, `swebench_verified.json`, `bird.json`, …)
> but is a **hand-authored 3–4-task miniature** that is *shaped like* that
> benchmark — not a slice of the real dataset. They exist to exercise each
> benchmark's own verifiable scorer end to end, offline and deterministically.
> They prove the *mechanism* is intact; they are **not** a real-world score.

See [../PROVENANCE.md](../PROVENANCE.md) for how this fits the three-track honesty
contract, and the [open evaluation plane](../../docs/concepts/open-evaluation-plane.md)
for the Recorded/Live tiers that run against *real* datasets.

## What each fixture is

- A tiny task list, each task carrying a `recorded` output so the adapter +
  scorer **replay** byte-identically with no model call. The scorer is the real
  one (SWE-bench's fail-to-pass/pass-to-pass transition, τ-bench's environment
  end state, GAIA's normalized exact match, BFCL's AST match, …).
- Most carry a `task_set_hash` the adapter recomputes and verifies on load, so a
  silent task-set change is caught.
- The `description` field on each fixture states plainly that it is
  "*-shaped*" / "a miniature of the real task".

## Running the real thing instead

These fixtures back the VincioBench `agentic_evals` / `data_analysis_conformance`
families and the open evaluation plane's Tier-S. To score a model on the **real**
public dataset, use the plane's Recorded/Live tiers with an official export — the
*identical* scoring code, a real task set:

```bash
python benchmarks/eval_live.py --provider anthropic --model claude-opus-4-8 \
  --benchmarks agent.gaia coding.swe_bench --tier live --dataset-dir ./datasets
```

## Files

`agentbench` · `bfcl` · `bird` · `dabench` · `ds_1000` · `gaia` ·
`infiagent_dabench` · `integrations` · `livecodebench` · `mmlu_pro` · `spider` ·
`swebench_verified` · `tau_bench` · `toolbench` · `webarena`
