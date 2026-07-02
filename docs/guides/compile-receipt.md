# Attach a compile receipt to a PR or incident

> A compact, text-light manifest of *why* a context packet was compiled — safe to
> share, deterministic, and diffable.

When a production run is surprising — a bad answer, stale memory, a privacy-scope
mismatch, budget trimming, or a replay divergence — you want one artifact that
explains the compile decision without exposing the underlying prompt or documents.
That is the [compile receipt](../concepts/compile-receipt.md).

## Get the receipt for a run

Every run links a receipt from its trace and returns it on the result:

```python
from vincio import ContextApp

app = ContextApp(name="support")
result = app.run("What is the refund window?")

receipt = result.metadata["compile_receipt"]   # a JSON-safe dict
print(receipt["receipt_hash"])
print([item["id"] for item in receipt["included"]])
print([(item["id"], item["reason"]) for item in receipt["excluded"]])
```

Or print it from the CLI for any stored run:

```bash
vincio trace receipt <trace_id>          # human summary (result.trace_id)
vincio trace receipt <trace_id> --json   # the full JSON receipt
```

## Build one directly and export it

From a compiled context (or a packet), build the typed `CompileReceipt`:

```python
from vincio.context import CompileReceipt

compiled = await app.context_compiler.compile(...)
receipt = compiled.receipt()

export = receipt.to_export()   # JSON-safe; carries no raw prompt/evidence text
assert receipt.verify()        # re-derives from bytes and checks its invariants
```

`to_export()` is the artifact you attach to a PR or an incident note. It is
text-light by construction — ids, content hashes, per-item scores, budget, privacy
posture, and conflict winners, never the underlying text.

## Diff two compiles

The receipt hash is stable over the compile *decision*, so a review can diff a
change against a baseline:

```python
baseline = old_compiled.receipt()
current = new_compiled.receipt()

if current.receipt_hash == baseline.receipt_hash:
    print("compile decision unchanged")
else:
    print(current.diverges_from(baseline))
    # {'included_added': [...], 'included_removed': [...],
    #  'excluded_added': [...], 'score_changes': [...],
    #  'used_tokens_delta': ..., 'render_changed': ...}
```

Attach the divergence to the PR that changed retrieval, scoring, or budgeting and
the reviewer sees exactly what moved into and out of the context — and why —
without ever seeing the documents.

See the runnable tour in
[`examples/17_compile_receipt.py`](../../examples/17_compile_receipt.py).

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Packet compile receipt](../concepts/compile-receipt.md)
- [Example: 17_compile_receipt.py](../../examples/17_compile_receipt.py)
- [Concept: Context packets & long-horizon governance](../concepts/context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
