# Universal web browsing & search

Some providers ship a hosted web-search tool, some ship a poor one, and local
models ship none. The `vincio.web` plane levels that: **any** model Vincio
serves — a hosted frontier model, an OpenAI-compatible gateway, or an
in-process GGUF with no function calling at all — gets the same two
Vincio-executed tools, `web_search` and `web_read`, governed like every other
action and provable after the fact. One line enables it:

```python
app.use_web_search()
```

The plane is built around three commitments.

## Token efficiency: a page is never forwarded

The median web page costs tens of thousands of tokens of markup; the fact the
model needs is usually one paragraph. `web_read` therefore returns only the
passages of a page relevant to the **model's own query**, packed under an
exact token budget:

1. a tolerant stdlib-parser pass collects text blocks with their nearest
   heading, skipping script/style/template subtrees;
2. navigation, header, footer, aside, and form subtrees are dropped, as is any
   block that is mostly link text (menus, tag clouds, "related articles"
   rails);
3. remaining blocks are ranked against the query by a self-contained BM25
   (with a light stemmer, plus a lead-position prior so a query-free read
   degrades to the article lead);
4. the best blocks are packed under `budget_tokens` and emitted in document
   order.

On the reference page the excerpts are two orders of magnitude cheaper than
the boilerplate-stripped page, and the pipeline is **pure**: the same bytes,
query, and budget always produce the same `PageExtract` — which is what makes
the evidence verifiable (below).

## Judgement: when to search ships as a skill

Giving a model a search tool is the easy half; the hard half is knowing when
the web helps, writing queries that find the fact, reading only what the
question needs, and stopping. That judgement ships as the built-in
[`browse_skill()`](../guides/agent-skills.md) — the Agent Skills shape — whose
summary line joins the always-disclosed skill index while its full
instructions surface through the skill library's **progressive disclosure**
only when the task looks web-relevant, scored and budgeted by the context
compiler like any other evidence. The same contract reaches every provider —
this is the first phase of teaching skills to models through the context
plane rather than through provider-specific system prompts.

The tool descriptions carry the compact version of the contract (search for
volatile, recent, niche, or citation-needing facts; 2–5 keyword queries; read
the most promising 1–2 results), so even a minimal integration inherits the
discipline.

## Every model, including those without function calling

Vincio's tool loop is driven by `ModelResponse.tool_calls`, which a provider
without native function calling can never populate. The
`ToolProtocolProvider` closes that gap by composition (the
`RetryingProvider` tradition): when a request carries tools and the wrapped
model does not claim `tool_calling`, it *lowers* the request — tool schemas
become a compact protocol block in the system message, prior tool turns fold
back into alternation-safe text — and *lifts* the reply, parsing fenced
`tool_call` JSON blocks into ordinary `ToolCallRequest`s. The runtime,
registry, permissions, budgets, and audit chain see exactly what a native
provider would have produced. `app.use_web_search()` applies the wrapper
automatically; a natively capable model passes through byte-untouched.

## Governed pre-egress, provable after

Web access is an external side effect, so it runs inside deterministic rails
checked **before any request leaves the process** (`WebPolicy`):

- schemes, allow/deny domain lists, and per-session search/fetch budgets;
- **private, loopback, and link-local hosts fail closed** — a model-directed
  fetcher must not become a server-side request forger against the network it
  runs in;
- robots.txt respected by default; byte and token ceilings per page.

Refusal is a typed `WebPolicyError` the model can read and adapt to ("search
budget exhausted; answer from what you have"), never a silent skip. Every
search and fetch records on the app's hash-chained audit log.

After the fact, the session is provable: every page read lands as a
`WebEvidence` content-bound to the SHA-256 of its snapshot, and because
extraction is pure, the excerpts re-derive offline from the snapshot bytes.
`app.web_browser.report()` returns the `WebSessionReport` whose `verify()`
checks the whole session from bytes — the same honesty contract charts and
narratives carry.

## The pieces

| Piece | Role |
|---|---|
| `DuckDuckGoBackend` | Default engine: keyless HTML endpoints, redirect decoding, ad dropping, typed rate-limit detection. |
| `SearchBackend` | The protocol any engine (SearXNG, Brave, an intranet index) implements to plug in. |
| `StaticSearchBackend` | Deterministic in-memory engine for tests, benchmarks, and air-gapped runs. |
| `extract_page` / `PageExtract` | The token-budgeted, deterministic reduction of a page. |
| `WebPolicy` | The pre-egress rails: domains, hosts, robots, budgets, ceilings. |
| `WebBrowser` | The governed session: search, read, snapshots, evidence, audit. |
| `WebEvidence` / `WebSessionReport` | Content-bound records that re-derive offline from snapshot bytes. |
| `browse_skill()` | The when/what/how judgement, progressively disclosed. |
| `ToolProtocolProvider` | Native-grade tool use for models without function calling. |
| `websearch` connector | Queries → cited, content-hashed `Document`s, making [deep research](agents.md) web-backed. |

Everything is offline-first: the whole plane is testable through an injected
transport, and `WebSearchBench` gates the token reduction, grounded recall,
SSRF refusal, offline verification, and the native-vs-protocol loop
equivalence.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Give any model the open web](../guides/web-search.md)
- [Example: 19_web_browser_search.py](../../examples/19_web_browser_search.py)
- [Concept: Prompt compiler](prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
