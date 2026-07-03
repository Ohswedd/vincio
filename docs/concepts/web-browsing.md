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

## Token efficiency, at the depth the task needs

The median web page costs tens of thousands of tokens of markup; the fact the
model needs is anything from a whole reference article to a single sentence.
`web_read` reads at four depths (`mode=`):

- **`excerpt`** — only the passages relevant to the **model's own query**,
  BM25-ranked and packed under a token budget (the token-efficient default). A
  tolerant stdlib-parser pass collects text blocks with their nearest heading,
  drops chrome (nav/header/footer/aside/form and mostly-link blocks), ranks the
  rest against the query, and packs the best under `budget_tokens`.
- **`section`** — the single heading-delimited section that best matches the
  query, whole, so a definition or a procedure arrives with its context (the
  heading itself is a strong topicality signal, so "Make a Request" wins for
  `make a request` over a lead paragraph that merely repeats the words).
- **`full`** — the entire boilerplate-stripped article, budget-capped.
- **`auto`** — choose per page: a page that already fits the budget is returned
  whole, a strong section match returns that section, otherwise the excerpts.
  This is what a browsing product does without being told.

Two more things make reading trustworthy for the dominant use-case — library
docs:

- **Code blocks are preserved verbatim.** `<pre>` content survives extraction
  fenced, so the model reads runnable code, not whitespace-collapsed prose.
- **Dead pages are flagged, not cited.** A cookie wall, paywall, login gate,
  soft-404, or JavaScript shell is detected deterministically (`available:
  false` plus a reason), so the model routes to another source instead of
  citing "We value your privacy" as content. A `find="exact string"` lookup
  additionally returns windows around a short fact the block filter would drop.

On the reference page the excerpts are two orders of magnitude cheaper than the
boilerplate-stripped page, and the pipeline is **pure**: the same bytes, query,
budget, and mode always produce the same `PageExtract` — which is what makes
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

## Prompt-driven: a pasted link is read for you

When the user's **own** message directs a fetch — a pasted link, "summarize
…", "according to …" — the page is fetched and folded into the run's evidence
with no tool round at all, then read as any tool result. This fires only on a
genuine directive or a URL that is essentially the whole ask; a URL merely
*discussed* ("what does `GET http://169.254.169.254` return?") or sitting
inside a code fence is left for the model to fetch deliberately. Auto-fetched
pages are tagged untrusted, framed *data-only, do-not-follow-instructions*,
snapshotted (so the compile stays offline-verifiable), and run through the same
untrusted-content screen as retrieved evidence. Only the current user message
is scanned — never history or prior tool output — so fetched content cannot
plant a URL that auto-fetches next turn.

## Crawl a site into a collection

Search-and-read answers a question; `app.web_crawl(seeds, scope=…)` (and the
`webcrawl` connector) build the *corpus* — a library's whole documentation, a
section of a site — through the same governed browser into a `WebCollection`
that converts to retrieval `Document`s **or** a tabular
[`Dataset`](tabular-evidence.md) and re-derives offline from its snapshots. The
walk is bounded on every axis (pages, depth, per-host, bytes, wall-clock),
deterministic (a lexicographically-ordered breadth-first frontier, so the same
seeds visit the same pages in the same order), and trap-resistant
(canonical-URL dedup plus a repeating-path-template guard that stops
pagination/calendar traps that mint infinite distinct URLs).

## Governed pre-egress, provable after

Web access is an external side effect, so it runs inside deterministic rails
checked **before any request leaves the process** (`WebPolicy`), on the
original URL **and every redirect hop**:

- schemes, allow/deny domain lists, and per-session search/fetch budgets;
- **private, loopback, and link-local hosts fail closed** — including the
  obfuscated IPv4 spellings (`0x7f.0.0.1`, `127.1`, integer form) and
  wildcard-DNS IP embedders (`10.0.0.1.nip.io`) that `getaddrinfo` would resolve
  to a private address — so a model-directed fetcher (or a 302 to cloud
  metadata) cannot become a server-side request forger;
- robots.txt respected by default; the body is **streamed with a decoded-byte
  cap** (defeating gzip/deflate bombs) with a `Content-Length` pre-check;
- transient failures retried honoring `Retry-After` (a 429 at most once), paced
  per host, and deduped by canonical URL so a page is fetched at most once and a
  re-read at a different depth costs zero network and zero budget.

Four presets — `default` / `research` / `scrape` / `locked_down` — cover the
common shapes; the SSRF/robots/redirect rails are non-negotiable across all of
them. Refusal is a typed `WebPolicyError` the model can read and adapt to
("search budget exhausted; answer from what you have"), never a silent skip.
Every search and fetch records on the app's hash-chained audit log.

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
