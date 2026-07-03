# Give any model the open web

> One line registers governed `web_search` / `web_read` tools that every model
> can call — natively, or through the text protocol when it has no function
> calling — with token-efficient reading and offline-verifiable evidence.

## Enable it

```python
from vincio import ContextApp

app = ContextApp(name="assistant", provider="anthropic", model="claude-sonnet-5")
app.use_web_search()

result = app.run("What is the latest stable Python release, and when did it ship?")
print(result.raw_text)                 # cites the URLs it used
print([t.tool_name for t in result.tool_results])   # e.g. ['web_search', 'web_read']
```

That one call registers the two tools on the standard registry (RBAC, budget,
cache, and audit included), loads the built-in browsing skill (when to search,
how to write queries, when to stop — progressively disclosed), and wraps the
provider in `ToolProtocolProvider`, so the *same* line works unchanged for a
local model with no native tool calling:

```python
app = ContextApp(name="assistant", provider="ollama", model="llama3.2")
app.use_web_search()   # identical loop, via the text protocol when needed
```

## Set the policy

Every `WebPolicy` field passes through as a keyword; refusal happens before
any request leaves the process, as a typed `WebPolicyError` the model can
read and adapt to:

```python
app.use_web_search(
    max_searches=4,                     # per-session search budget
    max_fetches=6,                      # per-session page budget
    excerpt_budget_tokens=600,          # what one web_read may cost
    allow_domains=["python.org", "pypi.org"],   # strict allowlist (empty = all)
    deny_domains=["tracker.example"],
)
```

Private, loopback, and link-local hosts are refused by default (SSRF
fail-closed), and robots.txt is respected. Every search and fetch lands on
`app.audit`.

## Presets and prompt-driven fetching

Start from a named `WebPolicy` and override any field; the SSRF/robots rails
stay fixed across every preset:

```python
app.use_web_search(preset="research")   # generous budgets and reading depth
app.use_web_search(preset="locked_down", allow_domains=["docs.internal"])
```

With web search enabled, a user message that *directs* a fetch is read for the
model automatically — no tool round:

```python
app.run("Summarize https://peps.python.org/pep-0703/ for me")   # fetched + cited
```

Only genuine fetch directives (or a pasted link that is the whole ask) trigger
this; a URL merely mentioned — `"is http://169.254.169.254 the metadata IP?"` —
is not fetched.

## Drive the browser directly, at any depth

The tools are a thin veneer over `WebBrowser`, which you can use standalone.
Pick the reading depth with `mode`, or pull a short fact with `find`:

```python
from vincio.web import WebBrowser, WebPolicy

browser = WebBrowser(policy=WebPolicy.preset("research"))
hits = browser.search_sync("python 3.13 release date", recency="y")   # DuckDuckGo
page = browser.read_sync(hits[0].url, query="release date", mode="section")
print(page.title, f"{page.reduction:.0f}x cheaper than the page")
for excerpt in page.excerpts:                # code blocks come back fenced
    print(f"[{excerpt.section}] {excerpt.text}")

if not page.available:                        # a wall / paywall / JS shell
    print("unavailable:", page.unavailable_reason)   # → read another result

hit = browser.read_sync(hits[0].url, find="3.13.0")   # windows around a short fact
print([m.text for m in hit.find_matches])
```

## Crawl a site into a collection or dataset

```python
collection = app.web_crawl("https://docs.example.com/", scope="subtree")
app.add_source("docs", documents=collection.to_documents())   # → RAG
dataset = collection.to_dataset()                             # → the data plane
assert collection.verify(app.web_browser.snapshots)           # offline-verifiable
```

The walk is bounded on every axis (pages, depth, per-host, bytes, wall-clock),
deterministic, and trap-resistant; the `webcrawl` connector wraps the same walk
for `app.add_source(..., connector=connect("webcrawl", seeds=[...]))`.

## Verify a session offline

Every read is snapshotted and content-hashed; excerpts are a pure function of
(snapshot, query, budget), so the session re-derives from bytes:

```python
report = app.web_browser.report()       # searches + reads, in order
assert report.verify(app.web_browser.snapshots)

evidence = report.reads[0]
evidence.content_hash                   # sha256 of the page snapshot
evidence.verify(stored_bytes)           # True iff hash + excerpts re-derive
```

## Feed search into RAG and deep research

The `websearch` connector turns queries into cited, content-hashed
`Document`s, so the [research agent](../concepts/agents.md) becomes
web-backed with zero new agent code:

```python
from vincio.connectors import connect

question = "What changed in Python 3.13 free-threading?"
app.add_source("web", connector=connect("websearch", queries=[question]))
report = app.research(question)
```

## Test and air-gap it

The plane is offline-first. Inject a transport and a static engine and the
whole loop — including the text protocol — runs deterministically with no
network:

```python
import httpx
from vincio.web import StaticSearchBackend, SearchResult

backend = StaticSearchBackend({"my query": [SearchResult(rank=1, title="T", url="https://example.org/", snippet="s")]})
client = httpx.AsyncClient(transport=httpx.MockTransport(my_handler))
app.use_web_search(backend=backend, client=client)
```

Any engine plugs in the same way: implement `SearchBackend.search()` (SearXNG,
Brave, an intranet index) and pass it as `backend=`.

## Choosing between Vincio-executed and provider-hosted search

`app.use_hosted_tools(["web_search"])` still surfaces OpenAI's server-executed
search where you want it. Prefer `use_web_search()` when you need the same
behavior across *all* providers and local models, pre-egress domain policy,
token-budgeted reading, or offline-verifiable citations; prefer the hosted
tool when you specifically want the provider's own crawl and are on the one
provider that ships it.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Universal web browsing & search](../concepts/web-browsing.md)
- [Example: 19_web_browser_search.py](../../examples/19_web_browser_search.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
