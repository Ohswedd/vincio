# Connect external data sources

The connector hub feeds the document engine from live systems. Every
connector returns plain `Document` objects with provenance (`source_uri`,
connector metadata, timestamps), so external content chunks, indexes,
budgets, and cites exactly like a local file.

## Quick start

```python
from vincio import ContextApp
from vincio.connectors import connect

app = ContextApp(name="kb", provider="openai", model="gpt-5.2")

# Web pages
app.add_source("docs", connector=connect("web", urls=[
    "https://example.com/handbook",
]))

# A GitHub repository (markdown + code, filtered by extension)
app.add_source("repo", connector=connect("github",
    repo="acme/handbook", token="ghp_...", path_prefix="docs/"))

# Rows from a database
app.add_source("faq", connector=connect("sql",
    query="SELECT id, question, answer FROM faq",
    url="sqlite:///kb.db", id_column="id", title_column="question"))
```

`add_source(connector=...)` loads, chunks, and indexes in one call; the
returned documents also work standalone: `docs = await connector.load()`.

## Built-in connectors

| Kind | Source | Dependencies |
|---|---|---|
| `web` | URLs (HTML stripped, title extracted) | none (core `httpx`) |
| `github` | repo files via the GitHub API | none |
| `sql` | query rows (SQLite built in; any DB-API connection) | none |
| `s3` | text objects in an S3 bucket | `pip install "vincio[s3]"` |
| `gcs` | text blobs in a GCS bucket | `pip install "vincio[gcs]"` |
| `notion` | database pages / page blocks | none |
| `confluence` | space pages (storage-format HTML) | none |
| `slack` | channel history (one document per channel) | none |
| `jira` | issues via the Jira Cloud REST API (ADF rendered to text) | none |
| `linear` | issues via the Linear GraphQL API | none |
| `gdrive` | files via the Drive API (Google-native docs exported) | none |
| `sharepoint` | document-library files via Microsoft Graph | none |
| `salesforce` | SOQL query results via the REST API | none |
| `zendesk` | Help Center articles via the REST API | none |
| `bigquery` | query rows | `pip install "vincio[bigquery]"` |
| `snowflake` | query rows | `pip install "vincio[snowflake]"` |

Notes:

- REST connectors (`web`, `github`, `notion`, `confluence`, `slack`, `jira`,
  `linear`, `gdrive`, `sharepoint`, `salesforce`, `zendesk`) run on the core
  `httpx` dependency and accept an injected `client=` (`httpx.AsyncClient`),
  use `httpx.MockTransport` for offline tests.
- `s3`/`gcs`/`bigquery` accept an injected client; `snowflake` an injected
  DB-API connection. The heavy SDK import is lazy, so they round-trip offline.
- For an injected sqlite3 connection that `add_source` will use, open it
  with `check_same_thread=False`.
- Authentication is per-connector: Jira/Zendesk take email + API token (Basic)
  or a bearer token; Linear an API key; Google Drive / SharePoint a bearer
  access token; Salesforce an instance URL + bearer token.

Third-party connectors can also register themselves on install via the
`vincio.connectors` entry-point group, see the [plugins guide](plugins.md).

## Custom connectors

Anything with an async `load() -> list[Document]` is a connector:

```python
from vincio.connectors import register_connector
from vincio.core.types import Document

@register_connector("tickets")
class TicketConnector:
    name = "tickets"

    def __init__(self, queue: str):
        self.queue = queue

    async def load(self) -> list[Document]:
        return [Document(text=..., source_uri=f"tickets://{self.queue}/{t.id}")
                for t in fetch(self.queue)]

app.add_source("support", connector=connect("tickets", queue="billing"))
```

## Keeping sources fresh

Pair connectors with a `LiveIndex` (upserts + TTL) to refresh changing
corpora without rebuilds, see [retrieval concepts](../concepts/retrieval.md#live-indexes).

## How it works

A connector's only job is to return `Document` objects with provenance
(`source_uri`, connector metadata, timestamps). From there the content is
identical to a local file: `add_source(connector=...)` runs the same
load → chunk → index path, and every chunk keeps its `source_uri`, so an answer
drawn from Notion or Snowflake cites back to the exact page or row. There is no
separate "external content" code path — provenance in, citations out.

## Best practice

- **Inject the client for tests.** Every REST connector accepts `client=` (an
  `httpx.AsyncClient`); wrap an `httpx.MockTransport` and the whole load → chunk →
  index loop runs offline and deterministically, exactly as the suite does.
- **Reach for `add_source(connector=...)` when you want a corpus indexed**, and
  the standalone `docs = await connector.load()` when you only want the
  `Document`s (to inspect, filter, or route elsewhere) — `load()` does not index.
- **Sample large tables instead of truncating.** The SQL-family connectors take
  an opt-in reservoir `sample=` that stands a representative sample in for a
  first-N cutoff.

## Gotchas

- **Warehouse extras are lazy but required.** `s3` / `gcs` / `bigquery` /
  `snowflake` import their heavy SDK only on use — so they round-trip offline
  with an injected client — but a *live* run raises a helpful
  `pip install "vincio[...]"` until the extra is installed.
- **Inject a sqlite connection with `check_same_thread=False`.** `add_source`
  may touch it off the calling thread; the default sqlite3 connection will raise.
- **Auth is per-connector, not global.** Jira/Zendesk take email + API token or a
  bearer; Linear an API key; Google Drive / SharePoint a bearer access token;
  Salesforce an instance URL + bearer — there is no single credential that
  unlocks all of them.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Retrieval (RAG)](../concepts/retrieval.md)
- [Guide: build a RAG app](build-rag-app.md)
- [Guide: Native video understanding & generation](video.md)
- [Example: 02_retrieval_rag.py](../../examples/02_retrieval_rag.py)
- [Concept: Context packets & long-horizon governance](../concepts/context-packets.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#knowledge)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
