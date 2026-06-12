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

Notes:

- REST connectors accept an injected `client=` (`httpx.AsyncClient`) — use
  `httpx.MockTransport` for offline tests.
- `s3`/`gcs` accept an injected boto3 / `google.cloud.storage` client.
- For an injected sqlite3 connection that `add_source` will use, open it
  with `check_same_thread=False`.

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
corpora without rebuilds — see [retrieval concepts](../concepts/retrieval.md#live-indexes).
