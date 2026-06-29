# Vincio applications — real-world backends

Small, production-shaped applications you can read in one sitting and copy as a
starting point. Each one is a *real* use of Vincio (not a feature tour), and each
runs **fully offline** on the bundled deterministic mock provider — point it at a
real model by setting one environment variable.

| Application | What it is | Shape |
|---|---|---|
| [`rag_service`](rag_service/) | A grounded document-QA HTTP microservice: answers strictly from a bundled knowledge base and returns the citations, cost, and trace id for every answer. | FastAPI |
| [`support_triage_api`](support_triage_api/) | A ticket-triage API: typed `Triage` output, per-user semantic memory, and an **approval-gated** escalation tool (writes are proposed for human sign-off, never auto-fired). | FastAPI |
| [`extraction_service`](extraction_service/) | A structured-extraction service: free invoice text in, a validated `Invoice` object out, with bounded structure-only self-correction. | FastAPI |
| [`cli_research_agent`](cli_research_agent/) | A bounded local research agent as a Unix-style command: `python app.py "your question"` retrieves from a notes folder and prints a cited, grounded answer. | single-file CLI |

## How they are structured

Each FastAPI application splits cleanly so the Vincio logic is testable without a
web framework:

- **`core.py`** — pure Vincio logic. It imports *nothing* from FastAPI, so it
  runs in CI with only the core library installed, fully offline. The one public
  function returns plain JSON-able dicts.
- **`main.py`** — a thin FastAPI shell that owns request/response typing, status
  codes, and error mapping, and delegates every decision to `core.py`.
- **`README.md`** — what it does, how to run it offline, how to point it at a
  real model, the endpoints, and copy-paste `curl` examples.

The CLI agent is a single `app.py` (no web framework).

## Run one offline

```bash
# FastAPI service (install the server extra once):
pip install "vincio[server]"
cd examples/applications/rag_service
uvicorn main:app --reload
# then: curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"what is the refund window?"}'

# or the CLI agent (no extra needed):
python examples/applications/cli_research_agent/app.py "what is the refund window?"
```

## Run against a real model

Set the provider and key in the environment before launching — nothing else
changes:

```bash
export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
```

Every application's `core.py` (and the CLI) is gated in CI by
`tests/test_example_apps.py`, which exercises it offline; the FastAPI shells are
additionally exercised with a test client when FastAPI is installed.
