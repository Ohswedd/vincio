# Support Triage API

A small, production-shaped FastAPI service that triages support tickets with
[Vincio](https://github.com/). Send it a ticket; get back a **typed verdict**
(category / priority / one-line summary), the reporting user's prior-ticket
context, and any **escalations pending human approval** — never auto-fired.

It runs **fully offline** by default on Vincio's deterministic mock provider
(no API keys, no network), and points at a real model with one environment
variable.

## Architecture

The service is split so the engine never depends on the web framework:

- **`core.py`** — pure Vincio logic. No `fastapi` import. Builds a `ContextApp`
  with an `output_schema=Triage` contract, user-scoped **semantic memory**, and
  an **approval-gated `escalate_ticket` write tool**. Exposes one function,
  `triage(ticket, user_id) -> dict`, returning plain JSON-able values. This file
  imports and runs with nothing but Vincio installed.
- **`main.py`** — the HTTP shell. The only file that imports FastAPI. Thin
  Pydantic-typed adapter that delegates to `core.triage`.

### How escalation stays safe

`escalate_ticket` is registered with `approval_required=True` and
`side_effects="write"`. The triage path **never executes it**. When a ticket is
`high`/`critical` priority, the response carries a `pending_approvals` entry
describing the exact action (tool + arguments) a human must approve — the
human-in-the-loop seam for irreversible actions.

> **Offline note:** the bundled mock fills the schema with placeholder values, so
> `priority` is never literally `"high"`/`"critical"` and the offline path returns
> an **empty** `pending_approvals`. Point the service at a real model (below) to
> see a genuine escalation proposed for approval. The example response below is
> illustrative of that real-model path.

## Run it offline

```bash
pip install vincio fastapi uvicorn
uvicorn main:app --reload
```

Or call the engine directly without HTTP (this is what CI does):

```bash
python -c "from core import triage; print(triage('I was double charged', 'u1'))"
```

## Point it at a real model

```bash
export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
uvicorn main:app
```

With no `VINCIO_PROVIDER` set, the bundled mock provider auto-generates a
schema-valid `Triage` so the whole service is deterministic and offline.

## Endpoints

| Method | Path      | Body                          | Returns                          |
|--------|-----------|-------------------------------|----------------------------------|
| GET    | `/health` | —                             | `{"status": "ok"}`               |
| POST   | `/triage` | `{"ticket": ..., "user_id": ...}` | typed triage verdict (below) |

### `POST /triage` response

```json
{
  "category": "billing",
  "priority": "high",
  "summary": "Customer reports being charged twice.",
  "pending_approvals": [
    {
      "tool": "escalate_ticket",
      "reason": "high priority billing ticket",
      "arguments": {"ticket_id": "trace-...", "reason": "...", "priority": "high"},
      "requires_human_approval": true
    }
  ],
  "prior_tickets": 0,
  "trace_id": "trace-...",
  "cost_usd": 0.0
}
```

`422` is returned for empty `ticket`/`user_id`; `502` for a provider/runtime
failure.

## Curl examples

```bash
# health
curl -s localhost:8000/health

# triage a billing issue
curl -s localhost:8000/triage \
  -H 'content-type: application/json' \
  -d '{"ticket": "I was double charged this month", "user_id": "u1"}'

# a second ticket from the same user — prior_tickets reflects the remembered history
curl -s localhost:8000/triage \
  -H 'content-type: application/json' \
  -d '{"ticket": "Still seeing the duplicate charge", "user_id": "u1"}'
```
