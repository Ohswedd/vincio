# Invoice Extraction Service

A small, production-shaped FastAPI backend that turns raw invoice text into a
**validated, structured invoice** using Vincio.

It demonstrates the real-world layering you'd ship:

- **`core.py`** — pure Vincio logic, *no web framework*. It builds a
  `ContextApp` whose output contract is a Pydantic `Invoice`
  (`vendor`, `total`, `currency`, `line_items`) and turns on **bounded
  self-correction** so a mis-shaped reply is repaired (structure-only, never
  invents facts) instead of discarded. It exposes one plain function,
  `extract(text) -> dict`. This module imports and runs with no HTTP
  dependency at all.
- **`main.py`** — the HTTP shell. A thin FastAPI app that maps endpoints onto
  `core.extract`, with typed Pydantic request/response bodies and proper error
  codes.
- **`README.md`** — this file.

## What it does

`POST /extract` with `{"text": "..."}` returns a JSON `Invoice`:

```json
{
  "vendor": "Acme Corp",
  "total": 1200.50,
  "currency": "USD",
  "line_items": ["widgets", "gadgets"]
}
```

The `Invoice` Pydantic model *is* the contract: its JSON schema drives the
provider's structured-output path, and the reply is parsed and validated back
into an `Invoice` before it leaves the service.

## Run it offline (no API keys, no network)

By default the service uses Vincio's deterministic **mock provider**, which
auto-generates a schema-valid `Invoice` — perfect for local dev, tests, and CI.

```bash
# from this directory
pip install fastapi uvicorn        # plus your Vincio install
uvicorn main:app --reload
```

Then:

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}

curl -s -X POST http://127.0.0.1:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Invoice from Acme Corp, total 1200.50 USD for widgets and gadgets"}'
```

You can also run the core logic directly, no server needed:

```bash
python core.py
```

## Point it at a real model

Set `VINCIO_PROVIDER` (and the matching API key) before starting the server —
**no code changes required**:

```bash
export VINCIO_PROVIDER=openai
export VINCIO_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
```

## Endpoints

| Method | Path       | Body                | Returns                          |
| ------ | ---------- | ------------------- | -------------------------------- |
| GET    | `/health`  | —                   | `{"status": "ok"}`               |
| POST   | `/extract` | `{"text": "<...>"}` | `Invoice` JSON                   |

### Errors

- **422** — empty or invalid `text`.
- **502** — extraction failed even after the bounded self-correction loop.

Interactive docs are available at `http://127.0.0.1:8000/docs` while the server
is running.
