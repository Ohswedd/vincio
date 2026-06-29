# Grounded RAG Service

A small, production-shaped FastAPI microservice that answers customer questions
**grounded in a knowledge base**, returning the supporting citations, the
per-call cost, and a trace id for every answer.

It is built on [Vincio](../../../README.md) (`from vincio import ContextApp`) and
is split into two files so the AI logic stays testable without a web server:

| File        | Responsibility                                                        |
|-------------|-----------------------------------------------------------------------|
| `core.py`   | Pure Vincio logic. No FastAPI import. Plain functions, JSON-able dicts.|
| `main.py`   | HTTP shell. FastAPI app, Pydantic request/response models, error mapping. |

## What it does

On first use, `core.build_app()`:

1. writes a tiny bundled corpus (refund policy, billing, support) to `./knowledge`;
2. wires a provider via `_provider()` — a deterministic in-process **mock by
   default**, a real model only when `VINCIO_PROVIDER` is set;
3. attaches the corpus with `add_source(..., retrieval="hybrid")` (keyword + vector);
4. sets `answer_only_from_sources = True` so the model may only use retrieved
   evidence; and
5. attaches a `groundedness` evaluator so every answer is scored.

`core.answer(question)` returns:

```json
{
  "answer": "...",
  "citations": ["doc_...:C0"],
  "cost_usd": 0.0,
  "trace_id": "trace_...",
  "groundedness": 1.0
}
```

## Run it offline (no API keys, no network)

The default provider is a deterministic mock, so the service runs with zero
configuration. From this directory:

```bash
uvicorn main:app --reload
```

Then in another terminal:

```bash
# Liveness probe
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}

# Ask a grounded question
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "what is the refund window?"}'
```

You can also exercise the core directly, without the HTTP layer:

```bash
python core.py
# or:  python -c "import core, json; print(json.dumps(core.answer('what is the refund window?')))"
```

Interactive docs are served at <http://127.0.0.1:8000/docs> once `uvicorn` is up.

## Point it at a real model

Set `VINCIO_PROVIDER` (and the matching API key + optional model) before
launching. The application code does not change:

```bash
export VINCIO_PROVIDER=openai
export VINCIO_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
uvicorn main:app
```

With the mock provider the answer text is a deterministic placeholder, but the
**citations, cost, and trace id are real** — the grounding/evaluation pipeline is
exactly what runs against a live model.

## Endpoints

| Method | Path      | Body                 | Returns                                                        |
|--------|-----------|----------------------|---------------------------------------------------------------|
| `GET`  | `/health` | —                    | `{"status": "ok"}`                                            |
| `POST` | `/ask`    | `{"question": str}`  | `{answer, citations, cost_usd, trace_id, groundedness}`       |

Error handling: an empty question is rejected by request validation with `422`;
a whitespace-only question returns `400`; a provider/retrieval failure returns
`502`. Internal exception detail is logged server-side, never echoed to the client.

## Dependencies

- Offline core: `vincio` only (what `core.py` needs).
- HTTP shell: `fastapi` + an ASGI server such as `uvicorn` (what `main.py` needs).
