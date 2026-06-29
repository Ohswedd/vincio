# CLI Research Agent

A bounded local research agent as a single-file command-line application — no
framework, no web server. It points a [Vincio](https://github.com/) `ContextApp`
at a small folder of your own markdown notes, answers a question *only* from that
evidence, and prints the answer with the exact citations that back it and the
cost of the run.

This is grounded retrieval-augmented generation (RAG) with guardrails, packaged
as a Unix-style command:

- `app.add_source("notes", ...)` indexes `./notes` (adaptive chunking + hybrid
  keyword/vector retrieval).
- `set_policy("answer_only_from_sources", True)` forbids the model from drawing
  on anything but the retrieved passages.
- `add_evaluator("groundedness")` scores how well the answer is actually
  supported by the evidence.

## Run it

```bash
python app.py "what is the refund window?"
```

With no argument it answers a sample question. On first run it writes a tiny
sample notes corpus into `./notes` (refund policy, subscription terms, support
SLA); drop more `.md` files in there and they are indexed automatically. Point
it at a different folder with `--notes`:

```bash
python app.py "how much downtime triggers a credit?" --notes ./notes
```

The output is the answer, the list of citations backing it, a groundedness
score, the run cost, and a trace id.

## Offline by default

Out of the box the app uses Vincio's bundled deterministic mock provider, so it
runs with **no API keys and no network access**. The answer text is a placeholder
from the mock, but the retrieval, citations, groundedness evaluation, trace, and
cost accounting are all real — which is what this example is demonstrating.

## Use a real model

Set `VINCIO_PROVIDER` (and the matching API key) to answer with a real model:

```bash
export VINCIO_PROVIDER=openai
export VINCIO_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python app.py "what is the refund window?"
```
