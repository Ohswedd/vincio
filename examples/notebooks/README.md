# Vincio notebooks (Google Colab-ready)

Five interactive notebooks that teach Vincio in the browser — **one `pip install`,
no API keys, no setup**. Every notebook runs fully offline on the bundled
deterministic mock provider; flip a single environment variable to run it against
a real model.

| # | Notebook | Open in Colab | What it teaches |
|---|---|---|---|
| 01 | [`01_quickstart.ipynb`](01_quickstart.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/01_quickstart.ipynb) | A first run, typed Pydantic output, grounded QA with citations, the trace id + cost on every result, and a short chat. |
| 02 | [`02_rag.ipynb`](02_rag.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/02_rag.ipynb) | Grounded RAG both ways — the one-line `rag()` front door and the verbose builder path — with citations and groundedness scores. |
| 03 | [`03_agents_and_tools.ipynb`](03_agents_and_tools.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/03_agents_and_tools.ipynb) | Permissioned tools, approval-gated writes, and the `tool_agent()` front door — a loop that cannot run away or fire a write unapproved. |
| 04 | [`04_evaluation.ipynb`](04_evaluation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/04_evaluation.ipynb) | Datasets, metrics, and CI gates — measure quality and block a regression, with `EvalRunner` and the `evaluation()` front door. |
| 05 | [`05_data_analysis.ipynb`](05_data_analysis.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohswedd/vincio/blob/main/examples/notebooks/05_data_analysis.ipynb) | The data plane — register a dataset, query it with cell-level provenance, analyze it into a cited narrative, and chart it, each `verify()`-ing offline. |

## Running locally

You can also open them in Jupyter or VS Code:

```bash
pip install vincio jupyter
jupyter notebook examples/notebooks/
```

Every notebook is gated in CI (`tests/test_example_notebooks.py`): its code cells
must parse and run offline, so a notebook can never drift from the working API.

## Running against a real model

Each notebook uses the shared `_provider()` helper, so to use a real model set,
before running the cells:

```python
import os
os.environ["VINCIO_PROVIDER"] = "openai"      # or anthropic / google / mistral
os.environ["VINCIO_MODEL"] = "gpt-4o-mini"
os.environ["OPENAI_API_KEY"] = "sk-..."
```
