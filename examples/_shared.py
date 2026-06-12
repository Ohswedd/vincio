"""Shared example helpers.

Every example runs fully offline by default using the deterministic mock
provider. Set VINCIO_PROVIDER (and the matching API key) to run against a
real model, e.g.:

    export VINCIO_PROVIDER=openai VINCIO_MODEL=gpt-5.2-mini OPENAI_API_KEY=sk-...
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from vincio.providers import MockProvider, build_provider
from vincio.providers.base import ModelProvider


def example_provider(default_responder=None, script=None) -> tuple[ModelProvider, str]:
    """Returns (provider, model) — mock offline, real when configured."""
    name = os.environ.get("VINCIO_PROVIDER", "mock")
    if name != "mock":
        return build_provider(name), os.environ.get("VINCIO_MODEL", "gpt-5.2-mini")
    return MockProvider(responder=default_responder, script=script), "mock-1"


def citing_responder(answer_template: str):
    """Mock responder that cites the first real evidence ref in the prompt."""

    def responder(request):
        text = "\n".join(m.text for m in request.messages)
        match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
        ref = match.group(1) if match else "E1"
        return answer_template.format(ref=ref)

    return responder


def json_responder(payload: dict):
    return lambda request: json.dumps(payload)


def write_sample_docs(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "refund_policy.md").write_text(
        "# Refund Policy\n\nCustomers on the Pro plan may request refunds within 30 days.\n\n"
        "## Fees\n\n| Plan | Window | Fee |\n|---|---|---|\n| Pro | 30 days | $0 |\n| Basic | 14 days | $5 |\n",
        encoding="utf-8",
    )
    (directory / "terms.md").write_text(
        "# Terms of Service\n\nThe subscription renews automatically unless terminated 60 days "
        "before renewal. The initial term is 24 months. Late payments accrue 1.5% monthly interest.\n",
        encoding="utf-8",
    )
    return directory
