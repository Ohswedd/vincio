"""Shared fixtures."""

from __future__ import annotations

import re

import pytest

from vincio import ContextApp, VincioConfig
from vincio.core.types import Document, EvidenceItem
from vincio.providers import MockProvider


@pytest.fixture()
def tmp_cwd(tmp_path, monkeypatch):
    """Run a test inside an isolated working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def sample_docs_dir(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "policy.md").write_text(
        "# Refund Policy\n\n"
        "Customers on the Pro plan may request refunds within 30 days.\n\n"
        "## Eligibility\n\n"
        "| Plan | Window | Fee |\n|------|--------|-----|\n| Pro | 30 days | $0 |\n| Basic | 14 days | $5 |\n\n"
        "## Process\n\nOpen a ticket with the invoice ID.\n",
        encoding="utf-8",
    )
    (docs / "terms.md").write_text(
        "# Terms\n\nThe subscription renews automatically unless terminated 60 days before renewal. "
        "The initial term is 24 months.\n",
        encoding="utf-8",
    )
    return docs


@pytest.fixture()
def citing_mock_provider():
    """Mock provider that cites the first evidence ref it sees in the prompt."""

    def responder(request):
        text = "\n".join(m.text for m in request.messages)
        match = re.search(r"\[([\w.:-]+:C\d+)\]", text)
        ref = match.group(1) if match else "E1"
        return f"The refund window for the Pro plan is 30 days. [{ref}]"

    return MockProvider(responder=responder)


@pytest.fixture()
def offline_config(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return config


@pytest.fixture()
def rag_app(sample_docs_dir, citing_mock_provider, offline_config, tmp_cwd):
    app = ContextApp(
        name="test_qa", provider=citing_mock_provider, model="mock-1", config=offline_config
    )
    app.add_source("docs", path=str(sample_docs_dir), retrieval="hybrid")
    app.set_policy("answer_only_from_sources", True)
    return app


@pytest.fixture()
def sample_document():
    return Document(
        text=(
            "The contract renews automatically unless terminated 60 days before renewal. "
            "The order form has a 24-month initial term. Late payments accrue 1.5% monthly interest. "
            "Either party may terminate for material breach with 30 days written notice."
        ),
        title="msa",
    )


@pytest.fixture()
def sample_evidence():
    return [
        EvidenceItem(
            id="e1",
            source_id="D1",
            text="The contract renews automatically unless terminated 60 days before renewal.",
            authority=0.9,
            relevance=0.9,
        ),
        EvidenceItem(
            id="e2",
            source_id="D2",
            text="The order form has a 24-month initial term.",
            authority=0.8,
            relevance=0.6,
        ),
        EvidenceItem(
            id="e3",
            source_id="D3",
            text="Bananas are rich in potassium.",
            authority=0.5,
            relevance=0.01,
        ),
    ]
