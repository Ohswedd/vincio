"""Codebase Q&A: code-aware chunking + repository import graph."""

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider

from vincio import ContextApp


def write_sample_repo() -> Path:
    repo = Path(tempfile.mkdtemp()) / "repo"
    repo.mkdir(parents=True)
    (repo / "billing.py").write_text(
        '"""Billing module."""\nimport datetime\n\n\n'
        "REFUND_WINDOW_DAYS = 30\n\n\n"
        "def is_refundable(paid_at, now=None):\n"
        '    """An invoice is refundable within REFUND_WINDOW_DAYS of payment."""\n'
        "    now = now or datetime.datetime.utcnow()\n"
        "    return (now - paid_at).days <= REFUND_WINDOW_DAYS\n",
    )
    (repo / "api.py").write_text(
        "import billing\n\n\ndef refund_endpoint(invoice):\n    return billing.is_refundable(invoice.paid_at)\n",
    )
    return repo


repo = write_sample_repo()
provider, model = example_provider(
    citing_responder("Refund eligibility is computed by is_refundable() in billing.py using a 30-day window. [{ref}]")
)

app = ContextApp(name="codebase_qa", provider=provider, model=model)
app.add_source("repo", path=str(repo), chunking="code_aware", retrieval="hybrid")

if __name__ == "__main__":
    result = app.run("Where is refund eligibility decided and what is the window?")
    print("answer:", result.output)
    print("evidence files:", sorted({e.metadata.get("source_uri", "") for e in result.evidence}))
