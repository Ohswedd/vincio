"""Grounded document QA with citations."""

import tempfile
from pathlib import Path

from _shared import citing_responder, example_provider, write_sample_docs

from vincio import ContextApp

docs_dir = write_sample_docs(Path(tempfile.mkdtemp()) / "docs")
provider, model = example_provider(
    citing_responder("The refund window for the Pro plan is 30 days with no fee. [{ref}]")
)

app = ContextApp(name="docs_qa", provider=provider, model=model)
app.add_source("docs", path=str(docs_dir), chunking="adaptive", retrieval="hybrid")
app.set_policy("answer_only_from_sources", True)
app.add_evaluator("groundedness")
app.add_evaluator("citation_accuracy")

if __name__ == "__main__":
    result = app.run("What is the refund window for the Pro plan?")
    print("answer:", result.output)
    print("citations:", result.citations)
    print("eval scores:", result.eval_scores)
    print("excluded context:", result.excluded_context)
