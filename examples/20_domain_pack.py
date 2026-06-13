"""Domain packs: a grounded support assistant in three lines (0.9).

``use_pack`` applies a domain bundle — role/objective/rules, a structured
output schema, recommended policies, evaluators, and a golden eval set — through
the public ``ContextApp`` API, so you can layer your own configuration on top.
"""

from _shared import example_provider, json_responder

from vincio import ContextApp, available_packs, load_pack

payload = {
    "category": "billing",
    "priority": "high",
    "response": "We've refunded the duplicate charge; it will post within 5 business days.",
    "needs_human": False,
}
provider, model = example_provider(json_responder(payload))

app = ContextApp(name="helpdesk", provider=provider, model=model).use_pack("support")
# Standalone demo: relax source-grounding (in production, attach your knowledge
# base with app.add_source(...) and keep answer_only_from_sources + citations on).
app.set_policy("answer_only_from_sources", False).set_policy("require_citations", False)


if __name__ == "__main__":
    print("available packs:", available_packs())
    pack = load_pack("support")
    print(f"schema: {pack.output_schema_name}  |  golden eval cases: {len(pack.eval_cases)}")
    result = app.run("I was charged twice this month")
    output = result.output
    output = output.model_dump() if hasattr(output, "model_dump") else output
    print("category:", output["category"], "| needs_human:", output["needs_human"])
    print("response:", output["response"])
