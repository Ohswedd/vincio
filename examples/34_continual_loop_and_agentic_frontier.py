"""The loop closes itself + the agentic frontier (1.10).

Continual, online, safe self-improvement on the same cited, grounded spine —
plus the agentic frontier the field now expects, all in-process and gated:

  1. Online improvement controller: live drift → a gated re-eval / re-optimization
     / rollback to the last known-good prompt, debounced, budgeted, audited.
  2. Real provider-backed reflective optimizer (GEPA proper): read the actual
     failing cases, cluster them into failure modes, propose the targeted edit.
  3. Autonomous experiment proposer + held-out growing golden regression suite.
  4. Deep-research agent: search → read → reflect → verify → synthesize, cited
     and budget-bounded by construction.
  5. Agent memory OS: self-editing memory as permissioned, audited tools.
  6. Computer-use behind hardened isolation + provider-native hosted tools.

Runs fully offline on the deterministic mock provider.
"""

from __future__ import annotations

import re

from vincio import (
    ContextApp,
    ContinuousImprovementController,
    GoldenRegressionSuite,
    ResearchBudget,
    VincioConfig,
)
from vincio.core.types import Document
from vincio.evals import Dataset, EvalCase
from vincio.optimize.pareto import objectives_from_weights
from vincio.optimize.reflective import LLMReflector, cluster_failures
from vincio.optimize.search import FitnessWeights
from vincio.prompts.registry import PromptRegistry
from vincio.prompts.templates import PromptSpec
from vincio.providers import MockProvider
from vincio.tools.sandbox import SubprocessIsolation, require_real_isolation


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _app(responder=None) -> ContextApp:
    cfg = VincioConfig()
    cfg.storage.metadata = "memory://"
    cfg.observability.exporter = "memory"
    cfg.security.audit_log = False
    return ContextApp(name="frontier", provider=MockProvider(responder=responder), model="mock-1", config=cfg)


def controller_demo() -> None:
    _section("1. Online improvement controller (drift → gated rollback)")
    app = _app()
    registry = PromptRegistry(".vincio/frontier_prompts")
    good = registry.push(app.prompt_spec, tags=["production"])
    registry.push(app.prompt_spec.model_copy(update={"objective": "a regressed head"}))
    app.prompt_spec = registry.get(app.prompt_spec.name).spec

    ctl: ContinuousImprovementController = app.continuous_improvement(
        metrics=["safety"], sustain=2, registry=registry, prompt_name=app.prompt_spec.name,
    )
    # First sustained safety-drift signal observes; the second acts.
    ctl.evaluate("safety", {"method": "cusum"})
    decision = ctl.evaluate("safety", {"method": "cusum"})
    print(f"  action: {decision.action} → restored {decision.rolled_back_to} (was {good.ref})")
    print("  audited + traced; controller state is restart-safe")


def reflector_demo() -> None:
    _section("2. Real reflector (GEPA proper): cluster failures, propose the fix")
    from vincio.evals import EvalReport
    from vincio.evals.reports import CaseResult

    report = EvalReport(cases=[
        CaseResult(case_id="c1", metrics={"groundedness": 0.2, "lexical_overlap": 0.3},
                   output_text="an uncited claim"),
    ])
    dataset = Dataset(cases=[EvalCase(id="c1", input="What is the refund window?", expected="30 days")])
    clusters = cluster_failures(report, dataset)
    print(f"  dominant failure mode: {clusters[0]['mode']} ({clusters[0]['count']} case)")

    def reflect_responder(req):
        import json
        return json.dumps({"diagnosis": "answers were under-cited",
                           "edits": [{"field": "citation_policy", "op": "set",
                                      "value": "Cite [Ek] for every claim.", "rationale": "low groundedness"}]})

    reflector = LLMReflector(MockProvider(responder=reflect_responder), "mock-1")
    reflection = reflector.reflect(PromptSpec(name="qa", objective="answer"), report,
                                   objectives=objectives_from_weights(FitnessWeights()), dataset=dataset)
    print(f"  diagnosis: {reflection.diagnosis}")
    print(f"  proposed edits: {[e.field for e in reflection.edits]}")


def experiment_proposer_demo() -> None:
    _section("3. Experiment proposer + held-out growing golden suite")
    app = _app()
    proposer = app.experiment_proposer(eval_budget=12)
    proposals = proposer.rank({"groundedness": 0.5, "schema_validity": 0.97})
    for p in proposals:
        print(f"  weakest: {p.target_metric} ({p.current}) → {p.kind} experiment, budget {p.eval_budget}")

    suite = GoldenRegressionSuite(".vincio/frontier_golden.jsonl")
    suite.add(EvalCase(id="g1", input="refund window?", expected="30 days"),
              fixed_by="qa@v2", guard_metric="lexical_overlap", guard_threshold=0.8)
    from vincio.evals import EvalReport
    from vincio.evals.reports import CaseResult

    regressing = EvalReport(cases=[CaseResult(case_id="g1", metrics={"lexical_overlap": 0.3})])
    print(f"  golden suite ({len(suite)} guard) blocks a regressing promotion: "
          f"{not suite.gate(regressing).passed}")


def deep_research_demo() -> None:
    _section("4. Deep-research agent (cited, grounded, budget-bounded)")

    def responder(req):
        text = "\n".join(m.text for m in req.messages)
        match = re.search(r"\[([\w.:-]+)\]", text)
        ref = match.group(1) if match else "E1"
        return f"The Pro plan refund window is 30 days. [{ref}]"

    app = _app(responder)
    app.add_source("docs", documents=[
        Document(id="refunds", title="Refund Policy",
                 text="The refund window for the Pro plan is 30 days from purchase. "
                      "Enterprise customers get a 60 day window."),
    ])
    report = app.research("What is the refund window for the Pro plan?",
                          budget=ResearchBudget(breadth=3, depth=1, max_sources=6))
    print(f"  answer: {report.answer[:80]}")
    print(f"  sources: {len(report.sources)} · citation coverage: "
          f"{report.metrics['citation_coverage']:.0%} · grounding: {report.metrics['grounding']:.0%}")


def memory_os_demo() -> None:
    _section("5. Agent memory OS (self-editing memory as audited tools)")
    app = _app()
    os = app.enable_memory_os(owner_id="agent-1", max_core_tokens=2000)
    mid = os.append("The user prefers concise, well-cited answers.")
    print(f"  appended memory {mid[:12]}… via the guarded write pipeline")
    print(f"  search: {os.search('answer preference')}")
    os.archive(mid)
    print(f"  archived (paged out of core); registered tools: "
          f"{[t for t in app.tool_registry.names if t.startswith('memory_')]}")


def computer_use_demo() -> None:
    _section("6. Computer-use behind hardened isolation + hosted tools")
    # Process isolation is not a security boundary; real workloads require one.
    try:
        require_real_isolation(SubprocessIsolation())
    except Exception as exc:  # noqa: BLE001 - demonstrating the guard
        print(f"  isolation guard: {type(exc).__name__} — subprocess refused for adversarial work")

    app = _app()
    app.enable_computer_use("mock")  # navigate / click / type / screenshot, permissioned + audited
    app.use_hosted_tools(["web_search", "code_interpreter"])  # provider-native, namespaced
    computer = [t for t in app.tool_registry.names if t.startswith("computer_")]
    hosted = [t for t in app.tool_registry.names if t.startswith("openai:")]
    print(f"  computer-use tools (approval-gated): {computer}")
    print(f"  hosted tools (server-executed, governed): {hosted}")


def main() -> None:
    controller_demo()
    reflector_demo()
    experiment_proposer_demo()
    deep_research_demo()
    memory_os_demo()
    computer_use_demo()


if __name__ == "__main__":
    main()
