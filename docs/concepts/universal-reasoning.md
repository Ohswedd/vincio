# Universal reasoning

Provider-native thinking is an optimization, not a portable reasoning system.
Vincio's universal reasoning engine puts an adaptive orchestration layer above
the normal run pipeline, so a local or hosted model with no reasoning control
still receives the same task assessment, decomposition, evidence gathering,
verification and bounded correction flow as a native reasoning model.

The engine uses a hybrid router. Exact syntax and confidently identified English
take the token-free deterministic path: Vincio's task taxonomy, structural
signals, constraints, supplied modalities, exact enabled-tool matching, and
specialized math, logic, causal, decision, temporal and spatial signals. For a
non-English or uncertain-language request, a compact validated classification
call asks the configured model to interpret the request natively. There is no
language allow-list: routing language coverage follows the selected model's own
language coverage.

The semantic call returns only task/depth/web/tool fields and confidence, never
reasoning text. It is traced, DLP-screened, token/cost-accounted and bounded by
the same run budget. Low-confidence or invalid output falls back to conservative
standard depth. It may recommend search or name an enabled tool, but cannot
bypass deterministic web policy, the registered-tool allow-list, permissions,
budgets or verification. Set `semantic_routing="off"` to force the offline-only
heuristic path or `"always"` to classify every eligible request semantically.

The combined decision selects:

- direct for formatting, rewriting, extraction and other simple work;
- standard for one material calculation, comparison or evidence-sensitive task;
- deep for logical, contradictory or genuinely multi-stage work.

It then selects a high-level strategy: decompose, evidence-first,
calculate-and-verify, logic-check, or tool-plan. These are operational receipts,
not chain-of-thought. Model prompts explicitly require private analysis and an
answer-only response; pass records contain run ids, validation and verifier
status, tokens and cost, never scratch work.

## Internal plan mode

A deep, genuinely multi-step request (comparison under constraints, root-cause
work, planning, coding, coupled decisions) additionally earns one bounded
internal planning call. The configured model returns a validated typed
decomposition — up to `plan_max_steps` `PlannedStep`s, each with an imperative
goal, a kind (`analyze`, `gather`, `compute`, `compare`, `decide`, `draft`,
`verify`), dependency indices, and the deterministic check its output should
survive — plus explicit assumptions and optional evidence queries. The merged
plan structures every candidate pass. Governance is one-directional: evidence
queries are honored only when the deterministic policy already selected search,
step goals are length-clipped, dangling dependencies are dropped, and a
low-confidence or invalid plan falls back to the heuristic decomposition. The
planning call is traced, cost-accounted and visible in the receipt
(`plan_mode_used`, `plan_steps`, `plan_tokens`). Set `plan_mode="off"` to
disable it or `"always"` to plan every eligible request; simple work never pays
for planning on `"auto"`.

## One reasoning and browsing system

Freshness is decided before generation. Explicit search requests, requested
URLs, high-stakes questions and unstable facts (current versions, prices, laws,
office holders, schedules and similar) route through the governed `WebBrowser`
when it is enabled. Words with local meanings—such as “current paragraph”,
“version control”, or “score this essay”—do not trigger browsing. An explicit
“do not browse” is preserved as `search_decision="user_declined"`; the model
must then acknowledge that live verification was unavailable. Search results are
read into small query-relevant excerpts, injection-screened, content-hashed and
compiled as untrusted evidence. Pasted pages are read directly rather than sent
through search, results prefer host diversity, and candidate calls cannot repeat
the engine-owned fetch. The final receipt separates content integrity
(`web_verified`) from answer checking (`answer_verification`). If fresh evidence
is unavailable, an unsupported current claim is refuted and withheld; an honest
statement of uncertainty may pass.
When evidence is unavailable in any language, the language-neutral
`[UNVERIFIED]` contract makes calibrated uncertainty mechanically observable.
Set `UniversalReasoningPolicy(web="required")` for a fail-closed path: if no
governed source can be read, the run raises `WebPolicyError` before generation.

## Validation and correction

Every candidate is a normal `ContextApp` run, so schema, citation, policy and
semantic validators still apply. The independent offline reasoning kernels then
recompute checkable arithmetic, units, dates, constraints and citation support.
Unicode tokenization, punctuation-aware sentence splitting and character
shingles for scripts without whitespace let evidence support work beyond Latin
text; deterministic arithmetic and capability/security checks remain
language-independent where their syntax permits.
A task-bound verifier also recomputes supported request shapes (one unambiguous
numeric expression, percentage-and-even-split problems, and explicit logical
inconsistency with a demonstrated witness)
before generation. Those verified facts become hard plan constraints, preventing
an internally valid calculation for the wrong interpretation from passing.
A fabricated-source check runs on every live-factual candidate: an answer that
attributes a claim to a URL or "according to …" domain present in neither the
attached evidence nor the request is refuted outright — even when the rest of
the answer is hedged — and the flagged sources are recorded in the receipt
(`fabricated_sources`). The check is precision-first: citing an attached
source, a request-mentioned URL, or a subdomain of either never triggers it,
and a bare product or organization name is never treated as a citation.
When every pass dies to a transient provider fault before producing an answer
(an empty upstream payload rather than a refusal), the engine spends its
reserved correction slot on one spaced salvage attempt — recorded as a
`salvage` pass — before giving up; a persistently unavailable upstream still
fails the run honestly.
A refuted answer cannot win candidate selection. When candidates disagree or a
kernel refutes the best one, one bounded correction pass receives only the
answers, verifier verdicts and governed evidence. Total passes, concurrency,
searches, pages and excerpt tokens all have explicit policy ceilings.
If every bounded attempt remains refuted, the engine returns a failed
`RunResult` with no output rather than emit a known-wrong answer. When the local
task verifier already proves the complete answer, it may instead synthesize a
minimal deterministic fallback containing only those proven facts.
It never substitutes that text fallback for a structured Pydantic contract.
If every bounded attempt still prints unrequested intermediate reasoning, the
same rule applies: return only a complete task-proven fallback or withhold it.

## Native reasoning models

When the provider declares native reasoning support, the same assessment also
sets its effort (`minimal`, `medium` or `high`). The surrounding evidence,
candidate and verifier architecture remains provider-independent. On a model
without native reasoning, bounded passes supply the missing test-time compute;
no provider field is required.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Guide: Universal reasoning, native thinking & the Responses API](../guides/reasoning.md)
- [Example: 22_universal_reasoning.py](../../examples/22_universal_reasoning.py)
- [Concept: Prompt compiler](prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
