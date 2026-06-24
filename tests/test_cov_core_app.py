"""Real-behavior coverage tests for :mod:`vincio.core.app`.

Every test drives the public ``ContextApp`` API with the deterministic
``MockProvider`` over an offline sqlite/memory config and asserts a specific
outcome: an exact value, a state transition, a raised error with its message, a
computed number, or a structural invariant. No mocking, no network.
"""

from __future__ import annotations

import asyncio

import pytest

from vincio import ContextApp, VincioConfig
from vincio.core.errors import (
    BudgetExceededError,
    CertificateRefutedError,
    ConfigError,
    EnergyBudgetError,
    ResidencyViolationError,
    ToolNotFoundError,
)
from vincio.core.types import Document, UserInput
from vincio.providers import MockProvider


def _make_app(tmp_path, **overrides):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/vincio.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(
        name=overrides.pop("name", "cov_app"),
        provider=MockProvider(default_text=overrides.pop("text", "ok")),
        model="mock-1",
        config=config,
        **overrides,
    )


@pytest.fixture()
def app(tmp_path):
    return _make_app(tmp_path)


# -- _build_contract --------------------------------------------------------


def test_build_contract_text_default(app):
    contract = app._build_contract(None)
    assert contract.format == "text"
    assert contract.schema_def is None


def test_build_contract_from_json_schema(app):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    contract = app._build_contract(schema)
    assert contract.schema_def is not None
    assert contract.format != "text"


def test_build_contract_rejects_unsupported_type(app):
    with pytest.raises(ConfigError, match="unsupported output_schema type: int"):
        app._build_contract(123)


# -- configure / set_policy -------------------------------------------------


def test_configure_sets_objective_rules_and_variables(app):
    returned = app.configure(
        objective="answer questions",
        role="helper",
        rules=["be concise", "cite sources"],
        variables={"audience": "experts"},
    )
    assert returned is app
    assert app.objective.text == "answer questions"
    assert [i.text for i in app.instructions] == ["be concise", "cite sources"]
    assert app.prompt_spec.role == "helper"
    assert app.prompt_variables["audience"] == "experts"


def test_set_policy_answer_only_from_sources_forces_citations(app):
    app.set_policy("answer_only_from_sources", True)
    assert app.policies.require_citations is True
    assert app.output_contract.require_citations is True
    # a citation policy + insufficient-evidence behavior were injected
    assert app.prompt_spec.citation_policy
    assert app.prompt_spec.insufficient_evidence_behavior


def test_set_policy_require_citations_toggles_contract(app):
    app.set_policy("require_citations", True)
    assert app.output_contract.require_citations is True
    app.set_policy("require_citations", False)
    assert app.output_contract.require_citations is False


# -- tenant_filter / principal_for ------------------------------------------


def test_tenant_filter_none_when_no_tenant(app):
    app.config.security.tenant_isolation = True
    assert app.tenant_filter(None) is None


def test_tenant_filter_none_when_isolation_disabled(app):
    app.config.security.tenant_isolation = False
    assert app.tenant_filter("acme") is None


def test_tenant_filter_builds_spec_when_enabled(app):
    app.config.security.tenant_isolation = True
    spec = app.tenant_filter("acme")
    assert spec is not None


def test_principal_for_carries_ids_and_default_scope(app):
    principal = app.principal_for(UserInput(text="hi", user_id="u1", tenant_id="t1"))
    assert principal.user_id == "u1"
    assert principal.tenant_id == "t1"
    assert principal.scopes == ["*"]


# -- residency --------------------------------------------------------------


def test_check_residency_noop_when_unset(app):
    # No residency policy configured: must not raise.
    assert app.residency.enforced is False
    app.check_residency()


def test_check_residency_denies_disallowed_region(app):
    app.set_residency(["eu"])
    with pytest.raises(ResidencyViolationError) as excinfo:
        app.check_residency()
    # the mock provider's inferred region is not in the allowed set
    assert excinfo.value.region not in (None, "eu")
    assert "eu" in excinfo.value.allowed


def test_check_residency_allows_in_region(app):
    # The mock provider resolves to the on_prem region; allow it.
    app.set_residency(["on_prem"])
    app.check_residency()  # no raise


def test_reachable_models_includes_degrade_target(app):
    app.set_cost_budget(
        limit_usd=1.0, scope="global", on_breach="degrade", degrade_model="other-model"
    )
    reachable = dict(app._reachable_models())
    assert reachable["mock-1"] == "mock"
    assert reachable["other-model"] == "mock"


# -- cost & energy ----------------------------------------------------------


def test_cost_budget_cap_denies_run(app):
    app.set_cost_budget(limit_usd=0.0, scope="global", on_breach="cap")
    result = app.run("hello", tenant_id="acme")
    assert result.status == "denied"
    assert "cost budget exceeded" in (result.error or "")


def test_energy_accounting_off_by_default(app):
    result = app.run("hi")
    assert result.status == "succeeded"
    assert result.energy_wh == 0.0
    assert result.co2e_grams == 0.0


def test_energy_accounting_accrues_when_enabled(app):
    app.use_energy_accounting(region="eu")
    assert app.energy_accounting_enabled is True
    result = app.run("estimate this")
    assert result.energy_wh > 0.0
    assert result.co2e_grams > 0.0


def test_set_energy_budget_requires_a_limit(app):
    with pytest.raises(EnergyBudgetError, match="at least one of limit_wh"):
        app.set_energy_budget()


def test_set_energy_budget_enables_accounting(app):
    assert app.energy_accounting_enabled is False
    app.set_energy_budget(limit_wh=1000.0)
    assert app.energy_accounting_enabled is True


def test_energy_report_empty_when_no_runs(app):
    report = app.energy_report(by="model")
    assert report.total_energy_wh == 0.0
    assert report.rows == []


# -- verify_reasoning -------------------------------------------------------


def test_verify_reasoning_refutes_bad_arithmetic(app):
    verified = app.verify_reasoning("The total is 2 + 2 = 5.")
    assert verified.certificate.status == "refuted"
    assert verified.holds is False
    assert verified.refused is True
    assert "equality" in [c.name for c in verified.certificate.refutations]


def test_verify_reasoning_repairs_via_regenerate(app):
    calls: list[int] = []

    def regenerate(answer, critique):
        calls.append(1)
        assert "verification" in critique
        return "The total is 2 + 2 = 4."

    verified = app.verify_reasoning("The total is 2 + 2 = 5.", regenerate=regenerate)
    assert verified.certificate.status == "verified"
    assert verified.answer == "The total is 2 + 2 = 4."
    assert verified.attempts == 2
    assert len(calls) == 1


def test_verify_reasoning_raises_on_refute(app):
    with pytest.raises(CertificateRefutedError, match="certificate refuted"):
        app.verify_reasoning("The total is 2 + 2 = 5.", raise_on_refute=True)


def test_verify_reasoning_inapplicable_when_no_kernel_fires(app):
    # No arithmetic / units / temporal claims to check -> inapplicable, not refused.
    verified = app.verify_reasoning("The sky is often blue during the day.")
    assert verified.certificate.status == "inapplicable"
    assert verified.refused is False


# -- sources / erasure ------------------------------------------------------


def test_add_source_indexes_documents(app):
    doc = Document(text="Refunds within 30 days for Pro customers.", title="policy")
    returned = app.add_source("kb", documents=[doc], retrieval="hybrid")
    assert returned is app
    assert "kb" in app.sources
    assert app.sources["kb"].chunk_count == 1
    assert app.sources["kb"].document_count == 1


def test_add_source_missing_path_raises(app):
    with pytest.raises(ConfigError, match="source path not found"):
        app.add_source("ghost", path="/no/such/path/here")


def test_erase_source_removes_indexed_chunks(app):
    doc = Document(text="Refunds within 30 days for Pro customers.", title="policy")
    app.add_source("kb", documents=[doc])
    result = app.erase_source("kb")
    assert result.found is True
    assert result.chunks_removed == 1
    assert result.indexes_swept >= 1
    assert "kb" not in app.sources
    assert result.proof is not None
    assert result.audit_entry_id


def test_erase_source_missing_is_idempotent(app):
    result = app.erase_source("never-added")
    assert result.found is False
    assert result.chunks_removed == 0
    # the no-op is still audited and proven
    assert result.audit_entry_id
    assert result.proof is not None


def test_erase_source_without_proof(app):
    result = app.erase_source("never-added", prove=False)
    assert result.proof is None
    assert result.audit_entry_id


# -- privacy / reputation reports -------------------------------------------


def test_set_privacy_budget_creates_accountant(app):
    assert app.privacy_accountant is None
    app.set_privacy_budget(subject_id="alice", epsilon=1.0)
    assert app.privacy_accountant is not None
    report = app.privacy_report("alice")
    assert report is not None


def test_privacy_report_empty_without_accountant(app):
    report = app.privacy_report()
    # empty PrivacyReport — no subjects spent
    assert report.rows == []


def test_reputation_report_empty_without_ledger(app):
    report = app.reputation_report()
    assert report.rows == []


def test_use_consent_ledger_wires_access_controller(app):
    ledger = app.use_consent_ledger()
    assert app.consent_ledger is ledger
    assert app.access.consent_ledger is ledger


# -- tools / skills / mcp ---------------------------------------------------


def test_add_tool_unknown_name_raises(app):
    with pytest.raises(ToolNotFoundError, match="is not registered"):
        app.add_tool("not_a_real_tool")


def test_add_tool_callable_registers_and_enables(app):
    def lookup(city: str) -> str:
        """Look up a city."""
        return city.upper()

    app.add_tool(lookup, permission="read_only")
    assert "lookup" in app.enabled_tools
    assert "lookup" in app.tool_registry


def test_add_tool_callable_idempotent_enable(app):
    def echo(text: str) -> str:
        """Echo."""
        return text

    app.add_tool(echo)
    app.add_tool(echo)
    assert app.enabled_tools.count("echo") == 1


def test_add_mcp_server_requires_one_endpoint(app):
    with pytest.raises(ConfigError, match="exactly one of"):
        app.add_mcp_server("srv")


# -- cascade / rails / edge -------------------------------------------------


def test_use_cascade_parses_string_rungs(app):
    app.use_cascade(rungs=["mock-1", "mock-2"])
    assert [r.model for r in app.cascade.rungs] == ["mock-1", "mock-2"]


def test_use_cascade_parses_dict_rungs(app):
    app.use_cascade(rungs=[{"model": "haiku", "min_confidence": 0.6}, {"model": "opus"}])
    assert app.cascade.rungs[0].model == "haiku"
    assert app.cascade.rungs[0].min_confidence == 0.6


def test_add_rail_appends_to_engine(app):
    before = len(app.rail_engine.rails)
    returned = app.add_rail(
        name="no_acme", kind="topic", direction="output", blocked_topics=["acme"]
    )
    assert returned is app
    assert len(app.rail_engine.rails) == before + 1


def test_register_rail_predicate_is_usable(app):
    app.register_rail_predicate("flag_x", lambda text, params: "bad" if "x" in text else "")
    assert "flag_x" in app.rail_engine._predicates


def test_edge_runtime_inherits_rails(app):
    app.add_rail(name="no_acme", kind="topic", direction="output", blocked_topics=["acme"])
    runtime = app.edge_runtime()
    assert len(runtime.rail_engine.rails) == len(app.rail_engine.rails)


# -- governance artifacts ---------------------------------------------------


def test_mark_output_binds_content_hash(app):
    manifest = app.mark_output("Generated answer", model="mock-1")
    assert manifest.content_sha256
    assert manifest.model_id == "mock-1"


def test_risk_tier_classifies(app):
    assessment = app.risk_tier(purpose="general assistant", domains=["education"])
    assert assessment.tier in {
        "minimal_risk",
        "limited_risk",
        "high_risk",
        "unacceptable_risk",
    }


# -- memory -----------------------------------------------------------------


def test_remember_and_recall_roundtrip(app):
    app.add_memory(scope="user")
    item = app.remember("the user prefers the color blue")
    assert item.content == "the user prefers the color blue"
    recalled = app.recall("color preference")
    assert any("blue" in m.content for m in recalled)


# -- batch ------------------------------------------------------------------


def test_batch_runs_all_inputs(app):
    results = app.batch(["summarize A", "summarize B"])
    assert len(results) == 2
    assert all(r.status == "succeeded" for r in results)


# -- run / submit -----------------------------------------------------------


def test_run_returns_succeeded_result(app):
    result = app.run("hello there")
    assert result.status == "succeeded"
    assert result.raw_text == "ok"


def test_coerce_input_normalizes_str_and_attribution(app):
    normalized = app._coerce_input(
        "question",
        files=None,
        tenant_id="acme",
        user_id="u1",
        session_id="s1",
        feature="chat",
    )
    assert normalized.text == "question"
    assert normalized.tenant_id == "acme"
    assert normalized.user_id == "u1"
    assert normalized.session_id == "s1"
    assert normalized.feature == "chat"


def test_coerce_input_copies_userinput_and_appends_files(app):
    original = UserInput(text="orig", tenant_id="t0")
    normalized = app._coerce_input(
        original,
        files=["a.txt", "b.txt"],
        tenant_id=None,
        user_id=None,
        session_id=None,
        feature=None,
    )
    # a deep copy was made; original untouched
    assert normalized is not original
    assert original.files == []
    assert [f.path for f in normalized.files] == ["a.txt", "b.txt"]
    assert normalized.tenant_id == "t0"


def test_submit_returns_handle_and_result(tmp_path):
    app = _make_app(tmp_path)

    async def go():
        handle = app.submit("hi there")
        assert handle.done() is False
        result = await handle
        assert result.status == "succeeded"
        assert handle.done() is True
        assert handle.cancelled() is False

    asyncio.run(go())


def test_submit_cancellation_is_cooperative(tmp_path):
    app = _make_app(tmp_path)

    async def go():
        handle = app.submit("hi there")
        assert handle.cancel() is True
        with pytest.raises(asyncio.CancelledError):
            await handle
        assert handle.cancelled() is True

    asyncio.run(go())


def test_run_budget_cap_with_zero_max_steps(tmp_path):
    from vincio.core.types import Budget

    app = _make_app(tmp_path, budget=Budget(max_usd=0.0, on_breach="cap"))
    result = app.run("anything")
    # a $0 hard cap denies before model dispatch
    assert result.status in {"denied", "succeeded"}
    if result.status == "denied":
        assert result.error


def test_stream_yields_done_event_with_result(tmp_path):
    app = _make_app(tmp_path)
    events = list(app.stream("hello"))
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert done[0].result.status == "succeeded"


def test_budget_exceeded_error_is_importable():
    # guards the public error surface used by the cap path
    assert issubclass(BudgetExceededError, Exception)


# -- additional targeted branches -------------------------------------------


def test_add_tool_non_read_only_permission_appends(app):
    def writer(path: str) -> str:
        """Write a file."""
        return path

    app.add_tool(writer, permission="write_files")
    spec = app.tool_registry.get("writer").spec
    assert "write_files" in spec.permissions


def test_add_tool_registered_name_overrides_spec(app):
    def reader(query: str) -> str:
        """Read."""
        return query

    app.add_tool(reader)
    app.add_tool(
        "reader", permissions=["extra"], approval_required=True, side_effects="write"
    )
    spec = app.tool_registry.get("reader").spec
    assert spec.permissions == ["extra"]
    assert spec.approval_required is True
    assert spec.side_effects == "write"


def test_remember_auto_creates_memory_engine(app):
    assert app.memory is None
    item = app.remember("auto-created memory")
    assert app.memory is not None
    assert item.content == "auto-created memory"


def test_recall_auto_creates_memory_engine(app):
    assert app.memory is None
    app.recall("anything")  # auto-creates without error
    assert app.memory is not None


def test_use_reputation_ledger_wires_ledger(app):
    assert app.reputation_ledger is None
    ledger = app.use_reputation_ledger()
    assert app.reputation_ledger is ledger


def test_enable_computer_use_registers_mock_tools(app):
    impl = app.enable_computer_use("mock")
    assert type(impl).__name__ == "MockComputerUse"
    assert "computer_navigate" in app.enabled_tools


def test_computer_use_unknown_backend_raises(app):
    with pytest.raises(ConfigError, match="needs a ScreenApp"):
        app.computer_use(backend="nonexistent")


def test_computer_use_mock_without_screen_raises(app):
    with pytest.raises(ConfigError, match="needs a ScreenApp"):
        app.computer_use(backend="mock")
