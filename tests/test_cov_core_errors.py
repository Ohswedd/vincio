"""Real-behavior coverage for the Vincio error hierarchy (vincio.core.errors).

These tests construct the real exception classes, assert on their stable codes,
structured fields, message formatting, catalog-resolved remediation/docs_url,
``to_dict`` serialization, and the catch-the-whole-family invariant. No mocks.
"""

from __future__ import annotations

import pytest

from vincio.core import errors as E
from vincio.core.error_catalog import (
    ERROR_CATALOG,
    docs_url_for,
    remediation_for,
    set_default_error_locale,
)

# --------------------------------------------------------------------------- #
# Base VincioError: construction, defaults, properties, to_dict               #
# --------------------------------------------------------------------------- #


def test_base_error_carries_message_and_default_code():
    err = E.VincioError("boom")
    assert err.code == "VINCIO_ERROR"
    assert err.message == "boom"
    # Exception base carries the message string for str()/args.
    assert str(err) == "boom"
    assert err.args == ("boom",)


def test_base_error_details_defaults_to_empty_dict_not_none():
    # ``details or {}`` must turn a None into a fresh empty dict.
    err = E.VincioError("x")
    assert err.details == {}
    # Each instance gets its own dict (no shared mutable default).
    err2 = E.VincioError("y")
    err.details["a"] = 1
    assert err2.details == {}


def test_base_error_details_passthrough():
    err = E.VincioError("x", details={"job_id": "j-42", "n": 3})
    assert err.details == {"job_id": "j-42", "n": 3}


def test_remediation_falls_back_to_catalog_when_no_hint():
    err = E.VincioError("x")
    # No instance hint -> catalog remediation for VINCIO_ERROR.
    assert err.remediation == remediation_for("VINCIO_ERROR")
    assert "VincioError" in err.remediation


def test_remediation_instance_override_wins_over_catalog():
    err = E.VincioError("x", hint="do the thing")
    assert err.remediation == "do the thing"
    # Empty-string hint is still an explicit override (not None), so it wins.
    err_empty = E.VincioError("x", hint="")
    assert err_empty.remediation == ""


def test_docs_url_falls_back_to_catalog_anchor():
    err = E.VincioError("x")
    assert err.docs_url == docs_url_for("VINCIO_ERROR")
    assert err.docs_url.endswith("#vincio_error")


def test_docs_url_instance_override_wins():
    err = E.VincioError("x", docs_url="https://example.test/custom")
    assert err.docs_url == "https://example.test/custom"


def test_unknown_code_yields_none_remediation_and_docs_url():
    class WeirdError(E.VincioError):
        code = "NOT_IN_CATALOG_XYZ"

    err = WeirdError("x")
    assert remediation_for("NOT_IN_CATALOG_XYZ") is None
    assert err.remediation is None
    assert err.docs_url is None


def test_to_dict_full_shape():
    err = E.VincioError("kaboom", details={"k": "v"})
    d = err.to_dict()
    assert d == {
        "code": "VINCIO_ERROR",
        "message": "kaboom",
        "details": {"k": "v"},
        "remediation": remediation_for("VINCIO_ERROR"),
        "docs_url": docs_url_for("VINCIO_ERROR"),
    }


def test_to_dict_reflects_instance_overrides():
    err = E.VincioError("m", hint="H", docs_url="U")
    d = err.to_dict()
    assert d["remediation"] == "H"
    assert d["docs_url"] == "U"


# --------------------------------------------------------------------------- #
# ProviderError family: provider/model/retryable fields + retry defaults      #
# --------------------------------------------------------------------------- #


def test_provider_error_fields_default_non_retryable():
    err = E.ProviderError("down", provider="openai", model="gpt-x")
    assert err.code == "PROVIDER_ERROR"
    assert err.provider == "openai"
    assert err.model == "gpt-x"
    assert err.retryable is False


def test_provider_error_details_threaded_through():
    err = E.ProviderError("d", details={"status": 503})
    assert err.details == {"status": 503}


def test_rate_limit_defaults_retryable_true_and_carries_retry_after():
    err = E.ProviderRateLimitError("slow down", retry_after_s=12.5, provider="anthropic")
    assert err.code == "PROVIDER_RATE_LIMIT"
    assert err.retryable is True  # set via kw.setdefault
    assert err.retry_after_s == 12.5
    assert err.provider == "anthropic"


def test_rate_limit_retry_after_optional_and_explicit_retryable_respected():
    err = E.ProviderRateLimitError("x", retryable=False)
    assert err.retry_after_s is None
    # Explicit retryable=False is NOT overridden by setdefault.
    assert err.retryable is False


def test_timeout_and_unavailable_default_retryable_true():
    t = E.ProviderTimeoutError("timed out")
    u = E.ProviderUnavailableError("gone")
    assert t.code == "PROVIDER_TIMEOUT" and t.retryable is True
    assert u.code == "PROVIDER_UNAVAILABLE" and u.retryable is True


def test_circuit_open_is_unavailable_subclass_but_non_retryable():
    err = E.CircuitOpenError("tripped")
    assert err.code == "CIRCUIT_OPEN"
    # Fails fast: non-retryable despite extending the retryable Unavailable.
    assert err.retryable is False
    assert isinstance(err, E.ProviderUnavailableError)
    assert isinstance(err, E.ProviderError)


def test_capability_mismatch_collects_missing_and_is_non_retryable():
    err = E.CapabilityMismatchError("no vision", missing=["vision", "tools"])
    assert err.code == "CAPABILITY_MISMATCH"
    assert err.missing == ["vision", "tools"]
    assert err.retryable is False
    # missing list is copied (not the caller's reference).
    src = ["a"]
    err2 = E.CapabilityMismatchError("m", missing=src)
    src.append("b")
    assert err2.missing == ["a"]


def test_capability_mismatch_missing_defaults_to_empty_list():
    err = E.CapabilityMismatchError("m")
    assert err.missing == []


def test_model_retired_non_retryable():
    err = E.ModelRetiredError("retire it")
    assert err.code == "MODEL_RETIRED"
    assert err.retryable is False


def test_batch_and_finetune_are_provider_errors():
    assert isinstance(E.BatchError("b"), E.ProviderError)
    assert E.BatchError("b").code == "BATCH_ERROR"
    assert isinstance(E.FineTuneError("f"), E.ProviderError)
    assert E.FineTuneError("f").code == "FINETUNE_ERROR"
    assert E.ProviderResponseError("r").code == "PROVIDER_RESPONSE"
    assert E.ProviderAuthError("a").code == "PROVIDER_AUTH"


# --------------------------------------------------------------------------- #
# Prompt / context families with structured payloads                          #
# --------------------------------------------------------------------------- #


def test_prompt_lint_carries_findings():
    err = E.PromptLintError("lint failed", findings=["rule-1", "rule-2"])
    assert err.code == "PROMPT_LINT"
    assert err.findings == ["rule-1", "rule-2"]
    assert isinstance(err, E.PromptError)


def test_prompt_lint_findings_default_empty():
    assert E.PromptLintError("x").findings == []


def test_budget_exceeded_used_and_limit():
    err = E.BudgetExceededError("over", used=1200, limit=1000)
    assert err.code == "BUDGET_EXCEEDED"
    assert err.used == 1200
    assert err.limit == 1000
    assert isinstance(err, E.ContextError)


def test_budget_exceeded_defaults_zero():
    err = E.BudgetExceededError("x")
    assert err.used == 0 and err.limit == 0


# --------------------------------------------------------------------------- #
# Tool family: .tool attribute and subclass codes                             #
# --------------------------------------------------------------------------- #


def test_tool_error_carries_tool_name():
    err = E.ToolError("boom", tool="search")
    assert err.code == "TOOL_ERROR"
    assert err.tool == "search"


def test_tool_error_tool_defaults_none():
    assert E.ToolError("x").tool is None


@pytest.mark.parametrize(
    "cls, code",
    [
        (E.ToolNotFoundError, "TOOL_NOT_FOUND"),
        (E.ToolPermissionError, "TOOL_PERMISSION"),
        (E.ToolValidationError, "TOOL_VALIDATION"),
        (E.ToolTimeoutError, "TOOL_TIMEOUT"),
        (E.ToolApprovalRequiredError, "TOOL_APPROVAL_REQUIRED"),
        (E.SandboxError, "SANDBOX_ERROR"),
        (E.ComputerUseError, "COMPUTER_USE_ERROR"),
        (E.ToolContractError, "TOOL_CONTRACT_VIOLATION"),
    ],
)
def test_tool_subclasses_codes_and_carry_tool(cls, code):
    err = cls("msg", tool="t")
    assert err.code == code
    assert err.tool == "t"
    assert isinstance(err, E.ToolError)


# --------------------------------------------------------------------------- #
# Agent / graph / workflow structured fields                                  #
# --------------------------------------------------------------------------- #


def test_agent_step_error_step_id():
    err = E.AgentStepError("step blew up", step_id="s3")
    assert err.code == "AGENT_STEP"
    assert err.step_id == "s3"
    assert isinstance(err, E.AgentEngineError)


def test_checkpoint_conflict_version_fields():
    err = E.CheckpointConflictError(
        "lost race", thread_id="th-1", expected_version=4, actual_version=5
    )
    assert err.code == "CHECKPOINT_CONFLICT"
    assert err.thread_id == "th-1"
    assert err.expected_version == 4
    assert err.actual_version == 5
    # Lifecycle: it is a GraphError, which is an AgentEngineError.
    assert isinstance(err, E.GraphError)
    assert isinstance(err, E.AgentEngineError)


def test_checkpoint_conflict_defaults_none():
    err = E.CheckpointConflictError("x")
    assert err.thread_id is None
    assert err.expected_version is None
    assert err.actual_version is None


def test_workflow_step_error_step():
    err = E.WorkflowStepError("step", step="ship")
    assert err.code == "WORKFLOW_STEP"
    assert err.step == "ship"
    assert isinstance(err, E.WorkflowError)


# --------------------------------------------------------------------------- #
# Output / generation / eval structured payloads                              #
# --------------------------------------------------------------------------- #


def test_output_schema_error_errors_list():
    err = E.OutputSchemaError("bad", errors=["field a missing"])
    assert err.code == "OUTPUT_SCHEMA"
    assert err.errors == ["field a missing"]
    assert E.OutputSchemaError("x").errors == []
    assert isinstance(err, E.OutputError)


def test_document_contract_error_violations():
    err = E.DocumentContractError("nope", violations=["no h1"])
    assert err.code == "DOCUMENT_CONTRACT"
    assert err.violations == ["no h1"]
    assert E.DocumentContractError("x").violations == []
    assert isinstance(err, E.GenerationError)


def test_gate_failed_error_failures():
    err = E.GateFailedError("gate", failures=["accuracy < 0.9"])
    assert err.code == "GATE_FAILED"
    assert err.failures == ["accuracy < 0.9"]
    assert E.GateFailedError("x").failures == []
    assert isinstance(err, E.EvalError)


# --------------------------------------------------------------------------- #
# Security / governance structured payloads                                   #
# --------------------------------------------------------------------------- #


def test_residency_violation_region_and_allowed():
    err = E.ResidencyViolationError("eu only", region="us-east", allowed=["eu-west", "eu-central"])
    assert err.code == "RESIDENCY_VIOLATION"
    assert err.region == "us-east"
    assert err.allowed == ["eu-west", "eu-central"]
    assert E.ResidencyViolationError("x").allowed == []
    assert isinstance(err, E.GovernanceError)


def test_governance_verification_error_counterexamples():
    err = E.GovernanceVerificationError("invariant broke", counterexamples=[{"state": "leak"}])
    assert err.code == "GOVERNANCE_INVARIANT_VIOLATED"
    assert err.counterexamples == [{"state": "leak"}]
    assert E.GovernanceVerificationError("x").counterexamples == []
    assert isinstance(err, E.GovernanceError)


def test_identity_error_is_security_error():
    err = E.IdentityError("bad DID")
    assert err.code == "IDENTITY_VERIFICATION_FAILED"
    assert isinstance(err, E.SecurityError)


def test_containment_and_egress_are_security_errors():
    assert isinstance(E.ContainmentError("x"), E.SecurityError)
    assert E.ContainmentError("x").code == "CONTAINMENT_BLOCKED"
    assert isinstance(E.EgressBlockedError("x"), E.SecurityError)
    assert E.EgressBlockedError("x").code == "EGRESS_BLOCKED"


# --------------------------------------------------------------------------- #
# Negotiation / choreography / contract structured payloads                   #
# --------------------------------------------------------------------------- #


def test_contract_error_breaches_and_is_negotiation_error():
    err = E.ContractError("breached", breaches=["price", "sla"])
    assert err.code == "CONTRACT_VIOLATION"
    assert err.breaches == ["price", "sla"]
    assert E.ContractError("x").breaches == []
    assert isinstance(err, E.NegotiationError)


def test_compensation_error_failures_and_is_choreography_error():
    err = E.CompensationError("did not unwind", failures=["step-2"])
    assert err.code == "COMPENSATION_FAILED"
    assert err.failures == ["step-2"]
    assert E.CompensationError("x").failures == []
    assert isinstance(err, E.ChoreographyError)


# --------------------------------------------------------------------------- #
# pytest.raises with message match through real raise sites                    #
# --------------------------------------------------------------------------- #


def test_raises_with_message_match():
    with pytest.raises(E.ProviderRateLimitError, match="quota exhausted"):
        raise E.ProviderRateLimitError("daily quota exhausted", retry_after_s=60)


def test_subclass_caught_as_base_vincio_error():
    with pytest.raises(E.VincioError) as excinfo:
        raise E.ToolNotFoundError("no such tool", tool="ghost")
    assert excinfo.value.code == "TOOL_NOT_FOUND"
    assert excinfo.value.tool == "ghost"


def test_provider_subclass_caught_as_provider_error():
    with pytest.raises(E.ProviderError) as excinfo:
        raise E.CircuitOpenError("open")
    assert excinfo.value.retryable is False


# --------------------------------------------------------------------------- #
# Hierarchy + catalog invariants across the whole exported family             #
# --------------------------------------------------------------------------- #


def test_every_exported_error_subclasses_vincio_error_and_has_string_code():
    for name in E.__all__:
        cls = getattr(E, name)
        assert issubclass(cls, E.VincioError), name
        assert isinstance(cls.code, str), name


def test_every_exported_error_code_resolves_in_catalog():
    # Each error's stable code must have an actionable catalog entry, so the
    # .remediation / .docs_url properties never silently return None on a
    # shipped error.
    for name in E.__all__:
        cls = getattr(E, name)
        assert cls.code in ERROR_CATALOG, f"{name} code {cls.code!r} not in catalog"
        assert docs_url_for(cls.code) is not None, name


def test_repr_includes_class_name_and_message():
    err = E.ProviderTimeoutError("slow")
    r = repr(err)
    assert "ProviderTimeoutError" in r
    assert "slow" in r


# --------------------------------------------------------------------------- #
# Locale-aware remediation flows through the property                          #
# --------------------------------------------------------------------------- #


def test_remediation_unknown_default_locale_falls_back_to_english(monkeypatch):
    # Switch the process default locale to one with no translations; the
    # property must still resolve via the English fallback, not return None.
    set_default_error_locale("zz")
    try:
        err = E.ConfigError("bad config")
        assert err.remediation == remediation_for("CONFIG_ERROR")
        assert err.remediation is not None
    finally:
        set_default_error_locale("en")
