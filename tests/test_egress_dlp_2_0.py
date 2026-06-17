"""2.0 mandatory egress DLP: a deterministic last-mile scan of the assembled
provider request, enforced at the provider boundary regardless of call-site."""

from __future__ import annotations

from vincio import ContextApp
from vincio.core.config import SecurityConfig, VincioConfig
from vincio.core.errors import EgressBlockedError
from vincio.core.types import Message, ModelRequest, ToolSpec
from vincio.security.policy import PolicyEngine

# A synthetic OpenAI-style credential assembled at runtime, so no contiguous
# secret-shaped token ever lives in source (committing one trips GitHub secret
# scanning as a "public leak"). Reassembled it still matches Vincio's
# `sk-[A-Za-z0-9_-]{16,}` api-key detector — which is exactly what the egress
# DLP scan must catch.
_SECRET = "sk-" + "live-" + "A" * 40


def _request_with_secret() -> ModelRequest:
    return ModelRequest(
        model="mock",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content=f"Use this credential: {_SECRET}"),
        ],
    )


def test_scan_egress_off_is_noop():
    engine = PolicyEngine(egress_dlp="off")
    result = engine.scan_egress(_request_with_secret())
    assert result.allowed is True
    assert result.violations == []


def test_scan_egress_warn_records_but_allows():
    engine = PolicyEngine(egress_dlp="warn")
    result = engine.scan_egress(_request_with_secret())
    assert result.allowed is True
    assert any(v.policy == "egress_secret" for v in result.violations)
    assert all(v.severity == "warn" for v in result.violations)


def test_scan_egress_block_blocks_credentials():
    engine = PolicyEngine(egress_dlp="block")
    result = engine.scan_egress(_request_with_secret())
    assert result.allowed is False
    assert result.blocking


def test_scan_egress_scans_tool_schemas():
    engine = PolicyEngine(egress_dlp="block")
    request = ModelRequest(
        model="mock",
        messages=[Message(role="user", content="hello")],
        tools=[
            ToolSpec(
                name="deploy",
                description="Deploy",
                input_schema={"default_token": _SECRET},
            )
        ],
    )
    result = engine.scan_egress(request)
    assert result.allowed is False


def test_scan_egress_allows_clean_request():
    engine = PolicyEngine(egress_dlp="block")
    request = ModelRequest(
        model="mock",
        messages=[Message(role="user", content="What is the capital of France?")],
    )
    result = engine.scan_egress(request)
    assert result.allowed is True
    assert result.violations == []


async def test_runtime_blocks_outbound_secret_end_to_end():
    cfg = VincioConfig(security=SecurityConfig(egress_dlp="block", audit_log=False))
    app = ContextApp(name="dlp", config=cfg)
    # The runtime catches VincioError into a FAILED RunResult (it never crashes
    # the caller); EgressBlockedError is a VincioError, so the run fails cleanly.
    result = await app.arun(f"Please remember my key {_SECRET} and echo it")
    assert result.status.value == "failed"
    assert "egress" in (result.error or "").lower()
    deny = [e for e in app.audit.entries if e.action == "egress_dlp" and e.decision == "deny"]
    assert deny


def test_egress_blocked_error_is_vincio_error():
    from vincio.core.errors import SecurityError, VincioError

    assert issubclass(EgressBlockedError, SecurityError)
    assert issubclass(EgressBlockedError, VincioError)


async def test_runtime_warn_mode_audits_without_blocking(tmp_path):
    cfg = VincioConfig(
        security=SecurityConfig(egress_dlp="warn", audit_dir=str(tmp_path / "audit"))
    )
    app = ContextApp(name="dlp_warn", config=cfg)
    result = await app.arun(f"Here is a token {_SECRET}")
    assert result.status.value in {"succeeded", "failed"}
    # The egress finding was recorded on the audit chain (warn, not blocked).
    egress_entries = [e for e in app.audit.entries if e.action == "egress_dlp"]
    assert egress_entries
    assert egress_entries[0].decision == "allow"
