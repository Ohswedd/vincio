"""Tests for security hardening: offline audit verification and sandbox limits."""

from __future__ import annotations

import json
import sys

import pytest

from vincio.security.audit import AuditLog, verify_audit_file
from vincio.tools.sandbox import SandboxedPython, run_subprocess_sandboxed


class TestAuditFileVerification:
    def test_intact_file_verifies(self, tmp_path):
        log = AuditLog(directory=tmp_path / "audit")
        for i in range(5):
            log.record("tool_call", user_id="u1", details={"n": i})
        result = log.verify_file()
        assert result.intact is True
        assert result.entries == 5
        assert result.broken_at is None

    def test_in_memory_log_verifies_trivially(self):
        log = AuditLog(directory=None)
        log.record("run", user_id="u1")
        assert log.verify_file().intact is True

    def test_missing_file_is_intact(self, tmp_path):
        assert verify_audit_file(tmp_path / "nope.jsonl").intact is True

    def test_tampered_detail_breaks_chain(self, tmp_path):
        log = AuditLog(directory=tmp_path / "audit")
        for i in range(4):
            log.record("tool_call", user_id="u1", details={"n": i})
        path = log.path
        lines = path.read_text().splitlines()
        # Tamper with the body of line 2 without fixing the hash chain.
        record = json.loads(lines[1])
        record["details"]["n"] = 999
        lines[1] = json.dumps(record)
        path.write_text("\n".join(lines) + "\n")

        result = verify_audit_file(path)
        assert result.intact is False
        assert result.broken_at == 2
        assert result.reason == "entry_hash mismatch"

    def test_deleted_row_breaks_chain(self, tmp_path):
        log = AuditLog(directory=tmp_path / "audit")
        for i in range(4):
            log.record("tool_call", details={"n": i})
        path = log.path
        lines = path.read_text().splitlines()
        del lines[1]  # drop a row -> prev_hash linkage breaks at the next row
        path.write_text("\n".join(lines) + "\n")

        result = verify_audit_file(path)
        assert result.intact is False
        assert result.reason == "prev_hash mismatch"


class TestSandboxLimits:
    async def test_basic_run_succeeds(self):
        result = await SandboxedPython().run("print('hello')")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_timeout_enforced(self):
        from vincio.core.errors import ToolTimeoutError

        with pytest.raises(ToolTimeoutError):
            await run_subprocess_sandboxed(
                [sys.executable, "-I", "-c", "import time; time.sleep(5)"],
                timeout_s=0.3,
            )

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="RLIMIT_AS is only reliably enforced on Linux",
    )
    async def test_memory_limit_kills_runaway_allocation(self):
        # Allocate well past a tight RLIMIT_AS; the child must die, not the host.
        code = "x = bytearray(400 * 1024 * 1024); print(len(x))"
        result = await SandboxedPython(
            max_memory_bytes=64 * 1024 * 1024, timeout_s=10
        ).run(code)
        assert result.exit_code != 0  # MemoryError / non-zero exit, host unharmed

    async def test_env_is_scrubbed(self, monkeypatch):
        monkeypatch.setenv("SECRET_TOKEN", "should-not-leak")
        result = await SandboxedPython().run(
            "import os; print('SECRET_TOKEN' in os.environ)"
        )
        assert "False" in result.stdout
