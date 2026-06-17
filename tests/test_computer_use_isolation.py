"""1.10 — computer-use / agentic browsing, pluggable isolation backends, and
provider-native hosted tools."""

import warnings

import pytest

from vincio import ContextApp, VincioConfig
from vincio.core.errors import SandboxError
from vincio.providers import MockProvider
from vincio.providers.hosted_tools import HOSTED_TOOLS, hosted_tool_specs, is_hosted
from vincio.providers.openai_responses import OpenAIResponsesProvider
from vincio.tools.computer_use import ComputerAction, MockComputerUse, computer_use_tools
from vincio.tools.sandbox import (
    ContainerIsolation,
    GVisorIsolation,
    SubprocessIsolation,
    WASMIsolation,
    get_isolation_backend,
    require_real_isolation,
)

warnings.simplefilter("ignore")


def _app(tmp_path):
    config = VincioConfig()
    config.storage.metadata = f"sqlite:///{tmp_path}/v.db"
    config.observability.exporter = "memory"
    config.security.audit_dir = str(tmp_path / "audit")
    return ContextApp(name="cu", provider=MockProvider(), model="mock-1", config=config)


# --------------------------------------------------------------------------- #
# pluggable isolation backends
# --------------------------------------------------------------------------- #


class TestIsolationBackends:
    def test_subprocess_is_not_a_security_boundary(self):
        assert SubprocessIsolation().real is False

    def test_real_backends_flagged(self):
        for backend in (ContainerIsolation(), GVisorIsolation(), WASMIsolation()):
            assert backend.real is True

    def test_require_real_isolation_blocks_subprocess(self):
        with pytest.raises(SandboxError):
            require_real_isolation(SubprocessIsolation())
        require_real_isolation(ContainerIsolation())  # no raise

    def test_registry_resolves_by_name(self):
        assert isinstance(get_isolation_backend("container"), ContainerIsolation)
        assert isinstance(get_isolation_backend("wasm"), WASMIsolation)
        with pytest.raises(SandboxError):
            get_isolation_backend("nope")

    def test_container_wraps_command(self):
        wrapped = ContainerIsolation()._wrap(["python", "x.py"], cwd="/work")
        assert wrapped[:3] == ["docker", "run", "--rm"]
        assert "--network=none" in wrapped and wrapped[-2:] == ["python", "x.py"]

    def test_gvisor_selects_runsc_runtime(self):
        wrapped = GVisorIsolation()._wrap(["python", "x.py"], cwd=None)
        assert "--runtime=runsc" in wrapped

    async def test_unavailable_backend_raises(self):
        backend = WASMIsolation(runtime="definitely-not-installed-xyz")
        assert backend.available() is False
        with pytest.raises(SandboxError):
            await backend.run(["echo", "hi"])

    async def test_subprocess_sandbox_runs_code(self):
        from vincio.tools.sandbox import SandboxedPython

        sandbox = SandboxedPython(timeout_s=10)
        result = await sandbox.run("print('hello from sandbox')")
        assert "hello from sandbox" in result.stdout

    def test_sandbox_require_isolation_refuses_subprocess(self):
        from vincio.tools.sandbox import SandboxedPython

        with pytest.raises(SandboxError):
            SandboxedPython(require_isolation=True)  # default backend is subprocess


# --------------------------------------------------------------------------- #
# computer-use action surface
# --------------------------------------------------------------------------- #


class TestComputerUse:
    async def test_mock_backend_actions(self):
        backend = MockComputerUse()
        nav = await backend.act(ComputerAction(action="navigate", url="https://example.com"))
        assert nav.ok and nav.url == "https://example.com"
        click = await backend.act(ComputerAction(action="click", selector="#submit"))
        assert "#submit" in click.text
        shot = await backend.act(ComputerAction(action="screenshot"))
        assert shot.screenshot_ref and shot.screenshot_ref.startswith("mock://")

    def test_computer_use_tools_are_callable(self):
        navigate, click, type_, screenshot = computer_use_tools(MockComputerUse())
        out = navigate("https://example.com")
        assert out["url"] == "https://example.com"
        assert screenshot()["screenshot_ref"].startswith("mock://")

    def test_enable_computer_use_registers_permissioned_tools(self, tmp_path):
        app = _app(tmp_path)
        app.enable_computer_use("mock")
        for tool in ("computer_navigate", "computer_click", "computer_type", "computer_screenshot"):
            assert tool in app.tool_registry
        spec = app.tool_registry.get("computer_navigate").spec
        assert spec.side_effects == "external"
        assert "computer:use" in spec.permissions
        assert spec.approval_required is True

    def test_require_isolation_blocks_subprocess(self, tmp_path):
        app = _app(tmp_path)
        with pytest.raises(SandboxError):
            app.enable_computer_use("mock", require_isolation=True)  # subprocess default

    def test_enable_computer_use_audited(self, tmp_path):
        app = _app(tmp_path)
        app.enable_computer_use("mock")
        assert app.audit.verify_chain()


# --------------------------------------------------------------------------- #
# provider-native hosted tools
# --------------------------------------------------------------------------- #


class TestHostedTools:
    def test_hosted_specs_namespaced_and_marked(self):
        specs = hosted_tool_specs(["web_search", "code_interpreter"])
        names = {s.name for s in specs}
        assert names == {"openai:web_search", "openai:code_interpreter"}
        assert all(is_hosted(s) for s in specs)

    def test_computer_use_hosted_requires_approval(self):
        spec = HOSTED_TOOLS["computer_use"]
        assert spec.approval_required is True
        assert spec.permissions == ["computer:use"]

    def test_unknown_hosted_tool_raises(self):
        with pytest.raises(KeyError):
            hosted_tool_specs(["nonexistent_tool"])

    def test_responses_provider_emits_builtin_descriptor(self):
        from vincio.core.types import ToolSpec

        provider = OpenAIResponsesProvider(api_key="x")
        hosted = hosted_tool_specs(["web_search"])[0]
        normal = ToolSpec(name="lookup", description="d", input_schema={"type": "object"})
        rendered = provider._render_tools([hosted, normal])
        assert rendered[0] == {"type": "web_search"}
        assert rendered[1]["type"] == "function" and rendered[1]["name"] == "lookup"

    def test_app_use_hosted_tools_registers_and_audits(self, tmp_path):
        app = _app(tmp_path)
        app.use_hosted_tools(["web_search", "file_search"])
        assert "openai:web_search" in app.tool_registry
        assert "openai:web_search" in app.enabled_tools
        assert app.audit.verify_chain()

    def test_hosted_tool_executes_server_side_not_locally(self, tmp_path):
        from vincio.core.errors import ToolNotFoundError

        app = _app(tmp_path)
        app.use_hosted_tools(["web_search"])
        handler = app.tool_registry.get("openai:web_search").handler
        with pytest.raises(ToolNotFoundError):
            handler(query="x")  # hosted tools have no local handler
