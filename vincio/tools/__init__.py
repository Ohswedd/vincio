"""Vincio tool engine."""

from .computer_use import (
    ComputerAction,
    ComputerObservation,
    ComputerUseBackend,
    MockComputerUse,
    PlaywrightComputerUse,
    ProviderComputerUse,
    computer_use_tools,
)
from .permissions import ApprovalRequest, ToolPermissionChecker, ToolPermissionDecision
from .registry import RegisteredTool, ToolRegistry
from .runtime import ToolRuntime, validate_against_schema
from .sandbox import (
    ContainerIsolation,
    GVisorIsolation,
    IsolationBackend,
    MicroVMIsolation,
    SandboxedPython,
    SandboxResult,
    SubprocessIsolation,
    WASMIsolation,
    get_isolation_backend,
    require_real_isolation,
    run_subprocess_sandboxed,
)

__all__ = [
    "ApprovalRequest",
    "ToolPermissionChecker",
    "ToolPermissionDecision",
    "RegisteredTool",
    "ToolRegistry",
    "ToolRuntime",
    "validate_against_schema",
    "SandboxedPython",
    "SandboxResult",
    "run_subprocess_sandboxed",
    # 1.10 — pluggable isolation backends
    "IsolationBackend",
    "SubprocessIsolation",
    "ContainerIsolation",
    "GVisorIsolation",
    "MicroVMIsolation",
    "WASMIsolation",
    "get_isolation_backend",
    "require_real_isolation",
    # 1.10 — computer-use / agentic browsing
    "ComputerAction",
    "ComputerObservation",
    "ComputerUseBackend",
    "MockComputerUse",
    "PlaywrightComputerUse",
    "ProviderComputerUse",
    "computer_use_tools",
]
