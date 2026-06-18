"""Sandboxed execution helpers (code execution sandbox).

`run_subprocess_sandboxed` executes a command in a separate process with a
timeout, output caps, a scrubbed environment, POSIX resource limits (CPU,
address space, open files via ``setrlimit``), and an optional working
directory jail. `SandboxedPython` runs Python snippets in a subprocess with
``-I`` (isolated mode) under conservative CPU/memory/fd limits by default.
These are OS-process isolation, not a security boundary against a hostile
kernel — appropriate for tool-grade isolation of generated code; harden
further with containers/seccomp for adversarial workloads.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..core.errors import SandboxError, ToolTimeoutError

__all__ = [
    "SandboxResult",
    "run_subprocess_sandboxed",
    "SandboxedPython",
    "IsolationBackend",
    "SubprocessIsolation",
    "ContainerIsolation",
    "MicroVMIsolation",
    "GVisorIsolation",
    "WASMIsolation",
    "ISOLATION_BACKENDS",
    "get_isolation_backend",
    "require_real_isolation",
]

_SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM")


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    truncated: bool = False


def _rlimit_preexec(
    max_cpu_seconds: int | None,
    max_memory_bytes: int | None,
    max_open_files: int | None,
) -> Callable[[], None] | None:
    """Build a POSIX ``preexec_fn`` that applies resource limits in the child.

    Returns ``None`` on platforms without the :mod:`resource` module (Windows),
    where these limits are not enforceable; the timeout and output caps still
    apply everywhere.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None

    if max_cpu_seconds is None and max_memory_bytes is None and max_open_files is None:
        return None

    def _set(which: int, value: int) -> None:
        # Best-effort: some platforms (notably macOS for RLIMIT_AS) don't honor
        # every limit. A limit we can't set must not crash the child — the
        # wall-clock timeout and output caps still bound it.
        try:
            soft, hard = resource.getrlimit(which)
            cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (cap, hard))
        except (ValueError, OSError):  # pragma: no cover - platform-dependent
            pass

    def _apply() -> None:  # pragma: no cover - runs in the forked child
        if max_cpu_seconds is not None:
            _set(resource.RLIMIT_CPU, max_cpu_seconds)
        if max_memory_bytes is not None and hasattr(resource, "RLIMIT_AS"):
            _set(resource.RLIMIT_AS, max_memory_bytes)
        if max_open_files is not None:
            _set(resource.RLIMIT_NOFILE, max_open_files)

    return _apply


async def run_subprocess_sandboxed(
    command: list[str],
    *,
    timeout_s: float = 30.0,
    max_output_bytes: int = 1_000_000,
    cwd: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
    stdin_data: str | None = None,
    max_cpu_seconds: int | None = None,
    max_memory_bytes: int | None = None,
    max_open_files: int | None = None,
) -> SandboxResult:
    """Run a command with a scrubbed environment and hard timeout.

    On POSIX, ``max_cpu_seconds`` (RLIMIT_CPU), ``max_memory_bytes``
    (RLIMIT_AS), and ``max_open_files`` (RLIMIT_NOFILE) are enforced in the
    child via ``setrlimit`` so a runaway snippet cannot exhaust CPU, memory, or
    file descriptors. These limits are best-effort on platforms without the
    ``resource`` module; the wall-clock timeout and output caps always apply.
    """
    import time

    env = {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}
    env.update(env_overrides or {})
    preexec = _rlimit_preexec(max_cpu_seconds, max_memory_bytes, max_open_files)
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        cwd=str(cwd) if cwd else None,
        env=env,
        preexec_fn=preexec,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(stdin_data.encode() if stdin_data is not None else None),
            timeout=timeout_s,
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise ToolTimeoutError(
            f"sandboxed command timed out after {timeout_s}s", tool=command[0]
        ) from exc
    truncated = len(stdout_bytes) > max_output_bytes or len(stderr_bytes) > max_output_bytes
    return SandboxResult(
        stdout=stdout_bytes[:max_output_bytes].decode("utf-8", errors="replace"),
        stderr=stderr_bytes[:max_output_bytes].decode("utf-8", errors="replace"),
        exit_code=process.returncode or 0,
        duration_ms=int((time.monotonic() - started) * 1000),
        truncated=truncated,
    )


class SandboxedPython:
    """Run Python snippets in an isolated subprocess (``python -I``).

    Isolated mode ignores environment variables and user site-packages; the
    snippet runs in a throwaway temp directory with no access to the parent
    process state.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 15.0,
        max_output_bytes: int = 500_000,
        python_executable: str | None = None,
        max_cpu_seconds: int | None = 10,
        max_memory_bytes: int | None = 512 * 1024 * 1024,
        max_open_files: int | None = 64,
        isolation: IsolationBackend | str | None = None,
        require_isolation: bool = False,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes
        self.python_executable = python_executable or sys.executable
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_bytes = max_memory_bytes
        self.max_open_files = max_open_files
        self.isolation = (
            get_isolation_backend(isolation) if isinstance(isolation, str) else isolation
        ) or SubprocessIsolation()
        # Code execution is a prime adversarial workload: when ``require_isolation``
        # is set, refuse to run on the zero-dep subprocess backend.
        if require_isolation:
            require_real_isolation(self.isolation)

    async def run(self, code: str) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="vincio_sandbox_") as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            return await self.isolation.run(
                [self.python_executable, "-I", str(script)],
                timeout_s=self.timeout_s,
                max_output_bytes=self.max_output_bytes,
                cwd=tmp,
                max_cpu_seconds=self.max_cpu_seconds,
                max_memory_bytes=self.max_memory_bytes,
                max_open_files=self.max_open_files,
            )


# ---------------------------------------------------------------------------
# Pluggable isolation backends
# ---------------------------------------------------------------------------


class IsolationBackend:
    """A pluggable isolation boundary behind the sandbox interface.

    The default :class:`SubprocessIsolation` is OS-process isolation with
    ``setrlimit`` — zero-dependency, but *not* a security boundary against a
    hostile kernel (``real`` is ``False``). Code-executing and computer-use
    workloads should run on a backend whose ``real`` is ``True``
    (container / microVM / gVisor / WASM); :func:`require_real_isolation`
    enforces it. Each real backend wraps a command in its runtime and shells out,
    degrading with a clear :class:`~vincio.core.errors.SandboxError` when the
    runtime binary is absent — so the abstraction is uniform whether or not the
    host has Docker/Firecracker/gVisor/Wasmtime installed.
    """

    name: str = "subprocess"
    level: str = "process"  # process | container | microvm | gvisor | wasm
    real: bool = False  # True only for backends that are a real security boundary
    runtime_binary: str | None = None

    def available(self) -> bool:
        if self.runtime_binary is None:
            return True
        return shutil.which(self.runtime_binary) is not None

    def _wrap(self, command: list[str], *, cwd: str | Path | None) -> list[str]:
        """Wrap *command* in this backend's runtime invocation. Override in
        real backends; the base runs the command unchanged (process isolation)."""
        return command

    async def run(self, command: list[str], **kwargs: Any) -> SandboxResult:
        if not self.available():
            raise SandboxError(
                f"isolation backend {self.name!r} requires {self.runtime_binary!r}, "
                "which is not installed; install it or choose another backend"
            )
        return await run_subprocess_sandboxed(self._wrap(command, cwd=kwargs.get("cwd")), **kwargs)


class SubprocessIsolation(IsolationBackend):
    """Zero-dependency process isolation (the default). Not a security boundary."""

    name = "subprocess"
    level = "process"
    real = False


class ContainerIsolation(IsolationBackend):
    """OCI-container isolation via Docker/Podman (``docker run --network none``)."""

    name = "container"
    level = "container"
    real = True

    def __init__(self, *, image: str = "python:3.12-slim", runtime: str = "docker",
                 network: str = "none") -> None:
        self.image = image
        self.runtime_binary = runtime
        self.network = network

    def _wrap(self, command: list[str], *, cwd: str | Path | None) -> list[str]:
        runtime = self.runtime_binary or "docker"
        mount = ["-v", f"{cwd}:{cwd}", "-w", str(cwd)] if cwd else []
        return [
            runtime, "run", "--rm", f"--network={self.network}",
            "--read-only", "--cap-drop=ALL", *mount, self.image, *command,
        ]


class GVisorIsolation(ContainerIsolation):
    """gVisor user-space-kernel isolation (``runsc`` via the Docker runtime)."""

    name = "gvisor"
    level = "gvisor"
    real = True

    def __init__(self, *, image: str = "python:3.12-slim", runtime: str = "docker") -> None:
        super().__init__(image=image, runtime=runtime)

    def _wrap(self, command: list[str], *, cwd: str | Path | None) -> list[str]:
        wrapped = super()._wrap(command, cwd=cwd)
        # Insert the gVisor runtime selector after `docker run`.
        return [*wrapped[:2], "--runtime=runsc", *wrapped[2:]]


class MicroVMIsolation(IsolationBackend):
    """microVM isolation via Firecracker-class runtimes (``ignite``/``firecracker``)."""

    name = "microvm"
    level = "microvm"
    real = True

    def __init__(self, *, runtime: str = "ignite", image: str = "python:3.12-slim") -> None:
        self.runtime_binary = runtime
        self.image = image

    def _wrap(self, command: list[str], *, cwd: str | Path | None) -> list[str]:
        runtime = self.runtime_binary or "ignite"
        return [runtime, "run", self.image, "--", *command]


class WASMIsolation(IsolationBackend):
    """WebAssembly isolation via Wasmtime (capability-based, deny-by-default)."""

    name = "wasm"
    level = "wasm"
    real = True

    def __init__(self, *, runtime: str = "wasmtime") -> None:
        self.runtime_binary = runtime

    def _wrap(self, command: list[str], *, cwd: str | Path | None) -> list[str]:
        runtime = self.runtime_binary or "wasmtime"
        mount = ["--dir", str(cwd)] if cwd else []
        return [runtime, "run", *mount, *command]


ISOLATION_BACKENDS: dict[str, type[IsolationBackend]] = {
    "subprocess": SubprocessIsolation,
    "container": ContainerIsolation,
    "gvisor": GVisorIsolation,
    "microvm": MicroVMIsolation,
    "wasm": WASMIsolation,
}


def get_isolation_backend(name: str) -> IsolationBackend:
    """Construct an isolation backend by name (``subprocess`` is the default)."""
    if name not in ISOLATION_BACKENDS:
        raise SandboxError(
            f"unknown isolation backend {name!r}; known: {sorted(ISOLATION_BACKENDS)}"
        )
    return ISOLATION_BACKENDS[name]()


def require_real_isolation(backend: IsolationBackend) -> None:
    """Raise unless *backend* is a real security boundary (not bare subprocess)."""
    if not backend.real:
        raise SandboxError(
            f"isolation backend {backend.name!r} is process-level only and is not a "
            "security boundary; code-executing / computer-use workloads require a real "
            "backend (container / microvm / gvisor / wasm)"
        )
