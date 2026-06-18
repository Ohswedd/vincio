"""Computer-use / agentic-browsing action surface.

A small, explicit action vocabulary — navigate / click / type / screenshot
(plus scroll / key / wait) — exposed as ordinary Vincio tools, so a computer-use
agent runs through the *same* permissioned, audited, budgeted tool runtime as any
local tool rather than a thin GUI adapter bolted on the side. Backends are
pluggable: a deterministic :class:`MockComputerUse` for offline tests, a
:class:`PlaywrightComputerUse` for real browser control, and provider-native
backends (Anthropic / OpenAI computer-use) that drive the model's own loop.

These workloads should run behind a real
:class:`~vincio.tools.sandbox.IsolationBackend` (container / microVM / gVisor /
WASM); :func:`~vincio.tools.sandbox.require_real_isolation` is enforced by
``ContextApp.enable_computer_use(require_isolation=True)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "ComputerAction",
    "ComputerObservation",
    "ComputerUseBackend",
    "MockComputerUse",
    "PlaywrightComputerUse",
    "ProviderComputerUse",
    "computer_use_tools",
]

ActionType = Literal["navigate", "click", "type", "screenshot", "scroll", "key", "wait"]


class ComputerAction(BaseModel):
    """One computer/browser action request."""

    action: ActionType
    url: str | None = None
    selector: str | None = None
    text: str | None = None
    x: int | None = None
    y: int | None = None
    key: str | None = None
    ms: int | None = None


class ComputerObservation(BaseModel):
    """The result of a computer action (the observe half of the loop)."""

    action: str
    ok: bool = True
    url: str = ""
    title: str = ""
    text: str = ""
    screenshot_ref: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComputerUseBackend:
    """Interface for a computer-use backend (subclass to plug a driver in)."""

    name: str = "computer_use"

    async def act(self, action: ComputerAction) -> ComputerObservation:  # pragma: no cover - iface
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None


class MockComputerUse(ComputerUseBackend):
    """Deterministic in-memory browser for offline tests and dry runs.

    Tracks a current URL, a visit log, and typed input, and returns stable,
    reproducible screenshot references — no network, no real browser."""

    name = "mock_computer_use"

    def __init__(self) -> None:
        self.url = "about:blank"
        self.visited: list[str] = []
        self.typed: list[str] = []
        self.clicks: list[str] = []
        self._shots = 0

    async def act(self, action: ComputerAction) -> ComputerObservation:
        if action.action == "navigate":
            self.url = action.url or self.url
            self.visited.append(self.url)
            return ComputerObservation(action="navigate", url=self.url, title=f"Page {self.url}")
        if action.action == "click":
            target = action.selector or f"({action.x},{action.y})"
            self.clicks.append(target)
            return ComputerObservation(action="click", url=self.url, text=f"clicked {target}")
        if action.action == "type":
            self.typed.append(action.text or "")
            return ComputerObservation(action="type", url=self.url, text=f"typed {action.text!r}")
        if action.action == "screenshot":
            self._shots += 1
            return ComputerObservation(
                action="screenshot", url=self.url,
                screenshot_ref=f"mock://screenshot/{self._shots}",
            )
        if action.action in ("scroll", "key", "wait"):
            return ComputerObservation(action=action.action, url=self.url)
        return ComputerObservation(action=action.action, ok=False, error="unsupported action")


class PlaywrightComputerUse(ComputerUseBackend):
    """Real browser control via Playwright (optional dependency, lazy import)."""

    name = "playwright_computer_use"

    def __init__(self, *, headless: bool = True, browser: str = "chromium") -> None:
        self.headless = headless
        self.browser = browser
        self._page: Any = None
        self._pw: Any = None

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            from ..core.errors import SandboxError

            raise SandboxError(
                "PlaywrightComputerUse requires Playwright: pip install playwright "
                "&& playwright install chromium"
            ) from exc
        self._pw = await async_playwright().start()
        engine = getattr(self._pw, self.browser)
        browser = await engine.launch(headless=self.headless)
        self._page = await browser.new_page()
        return self._page

    async def act(self, action: ComputerAction) -> ComputerObservation:  # pragma: no cover - needs browser
        page = await self._ensure_page()
        if action.action == "navigate" and action.url:
            await page.goto(action.url)
            return ComputerObservation(action="navigate", url=page.url, title=await page.title())
        if action.action == "click" and action.selector:
            await page.click(action.selector)
            return ComputerObservation(action="click", url=page.url)
        if action.action == "type" and action.selector is not None:
            await page.fill(action.selector, action.text or "")
            return ComputerObservation(action="type", url=page.url)
        if action.action == "screenshot":
            data = await page.screenshot()
            return ComputerObservation(
                action="screenshot", url=page.url,
                screenshot_ref=f"bytes://{len(data)}", metadata={"bytes": len(data)},
            )
        return ComputerObservation(action=action.action, url=page.url)

    async def close(self) -> None:  # pragma: no cover - needs browser
        if self._pw is not None:
            await self._pw.stop()


class ProviderComputerUse(ComputerUseBackend):
    """Provider-native computer-use (Anthropic / OpenAI) adapter.

    Delegates each action to the provider's hosted computer-use capability. The
    provider runs the screenshot→action→observe loop server-side; this adapter
    keeps the action vocabulary uniform with the local backends so the agent code
    is identical regardless of where the loop runs."""

    name = "provider_computer_use"

    def __init__(self, provider: Any, model: str, *, namespace: str = "anthropic") -> None:
        self.provider = provider
        self.model = model
        self.namespace = namespace

    async def act(self, action: ComputerAction) -> ComputerObservation:  # pragma: no cover - needs provider
        executor = getattr(self.provider, "computer_use", None)
        if executor is None:
            from ..core.errors import SandboxError

            raise SandboxError(
                f"provider {getattr(self.provider, 'name', '?')!r} does not expose a "
                "native computer-use capability"
            )
        result = await executor(self.model, action.model_dump(exclude_none=True))
        return ComputerObservation.model_validate(result)


def computer_use_tools(backend: ComputerUseBackend) -> list[Callable[..., Any]]:
    """The browser action vocabulary as named callables for ``app.add_tool``.

    Register these as ``external`` write tools with a ``computer:use`` permission
    so the agent's GUI loop rides the same RBAC + audit + budget path as any tool.
    """
    from ..providers.base import run_sync

    def computer_navigate(url: str) -> dict[str, Any]:
        """Navigate the browser to a URL."""
        return run_sync(backend.act(ComputerAction(action="navigate", url=url))).model_dump()

    def computer_click(selector: str = "", x: int = 0, y: int = 0) -> dict[str, Any]:
        """Click an element by selector or coordinates."""
        return run_sync(
            backend.act(ComputerAction(action="click", selector=selector or None, x=x or None, y=y or None))
        ).model_dump()

    def computer_type(text: str, selector: str = "") -> dict[str, Any]:
        """Type text, optionally into a selector."""
        return run_sync(
            backend.act(ComputerAction(action="type", text=text, selector=selector or None))
        ).model_dump()

    def computer_screenshot() -> dict[str, Any]:
        """Capture a screenshot of the current page."""
        return run_sync(backend.act(ComputerAction(action="screenshot"))).model_dump()

    return [computer_navigate, computer_click, computer_type, computer_screenshot]
