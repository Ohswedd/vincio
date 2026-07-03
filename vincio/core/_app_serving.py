"""Computer-use, tool, skill, MCP, realtime/voice, and A2A serving verbs — a private mixin of
:class:`~vincio.core.app.ContextApp`.

Extracted verbatim from ``vincio/core/app.py`` (v7.5 structure line): method
source, decorators, comments, and docstrings are unchanged. ``ContextApp``
composes this class, so every method here remains an ``app.*`` verb; the
``self: ContextApp`` annotations keep attribute access type-checked against
the composed app. The standing hygiene lints (:mod:`vincio._error_contract`,
:mod:`vincio._observable_failure`, :mod:`vincio._assert_robustness`)
deliberately keep ``vincio/core/_app_*.py`` in scope despite the private
filename, so the verb surface stays guarded after the split.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..security.access import Principal
from ..skills.library import SkillLibrary
from .errors import (
    ConfigError,
    ToolNotFoundError,
)

if TYPE_CHECKING:
    from ..providers.base import ModelProvider
    from .app import ContextApp


class _ServingVerbs:
    """Computer-use, tool, skill, MCP, realtime/voice, and A2A serving verbs. Mixed into :class:`~vincio.core.app.ContextApp`."""

    if TYPE_CHECKING:
        # ContextApp state this mixin's verbs assign. mypy would otherwise
        # attribute the unannotated ``self.X = ...`` assignments to this class
        # and clash with ContextApp.__init__; the declarations (type-checking
        # only, no runtime effect) keep the split typing identical to the
        # monolith's.
        skill_library: SkillLibrary | None
        web_browser: Any
        _provider_instance: ModelProvider | None


    def enable_computer_use(  # type: ignore[misc]
        self: ContextApp,
        backend: str = "mock",
        *,
        isolation: str | None = None,
        require_isolation: bool = False,
        permission: str = "computer:use",
        approval_required: bool = True,
        **backend_kwargs: Any,
    ):
        """Register a computer-use action surface (navigate / click / type /
        screenshot) as audited, permissioned tools.

        ``backend`` is ``"mock"`` (deterministic, offline), ``"playwright"`` (real
        browser), or ``"provider"`` (provider-native computer-use). With
        ``require_isolation=True`` the workload must run behind a real
        :class:`~vincio.tools.sandbox.IsolationBackend` (container / microVM /
        gVisor / WASM) — subprocess-only hosts are refused::

            app.enable_computer_use("mock")
            agent = app.agent(tools=["computer_navigate", "computer_screenshot"])
        """
        from ..tools.computer_use import (
            MockComputerUse,
            PlaywrightComputerUse,
            ProviderComputerUse,
            computer_use_tools,
        )

        if require_isolation:
            from ..tools.sandbox import get_isolation_backend, require_real_isolation

            require_real_isolation(get_isolation_backend(isolation or "subprocess"))
        if backend == "playwright":
            impl: Any = PlaywrightComputerUse(**backend_kwargs)
        elif backend == "provider":
            impl = ProviderComputerUse(self._base_provider(), self.model, **backend_kwargs)
        else:
            impl = MockComputerUse()
        for tool in computer_use_tools(impl):
            self.add_tool(
                tool,
                permissions=[permission],
                side_effects="external",
                approval_required=approval_required,
            )
        self.audit.record(
            "computer_use_enabled",
            decision="allow",
            details={
                "backend": backend,
                "isolation": isolation,
                "require_isolation": require_isolation,
            },
        )
        return impl

    def computer_use(  # type: ignore[misc]
        self: ContextApp,
        backend: str = "mock",
        *,
        screen: Any = None,
        policy: Any = None,
        approve: Callable[..., bool] | None = None,
        auto_undo: bool = True,
        max_steps: int = 50,
        isolation: str | None = None,
        require_isolation: bool = False,
        **backend_kwargs: Any,
    ) -> Any:
        """Open a grounded, verified, reversible computer-use **action plane**.

        Returns a :class:`~vincio.tools.ComputerEnvironment` that perceives a screen
        as typed, addressable :class:`~vincio.tools.UIElement`\\ s, grounds an intent
        to a stable selector, **pre-gates** each action against an
        :class:`~vincio.tools.ActionPolicy` (a destructive or out-of-scope action is
        gated like a write tool, with an ``approve`` callback), acts, **post-verifies**
        the effect, and **undoes** it on divergence — every action recorded on this
        app's hash-chained audit log.

        ``backend`` is ``"mock"`` (deterministic, offline; pass a
        :class:`~vincio.tools.ScreenApp`/:class:`~vincio.tools.MockScreen` as
        ``screen``), ``"playwright"`` (a real browser / CDP), ``"accessibility"`` (an
        OS accessibility tree), or ``"remote_desktop"`` (a remote machine); the real
        adapters need ``vincio[computer-use]``. With ``require_isolation=True`` the
        workload must run behind a real
        :class:`~vincio.tools.sandbox.IsolationBackend`::

            app_spec, task = build_web_checkout()
            env = app.computer_use(screen=app_spec, policy=ActionPolicy(allow_urls=["https://shop.test"]))
            run = env.run(my_policy, task)
            run.success and run.safe  # verified end-state, no unapproved destructive action
        """
        from ..tools.computer_environment import (
            AccessibilityScreen,
            ActionPolicy,
            ComputerEnvironment,
            MockScreen,
            PlaywrightScreen,
            RemoteDesktopScreen,
            ScreenApp,
            ScreenBackend,
        )

        if require_isolation:
            from ..tools.sandbox import get_isolation_backend, require_real_isolation

            require_real_isolation(get_isolation_backend(isolation or "subprocess"))

        if isinstance(screen, ScreenBackend):
            impl: ScreenBackend = screen
        elif isinstance(screen, ScreenApp):
            impl = MockScreen(screen)
        elif backend == "playwright":
            impl = PlaywrightScreen(**backend_kwargs)
        elif backend == "accessibility":
            impl = AccessibilityScreen(**backend_kwargs)
        elif backend == "remote_desktop":
            impl = RemoteDesktopScreen(**backend_kwargs)
        elif isinstance(screen, dict):
            impl = MockScreen(ScreenApp.model_validate(screen))
        else:
            raise ConfigError(
                f"computer_use backend {backend!r} needs a ScreenApp/MockScreen via screen=; "
                "the deterministic offline backend is 'mock'"
            )

        self.audit.record(
            "computer_use_session",
            decision="allow",
            details={
                "backend": getattr(impl, "name", backend),
                "isolation": isolation,
                "require_isolation": require_isolation,
                "auto_undo": auto_undo,
            },
        )
        return ComputerEnvironment(
            impl,
            app=self,
            policy=policy if isinstance(policy, ActionPolicy) else (ActionPolicy(**policy) if isinstance(policy, dict) else None),
            approve=approve,
            auto_undo=auto_undo,
            max_steps=max_steps,
        )

    def use_hosted_tools(  # type: ignore[misc]
        self: ContextApp, names: list[str] | None = None, *, namespace: str = "openai"
    ) -> ContextApp:
        """Surface provider-native hosted tools (``web_search`` / ``file_search`` /
        ``code_interpreter`` / ``computer_use``) as namespaced Vincio tools.

        They register on the tool registry with explicit permissions and ride the
        same RBAC + audit path as any local tool; the Responses adapter emits each
        as its provider-native built-in descriptor::

            app.use_hosted_tools(["web_search", "code_interpreter"])
        """
        from ..providers.hosted_tools import hosted_tool_specs

        for spec in hosted_tool_specs(names, namespace=namespace):
            self.tool_registry.register_spec(spec)
            if spec.name not in self.enabled_tools:
                self.enabled_tools.append(spec.name)
        self.audit.record(
            "hosted_tools_enabled",
            decision="allow",
            details={
                "namespace": namespace,
                "tools": [s.name for s in hosted_tool_specs(names, namespace=namespace)],
            },
        )
        return self

    def use_web_search(  # type: ignore[misc]
        self: ContextApp,
        backend: Any | None = None,
        *,
        preset: str | None = None,
        policy: Any | None = None,
        client: Any | None = None,
        skill: bool = True,
        today: str | None = None,
        tool_protocol: bool | str = True,
        **policy_fields: Any,
    ) -> ContextApp:
        """Give this app's model — **any** model — governed access to the open web.

        Registers Vincio-executed ``web_search`` / ``web_read`` tools (DuckDuckGo
        by default; any :class:`~vincio.web.SearchBackend` via ``backend``),
        loads the built-in browsing skill (when to search, how to write queries,
        when to stop — progressively disclosed, and stamped with ``today`` so the
        model knows what "current" means), and wraps the provider in
        :class:`~vincio.providers.ToolProtocolProvider` so a model without
        native function calling gets the same two tools through a text protocol.
        When the user's own message directs a fetch (a pasted link, "summarize
        …"), the page is fetched and folded in as untrusted, screened, offline-
        verifiable evidence with no tool round.

        ``preset`` picks a starting :class:`~vincio.web.WebPolicy`
        (``"default"`` / ``"research"`` / ``"scrape"`` / ``"locked_down"``); any
        policy field overrides it as a keyword. Every search and fetch is
        policy-gated pre-egress and lands on the audit chain; the session's
        evidence re-derives offline via ``app.web_browser.report()``::

            app.use_web_search(preset="research", deny_domains=["tracker.example"])

        ``tool_protocol=False`` leaves the provider unwrapped (native-only);
        ``"force"`` applies the text protocol even to natively capable models.
        """
        from ..providers.tool_protocol import ToolProtocolProvider
        from ..web.browser import WebBrowser
        from ..web.policy import WebPolicy
        from ..web.skill import browse_skill

        if isinstance(policy, WebPolicy):
            resolved_policy = policy
        else:
            base = dict(policy) if isinstance(policy, dict) else {}
            fields = {**base, **policy_fields}
            resolved_policy = (
                WebPolicy.preset(preset, **fields) if preset else WebPolicy(**fields)
            )
        browser = WebBrowser(backend, policy=resolved_policy, client=client, audit=self.audit)
        self.web_browser = browser
        for spec, handler in browser.tool_handlers():
            self.tool_registry.register_spec(spec, handler=handler)
            if spec.name not in self.enabled_tools:
                self.enabled_tools.append(spec.name)
        if skill:
            self.add_skill(browse_skill(today=today))
        if tool_protocol:
            self._provider_instance = ToolProtocolProvider(
                self._base_provider(), force=tool_protocol == "force"
            )
        self.audit.record(
            "web_search_enabled",
            decision="allow",
            details={
                "backend": getattr(browser.backend, "name", type(browser.backend).__name__),
                "policy": resolved_policy.model_dump(),
                "tool_protocol": str(tool_protocol),
            },
        )
        return self

    def web_crawl(  # type: ignore[misc]
        self: ContextApp,
        seeds: str | list[str],
        *,
        scope: str = "subtree",
        query: str = "",
        max_pages: int | None = None,
        max_depth: int | None = None,
        policy: Any | None = None,
        client: Any | None = None,
        mode: str = "full",
    ) -> Any:
        """Crawl a site into a governed, offline-verifiable
        :class:`~vincio.web.WebCollection`.

        Walks outward from ``seeds`` (``scope`` = ``"page"`` / ``"subtree"`` /
        ``"domain"``) through a :class:`~vincio.web.WebBrowser`, so every fetch
        keeps the SSRF rails, robots, size caps, and snapshotting, and the walk
        is bounded on every axis (pages, depth, per-host, bytes, wall-clock) with
        trap-template defense. The result converts to retrieval documents
        (:meth:`~vincio.web.WebCollection.to_documents`) or a tabular
        :class:`~vincio.data.Dataset` (:meth:`~vincio.web.WebCollection.to_dataset`)
        and re-derives offline via ``collection.verify(browser.snapshots)``::

            collection = app.web_crawl("https://docs.example.com/", scope="subtree")
            app.add_source("docs", documents=collection.to_documents())
        """
        from ..providers.base import run_sync
        from ..web.crawl import WebCrawler
        from ..web.policy import WebPolicy

        resolved_policy = policy if isinstance(policy, WebPolicy) else WebPolicy.preset("scrape")
        crawler = WebCrawler(
            policy=resolved_policy, client=client, mode=mode  # type: ignore[arg-type]
        )
        collection = run_sync(
            crawler.crawl(
                seeds, scope=scope, query=query,  # type: ignore[arg-type]
                max_pages=max_pages, max_depth=max_depth,
            )
        )
        self.audit.record(
            "web_crawl",
            decision="allow",
            details={
                "seeds": [seeds] if isinstance(seeds, str) else list(seeds),
                "scope": scope,
                "pages": collection.pages_fetched,
                "stopped": collection.stopped_reason,
            },
        )
        return collection

    # -- tools ------------------------------------------------------------------------------------

    def add_tool(  # type: ignore[misc]
        self: ContextApp,
        tool: str | Callable,
        *,
        permissions: list[str] | None = None,
        permission: str | None = None,
        approval_required: bool = False,
        side_effects: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> ContextApp:
        """Enable a tool: a callable (registered now) or the name of a tool
        already registered on app.tool_registry."""
        if permission is not None:  # default permission is "read_only"
            if permission == "read_only":
                side_effects = side_effects or "read"
            else:
                permissions = [*(permissions or []), permission]
        if callable(tool):
            self.tool_registry.register(
                tool,
                permissions=permissions or [],
                approval_required=approval_required,
                side_effects=side_effects or "read",
                description=description,
                **kwargs,
            )
            name = kwargs.get("name") or tool.__name__
        else:
            name = tool
            if name not in self.tool_registry:
                raise ToolNotFoundError(
                    f"tool {name!r} is not registered; pass a callable or register it via "
                    "app.tool_registry.register(...)",
                    tool=name,
                )
            spec = self.tool_registry.get(name).spec
            if permissions:
                spec.permissions = permissions
            if approval_required:
                spec.approval_required = True
            if side_effects:
                spec.side_effects = side_effects  # type: ignore[assignment]
        if name not in self.enabled_tools:
            self.enabled_tools.append(name)
        return self

    # -- skills ---------------------------------------------------------------------------------

    def add_skill(self: ContextApp, skill: str | Any, *, register_scripts: bool = False) -> ContextApp:  # type: ignore[misc]
        """Load an Agent Skill (``SKILL.md`` path or a :class:`Skill`) and inject
        it through the compiler with progressive disclosure: a one-line summary
        is always available; the full body is included only when a run's task is
        relevant. Set ``register_scripts=True`` to expose bundled scripts as
        sandboxed, permissioned tools."""
        from ..skills import Skill, load_skill, register_skill_scripts

        loaded = skill if isinstance(skill, Skill) else load_skill(skill)
        if self.skill_library is None:
            self.skill_library = SkillLibrary()
        self.skill_library.add(loaded)
        if register_scripts and loaded.scripts:
            for name in register_skill_scripts(self.tool_registry, loaded):
                if name not in self.enabled_tools:
                    self.enabled_tools.append(name)
        return self

    # -- MCP ------------------------------------------------------------------------------------

    def add_mcp_server(  # type: ignore[misc]
        self: ContextApp,
        name: str,
        *,
        command: list[str] | None = None,
        url: str | None = None,
        server: Any | None = None,
        transport: Any | None = None,
        headers: dict[str, str] | None = None,
        http_client: Any | None = None,
        auth: str | None = None,
        tools: bool = True,
        resources: bool = True,
        prompts: bool = False,
        permissions: list[str] | None = None,
        sampling: bool = True,
        elicitation: Any | None = None,
        elicitation_policy: Any | None = None,
        elicitation_approval: Any | None = None,
    ) -> ContextApp:
        """Connect to an MCP server and register its tools/resources/prompts.

        Provide exactly one of ``command`` (stdio), ``url`` (Streamable HTTP),
        ``server`` (an in-process :class:`MCPServer`), or ``transport``. MCP
        tools register through the existing permissioned, sandboxed, audited
        runtime (namespaced ``<name>.<tool>``); resources become evidence with
        ``origin: mcp:<name>``. Server-initiated sampling routes to this app's
        provider.

        A server's mid-call **elicitation** request routes to a governed
        :class:`~vincio.mcp.apps.ElicitationGate` built from ``elicitation`` (the
        collector that obtains the user's value): the collected value is screened
        through this app's input rails and tainted *untrusted*, so it is contained
        like any other untrusted input. Pass an ``elicitation_approval`` callable
        (``ElicitationRequest -> bool``) to additionally gate the request behind an
        approval — the way a write tool is gated — or an
        :class:`~vincio.mcp.apps.ElicitationPolicy` for full control. Connect
        happens now (synchronously); the live client is kept on
        ``app.mcp_clients[name]``.
        """
        from ..mcp import (
            InProcessTransport,
            MCPClient,
            StdioTransport,
            StreamableHTTPTransport,
        )
        from ..providers.base import run_sync

        if transport is None:
            provided = [x for x in (command, url, server) if x is not None]
            if len(provided) != 1:
                raise ConfigError(
                    "add_mcp_server requires exactly one of command=, url=, server=, or transport="
                )
            if command is not None:
                transport = StdioTransport(command)
            elif url is not None:
                transport = StreamableHTTPTransport(url, headers=headers, client=http_client)
            else:
                transport = InProcessTransport(server, auth=auth)
        elicitation_gate = None
        if elicitation is not None or elicitation_policy is not None:
            from ..mcp.apps import ElicitationGate, ElicitationPolicy

            policy = elicitation_policy
            if policy is None:
                policy = ElicitationPolicy(require_approval=callable(elicitation_approval))
            elicitation_gate = ElicitationGate(
                elicitation,
                policy=policy,
                rail_engine=self.rail_engine,
                approver=elicitation_approval if callable(elicitation_approval) else None,
                audit=self.audit,
            )
        client = MCPClient(
            transport,
            name=name,
            sampling_provider=self.resolve_provider() if sampling else None,
            sampling_model=self.model,
            elicitation_gate=elicitation_gate,
        )
        run_sync(
            client.register_into(
                self,
                tools=tools,
                resources=resources,
                prompts=prompts,
                permissions=permissions,
            )
        )
        self.mcp_clients[name] = client
        return self

    def add_mcp_from_registry(  # type: ignore[misc]
        self: ContextApp,
        name: str,
        *,
        registry: Any,
        directory: Any | None = None,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        server: Any | None = None,
        transport: Any | None = None,
        headers: dict[str, str] | None = None,
        http_client: Any | None = None,
        tools: bool = True,
        resources: bool = True,
        prompts: bool = False,
        permissions: list[str] | None = None,
        principal: Any | None = None,
    ) -> ContextApp:
        """Discover an MCP server from a registry and land its tools in the
        permissioned runtime — one governed call (the marketplace bridge).

        Three concerns compose: **discovery** (an
        :class:`~vincio.registry.MCPRegistryClient` — the official MCP Registry
        or an offline catalog — finds the server), **governance** (a governed
        :class:`~vincio.registry.AgentDirectory` under an
        :class:`~vincio.security.access.AllowListGate` decides reachability and
        records the decision on this app's audit chain), and **connection**
        (:meth:`add_mcp_server` runs the server's tools through the existing
        permissioned, sandboxed, audited runtime).

        Pass ``directory=`` to reuse an existing governed directory, or
        ``allow`` / ``deny`` globs to build one (fail-closed; defaults to
        allowing exactly ``name``). For offline / in-process use, pass
        ``server=`` (an in-process :class:`~vincio.mcp.MCPServer`) or
        ``transport=``; otherwise the resolved server's URL or stdio command is
        used. Raises :class:`~vincio.core.errors.AccessDeniedError` if the gate
        denies the server.
        """
        from ..providers.base import run_sync

        if directory is None:
            directory = self.agent_directory(
                allow=allow if allow is not None else [name], deny=deny
            )
        # Discovery registers candidate servers into the directory as governed,
        # audited AgentRecords (protocol="mcp").
        run_sync(registry.register_into_directory(directory))
        # The governed resolution is the audited access decision.
        record = directory.resolve(name, principal=principal)
        srv = run_sync(registry.get_server(name))

        conn: dict[str, Any] = {}
        if transport is not None:
            conn["transport"] = transport
        elif server is not None:
            conn["server"] = server
        elif srv is not None and srv.url:
            conn["url"] = srv.url
        elif srv is not None and srv.command:
            conn["command"] = srv.command
        elif record.url:
            conn["url"] = record.url
        else:
            raise ConfigError(
                f"MCP server {name!r} has no url/command in the registry; pass server= or transport="
            )
        return self.add_mcp_server(
            name,
            headers=headers,
            http_client=http_client,
            tools=tools,
            resources=resources,
            prompts=prompts,
            permissions=permissions,
            **conn,
        )

    def serve_mcp(  # type: ignore[misc]
        self: ContextApp,
        *,
        name: str | None = None,
        expose_resources: bool = True,
        expose_prompts: bool = True,
        ui_resources: list[Any] | None = None,
        token_validator: Any | None = None,
    ) -> Any:
        """Expose this app as an MCP server (returns an :class:`MCPServer`).

        Registered tools become MCP tools (run through the permissioned,
        sandboxed, audited runtime); evidence/sources become resources; the
        prompt spec becomes a prompt. Pass ``ui_resources`` — a list of
        :class:`~vincio.mcp.MCPUIResource` — to also serve MCP-UI / AG-UI
        resources for generative-UI hosts. Run it over stdio with
        ``vincio.mcp.serve_stdio(server)`` or the ``vincio mcp serve`` CLI.
        """
        from ..mcp import build_app_server

        return build_app_server(
            self,
            name=name,
            expose_resources=expose_resources,
            expose_prompts=expose_prompts,
            ui_resources=ui_resources,
            token_validator=token_validator,
        )

    def mcp_app(self: ContextApp, name: str, *, max_render_tokens: int = 4096) -> Any:  # type: ignore[misc]
        """Bridge a consumed MCP server's UI resources onto the AG-UI channel.

        Returns an :class:`~vincio.mcp.apps.MCPAppBridge` over the client
        connected as ``name`` (via :meth:`add_mcp_server`). The bridge reads the
        server's server-rendered ``ui://`` resources and lowers each into an
        :class:`~vincio.server.agui.AGUIEvent` — token-metered against
        ``max_render_tokens`` and recorded on this app's audit chain — so MCP
        Apps UI rides the *same* governed generative-UI stream as the run.
        """
        from ..mcp.apps import MCPAppBridge

        client = self.mcp_clients.get(name)
        if client is None:
            raise ConfigError(
                f"no MCP server connected as {name!r}; call add_mcp_server({name!r}, ...) first"
            )
        return MCPAppBridge(client, audit=self.audit, max_render_tokens=max_render_tokens)

    # -- realtime / voice (optional module) ------------------------------------------------------

    def realtime_session(  # type: ignore[misc]
        self: ContextApp,
        *,
        backend: str = "inprocess",
        config: Any | None = None,
        **backend_kwargs: Any,
    ) -> Any:
        """Open a voice/realtime session (returns a :class:`RealtimeSession`).

        In-session tool calls route through this app's **permissioned,
        sandboxed, audited** tool runtime — exactly like a native tool call.
        ``backend`` is ``inprocess`` (offline default), ``openai`` (OpenAI
        Realtime), or ``gemini`` (Gemini Live); the hosted backends need
        ``pip install "vincio[realtime]"``. Optional module — see
        :mod:`vincio.realtime`.
        """
        from ..core.types import ToolCall
        from ..realtime import RealtimeConfig, connect_realtime

        async def _dispatch(name: str, arguments: dict[str, Any]) -> Any:
            # Route through the permissioned runtime exactly like a native tool
            # call: validation, scopes, and the approval gate all apply. We do
            # NOT pre-approve — an approval-required tool hits the same gate
            # (and raises ToolApprovalRequiredError, surfaced as an error event)
            # as on the text path, so voice cannot auto-run a write tool.
            result = await self.tool_runtime.execute(
                ToolCall(tool_name=name, arguments=arguments),
                principal=Principal(scopes=list(self.policies.custom.get("scopes", ["*"]))),
            )
            return result.output if result.status == "ok" else {"error": result.error}

        if config is None:
            config = RealtimeConfig(model=backend_kwargs.pop("model", "gpt-realtime"))
        return connect_realtime(backend, config=config, tool_dispatcher=_dispatch, **backend_kwargs)

    def voice_agent(  # type: ignore[misc]
        self: ContextApp,
        *,
        backend: str = "inprocess",
        config: Any | None = None,
        research: bool = True,
        memory_os: bool = True,
        rails: bool = True,
        owner_id: str = "voice",
        **backend_kwargs: Any,
    ) -> Any:
        """Open an end-to-end :class:`~vincio.realtime.VoiceAgent`.

        A realtime session wired to the full stack: the deep-research agent (as
        an in-session ``research`` tool), the self-editing memory OS, and the
        app's deterministic input/output rails over every spoken transcript and
        reply — so a spoken assistant inherits the same grounding, budget, and
        audit guarantees as the text path. In-session tool calls route through
        this app's permissioned, sandboxed, audited runtime::

            app.add_source("kb", documents=[...])
            agent = app.voice_agent()
            async with agent:
                await agent.send_text("What is the refund window?")
                await agent.commit()
                async for event in agent.events():
                    ...

        ``backend`` is ``inprocess`` (offline default), ``openai``, or ``gemini``
        (hosted backends need ``pip install "vincio[realtime]"``). Optional
        module — see :mod:`vincio.realtime`.
        """
        from ..realtime.voice_agent import VoiceAgent

        return VoiceAgent(
            self,
            backend=backend,
            config=config,
            research=research,
            memory_os=memory_os,
            rails=rails,
            owner_id=owner_id,
            **backend_kwargs,
        )

    # -- A2A ------------------------------------------------------------------------------------

    def serve_a2a(  # type: ignore[misc]
        self: ContextApp,
        target: Any | None = None,
        *,
        name: str | None = None,
        url: str = "",
        description: str = "",
        token_validator: Any | None = None,
    ) -> Any:
        """Expose a crew, a compiled graph, or this app over A2A.

        Pass a :class:`Crew`, a compiled :class:`StateGraph`, or ``None`` (the
        app itself). Returns an :class:`A2AServer` whose Agent Card is served at
        ``/.well-known/agent.json``; delegation stays bounded and traced. Run it
        over HTTP behind the FastAPI server or consume it in-process.
        """
        from ..a2a import app_a2a_server, crew_a2a_server, graph_a2a_server
        from ..agents.crew import Crew
        from ..agents.graph import CompiledGraph

        if target is None:
            return app_a2a_server(
                self, name=name, url=url, description=description, token_validator=token_validator
            )
        if isinstance(target, Crew):
            return crew_a2a_server(
                target,
                name=name,
                url=url,
                description=description,
                token_validator=token_validator,
                audit=self.audit,
            )
        if isinstance(target, CompiledGraph):
            return graph_a2a_server(
                target,
                name=name or "graph",
                url=url,
                description=description,
                tracer=self.tracer,
                token_validator=token_validator,
                audit=self.audit,
            )
        raise ConfigError(
            "serve_a2a target must be a Crew, a compiled StateGraph, or None (the app)"
        )

    def agent_directory(  # type: ignore[misc]
        self: ContextApp,
        *,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        default_allow: bool = False,
    ) -> Any:
        """A governed, audited :class:`~vincio.registry.AgentDirectory` for this app.

        Resolutions pass an allow-list gate (``allow`` / ``deny`` fnmatch globs;
        fail-closed by default) and are recorded as access decisions on this app's
        hash-chained audit log, so the agent fabric is as accountable as a local
        tool call. Register A2A Agent Cards directly, or discover agents from an
        AGNTCY/ACP or MCP registry into it.
        """
        from ..registry import AgentDirectory
        from ..security.access import AllowListGate

        gate = None
        if allow is not None or deny is not None or default_allow is False:
            gate = AllowListGate(allow=allow, deny=deny, default_allow=default_allow)
        return AgentDirectory(allow_list=gate, audit=self.audit)
