# Guide: the governed agent fabric (registry & discovery)

Point-to-point delegation, one agent calling another it already knows, does not
scale to an organization. As soon as agents and tool servers proliferate, you need
to **discover** them by capability and **govern** which ones are reachable. Vincio's
`vincio.registry` turns the existing A2A Agent Card into a *discoverable, governed
fabric* that spans both interop camps (A2A and **AGNTCY/ACP**) and the official
**MCP Registry**, with every resolution recorded as an access decision on the
hash-chained audit log.

This is additive; it changes nothing about how a single agent runs.

## The directory

An `AgentDirectory` indexes protocol-neutral `AgentRecord`s, normalized from A2A
Agent Cards, ACP manifests, or MCP server records, and answers capability queries.

```python
from vincio.a2a.protocol import AgentCard, AgentSkill
from vincio.registry import AgentDirectory
from vincio.security.access import AllowListGate
from vincio.security.audit import AuditLog

directory = AgentDirectory(
    allow_list=AllowListGate(allow=["researcher", "*.trusted.example"], deny=["evil*"]),
    audit=AuditLog(),
)

directory.register(
    AgentCard(
        name="researcher",
        description="web research agent",
        url="https://researcher.example",
        skills=[AgentSkill(id="research", name="research", tags=["research", "web"])],
    ),
    url="https://researcher.example",
)

# Discover by capability / tag / free text.
directory.find(tag="research")          # -> [AgentRecord(name="researcher", ...)]
directory.find(capability="research")
directory.find(query="web")
```

Or build one wired to an app's audit chain in one call:

```python
directory = app.agent_directory(allow=["researcher"], deny=["evil*"])
```

## Governed resolution

`resolve(name)` passes the name through the `AllowListGate` and records the decision
on the audit chain, whether it allows or denies. An unlisted (or explicitly denied)
agent is not reachable.

```python
record = directory.resolve("researcher")     # AccessDecision recorded as "agent_resolve" (allow)

directory.resolve("evil-bot")                 # raises AccessDeniedError; decision recorded (deny)

# Non-raising variant returns the decision:
res = directory.try_resolve("researcher")
res.allowed, res.decision.reason
```

`AllowListGate` is **fail-closed**: deny patterns evaluate first, then allow
patterns, and anything matching neither falls through to `default_allow` (False).
Patterns are fnmatch globs over the agent name or URL. It is a thin view over the
same `AccessController` the data plane uses, so a resolution decision is as
explainable and auditable as any other access check.

## Spanning both interop camps

A2A speaks JSON-RPC; **AGNTCY/ACP** is the REST-native camp. The ACP adapter maps
manifests to/from the Agent Card so one directory indexes agents from either:

```python
from vincio.registry import ACPClient, ACPAgentManifest

catalog = [ACPAgentManifest(id="planner", name="planner", capabilities=["planning"])]
await ACPClient(catalog=catalog).register_into_directory(directory)   # offline
# or: ACPClient(base_url="https://acp.example").register_into_directory(directory)
```

The **MCP Registry** discovery client resolves MCP servers under the same allow-list:

```python
from vincio.registry import MCPRegistryClient

await MCPRegistryClient(base_url="https://registry.modelcontextprotocol.io")\
    .register_into_directory(directory)        # protocol="mcp" records, gated like agents
```

Both clients accept an in-process `catalog=` for fully offline use (tests, air-gapped
deployments) and an HTTP `base_url=` for live discovery, the same in-process/HTTP
duality the MCP and A2A clients already use.

## Why this is governed by construction

- **Discovery never auto-trusts.** A discovered agent becomes an `AgentRecord`; it is
  reachable only if `resolve` passes the allow-list.
- **Every resolution is on the audit chain.** `audit.query(action="agent_resolve")`
  returns the allow/deny decisions, so the fabric is accountable end to end.
- **It is yours.** The directory is a governed catalog you run on your own
  infrastructure, never a hosted control plane.

## Gotchas

- **Fail-closed means you must allow explicitly.** `default_allow` is `False`, so
  an agent matching neither an allow nor a deny pattern is unreachable. An empty
  allow-list denies everything — the safe default, but a silent one if you forgot
  to list your agents.
- **Deny wins.** Deny patterns evaluate before allow, so `deny=["evil*"]` blocks
  `evil-researcher` even if `allow=["*"]`. Patterns are fnmatch globs over the
  agent **name or URL** — pin the URL suffix (`*.trusted.example`) when the name
  alone is spoofable.
- **`resolve` raises; `try_resolve` returns a decision.** Use `try_resolve` on a
  hot path where a denied agent is expected and you want the `AccessDecision`
  reason without an exception.
- **Discovery is not trust.** Registering a card (or importing an ACP/MCP
  catalog) only makes an agent *findable*; it becomes *reachable* only after
  `resolve` passes the gate — and every allow/deny lands on the audit chain, so
  `audit.query(action="agent_resolve")` is the accountability record.
- **Offline vs live is one switch.** Every client takes `catalog=` (in-process,
  for tests and air-gapped runs) or `base_url=` (live HTTP) — the same duality
  the MCP and A2A clients use, so a directory built in CI matches production.

See the [API reference](../reference/api.md) (`vincio.registry`) and the runnable
[`10_interop_and_protocols.py`](../../examples/10_interop_and_protocols.py) example.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Example: 10_interop_and_protocols.py](../../examples/10_interop_and_protocols.py)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#serving)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
