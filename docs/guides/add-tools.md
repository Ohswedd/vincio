# Guide: add tools

A tool is not just a Python callable you hand to a model — it is a *governed*
capability. Vincio's tool layer is a permissioned registry: every call is
checked against identity, scopes, attribute rules, tenant boundary, and data
sensitivity **in code, before it runs**, never gated on model judgment. This
guide is that layer, from a one-liner to the write guardrails and the sandbox.

## Register and enable

```python
from vincio import ContextApp

app = ContextApp(name="support_refunds")

def billing_lookup(invoice_id: str) -> dict:
    """Look up a billing record."""           # becomes the tool description
    return crm.get_invoice(invoice_id)        # schema derived from type hints

app.add_tool(billing_lookup, permissions=["billing:read"])
app.add_tool(refund_create, permissions=["billing:write"],
             side_effects="write", approval_required=True)
```

The **input schema is derived from the type hints** — parameter names, types,
and defaults become a JSON schema the model is constrained to, and the first
line of the docstring becomes the tool description the model reads. You write a
normal typed function; the contract is inferred.

Explicit contracts when inference isn't enough (rich validation, an output
contract, a pre/post-condition `contract`):

```python
app.tool_registry.register(
    handler=crm_lookup,
    name="crm_lookup",
    description="Look up customer CRM record",
    input_schema=CrmLookupInput,      # Pydantic model or JSON schema
    output_schema=CrmRecord,
    permissions=["crm:read"],
    side_effects="read",              # read | write | external
    timeout_ms=10_000,
    idempotent=False,
)
```

`side_effects` is the load-bearing field: `read` tools are cacheable and run
freely; `write` tools get idempotency keys and the approval gate; `external`
tools can be disabled wholesale by policy (`allow_external_tools`).

## How it works: the permission pipeline

Every call runs the same deterministic `ToolPermissionChecker.check(...)`
sequence before the handler is ever invoked. Each step short-circuits to a
denied `ToolPermissionDecision`, and the whole checklist lands on the trace:

```
1. RBAC scopes    — the run's principal must hold every scope in the tool's
                    permissions=[...] (wildcard-aware: "billing:*" grants
                    "billing:read")
2. ABAC rules     — evaluate(action="tool:<side_effect>", resource="tool:<name>")
                    against attribute conditions; a matching deny wins
3. Tenant boundary— the principal's tenant must match the resource tenant
4. External policy— an "external" tool is refused when allow_external is off
5. Sensitivity    — arguments are scanned for secrets; credentials passed as
                    tool arguments are ALWAYS denied
6. Capability     — (opt-in) a side-effecting tool on untrusted-tainted input
                    must present a CapabilityToken minted from the user request,
                    else it is routed to approval instead of escalating silently
7. Approval gate  — approval_required (or a failed capability check) suspends
                    the call for a human decision
```

Then, and only then: execute (timeout enforced) → `validate_output` →
`sanitize_output` (secrets redacted, injection-looking text wrapped as
untrusted) → trace → cache. Nothing about this sequence depends on what the
model said.

## Permissioned registry: RBAC and ABAC

**RBAC (scopes).** A run's principal carries scopes; a tool declares the scopes
it needs. The default run principal holds `["*"]` (unrestricted) — narrow it per
app and the checker enforces it:

```python
app.set_policy("scopes", ["billing:read", "crm:read"])   # this app can read, not write
app.run("Look up invoice INV-9", tenant_id="acme")       # refund_create now denied at check time
```

Scope matching is wildcard-aware, so a role with `billing:*` satisfies both
`billing:read` and `billing:write`. Group scopes into named roles on the app's
access controller:

```python
from vincio.security.access import Role, AccessRule

app.access.roles["billing_agent"] = Role(name="billing_agent", scopes=["billing:read"])
```

**ABAC (attribute rules).** For decisions that depend on *context* rather than a
static scope — time of day, data classification, a caller attribute — add an
`AccessRule`. Rules match on the `tool:<side_effect>` action and `tool:<name>`
resource and are evaluated lowest-`priority`-first, so a low-priority deny wins:

```python
app.access.rules.append(AccessRule(
    id="no-writes-for-contractors",
    effect="deny", priority=10,
    actions=["tool:write"], resources=["tool:*"],
    condition={"employment": "contractor"},   # matched against principal.attributes
))
```

## Write-action guardrails

Write tools require explicit permission and (when `approval_required`) a granted
approval. Every write also gets an **idempotency key** — a stable hash over
`(tool, arguments, tenant, user)` — so a replay returns the original result
instead of double-charging or double-shipping:

```python
turn_one = await app.tool_runtime.execute(call)     # refund issued, result cached by key
turn_two = await app.tool_runtime.execute(call)     # same args → cached result, no second refund
```

The approval gate is a callback returning `True`/`False`; deny is the default
when no callback is set, so a chat reply can never silently trigger a side
effect. The `ApprovalRequest` carries everything a UI needs to decide:

```python
from vincio.tools import ApprovalRequest

async def approve(request: ApprovalRequest) -> bool:   # human-in-the-loop hook
    # request.tool, request.arguments, request.principal_user, request.side_effects,
    # request.idempotency_key
    return await my_ui.confirm(f"{request.tool}({request.arguments})")

app.tool_runtime.permissions.approval_callback = approve
```

(The higher-level [Assistant](assistant.md) surfaces this same gate as pending
approvals you `chat.approve(...)`, on the identical runtime.)

## Best practice

- **Model side effects honestly.** Set `side_effects="write"` on anything that
  mutates state and `approval_required=True` on anything irreversible or costly.
  A tool mislabeled `read` skips idempotency and the approval gate.
- **Grant the narrowest scope that works,** and prefer wildcard roles
  (`billing:*`) over enumerating every scope.
- **Keep the handler pure and typed.** Type hints *are* the schema; a `dict`
  parameter gives the model no contract to fill.
- **Avoid** stuffing credentials into arguments — the sensitivity check denies
  it every time; pass secrets through the environment or an injected client.

## Gotchas

- **The default principal holds `["*"]`.** Until you `set_policy("scopes", ...)`,
  every tool passes the RBAC check — restriction is opt-in.
- **ABAC deny needs a low priority.** Rules evaluate lowest-priority-first and
  the first match wins; a deny rule at high priority can be pre-empted by an
  earlier allow.
- **Approval defaults to *deny*.** With `approval_required=True` and no
  `approval_callback`, the tool is refused, not run — wire the callback (or the
  Assistant's approval surface) before you ship.
- **Idempotency replay is per-process.** The key cache lives on the runtime, so
  a restart can re-execute; make the underlying write idempotent on the vendor
  side for hard guarantees.

## Reliability scoring

Every call feeds rolling per-tool stats, so you can retire or down-rank a flaky
tool from data instead of anecdote:

```python
app.tool_registry.reliability("billing_lookup")
# {"reliability": 0.99, "avg_latency_ms": 142.0, "usefulness": 0.4, "calls": 1250}
```

`reliability` is the success ratio, `usefulness` the measured quality lift when
the tool was used, `avg_latency_ms` the mean wall-clock cost.

## Sandboxed code execution

Tools that run generated code go through `SandboxedPython`: an isolated
subprocess (`python -I` — no env vars, no user site-packages, a throwaway temp
cwd) under conservative resource limits.

```python
from vincio.tools import SandboxedPython

result = await SandboxedPython(timeout_s=15).run("print(2 + 2)")
print(result.stdout, result.exit_code)     # "4\n", 0
```

By default the sandbox enforces `max_cpu_seconds=10`, `max_memory_bytes=512 MiB`,
and `max_open_files=64` via POSIX `setrlimit`, plus a hard wall-clock timeout and
output caps. This is **OS-process isolation, not a kernel sandbox** — for
adversarial code, pass a real isolation backend and set `require_isolation=True`
to refuse the zero-dependency subprocess path, or run the tool in a container/VM.

<!-- BEGIN GENERATED: related (vincio._docmap) -->

## Related

- [Concept: Agents & orchestration](../concepts/agents.md)
- [Guide: How-to: orchestrate multi-agent systems](orchestrate-agents.md)
- [Guide: Agent Skills](agent-skills.md)
- [Example: 04_agents_and_tools.py](../../examples/04_agents_and_tools.py)
- [Example: 05_orchestration.py](../../examples/05_orchestration.py)
- [Concept: Prompt compiler](../concepts/prompt-compiler.md)
- [Reference: capability map](../reference/capability-map.md)
- [Reference: API](../reference/api.md#runs)
- [Documentation index](../README.md)
- [Learning path](../learning-path.md)

<!-- END GENERATED: related -->
