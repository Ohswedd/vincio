# Guide: add tools

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

Explicit contracts when you need them:

```python
app.tool_registry.register(
    name="crm_lookup",
    handler=crm_lookup,
    description="Look up customer CRM record",
    input_schema=CrmLookupInput,      # Pydantic model or JSON schema
    output_schema=CrmRecord,
    permissions=["crm:read"],
    side_effects="read",
    timeout_ms=10_000,
)
```

## Execution lifecycle

validate_arguments → check_permissions (RBAC scopes, ABAC rules, tenant
boundary, sensitivity scan — credentials in arguments are always denied) →
approval gate → execute (timeout enforced) → validate_output →
sanitize_output (secrets redacted; injection-looking text wrapped as
untrusted) → trace → cache.

## Write-action guardrails

Write tools require explicit permission and (when `approval_required`)
a granted approval; every write gets an **idempotency key** — replays return
the original result instead of double-executing:

```python
from vincio.tools import ToolPermissionChecker

async def approve(request):       # human-in-the-loop hook
    return await my_ui.confirm(f"{request.tool}({request.arguments})")

app.tool_runtime.permissions.approval_callback = approve
```

## Reliability scoring

```python
app.tool_registry.reliability("billing_lookup")
# {"reliability": 0.99, "avg_latency_ms": 142.0, "usefulness": 0.4, "calls": 1250}
```

## Sandboxed code execution

```python
from vincio.tools import SandboxedPython
result = await SandboxedPython(timeout_s=15).run("print(2 + 2)")
```

Runs in an isolated subprocess (`python -I`, scrubbed environment, output
caps, hard timeout).
