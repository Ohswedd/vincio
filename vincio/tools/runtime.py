"""Tool execution runtime.

Lifecycle: validate_arguments → check_permissions → (approval) → execute
(with timeout) → validate_output → sanitize_output → trace → cache.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from ..core.errors import (
    ToolApprovalRequiredError,
    ToolContractError,
    ToolTimeoutError,
    ToolValidationError,
)
from ..core.types import ToolCall, ToolResult, TrustLevel
from ..core.utils import stable_hash
from ..observability.traces import Tracer
from ..security.access import Principal
from ..security.injection import InjectionDetector, wrap_untrusted
from ..security.secrets import SecretScanner
from .permissions import ApprovalRequest, ToolPermissionChecker
from .registry import RegisteredTool, ToolRegistry

__all__ = ["validate_against_schema", "ToolRuntime"]


def validate_against_schema(value: Any, schema: dict[str, Any], *, path: str = "$") -> list[str]:
    """Minimal deterministic JSON-schema validation (type/required/properties/
    enum/items/bounds/pattern). Returns a list of error strings."""
    errors: list[str] = []
    if not schema:
        return errors
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "null" in schema_type and value is None:
            return errors
        schema_type = next((t for t in schema_type if t != "null"), None)
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
        return errors
    if "anyOf" in schema:
        candidate_errors = [validate_against_schema(value, option, path=path) for option in schema["anyOf"]]
        if not any(not e for e in candidate_errors):
            errors.append(f"{path}: matches no anyOf branch")
        return errors
    type_checks = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if schema_type in type_checks:
        expected = type_checks[schema_type]
        if schema_type == "integer" and isinstance(value, bool):
            errors.append(f"{path}: expected integer, got bool")
            return errors
        if not isinstance(value, expected):  # type: ignore[arg-type]
            errors.append(f"{path}: expected {schema_type}, got {type(value).__name__}")
            return errors
    if schema_type == "object" or "properties" in schema:
        if isinstance(value, dict):
            for required in schema.get("required", []):
                if required not in value:
                    errors.append(f"{path}.{required}: required property missing")
            properties = schema.get("properties", {})
            for key, item in value.items():
                if key in properties:
                    errors.extend(validate_against_schema(item, properties[key], path=f"{path}.{key}"))
                elif schema.get("additionalProperties") is False:
                    errors.append(f"{path}.{key}: additional property not allowed")
    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(validate_against_schema(item, item_schema, path=f"{path}[{index}]"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    return errors


class ToolRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_checker: ToolPermissionChecker | None = None,
        tracer: Tracer | None = None,
        cache_enabled: bool = True,
        cache_ttl_s: float = 3600.0,
        max_cache_entries: int = 4096,
        injection_detector: InjectionDetector | None = None,
        secret_scanner: SecretScanner | None = None,
        content_policy: Any = None,
        shield: Any = None,
    ) -> None:
        self.registry = registry
        self.permissions = permission_checker or ToolPermissionChecker()
        self.tracer = tracer or Tracer()
        self.cache_enabled = cache_enabled
        self.cache_ttl_s = cache_ttl_s
        self.max_cache_entries = max_cache_entries
        self._cache: dict[str, tuple[float, ToolResult]] = {}
        self.injection = injection_detector or InjectionDetector()
        self.secrets = secret_scanner or SecretScanner()
        self._idempotency_seen: dict[str, ToolResult] = {}
        # optional content-capture gate. ``None`` keeps the full output on
        # the span (the replay-faithful default); a
        # ``ContentCapturePolicy(capture=False)`` redacts/truncates — or drops —
        # tool output before it ever lands on a span the exporters read.
        self.content_policy = content_policy
        # optional behavioural shield (``vincio.verify.Shield``, duck-typed). When
        # set, a tool call is checked against the shield's ``BehaviorSpec``\\ s
        # *before* it executes; a blocked call is refused like a denied permission.
        self.shield = shield

    # -- caching -----------------------------------------------------

    def _cache_key(self, tool_name: str, arguments: dict[str, Any], principal: Principal) -> str:
        return stable_hash(
            {
                "tool": tool_name,
                "args": arguments,
                "tenant": principal.tenant_id,
                "scopes": sorted(self.permissions.access.effective_scopes(principal)),
            }
        )

    def invalidate_cache(self, tool_name: str | None = None) -> int:
        if tool_name is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        keys = [k for k, (_, result) in self._cache.items() if result.tool_name == tool_name]
        for key in keys:
            del self._cache[key]
        return len(keys)

    # -- sanitization -------------------------------------------------------------

    def _sanitize_output(self, output: Any, tool_name: str) -> tuple[Any, dict[str, Any]]:
        notes: dict[str, Any] = {}
        if isinstance(output, str):
            redacted = self.secrets.redact_text(output)
            if redacted != output:
                notes["secrets_redacted"] = True
                output = redacted
            verdict = self.injection.detect(output)
            if verdict.detected:
                notes["injection_wrapped"] = True
                notes["injection_risk"] = verdict.risk
                output = wrap_untrusted(output, source=f"tool:{tool_name}", trust=TrustLevel.UNTRUSTED_TOOL)
        return output, notes

    # -- execution ----------------------------------------------------------------

    async def execute(
        self,
        call: ToolCall,
        *,
        principal: Principal | None = None,
        resource_tenant_id: str | None = None,
        approved: bool = False,
        capability: Any = None,
    ) -> ToolResult:
        principal = principal or Principal()
        tool: RegisteredTool = self.registry.get(call.tool_name)
        spec = tool.spec

        with self.tracer.span(call.tool_name, type="tool_call") as span:
            span.set(tool=call.tool_name, arguments=call.arguments, side_effects=spec.side_effects)

            # 1. validate arguments
            errors = validate_against_schema(call.arguments, spec.input_schema)
            if errors:
                span.add_event("validation_failed", errors=errors)
                raise ToolValidationError(
                    f"invalid arguments for {call.tool_name}: {errors}", tool=call.tool_name
                )

            # 1b. pre-condition contract (a contract on behaviour, not just schema)
            if tool.contract is not None:
                pre_breaches = tool.contract.check_pre(call.arguments)
                if pre_breaches:
                    span.add_event("contract_pre_failed", breaches=pre_breaches)
                    raise ToolContractError(
                        f"tool {call.tool_name} precondition breached: {pre_breaches}",
                        tool=call.tool_name,
                        details={"breaches": pre_breaches},
                    )

            # 2. permissions
            decision = self.permissions.check(
                spec, call.arguments, principal, resource_tenant_id=resource_tenant_id,
                capability=capability,
            )
            span.set(permission_checks=decision.checks)
            if not decision.allowed:
                span.add_event("denied", reason=decision.reason)
                result = ToolResult(
                    call_id=call.id, tool_name=call.tool_name, status="denied", error=decision.reason
                )
                return result

            # 2b. behavioural shield — block a policy-violating action before it runs.
            # ``approved`` is the caller's explicit approval flag, so a spec that
            # forbids an unapproved write is enforced even for a tool the permission
            # layer would not have gated on its own.
            if self.shield is not None:
                shield_decision = self.shield.guard_tool_call(
                    call.tool_name,
                    side_effects=spec.side_effects,
                    approved=approved,
                    arguments=call.arguments,
                )
                if not shield_decision.allowed:
                    span.add_event("shield_blocked", reason=shield_decision.reason)
                    return ToolResult(
                        call_id=call.id, tool_name=call.tool_name, status="denied",
                        error=f"blocked by shield: {shield_decision.reason}",
                    )

            # 3. approval gate
            idempotency_key = self.permissions.idempotency_key(spec, call.arguments, principal)
            if decision.requires_approval and not approved:
                request = ApprovalRequest(
                    tool=call.tool_name,
                    arguments=call.arguments,
                    principal_user=principal.user_id,
                    principal_tenant=principal.tenant_id,
                    idempotency_key=idempotency_key,
                    side_effects=spec.side_effects,
                )
                granted = await self.permissions.request_approval(request)
                if not granted:
                    span.add_event("approval_required")
                    raise ToolApprovalRequiredError(
                        f"tool {call.tool_name} requires approval", tool=call.tool_name,
                        details={"idempotency_key": idempotency_key},
                    )

            # Idempotency replay for write tools.
            if spec.side_effects == "write" and idempotency_key in self._idempotency_seen:
                cached = self._idempotency_seen[idempotency_key]
                span.add_event("idempotent_replay")
                return cached.model_copy(update={"call_id": call.id, "cached": True})

            # Cache for read-only tools.
            cache_key = self._cache_key(call.tool_name, call.arguments, principal)
            if self.cache_enabled and spec.is_cacheable:
                hit = self._cache.get(cache_key)
                if hit is not None and (time.monotonic() - hit[0]) < self.cache_ttl_s:
                    span.add_event("cache_hit")
                    return hit[1].model_copy(update={"call_id": call.id, "cached": True})

            # 4. execute with timeout
            started = time.monotonic()
            try:
                if tool.is_async:
                    output = await asyncio.wait_for(
                        tool.handler(**call.arguments), timeout=spec.timeout_ms / 1000
                    )
                else:
                    output = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, lambda: tool.handler(**call.arguments)
                        ),
                        timeout=spec.timeout_ms / 1000,
                    )
            except TimeoutError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                self.registry.record_call(call.tool_name, success=False, duration_ms=duration_ms)
                span.add_event("timeout")
                raise ToolTimeoutError(
                    f"tool {call.tool_name} timed out after {spec.timeout_ms}ms", tool=call.tool_name
                ) from exc
            except Exception as exc:  # noqa: BLE001 - tool errors become results
                duration_ms = int((time.monotonic() - started) * 1000)
                self.registry.record_call(call.tool_name, success=False, duration_ms=duration_ms)
                span.add_event("error", message=str(exc))
                return ToolResult(
                    call_id=call.id,
                    tool_name=call.tool_name,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                )
            duration_ms = int((time.monotonic() - started) * 1000)

            # 5. validate output
            if isinstance(output, dict) and spec.output_schema:
                output_errors = validate_against_schema(output, spec.output_schema)
                if output_errors:
                    self.registry.record_call(call.tool_name, success=False, duration_ms=duration_ms)
                    raise ToolValidationError(
                        f"tool {call.tool_name} returned invalid output: {output_errors}",
                        tool=call.tool_name,
                    )
            if hasattr(output, "model_dump"):
                output = output.model_dump(mode="json")

            # 5b. post-condition contract — refuse an out-of-contract result
            if tool.contract is not None:
                post_breaches = tool.contract.check_post(call.arguments, output)
                if post_breaches:
                    self.registry.record_call(call.tool_name, success=False, duration_ms=duration_ms)
                    span.add_event("contract_post_failed", breaches=post_breaches)
                    raise ToolContractError(
                        f"tool {call.tool_name} postcondition breached: {post_breaches}",
                        tool=call.tool_name,
                        details={"breaches": post_breaches},
                    )

            # 6. sanitize
            output, sanitize_notes = self._sanitize_output(output, call.tool_name)

            self.registry.record_call(call.tool_name, success=True, duration_ms=duration_ms)
            # ``output`` stays truncated for the human-facing trace view;
            # ``output_full`` carries the structured (or, for strings, generously
            # capped) value so the trace-replay executor can pin a faithful tool
            # output instead of a 500-char preview.
            output_full = output if not isinstance(output, str) else output[:50_000]
            content = {"output": str(output)[:500], "output_full": output_full}
            if self.content_policy is not None:
                content = {
                    key: self.content_policy.apply(value) for key, value in content.items()
                }
                content = {key: value for key, value in content.items() if value is not None}
            span.set(duration_ms=duration_ms, **content, **sanitize_notes)

            result = ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                status="ok",
                output=output,
                duration_ms=duration_ms,
                trust_level=TrustLevel.UNTRUSTED_TOOL,
                metadata=sanitize_notes,
            )
            if self.cache_enabled and spec.is_cacheable:
                if len(self._cache) >= self.max_cache_entries:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = (time.monotonic(), result)
            if spec.side_effects == "write":
                self._idempotency_seen[idempotency_key] = result
            return result
