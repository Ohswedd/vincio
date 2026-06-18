"""Typed signatures.

DSPy-style input → output signatures over the prompt AST. A signature
declares *what* a model call computes — typed inputs, typed outputs, and an
instruction — and compiles to a :class:`~vincio.prompts.templates.PromptSpec`
(and therefore to the prompt AST), so every signature is automatically a
prompt-optimization target and gets the full validation pipeline on its
outputs.

::

    class Triage(Signature):
        \"\"\"Classify a support ticket.\"\"\"

        ticket: str = InputField(desc="the raw ticket text")
        label: str = OutputField(desc="bug | billing | feature | other")
        confidence: float = OutputField()

    predict = Predict(Triage, provider=provider, model="gpt-5.2")
    result = predict(ticket="The export button 500s")
    result.label, result.confidence

String form: ``signature("ticket -> label, confidence: float")``.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, get_type_hints

from pydantic import BaseModel, Field, create_model

from ..core.errors import OutputSchemaError, PromptError
from ..core.types import ModelRequest
from ..output.schemas import OutputContract, OutputSchema
from ..output.validators import OutputValidator
from .templates import PromptSpec

__all__ = ["InputField", "OutputField", "Signature", "signature", "Predict", "PredictResult"]

_IO_KEY = "__vincio_io__"


def InputField(*, desc: str = "", default: Any = ..., **kwargs: Any) -> Any:
    """Declare a signature input field."""
    return Field(default, description=desc or None, json_schema_extra={_IO_KEY: "input"}, **kwargs)


def OutputField(*, desc: str = "", **kwargs: Any) -> Any:
    """Declare a signature output field."""
    return Field(..., description=desc or None, json_schema_extra={_IO_KEY: "output"}, **kwargs)


class Signature(BaseModel):
    """Base class for typed input → output signatures.

    Subclass with annotated fields marked :func:`InputField` /
    :func:`OutputField`; the docstring becomes the instruction.
    """

    # Per-subclass cache of the derived output model (set lazily in output_model()).
    _output_model: ClassVar[type[BaseModel] | None] = None

    @classmethod
    def _io_fields(cls, direction: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for name, info in cls.model_fields.items():
            extra = info.json_schema_extra
            marker = extra.get(_IO_KEY) if isinstance(extra, dict) else None
            if marker is None:
                # Unmarked fields default to inputs (ergonomic for plain annotations).
                marker = "input"
            if marker == direction:
                fields[name] = info
        return fields

    @classmethod
    def input_fields(cls) -> dict[str, Any]:
        return cls._io_fields("input")

    @classmethod
    def output_fields(cls) -> dict[str, Any]:
        return cls._io_fields("output")

    @classmethod
    def instructions(cls) -> str:
        return (cls.__doc__ or "").strip()

    @classmethod
    def output_model(cls) -> type[BaseModel]:
        """A Pydantic model holding only the output fields."""
        cached = cls.__dict__.get("_output_model")
        if cached is not None:
            return cached
        hints = get_type_hints(cls)
        fields = {
            name: (hints.get(name, str), Field(..., description=info.description))
            for name, info in cls.output_fields().items()
        }
        if not fields:
            raise PromptError(f"signature {cls.__name__} declares no output fields")
        model = create_model(f"{cls.__name__}Output", **fields)  # type: ignore[call-overload]
        cls._output_model = model
        return model

    @classmethod
    def output_schema(cls) -> OutputSchema:
        return OutputSchema.from_pydantic(cls.output_model(), name=cls.__name__)

    @classmethod
    def to_prompt_spec(cls, *, name: str | None = None) -> PromptSpec:
        """Compile the signature to a PromptSpec (and thus the prompt AST).

        The resulting spec is a drop-in target for the prompt optimizer:
        instruction rewrites, format selection, and example search all apply.
        """
        inputs = cls.input_fields()
        input_lines = [
            f"- {field_name}: {info.description or 'input'}" for field_name, info in inputs.items()
        ]
        output_lines = [
            f"- {field_name}: {info.description or 'output'}"
            for field_name, info in cls.output_fields().items()
        ]
        rules = []
        if input_lines:
            rules.append("Inputs:\n" + "\n".join(input_lines))
        rules.append("Produce exactly these outputs:\n" + "\n".join(output_lines))
        return PromptSpec(
            name=name or cls.__name__,
            objective=cls.instructions() or f"Compute {cls.__name__}",
            rules=rules,
            output_schema=cls.output_schema().json_schema,
            output_format="json",
        )

    @classmethod
    def render_inputs(cls, **values: Any) -> str:
        """Render and type-check input values as the user task block."""
        inputs = cls.input_fields()
        unknown = set(values) - set(inputs)
        if unknown:
            raise PromptError(
                f"unknown signature inputs {sorted(unknown)}; expected {sorted(inputs)}"
            )
        missing = [
            name for name, info in inputs.items() if info.is_required() and name not in values
        ]
        if missing:
            raise PromptError(f"missing signature inputs: {missing}")
        hints = get_type_hints(cls)
        lines: list[str] = []
        for name in inputs:
            if name not in values:
                continue
            value = values[name]
            expected = hints.get(name)
            if expected in (str, int, float, bool) and not isinstance(value, expected):
                if not (expected is float and isinstance(value, int)):
                    raise PromptError(
                        f"signature input {name!r} expected {expected.__name__}, "
                        f"got {type(value).__name__}"
                    )
            lines.append(f"{name}: {value}")
        return "\n".join(lines)


_SIG_FIELD_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*([a-zA-Z_][a-zA-Z0-9_]*))?\s*$")
_SIG_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}


def signature(spec: str, *, instructions: str = "", name: str = "signature") -> type[Signature]:
    """Build a Signature type from a DSPy-style string spec::

        QA = signature("question, context -> answer, confidence: float")
    """
    if "->" not in spec:
        raise PromptError("signature spec must contain '->' separating inputs from outputs")
    input_part, output_part = spec.split("->", 1)

    def parse(part: str, direction: str) -> dict[str, tuple[type, Any]]:
        fields: dict[str, tuple[type, Any]] = {}
        for chunk in part.split(","):
            if not chunk.strip():
                continue
            match = _SIG_FIELD_RE.match(chunk)
            if match is None:
                raise PromptError(f"invalid signature field: {chunk.strip()!r}")
            field_name, type_name = match.group(1), match.group(2) or "str"
            if type_name not in _SIG_TYPES:
                raise PromptError(
                    f"unknown signature field type {type_name!r}; known: {sorted(_SIG_TYPES)}"
                )
            marker = InputField() if direction == "input" else OutputField()
            fields[field_name] = (_SIG_TYPES[type_name], marker)
        return fields

    inputs = parse(input_part, "input")
    outputs = parse(output_part, "output")
    if not inputs or not outputs:
        raise PromptError("signature spec needs at least one input and one output")
    model = create_model(name, __base__=Signature, **{**inputs, **outputs})  # type: ignore[call-overload]
    if instructions:
        model.__doc__ = instructions
    return model


class PredictResult:
    """Typed result of a signature prediction: output fields are attributes."""

    def __init__(self, output: BaseModel, *, raw_text: str, report: Any, response: Any) -> None:
        self.output = output
        self.raw_text = raw_text
        self.report = report
        self.response = response

    def __getattr__(self, name: str) -> Any:
        return getattr(self.output, name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PredictResult({self.output!r})"


class Predict:
    """Execute a signature against a provider with full output validation.

    The signature compiles to a PromptSpec; the output schema rides the
    provider's native constrained decoding when supported and the robust
    parser + structural repair otherwise.
    """

    def __init__(
        self,
        sig: type[Signature],
        *,
        provider: Any,
        model: str,
        temperature: float = 0.0,
        prompt_spec: PromptSpec | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.sig = sig
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.spec = prompt_spec or sig.to_prompt_spec()
        self.max_output_tokens = max_output_tokens
        schema = sig.output_schema()
        self.validator = OutputValidator(OutputContract.from_schema(schema), schema=schema)

    def _request(self, **inputs: Any) -> ModelRequest:
        from .compiler import PromptCompiler

        compiled = PromptCompiler().compile(
            self.spec,
            user_task=self.sig.render_inputs(**inputs),
            provider_enforces_schema=self.provider.capabilities(self.model).structured_output,
        )
        schema = self.sig.output_schema()
        return ModelRequest(
            model=self.model,
            messages=list(compiled.messages),
            output_schema=schema.json_schema,
            output_schema_name=schema.name,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    async def acall(self, **inputs: Any) -> PredictResult:
        response = await self.provider.generate(self._request(**inputs))
        report = await self.validator.validate(response.text, structured=response.structured)
        if not report.valid:
            raise OutputSchemaError(
                f"signature {self.sig.__name__} output failed validation: "
                + "; ".join(report.errors),
            )
        output = report.output
        if not isinstance(output, BaseModel):
            output = self.sig.output_model().model_validate(output)
        return PredictResult(
            output, raw_text=response.text, report=report, response=response
        )

    def __call__(self, **inputs: Any) -> PredictResult:
        from ..providers.base import run_sync

        return run_sync(self.acall(**inputs))
