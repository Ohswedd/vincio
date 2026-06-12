"""Output schemas: typed output contracts.

:class:`OutputSchema` wraps either a Pydantic model or a raw JSON schema and
provides validation, instance parsing, and provider-ready schema dicts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..core.errors import OutputSchemaError

__all__ = ["OutputSchema", "ValidatorSpec", "RepairPolicy", "OutputContract", "SchemaRegistry"]


class OutputSchema:
    def __init__(
        self,
        *,
        name: str,
        json_schema: dict[str, Any],
        model: type[BaseModel] | None = None,
    ) -> None:
        self.name = name
        self.json_schema = json_schema
        self.model = model

    @classmethod
    def from_pydantic(cls, model: type[BaseModel], *, name: str | None = None) -> OutputSchema:
        return cls(
            name=name or model.__name__,
            json_schema=model.model_json_schema(),
            model=model,
        )

    @classmethod
    def from_json_schema(cls, schema: dict[str, Any], *, name: str = "output") -> OutputSchema:
        return cls(name=name, json_schema=schema, model=None)

    def validate(self, data: Any) -> Any:
        """Validate and coerce. Returns a model instance when a Pydantic model
        backs the schema, else the (schema-checked) raw data."""
        if self.model is not None:
            try:
                return self.model.model_validate(data)
            except ValidationError as exc:
                raise OutputSchemaError(
                    f"output does not match schema {self.name!r}: {exc.error_count()} error(s)",
                    errors=exc.errors(),
                ) from exc
        from ..tools.runtime import validate_against_schema

        errors = validate_against_schema(data, self.json_schema)
        if errors:
            raise OutputSchemaError(
                f"output does not match schema {self.name!r}", errors=errors
            )
        return data

    def is_valid(self, data: Any) -> bool:
        try:
            self.validate(data)
            return True
        except OutputSchemaError:
            return False


class ValidatorSpec(BaseModel):
    """Named semantic validator reference carried by the contract."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    blocking: bool = True


class RepairPolicy(BaseModel):
    """What repair may and may not touch."""

    allow_json_repair: bool = True
    allow_fill_optional: bool = True
    allow_type_coercion: bool = True
    allow_markdown_formatting: bool = True
    allow_llm_repair: bool = False  # model-based reserialization of malformed output
    max_repair_attempts: int = 2


class OutputContract(BaseModel):
    """The full output contract."""

    model_config = {"arbitrary_types_allowed": True}

    schema_def: dict[str, Any] | None = None
    schema_name: str = "output"
    schema_obj: Any = Field(default=None, exclude=True)  # OutputSchema (keeps the model class)
    format: Literal["json", "markdown", "text", "tool", "native"] = "text"
    validators: list[ValidatorSpec] = Field(default_factory=list)
    repair_policy: RepairPolicy = Field(default_factory=RepairPolicy)
    require_citations: bool = False

    @classmethod
    def from_schema(cls, schema: OutputSchema, **kwargs: Any) -> OutputContract:
        return cls(
            schema_def=schema.json_schema,
            schema_name=schema.name,
            schema_obj=schema,
            format=kwargs.pop("format", "json"),
            **kwargs,
        )

    def output_schema(self) -> OutputSchema | None:
        if isinstance(self.schema_obj, OutputSchema):
            return self.schema_obj
        if self.schema_def is not None:
            return OutputSchema.from_json_schema(self.schema_def, name=self.schema_name)
        return None


class SchemaRegistry:
    def __init__(self) -> None:
        self._schemas: dict[str, OutputSchema] = {}

    def register(self, schema: OutputSchema | type[BaseModel]) -> OutputSchema:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema = OutputSchema.from_pydantic(schema)
        self._schemas[schema.name] = schema
        return schema

    def get(self, name: str) -> OutputSchema:
        if name not in self._schemas:
            raise OutputSchemaError(f"schema {name!r} not registered")
        return self._schemas[name]

    def __contains__(self, name: str) -> bool:
        return name in self._schemas
