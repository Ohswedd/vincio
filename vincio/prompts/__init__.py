"""Vincio prompt engine: typed prompts, AST, compiler, lint."""

from .ast import (
    DefinitionNode,
    EvidenceBlockNode,
    ExampleNode,
    MemoryBlockNode,
    ObjectiveNode,
    OutputContractNode,
    PromptAST,
    PromptNode,
    RuleNode,
    SafetyPolicyNode,
    SystemRoleNode,
    ToolResultBlockNode,
    UserTaskNode,
)
from .compiler import (
    COMPILER_VERSION,
    CompiledPrompt,
    CompilerOptions,
    PromptCompiler,
    RenderFormat,
)
from .lint import LINT_RULES, LintFinding, lint_ast, lint_spec
from .optimizers import PromptVariant, diff_rendered, diff_specs, generate_variants
from .registry import PromptRegistry, PromptVersion
from .signatures import InputField, OutputField, Predict, PredictResult, Signature, signature
from .templates import PromptSpec, PromptVariable

__all__ = [
    "PromptAST",
    "PromptNode",
    "SystemRoleNode",
    "ObjectiveNode",
    "RuleNode",
    "DefinitionNode",
    "SafetyPolicyNode",
    "OutputContractNode",
    "ExampleNode",
    "MemoryBlockNode",
    "EvidenceBlockNode",
    "ToolResultBlockNode",
    "UserTaskNode",
    "CompiledPrompt",
    "CompilerOptions",
    "PromptCompiler",
    "RenderFormat",
    "COMPILER_VERSION",
    "LintFinding",
    "LINT_RULES",
    "lint_ast",
    "lint_spec",
    "PromptRegistry",
    "PromptSpec",
    "PromptVariable",
    "PromptVariant",
    "PromptVersion",
    "generate_variants",
    "diff_specs",
    "diff_rendered",
    "Signature",
    "InputField",
    "OutputField",
    "signature",
    "Predict",
    "PredictResult",
]
