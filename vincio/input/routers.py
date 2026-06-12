"""Input router: converts raw input into structured task state.

Pipeline: normalize → language detect → classify task → classify files →
PII/secret pre-scan → injection detection → scope resolution → ambiguity
detection → trust tagging. The result is a :class:`RoutedInput` consumed by
the runtime.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.types import Objective, TaskType, TrustLevel, UserInput
from ..security.injection import InjectionDetector, InjectionVerdict
from ..security.pii import PIIDetector, PIIMatch
from ..security.secrets import SecretFinding, SecretScanner
from .classifiers import (
    AmbiguityReport,
    TaskClassification,
    classify_file,
    classify_task,
    detect_ambiguity,
)
from .normalizers import detect_language, normalize_text

__all__ = ["RoutedInput", "InputRouter"]


class RoutedInput(BaseModel):
    input: UserInput
    objective: Objective
    task: TaskClassification
    language: str = "en"
    file_kinds: dict[str, str] = Field(default_factory=dict)
    pii_matches: list[PIIMatch] = Field(default_factory=list)
    secret_findings: list[SecretFinding] = Field(default_factory=list)
    injection: InjectionVerdict | None = None
    ambiguity: AmbiguityReport | None = None
    trust_level: TrustLevel = TrustLevel.USER
    metadata: dict[str, Any] = Field(default_factory=dict)


class InputRouter:
    def __init__(
        self,
        *,
        pii_detector: PIIDetector | None = None,
        secret_scanner: SecretScanner | None = None,
        injection_detector: InjectionDetector | None = None,
        default_objective: str = "Assist the user with their request",
    ) -> None:
        self.pii = pii_detector or PIIDetector()
        self.secrets = secret_scanner or SecretScanner()
        self.injection = injection_detector or InjectionDetector()
        self.default_objective = default_objective

    def route(
        self,
        raw: UserInput | str,
        *,
        objective: Objective | str | None = None,
        has_sources: bool = False,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> RoutedInput:
        user_input = UserInput(text=raw) if isinstance(raw, str) else raw.model_copy(deep=True)

        # 8. scope resolution: explicit args override, input fields otherwise.
        if tenant_id is not None:
            user_input.tenant_id = tenant_id
        if user_id is not None:
            user_input.user_id = user_id

        # 1. normalize
        text = normalize_text(user_input.text or "")
        user_input.text = text

        # 2. language
        language = user_input.locale or detect_language(text)
        user_input.locale = user_input.locale or language

        # 3-4. intent / task routing
        task = classify_task(user_input, has_sources=has_sources)

        # objective resolution
        if isinstance(objective, Objective):
            resolved_objective = objective.model_copy(deep=True)
            if resolved_objective.task_type == TaskType.GENERAL:
                resolved_objective.task_type = task.task_type
        elif isinstance(objective, str) and objective:
            resolved_objective = Objective(text=objective, task_type=task.task_type)
        else:
            resolved_objective = Objective(
                text=text[:200] or self.default_objective, task_type=task.task_type
            )

        # 5. file/media classification
        file_kinds = {f.path: classify_file(f) for f in user_input.files}
        for image in user_input.images:
            if image.path:
                file_kinds[image.path] = "image"
        for audio in user_input.audio:
            if audio.path:
                file_kinds[audio.path] = "audio"

        # 6. PII/secrets pre-scan
        pii_matches = self.pii.detect(text)
        secret_findings = self.secrets.scan_text(text)

        # 7. injection detection (user text is trusted to instruct, but the
        # verdict is recorded so policies can escalate in strict mode).
        injection = self.injection.detect(text)

        # 9. ambiguity detection
        ambiguity = detect_ambiguity(user_input)

        return RoutedInput(
            input=user_input,
            objective=resolved_objective,
            task=task,
            language=language,
            file_kinds=file_kinds,
            pii_matches=pii_matches,
            secret_findings=secret_findings,
            injection=injection,
            ambiguity=ambiguity,
            trust_level=TrustLevel.USER,
        )
