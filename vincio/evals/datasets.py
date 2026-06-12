"""Eval datasets: JSONL golden datasets with tags and rubrics."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import DatasetError

__all__ = ["EvalCase", "Dataset"]


class EvalCase(BaseModel):
    id: str
    input: str | dict[str, Any]
    context: dict[str, Any] = Field(default_factory=dict)
    expected: Any = None
    rubric: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    difficulty: str = "medium"  # easy | medium | hard
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def input_text(self) -> str:
        if isinstance(self.input, str):
            return self.input
        return str(self.input.get("text") or self.input.get("input") or json.dumps(self.input))


class Dataset(BaseModel):
    name: str = "dataset"
    cases: list[EvalCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None) -> Dataset:
        path = Path(path)
        if not path.is_file():
            raise DatasetError(f"dataset file not found: {path}")
        cases: list[EvalCase] = []
        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                record.setdefault("id", f"case_{line_number:04d}")
                try:
                    cases.append(EvalCase.model_validate(record))
                except ValueError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid case: {exc}") from exc
        return cls(name=name or path.stem, cases=cases)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def filter(
        self,
        *,
        tags: list[str] | None = None,
        difficulty: str | None = None,
        ids: list[str] | None = None,
    ) -> Dataset:
        cases = [
            case
            for case in self.cases
            if (tags is None or any(tag in case.tags for tag in tags))
            and (difficulty is None or case.difficulty == difficulty)
            and (ids is None or case.id in ids)
        ]
        return Dataset(name=self.name, cases=cases, metadata=self.metadata)

    def sample(self, n: int, *, seed: int = 42) -> Dataset:
        if n >= len(self.cases):
            return self
        rng = random.Random(seed)
        return Dataset(name=f"{self.name}_sample{n}", cases=rng.sample(self.cases, n))

    def split(self, fraction: float = 0.8, *, seed: int = 42) -> tuple[Dataset, Dataset]:
        rng = random.Random(seed)
        shuffled = list(self.cases)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * fraction)
        return (
            Dataset(name=f"{self.name}_train", cases=shuffled[:cut]),
            Dataset(name=f"{self.name}_held", cases=shuffled[cut:]),
        )
