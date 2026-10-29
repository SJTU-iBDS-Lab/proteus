from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RedTeamModelsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    mutator: str
    extractor: str
    evolver: str


class BlueTeamModelsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    gatekeeper: str
    alignment: str
    flow_sim: str
    fallback: str


class SandboxModelsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    target: str


class AIGModelsConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str


class ModelsConfig(BaseModel):
    """Typed view of the canonical role-based ``configs/models.yaml``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    red_team: RedTeamModelsConfig
    blue_team: BlueTeamModelsConfig
    sandbox: SandboxModelsConfig
    aig: AIGModelsConfig


class RedTeamConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    max_rounds: int = 20
    max_cost_usd: float = 50.0
    max_api_calls: int = 100
    round_budget: float = 2.0
    ablation_mode: str = "full"
    cache_dir: Path = Field(default_factory=lambda: Path(".cache"))
    stop_conditions: dict[str, object] = Field(
        default_factory=lambda: {
            "max_consecutive_failures": 3,
            "min_evasion_delta": 0.01,
            "budget_warn_ratio": 0.8,
        }
    )
    models: ModelsConfig | None = None


TModel = TypeVar("TModel", bound=BaseModel)


def load_yaml_document(path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    data = cast(object, yaml.safe_load(file_path.read_text(encoding="utf-8")))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at {file_path}")
    mapping = cast(dict[object, object], data)
    return {str(key): value for key, value in mapping.items()}


def load_yaml_config(path: str | Path, model_type: type[TModel]) -> TModel:
    return model_type.model_validate(load_yaml_document(path))


def load_models_config(path: str | Path) -> ModelsConfig:
    return load_yaml_config(path, ModelsConfig)
