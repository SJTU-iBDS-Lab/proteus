from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import (
    ModelsConfig,
    load_models_config,
    load_yaml_config,
    load_yaml_document,
)


def test_load_models_config_parses_role_based_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _ = config_path.write_text(
        """
red_team:
  mutator: deepseek-v4-pro
  extractor: deepseek-v4-pro
  evolver: deepseek-v4-pro
blue_team:
  gatekeeper: glm-5
  alignment: glm-5
  flow_sim: glm-5
  fallback: glm-5
sandbox:
  target: glm-5
aig:
  model: glm-5
""".strip(),
        encoding="utf-8",
    )

    config = load_models_config(config_path)

    assert config.red_team.mutator == "deepseek-v4-pro"
    assert config.blue_team.gatekeeper == "glm-5"
    assert config.sandbox.target == "glm-5"
    assert config.aig.model == "glm-5"


def test_load_yaml_document_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    _ = config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected mapping"):
        load_yaml_document(config_path)


def test_load_yaml_config_supports_generic_model_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _ = config_path.write_text(
        """
red_team:
  mutator: deepseek-v4-pro
  extractor: deepseek-v4-pro
  evolver: deepseek-v4-pro
blue_team:
  gatekeeper: glm-5
  alignment: glm-5
  flow_sim: glm-5
  fallback: glm-5
sandbox:
  target: glm-5
aig:
  model: glm-5
""".strip(),
        encoding="utf-8",
    )

    config = load_yaml_config(config_path, ModelsConfig)

    assert isinstance(config, ModelsConfig)
    assert config.red_team.mutator == "deepseek-v4-pro"


def test_repo_bootstrap_models_yaml_matches_config_schema() -> None:
    config = load_models_config(Path("configs/models.yaml"))

    assert config.red_team.mutator == "deepseek-v4-pro"
    assert config.blue_team.fallback == "glm-5"
    assert config.sandbox.target == "glm-5"
    assert config.aig.model == "glm-5"
