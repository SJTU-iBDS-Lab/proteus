from src.core.config import ModelsConfig, RedTeamConfig


def test_models_config_parsing():
    data = {
        "red_team": {
            "mutator": "deepseek-v4-pro",
            "extractor": "deepseek-v4-pro",
            "evolver": "deepseek-v4-pro",
        },
        "blue_team": {
            "gatekeeper": "glm-5",
            "alignment": "glm-5",
            "flow_sim": "glm-5",
            "fallback": "glm-5",
        },
        "sandbox": {"target": "glm-5"},
        "aig": {"model": "glm-5"},
    }
    config = ModelsConfig.model_validate(data)
    assert config.red_team.mutator == "deepseek-v4-pro"
    assert config.blue_team.gatekeeper == "glm-5"
    assert config.sandbox.target == "glm-5"
    assert config.aig.model == "glm-5"


def test_red_team_config_parsing():
    data = {
        "max_rounds": 10,
        "max_cost_usd": 100.0,
        "max_api_calls": 200,
        "round_budget": 5.0,
        "ablation_mode": "no_knowledge",
        "cache_dir": ".test_cache",
        "stop_conditions": {
            "max_consecutive_failures": 5,
            "min_evasion_delta": 0.05,
            "budget_warn_ratio": 0.9,
        },
    }
    config = RedTeamConfig.model_validate(data)
    assert config.max_rounds == 10
    assert config.max_cost_usd == 100.0
    assert config.ablation_mode == "no_knowledge"
    assert str(config.cache_dir) == ".test_cache"
    assert config.stop_conditions["max_consecutive_failures"] == 5
