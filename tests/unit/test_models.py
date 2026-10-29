import pytest
from src.core.models import MutationCandidate, AttackLogRecord


def test_mutation_candidate():
    candidate = MutationCandidate(
        parent_skill_id="test_skill",
        mutation_rationale="test rationale",
        mutated_skill_doc="test doc",
        token_usage={"input_tokens": 100, "output_tokens": 50},
    )
    assert candidate.parent_skill_id == "test_skill"
    assert candidate.token_usage["input_tokens"] == 100

    data = candidate.to_dict()
    assert data["parent_skill_id"] == "test_skill"
    token_usage = data["token_usage"]
    assert isinstance(token_usage, dict)
    assert token_usage["output_tokens"] == 50


def test_attack_log_record():
    record = AttackLogRecord(
        round_id="round_001",
        skill_id="test_skill",
        target_model="gpt-5.4",
        cost_usd=0.05,
    )
    assert record.round_id == "round_001"
    assert record.cost_usd == 0.05

    data = record.to_dict()
    assert data["round_id"] == "round_001"
    assert data["cost_usd"] == 0.05
