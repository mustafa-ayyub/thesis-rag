import pytest
from pydantic import ValidationError

from src.settings import Settings, load_settings
from src.systems.system_b.types import AgentDecision


def test_settings_defaults_include_system_b():
    cfg = Settings()

    assert cfg.system_b.max_retrieval_calls == 3
    assert cfg.system_b.max_final_chunks == 5
    assert cfg.system_b.results_dir == "data/results/system_b"


def test_load_settings_reads_system_b_config():
    cfg = load_settings("config/settings.yaml")

    assert cfg.system_b.max_retrieval_calls == 3
    assert cfg.system_b.max_final_chunks == 5
    assert cfg.system_b.agent_max_tokens == 800


def test_agent_decision_requires_query_for_retrieve_more():
    with pytest.raises(ValidationError, match="next_query"):
        AgentDecision(decision="retrieve_more", next_query="   ")


def test_agent_decision_rejects_query_for_stop():
    with pytest.raises(ValidationError, match="next_query"):
        AgentDecision(decision="stop", next_query="query")


def test_agent_decision_allows_empty_selection_for_stop():
    decision = AgentDecision(decision="stop", reason="not answerable")

    assert decision.selected_evidence_ids == []


def test_agent_decision_coerces_string_list_fields():
    # The agent occasionally emits a single string (or null) where a list is
    # expected; it must be coerced rather than rejected (regression for Q093).
    decision = AgentDecision(
        covered="The evidence pool covers covered bonds.",
        missing="",
        decision="stop",
        reason="no_retrieval",
        selected_evidence_ids="E1",
    )

    assert decision.covered == ["The evidence pool covers covered bonds."]
    assert decision.missing == []
    assert decision.selected_evidence_ids == ["E1"]


def test_agent_decision_coerces_null_list_fields():
    decision = AgentDecision.model_validate(
        {"covered": None, "missing": None, "decision": "stop"}
    )

    assert decision.covered == []
    assert decision.missing == []
