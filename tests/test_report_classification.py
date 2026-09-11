"""`eval.report.response_only_injections` must not treat an empty ground-truth
list as "no tool call". AgentDojo leaves ground truth unwritten for workspace
injection_task_6 to 13, whose success check reads the environment."""

from __future__ import annotations

from eval.report import response_only_injections


def test_environment_scored_attacks_with_no_ground_truth_are_tool_attacks():
    assert response_only_injections("workspace") == set()


def test_the_one_text_scored_attack_is_still_response_only():
    assert response_only_injections("travel") == {"injection_task_6"}
