import pytest

from eval.scenarios.hand_crafted import SCENARIOS
from middleware.melon.engine import evaluate_pair


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_evaluate_pair_matches_expected_verdict(scenario):
    result = evaluate_pair(scenario.original_calls, scenario.masked_calls)
    assert result.ran is True
    assert result.verdict == scenario.expected_verdict, (
        f"{scenario.name}: expected {scenario.expected_verdict}, got "
        f"{result.verdict} (distance={result.distance})\n{scenario.description}"
    )


def test_evaluate_pair_no_original_calls_is_trivially_safe():
    result = evaluate_pair([], [])
    assert result.verdict == "safe"
    assert result.distance is None
