from eval.harness import CaseResult
from eval.metrics import compute_metrics
from middleware.melon.types import MelonVerdict


def _verdict(v: str) -> MelonVerdict:
    return MelonVerdict(ran=True, verdict=v, distance=0.0)


def test_attack_prevention_rate_counts_only_successful_attacks():
    results = [
        CaseResult("t1", "inj1", ground_truth_attack_succeeded=True, melon_verdict=_verdict("block")),
        CaseResult("t2", "inj1", ground_truth_attack_succeeded=True, melon_verdict=_verdict("safe")),
        CaseResult("t3", "inj1", ground_truth_attack_succeeded=False, melon_verdict=_verdict("safe")),
    ]
    report = compute_metrics(results)
    assert report.attack_cases == 3
    assert report.attacks_actually_succeeded == 2
    assert report.attacks_blocked == 1
    assert report.attack_prevention_rate == 0.5


def test_false_positive_rate_only_over_benign_cases():
    results = [
        CaseResult("t1", None, ground_truth_attack_succeeded=None, melon_verdict=_verdict("safe")),
        CaseResult("t2", None, ground_truth_attack_succeeded=None, melon_verdict=_verdict("block")),
        CaseResult("t3", "inj1", ground_truth_attack_succeeded=True, melon_verdict=_verdict("block")),
    ]
    report = compute_metrics(results)
    assert report.benign_cases == 2
    assert report.false_positive_rate == 0.5


def test_none_rates_when_no_relevant_cases():
    results = [
        CaseResult("t1", "inj1", ground_truth_attack_succeeded=False, melon_verdict=_verdict("safe")),
    ]
    report = compute_metrics(results)
    assert report.attack_prevention_rate is None  # no successful attacks to measure against
    assert report.false_positive_rate is None  # no benign cases at all


def test_empty_results():
    report = compute_metrics([])
    assert report.total_cases == 0
    assert report.attack_prevention_rate is None
    assert report.false_positive_rate is None
