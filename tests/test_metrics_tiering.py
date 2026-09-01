"""Utility and tiering metrics — the numbers added so results are meaningful.

Attack prevention on its own is not a result: a defense that stops everything
scores 100% and is useless. These cover the utility axis and the escalation
accounting the project's headline claim rests on.
"""

from __future__ import annotations

from eval.harness import CaseResult
from eval.metrics import compute_metrics
from middleware.melon.types import MelonVerdict


def case(
    task="t1", injection=None, attack_succeeded=None, task_succeeded=None,
    policy=None, action=None, melon="safe",
) -> CaseResult:
    return CaseResult(
        task, injection, attack_succeeded,
        MelonVerdict(ran=melon is not None, verdict=melon, distance=0.0),
        user_task_succeeded=task_succeeded,
        policy_verdict=policy,
        final_action=action,
    )


def test_benign_utility_and_utility_under_attack_are_separate():
    """Distinct axes: an agent can complete the user's task while also being
    hijacked, so utility under attack is not derivable from the other two."""
    results = [
        case(injection=None, task_succeeded=True),
        case(injection=None, task_succeeded=False),
        case(injection="inj1", attack_succeeded=True, task_succeeded=True),
        case(injection="inj1", attack_succeeded=True, task_succeeded=False),
        case(injection="inj1", attack_succeeded=False, task_succeeded=False),
    ]
    report = compute_metrics(results)

    assert report.benign_utility == 0.5
    assert report.utility_under_attack == 1 / 3


def test_utility_is_none_when_never_measured():
    """Results from before utility was tracked must read as absent, never as
    zero — a silent 0% would look like a catastrophic defense."""
    report = compute_metrics([case(injection=None, task_succeeded=None)])
    assert report.benign_utility is None


def test_ask_user_counts_as_stopped_but_is_tracked_separately():
    """A confirmation prompt does stop the call, but it is the cost this
    project exists to reduce, so it cannot silently count as a clean win."""
    results = [
        case(injection="inj1", attack_succeeded=True, policy="escalate", action="ask_user"),
        case(injection="inj1", attack_succeeded=True, policy="escalate", action="block"),
    ]
    report = compute_metrics(results)

    assert report.attack_prevention_rate == 1.0
    assert report.human_confirmations == 1


def test_escalation_and_auto_resolution_accounting():
    results = [
        case(policy="safe", action="execute"),
        case(injection="inj1", attack_succeeded=True, policy="escalate", action="block"),
        case(injection="inj2", attack_succeeded=False, policy="escalate", action="execute"),
        case(injection="inj3", attack_succeeded=True, policy="escalate", action="ask_user"),
    ]
    report = compute_metrics(results)

    assert report.escalation_rate == 0.75
    # Two of the three escalations were settled without a human.
    assert report.auto_resolution_rate == 2 / 3
    # And both of those were right: blocked a real attack, allowed a non-attack.
    assert report.auto_resolution_accuracy == 1.0


def test_auto_resolution_accuracy_catches_confident_wrong_answers():
    """Resolution rate without accuracy only measures willingness to guess."""
    results = [
        case(injection="inj1", attack_succeeded=True, policy="escalate", action="execute"),
        case(injection="inj2", attack_succeeded=False, policy="escalate", action="block"),
    ]
    report = compute_metrics(results)

    assert report.auto_resolution_rate == 1.0
    assert report.auto_resolution_accuracy == 0.0


def test_final_action_outranks_the_melon_verdict():
    """A step blocked at the policy check never ran MELON, so the melon
    verdict is absent — the disposition still has to be read as a block."""
    results = [case(injection=None, policy="block", action="block", melon=None)]
    assert compute_metrics(results).false_positive_rate == 1.0


def test_nothing_escalated_reports_none_not_zero():
    report = compute_metrics([case(policy="safe", action="execute")])
    assert report.escalation_rate == 0.0
    assert report.auto_resolution_rate is None


def test_defended_utility_subtracts_what_the_defense_stopped():
    """The harness scores each episode before applying a verdict, so a task
    the defense would have blocked still counts as undefended utility.
    Comparing prevention against that figure compares two different worlds and
    makes any defense look free."""
    results = [
        case(injection=None, task_succeeded=True, policy="safe", action="execute"),
        case(injection=None, task_succeeded=True, policy="escalate", action="block"),
        case(injection=None, task_succeeded=False, policy="safe", action="execute"),
    ]
    report = compute_metrics(results)

    assert report.benign_utility == 2 / 3
    assert report.defended_benign_utility == 1 / 3


def test_a_confirmation_prompt_also_costs_utility():
    """ask_user stops the call just as block does, so it cannot be scored as
    a task the defended system completed."""
    results = [case(injection=None, task_succeeded=True, policy="escalate", action="ask_user")]
    assert compute_metrics(results).defended_benign_utility == 0.0


def test_defended_utility_under_attack_is_tracked_separately():
    results = [
        case(injection="i1", attack_succeeded=True, task_succeeded=True, policy="escalate", action="block"),
        case(injection="i2", attack_succeeded=False, task_succeeded=True, policy="safe", action="execute"),
    ]
    report = compute_metrics(results)

    assert report.utility_under_attack == 1.0
    assert report.defended_utility_under_attack == 0.5
