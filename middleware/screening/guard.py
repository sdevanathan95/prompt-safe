"""Stages 1-2, and the escalation into Stage 3.

Exposed as two calls because RTBAS's per-step algorithm has two distinct
moments and they happen on opposite sides of the agent's own generation:

    screen_step()  -> before the agent generates, to decide what it may see
    check_calls()  -> after it proposes calls, before any of them execute

Wiring both is what makes redaction real. A pipeline that only calls
check_calls() still enforces the policy, but the agent generated from the
unredacted history, so the masking half of the defense is inert — worth
knowing when reading numbers produced that way.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from middleware.melon.types import MelonVerdict, ToolCall
from middleware.screening import policy
from middleware.screening.labels import Label
from middleware.screening.redactor import RedactionResult, redact
from middleware.screening.alignment import check_alignment
from middleware.screening.provenance import (
    call_label,
    explain_call_label,
    source_regions_for_call,
)
from middleware.screening.regions import Region, build_regions, labels_by_id
from middleware.screening.screener import JudgeFn, ScreenResult, screen
from middleware.trace.schema import FinalAction, ScreenedRegions, StepTrace

# Called only for steps the policy check could not settle. Returns Track B's
# verdict for the proposed calls.
EscalateFn = Callable[[list[ToolCall]], MelonVerdict]

# Ordering used to reduce several per-call verdicts to one step verdict: the
# most severe wins, so a single blockable call is not waved through by safe
# ones beside it.
_SEVERITY = {"safe": 0, "escalate": 1, "block": 2}


@dataclass
class StageTimings:
    """Wall-clock milliseconds per stage of one step.

    The project's cost argument is that Stage 1 is cheap and always on while
    Stage 3 is expensive and rare, so the average turn pays far less than the
    worst one. That is a claim about a distribution, and it cannot be checked
    without measuring the stages separately.
    """

    screen_ms: float = 0.0
    policy_ms: float = 0.0
    melon_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.screen_ms + self.policy_ms + self.melon_ms

    def to_dict(self) -> dict:
        return {
            "screen_ms": round(self.screen_ms, 2),
            "policy_ms": round(self.policy_ms, 2),
            "melon_ms": round(self.melon_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class StepResult:
    trace: StepTrace
    redaction: RedactionResult
    decisions: list[policy.PolicyDecision]
    melon_verdict: MelonVerdict | None
    timings: StageTimings = field(default_factory=StageTimings)


@dataclass
class ScreenedStep:
    regions: list[Region]
    screen_result: ScreenResult
    redaction: RedactionResult
    task_description: str = ""
    screen_ms: float = 0.0

    @property
    def label(self) -> Label:
        return self.screen_result.label


def screen_step(
    tool_outputs: list[tuple[str, str]],
    task_description: str,
    judge_fn: JudgeFn,
    start_index: int = 1,
    trusted_authors: frozenset[str] = frozenset(),
) -> ScreenedStep:
    """Stage 1: tag, screen, redact. Call before the agent generates."""
    started = time.perf_counter()
    regions = build_regions(
        tool_outputs, start_index=start_index, trusted_authors=trusted_authors
    )
    screen_result = _screen_if_it_can_change_anything(regions, task_description, judge_fn)
    redaction = redact(regions, screen_result.label)
    return ScreenedStep(
        regions=regions,
        screen_result=screen_result,
        redaction=redaction,
        task_description=task_description,
        screen_ms=(time.perf_counter() - started) * 1000.0,
    )


def _screen_if_it_can_change_anything(
    regions: list[Region],
    task_description: str,
    judge_fn: JudgeFn,
) -> ScreenResult:
    """Skip the judge call when its answer cannot affect the outcome.

    The dependency label is the join of the relevant regions' labels. When
    every region carries the same label, that join is that label for any
    non-empty subset the judge could name, and the redactor keeps everything
    because each region's label flows to it. The screener is then a paid model
    call whose result is already determined.

    This is not an approximation — it is the same answer, and it removes the
    always-on cost from every step whose history is uniformly labeled, which
    is most of them on suites where regions carry no author information.
    """
    if not regions:
        return screen([], task_description, judge_fn)

    distinct_labels = {region.label for region in regions}
    if len(distinct_labels) == 1:
        only = next(iter(distinct_labels))
        return ScreenResult(
            relevant_ids=[region.id for region in regions],
            label=only,
            reasoning=(
                "Every region carries the same label, so which of them the "
                "next decision depends on cannot change the outcome; the "
                "screening model call was skipped."
            ),
        )

    return screen(regions, task_description, judge_fn)


def check_calls(
    step: int,
    screened: ScreenedStep,
    proposed_calls: list[ToolCall],
    escalate_fn: EscalateFn | None = None,
    enforce_confidentiality: bool = policy.ENFORCE_CONFIDENTIALITY_BY_DEFAULT,
    alignment_judge_fn=None,
) -> StepResult:
    """Stage 2, escalating to Stage 3 only for the ambiguous bucket."""
    # Per-argument provenance rather than the step's joined label. The join
    # makes every call in a turn as untrusted as the most untrusted region
    # that turn depended on, even when this particular call's arguments all
    # came from the user; that is the bulk of the escalation volume, and each
    # escalation costs a second model call.
    policy_started = time.perf_counter()
    call_labels = [
        call_label(call.arguments, screened.regions, screened.task_description, screened.label)
        for call in proposed_calls
    ]
    decisions = [
        policy.check(call.name, label, enforce_confidentiality)
        for call, label in zip(proposed_calls, call_labels)
    ]
    verdict = _worst_verdict(decisions)
    driving = _driving_decision(decisions, verdict)

    # Stage 2.5: an escalation only means untrusted content reached a
    # sensitive action, which is also what a user pointing the agent at a
    # document looks like. Ask whether the call serves the request before
    # paying for a masked re-execution. Can only downgrade, never permit
    # something already blocked.
    alignment = None
    if verdict == "escalate" and alignment_judge_fn is not None and driving is not None:
        driving_call = proposed_calls[decisions.index(driving)]
        alignment = check_alignment(
            screened.task_description,
            driving_call.name,
            driving_call.arguments,
            source_regions_for_call(driving_call.arguments, screened.regions),
            alignment_judge_fn,
        )
        if alignment.clears_escalation:
            verdict = "safe"

    timings = StageTimings(
        screen_ms=screened.screen_ms,
        policy_ms=(time.perf_counter() - policy_started) * 1000.0,
    )

    melon_verdict: MelonVerdict | None = None
    final_action: FinalAction
    explanation: str

    if verdict == "safe":
        final_action = "execute"
        if alignment is not None and alignment.clears_escalation:
            explanation = (
                "This step depends on content that came from outside, but the "
                "user's own request pointed the agent at that content and this "
                f"action is what it asks for. {alignment.reasoning}"
            )
        else:
            explanation = driving.explanation if driving else (
                "This step proposed no tool calls, so there was nothing to check."
            )
    elif verdict == "block":
        final_action = "block"
        explanation = driving.explanation
    elif escalate_fn is None:
        # Stage 3 not wired: fall back to RTBAS's own behavior rather than
        # guessing, so a run without Track B is still sound, just costlier.
        final_action = "ask_user"
        explanation = (
            f"{driving.explanation} No automated counterfactual test was "
            "available for this run, so it falls back to asking the user."
        )
    else:
        melon_started = time.perf_counter()
        melon_verdict = escalate_fn(proposed_calls)
        timings.melon_ms = (time.perf_counter() - melon_started) * 1000.0
        final_action, explanation = _resolve_escalation(melon_verdict, driving)

    driving_label = (
        call_labels[decisions.index(driving)] if driving is not None else screened.label
    )
    trace = StepTrace(
        step=step,
        context_label=driving_label.to_dict(),
        policy_label=(driving.policy_label.to_dict() if driving else screened.label.to_dict()),
        screened_regions=ScreenedRegions(
            relevant=list(screened.screen_result.relevant_ids),
            masked=list(screened.redaction.masked_ids),
            labels=labels_by_id(screened.regions),
        ),
        policy_verdict=verdict,
        melon_check=melon_verdict.to_trace_dict() if melon_verdict else None,
        final_action=final_action,
        explanation=explanation,
    )
    if driving is not None and verdict != "safe":
        trace.explanation += " " + explain_call_label(
            proposed_calls[decisions.index(driving)].arguments,
            screened.regions,
            screened.task_description,
            screened.label,
        )

    return StepResult(trace, screened.redaction, decisions, melon_verdict, timings)


def _worst_verdict(decisions: list[policy.PolicyDecision]) -> policy.Verdict:
    if not decisions:
        return "safe"
    return max((d.verdict for d in decisions), key=lambda v: _SEVERITY[v])


def _driving_decision(
    decisions: list[policy.PolicyDecision], verdict: policy.Verdict
) -> policy.PolicyDecision | None:
    """The call responsible for the step's verdict — the one whose reasoning
    the trace should show."""
    return next((d for d in decisions if d.verdict == verdict), None)


def _resolve_escalation(
    melon_verdict: MelonVerdict, driving: policy.PolicyDecision
) -> tuple[FinalAction, str]:
    if melon_verdict.verdict == "block":
        return "block", (
            f"{driving.explanation} The counterfactual test then found the "
            f"same action still happened with the user's task removed, so the "
            f"tool output caused it, not the user. {melon_verdict.explanation}"
        )
    if melon_verdict.verdict == "safe":
        return "execute", (
            f"{driving.explanation} The counterfactual test then found the "
            f"action disappeared once the user's task was removed, so the "
            f"user's request is what drove it. {melon_verdict.explanation}"
        )
    # A verdict of None means Stage 3 ran and could not decide. This is the
    # only path that should reach a human.
    return "ask_user", (
        f"{driving.explanation} The counterfactual test was inconclusive, so "
        "this is the rare case that still needs a person to confirm."
    )
