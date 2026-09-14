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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from middleware.melon.types import MelonVerdict, ToolCall
from middleware.screening import policy
from middleware.screening.alignment import check_alignment, designated_regions
from middleware.screening.labels import Integrity, Label
from middleware.screening.output_check import (
    CallVerdict,
    check_answer,
    check_call,
    taken_only_from,
    what_the_call_carries,
)
from middleware.screening.provenance import (
    call_label,
    explain_call_label,
)
from middleware.screening.redactor import RedactionResult, redact
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
    preset_regions: list[Region] | None = None,
) -> ScreenedStep:
    """Stage 1: tag, screen, redact. Call before the agent generates.

    `preset_regions` substitutes an already-labelled region list for the one
    `build_regions` would derive. It exists for `taint.TaintStore.relabel`,
    whose labels come from the session's write history rather than from the
    transcript, and which therefore cannot be expressed as a tool-name or
    author rule inside `regions.py`.
    """
    started = time.perf_counter()
    regions = preset_regions if preset_regions is not None else build_regions(
        tool_outputs, start_index=start_index, trusted_authors=trusted_authors
    )
    screen_result = _screen_if_it_can_change_anything(
        regions, task_description, judge_fn
    )
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
    alignment_results: list | None = None,
    original_response: str = "",
    answer_judge_fn: JudgeFn | None = None,
    check_response_channel: bool = False,
) -> StepResult:
    """Stage 2, escalating to Stage 3 only for the ambiguous bucket."""
    # Per-argument provenance rather than the step's joined label. The join
    # makes every call in a turn as untrusted as the most untrusted region
    # that turn depended on, even when this particular call's arguments all
    # came from the user; that is the bulk of the escalation volume, and each
    # escalation costs a second model call.
    policy_started = time.perf_counter()
    call_labels = [
        call_label(
            call.arguments, screened.regions, screened.task_description, screened.label
        )
        for call in proposed_calls
    ]
    decisions = [
        policy.check(
            call.name,
            label,
            enforce_confidentiality,
            task_description=screened.task_description,
            arguments=call.arguments,
        )
        for call, label in zip(proposed_calls, call_labels)
    ]
    verdict = _worst_verdict(decisions)
    driving = _driving_decision(decisions, verdict)
    # The calls Stage 3 will be asked about: those the policy escalated that
    # nothing since has settled.
    still_escalated = [
        call
        for call, decision in zip(proposed_calls, decisions)
        if decision.verdict == "escalate"
    ]

    # Stage 2.5: an escalation only means untrusted content reached a
    # sensitive action, which is also what a user pointing the agent at a
    # document looks like. Ask whether the call serves the request before
    # paying for a masked re-execution. Can only downgrade, never permit
    # something already blocked.
    #
    # Every escalating call is checked, not just the one that drove the
    # verdict. Clearing a step on one aligned call lets any other call in the
    # same step through with it — measured: a travel step was cleared on a
    # legitimate calendar event while the injected send_email rode along
    # beside it. A step is only cleared if nothing in it needs escalating.
    alignment = None
    if verdict == "escalate" and alignment_judge_fn is not None:
        escalating = [
            (call, decision)
            for call, decision in zip(proposed_calls, decisions)
            if decision.verdict == "escalate"
        ]

        def aligned(call: ToolCall):
            return check_alignment(
                screened.task_description,
                call.name,
                call.arguments,
                screened.regions,
                alignment_judge_fn,
            )

        calls_to_check = [call for call, _ in escalating]
        if alignment_results is not None:
            # Precomputed by the caller, concurrently with Stage 1. The
            # alignment question needs only the task, the call and the regions
            # the call's values came from — none of which depend on the
            # screener — so waiting for Stage 2 to ask it puts a whole model
            # round trip in series for no reason.
            index_of = {id(call): i for i, call in enumerate(proposed_calls)}
            results = [alignment_results[index_of[id(call)]] for call in calls_to_check]
        elif len(calls_to_check) <= 1:
            results = [aligned(call) for call in calls_to_check]
        else:
            # Independent questions about independent calls. Sequentially they
            # would put a model round trip per call onto the one budget that
            # has to stay small.
            with ThreadPoolExecutor(max_workers=min(len(calls_to_check), 8)) as pool:
                results = list(pool.map(aligned, calls_to_check))
        if results and all(r.clears_escalation for r in results):
            alignment = results[0]
            verdict = "safe"
        else:
            # A call the user's delegation covers is no evidence for the
            # counterfactual test to weigh: a masked run repeats a delegated
            # action by design. The live Session only ever hands Stage 3 the
            # one escalated call; a post-hoc episode would otherwise hand it all.
            still_escalated = [
                call
                for call, result in zip(calls_to_check, results)
                if not result.clears_escalation
            ]
            # The trace explains a call Stage 3 is asked about, not one the
            # delegation already cleared.
            driving = next(
                decision
                for call, decision in zip(proposed_calls, decisions)
                if any(call is remaining for remaining in still_escalated)
            )

    timings = StageTimings(
        screen_ms=screened.screen_ms,
        policy_ms=(time.perf_counter() - policy_started) * 1000.0,
    )

    melon_verdict: MelonVerdict | None = None
    call_checks: list[CallVerdict] = []
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
            explanation = (
                driving.explanation
                if driving
                else (
                    "This step proposed no tool calls, so there was nothing to check."
                )
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
        melon_verdict = escalate_fn(still_escalated)
        final_action, explanation = _resolve_escalation(melon_verdict, driving)
        if alignment_judge_fn is not None:
            final_action, explanation, call_checks = _second_look(
                screened,
                still_escalated,
                melon_verdict,
                driving,
                alignment_judge_fn,
                (final_action, explanation),
            )
        timings.melon_ms = (time.perf_counter() - melon_started) * 1000.0

    # Response channel. An injection whose goal is met by what the agent says
    # calls no tool, so every check above clears it -- which is why this cannot
    # be gated on the tool-call path having escalated. It asks whether the
    # answer carried out an instruction planted in content the agent read; see
    # screening/output_check.py. It can only add a block, never clear one.
    response_verdict = None
    if check_response_channel and original_response and answer_judge_fn is not None:
        response_verdict = check_answer(
            screened.task_description,
            original_response,
            screened.regions,
            answer_judge_fn,
        )
        if (
            response_verdict is not None
            and response_verdict.flagged
            and final_action == "execute"
        ):
            final_action = "block"
            explanation = response_verdict.explanation

    driving_label = (
        call_labels[decisions.index(driving)] if driving is not None else screened.label
    )
    trace = StepTrace(
        step=step,
        context_label=driving_label.to_dict(),
        policy_label=(
            driving.policy_label.to_dict() if driving else screened.label.to_dict()
        ),
        screened_regions=ScreenedRegions(
            relevant=list(screened.screen_result.relevant_ids),
            masked=list(screened.redaction.masked_ids),
            labels=labels_by_id(screened.regions),
        ),
        policy_verdict=verdict,
        melon_check=melon_verdict.to_trace_dict() if melon_verdict else None,
        response_check=response_verdict.to_trace_dict() if response_verdict else None,
        call_checks=[c.to_trace_dict() for c in call_checks] or None,
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


def _second_look(
    screened: ScreenedStep,
    calls: list[ToolCall],
    melon_verdict: MelonVerdict,
    driving: policy.PolicyDecision,
    judge_fn: JudgeFn,
    resolved: tuple[FinalAction, str],
) -> tuple[FinalAction, str, list[CallVerdict]]:
    """The planted-instruction question, asked of the calls Stage 3 ruled on.

    A masked run misleads in two ways, one per verdict:

    - It clears a call it simply declined to repeat. Measured: an injected
      "visit this link" was fetched by the real run while the masked run only
      summarized. A call that carries out an instruction planted for the
      assistant -- one naming what the call acts on -- is blocked anyway.
    - It blocks a call it repeated only because the user delegated it. When
      everything the call acts on came from the source the user pointed at,
      the masked run repeats it by design -- MELON's documented false-positive
      class -- so the block needs a planted instruction behind it too. If the
      judge finds none naming what the call acts on, the delegation explains
      the convergence and the call runs.

    A judge that fails leaves the counterfactual test's answer as it was.
    """
    task = screened.task_description
    if melon_verdict.verdict == "safe":
        checks = _planted_instruction_checks(task, calls, screened.regions, judge_fn)
        flagged = next((c for c in checks if c.flagged), None)
        if flagged is not None:
            return (
                "block",
                f"{driving.explanation} The counterfactual test did not reproduce "
                "the action, but a second check found why it happened: "
                f"{flagged.explanation}",
                checks,
            )
        return (*resolved, checks)

    reproduced = melon_verdict.reproduced_calls
    if melon_verdict.verdict != "block" or not reproduced:
        return (*resolved, [])
    designated = designated_regions(task, screened.regions)
    if not designated or not all(
        taken_only_from(what_the_call_carries(c.arguments, task), designated)
        for c in reproduced
    ):
        return (*resolved, [])
    checks = _planted_instruction_checks(task, reproduced, screened.regions, judge_fn)
    if len(checks) == len(reproduced) and all(c.judged and not c.grounded for c in checks):
        return (
            "execute",
            f"{driving.explanation} The counterfactual test repeated the action, "
            "but everything it acts on comes from the source the user pointed "
            "the agent at, and no instruction planted there for the assistant "
            "asks for it -- so the user's delegation, not an injection, is why "
            f"both runs did it. {melon_verdict.explanation}",
            checks,
        )
    return (*resolved, checks)


def _planted_instruction_checks(
    task_description: str,
    calls: list[ToolCall],
    regions: list[Region],
    judge_fn: JudgeFn,
) -> list[CallVerdict]:
    """check_call for each call, concurrently -- independent questions that
    would otherwise put a model round trip per call on the escalated path."""

    def one(call: ToolCall) -> CallVerdict | None:
        return check_call(task_description, call.name, call.arguments, regions, judge_fn)

    if len(calls) <= 1:
        found = [one(call) for call in calls]
    else:
        with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
            found = list(pool.map(one, calls))
    return [verdict for verdict in found if verdict is not None]
