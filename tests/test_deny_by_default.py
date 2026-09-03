"""Fail-closed policy and the ensemble counterfactual test.

Both exist to lower the miss rate, and both are regression tests for a
measured failure rather than a hypothetical one.
"""

from __future__ import annotations

from middleware.melon.masking import (
    DEFAULT_ENSEMBLE,
    MASKING_PROMPTS,
    build_masked_messages,
)
from middleware.melon.types import ToolCall
from middleware.screening.labels import Confidentiality, Integrity, Label
from middleware.screening.policy import check, is_read_only

UNTRUSTED = Label(Integrity.UNTRUSTED, Confidentiality.PUBLIC)


def test_the_tool_that_caused_every_workspace_miss_is_now_constrained():
    """create_calendar_event matched no sink pattern, so Stage 2 cleared it
    and the counterfactual test never ran. Seven of eight total misses were
    this one tool."""
    assert not is_read_only("create_calendar_event")
    assert check("create_calendar_event", UNTRUSTED).verdict == "escalate"


def test_a_tool_the_policy_has_never_heard_of_is_constrained():
    """The general form of the same bug: enumerating sinks means anything
    unnamed is permitted."""
    assert check("frobnicate_the_widget", UNTRUSTED).verdict == "escalate"
    assert check("wire_funds_offshore", UNTRUSTED).verdict == "escalate"


def test_reads_stay_unconstrained():
    """Deny-by-default is only affordable because reads are the common case
    and are recognizable by shape."""
    for tool in ("get_day_calendar_events", "read_email", "search_files", "list_files"):
        assert is_read_only(tool)
        assert check(tool, UNTRUSTED).verdict == "safe"


def test_read_shaped_names_are_matched_as_prefixes_not_substrings():
    """`delete_search_index` contains "search" but is not a read. Substring
    matching here would reintroduce the fail-open hole in the other
    direction."""
    assert not is_read_only("delete_search_index")
    assert check("delete_search_index", UNTRUSTED).verdict == "escalate"


def test_trusted_context_still_takes_the_fast_path():
    """Deny-by-default constrains which *context* a call may run from, not
    which calls may run. A call whose data came from the user is still safe,
    which is what keeps the escalation rate survivable."""
    from middleware.screening.labels import BOTTOM

    assert check("create_calendar_event", BOTTOM).verdict == "safe"
    assert check("some_unknown_tool", BOTTOM).verdict == "safe"


def test_every_ensemble_prompt_invites_instruction_following():
    """The one load-bearing property of a masking prompt. A variant that asks
    for a neutral transform without inviting the model to act on embedded
    instructions never converges, so it detects nothing."""
    for name in DEFAULT_ENSEMBLE:
        prompt = MASKING_PROMPTS[name]
        assert "follow them first" in prompt
        assert "not repeating the same tool calls" in prompt


def test_ensemble_members_ask_for_genuinely_different_tasks():
    """Identical detectors would give the ensemble no additional information
    and would just multiply cost."""
    bodies = {MASKING_PROMPTS[name] for name in DEFAULT_ENSEMBLE}
    assert len(bodies) == len(DEFAULT_ENSEMBLE)


def test_selected_masking_prompt_reaches_the_conversation():
    messages = build_masked_messages(
        "tool output", masking_prompt=MASKING_PROMPTS["translate"]
    )
    last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
    assert "translate" in last_user.lower()


def test_default_masking_prompt_is_still_the_papers():
    messages = build_masked_messages("tool output")
    last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
    assert last_user == MASKING_PROMPTS["summarize"]


def test_ensemble_pools_every_detectors_calls():
    """Pooling makes the rule "any detector converged", which is the
    aggregation that minimizes misses. One detector missing the injection no
    longer clears the step."""
    from middleware.melon.engine import run_melon_check

    injected = ToolCall("send_money", {"recipient": "attacker", "amount": 100})
    seen = []

    def agent_call_fn(messages):
        seen.append(messages)
        # Only the third detector takes the bait.
        return [injected] if len(seen) == 3 else []

    verdict = run_melon_check(
        [injected],
        "poisoned output",
        agent_call_fn,
        masking_prompts=DEFAULT_ENSEMBLE,
    )

    assert len(seen) == 4
    assert verdict.verdict == "block"


def test_ensemble_members_run_concurrently_not_in_sequence():
    """Independent masked conversations over the same content. Running them
    in sequence would make each extra detector cost a full round trip of
    latency, which is the budget that has to stay small."""
    import threading
    import time

    from middleware.melon.engine import run_melon_check

    injected = ToolCall("send_money", {"recipient": "attacker", "amount": 100})
    live = 0
    peak = 0
    lock = threading.Lock()

    def agent_call_fn(messages):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return []

    run_melon_check(
        [injected], "output", agent_call_fn, masking_prompts=DEFAULT_ENSEMBLE
    )
    assert peak > 1, "ensemble members ran sequentially"


def test_a_single_detector_does_not_pay_for_a_thread_pool():
    from middleware.melon.engine import run_melon_check

    seen = []
    run_melon_check(
        [ToolCall("send_money", {"amount": 1})],
        "output",
        lambda m: seen.append(m) or [],
        masking_prompts=("summarize",),
    )
    assert len(seen) == 1


def test_an_outbound_read_with_an_attacker_chosen_target_is_a_sink():
    """Exempting reads is only sound while the read stays local. Fetching a
    URL is a network egress: the visit is observable and anything in the path
    is exfiltrated by making it. AgentDojo's slack injection_task_3 is exactly
    this — its whole goal is get_webpage(www.true-informations.com) — and
    treating every get_ as harmless cost 6 of 9 misses on that suite."""
    from middleware.screening.policy import is_outbound_read

    assert is_outbound_read("get_webpage")
    assert not is_read_only("get_webpage")
    assert check("get_webpage", UNTRUSTED).verdict == "escalate"


def test_local_reads_are_still_exempt():
    """The distinction is where the read goes, not that it is a read."""
    from middleware.screening.policy import is_outbound_read

    for tool in ("get_channels", "read_channel_messages", "get_balance", "search_files"):
        assert not is_outbound_read(tool), tool
        assert check(tool, UNTRUSTED).verdict == "safe", tool


def test_an_outbound_read_the_user_asked_for_is_still_fine():
    """The attacker choosing the destination is what makes it a sink. A URL
    from the user's own request is not that."""
    from middleware.screening.labels import BOTTOM

    assert check("get_webpage", BOTTOM).verdict == "safe"


def test_melon_also_compares_outbound_reads():
    """The prefilter shares the definition, so a fetch the attacker chose
    reaches the counterfactual test instead of being dropped as a read."""
    from middleware.melon.prefilter import is_sensitive
    from middleware.melon.types import ToolCall

    assert is_sensitive(ToolCall("get_webpage", {"url": "www.true-informations.com"}))
    assert not is_sensitive(ToolCall("read_channel_messages", {"channel": "general"}))
