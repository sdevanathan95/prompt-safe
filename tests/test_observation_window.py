"""The masked run must not be shown the outcome of the call it is testing.

MELON evaluates step t against the observations that preceded it. Handing the
masked run the finished transcript includes the result of the very action
under test, and the masking prompt ends with "make sure that you are not
repeating the same tool calls" — so a masked run shown a completed transfer
declines to repeat it, produces nothing, and the step scores as maximum
divergence. Every remaining in-scope miss on the banking suite had this shape.
"""

from __future__ import annotations

from eval.harness import _observations_before_final_decision


def _tool(name, content):
    return {
        "role": "tool",
        "content": content,
        "tool_call": type("C", (), {"function": name})(),
    }


def _assistant(*, calls):
    return {"role": "assistant", "content": "", "tool_calls": calls or None}


def test_the_decision_under_test_and_its_result_are_excluded():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant(calls=["read"]),
        _tool("get_most_recent_transactions", "history"),
        _assistant(calls=["send"]),  # <- the decision being tested
        _tool("send_money", "transfer done"),  # <- its own aftermath
    ]
    kept = _observations_before_final_decision(messages)

    names = [m["tool_call"].function for m in kept if m["role"] == "tool"]
    assert names == ["get_most_recent_transactions"]
    assert not any(
        m["role"] == "tool" and m["tool_call"].function == "send_money" for m in kept
    )


def test_earlier_decisions_and_their_results_are_kept():
    """Only the final decision is under test. Everything the agent legitimately
    observed before it is exactly the context the masked run needs."""
    messages = [
        _assistant(calls=["a"]),
        _tool("read_file", "bill"),
        _assistant(calls=["b"]),
        _tool("get_balance", "412.19"),
        _assistant(calls=["c"]),
        _tool("send_money", "done"),
    ]
    names = [
        m["tool_call"].function
        for m in _observations_before_final_decision(messages)
        if m["role"] == "tool"
    ]
    assert names == ["read_file", "get_balance"]


def test_a_transcript_with_no_tool_calls_is_returned_whole():
    messages = [{"role": "user", "content": "hi"}, _assistant(calls=None)]
    assert _observations_before_final_decision(messages) == messages
