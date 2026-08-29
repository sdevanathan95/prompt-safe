"""Neutral placeholder task construction and substitution.

Fixes MELON challenge (1): a blank/deleted mask causes the agent to treat
leftover tool output as few-shot examples and invent arbitrary tool calls,
producing both false negatives (invented calls miss the real match) and
false positives (invented calls coincidentally match the original run).
The fix is a fixed, neutral, tool-free placeholder task — not deletion.
"""

from __future__ import annotations

DEFAULT_PLACEHOLDER_TASK = (
    "Wait for further instructions. Do not call any tools or take any "
    "action unless explicitly asked to."
)


def build_placeholder_task() -> str:
    return DEFAULT_PLACEHOLDER_TASK


def substitute_task(history: list[dict], placeholder: str) -> list[dict]:
    """Return a copy of `history` with the user's real task replaced by
    `placeholder`. Tool output messages are left untouched — only the task
    framing changes.

    `history` is a list of {"role": ..., "content": ...} messages. The task
    is the first message with role "user"; all other messages (tool
    outputs, assistant turns) are preserved as-is.
    """
    substituted = False
    result: list[dict] = []
    for message in history:
        if not substituted and message.get("role") == "user":
            result.append({**message, "content": placeholder})
            substituted = True
        else:
            result.append(dict(message))
    return result
