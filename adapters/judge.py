"""Provider adapters for the middleware's own internal LLM calls.

The screener needs a model to answer one structured question, and forcing a
tool call is what keeps the answer well-formed. The two providers spell that
differently — OpenAI puts the system message in the message list and wraps the
schema in a "function" envelope, Anthropic takes the system message as its own
argument and uses the schema directly — so each gets a small adapter and the
middleware stays provider-agnostic.

This is the concrete content of the "works with any LLM behind an API" claim:
nothing here depends on model internals, only on tool-calling, which every
major provider exposes.
"""

from __future__ import annotations

import json
from typing import Callable

# Cheap models are the right default for the judge: it answers a narrow
# relevance question, and RTBAS's own overhead table shows the screener is the
# dominant added cost when run on a frontier model.
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def openai_judge(model: str = DEFAULT_OPENAI_JUDGE_MODEL, client=None) -> Callable:
    """A JudgeFn backed by OpenAI chat completions with a forced tool call."""
    if client is None:
        import openai

        client = openai.OpenAI()

    def judge_fn(messages: list[dict], tool_schema: dict) -> dict:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            tools=[{"type": "function", "function": tool_schema}],
            tool_choice={"type": "function", "function": {"name": tool_schema["name"]}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError(
                f"Judge model {model} returned no tool call despite tool_choice "
                "forcing one."
            )
        return json.loads(tool_calls[0].function.arguments)

    return judge_fn


def anthropic_judge(model: str = DEFAULT_ANTHROPIC_JUDGE_MODEL, client=None) -> Callable:
    """A JudgeFn backed by Anthropic messages with a forced tool call."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    def judge_fn(messages: list[dict], tool_schema: dict) -> dict:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=system or None,
            messages=turns,
            tools=[
                {
                    "name": tool_schema["name"],
                    "description": tool_schema["description"],
                    "input_schema": tool_schema["parameters"],
                }
            ],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError(
            f"Judge model {model} returned no tool_use block despite tool_choice "
            "forcing one."
        )

    return judge_fn
