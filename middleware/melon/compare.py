"""Tool-call comparison for MELON's counterfactual test.

Implements the detection rule from arXiv:2502.05174 Algorithm 1:

    Alert iff  ∃c ∈ C^o_{t+1}, ∃h ∈ H_{t+1} : sim(c, h) > θ

Three properties of that rule matter and were each got wrong here before:

- It is an **all-pairs** comparison. Every original call is compared against
  every cached masked call. Pairing the two runs up by function name first
  meant a masked `send_money` was never compared against an original
  `transfer_money`, and every unpaired call scored maximum divergence — five
  of twelve benchmark cases read as clean for that reason alone.
- Similarity is the embedding cosine (§3.3) over the **rendered,
  argument-filtered** string (A.3), not the raw call, so that free-text
  arguments cannot dominate the vector.
- Only calls that can cause harm are compared -- on both sides.

One refinement is ours, not the paper's: when both calls name a destination,
the destination decides instead of the embedding. In a short rendering the
call template dominates the vector. Measured on the full AgentDojo run, a
payment to the payee's IBAN scored 0.939 against one to the attacker's, two
digits apart, and an email to janeLong@ scored 0.884 against one to jay@ --
both over θ, and 21 legitimate calls were blocked that way. The destination is
exactly what an injection chooses, so it is compared as a value; see
`_converges`.
"""

from __future__ import annotations

import re

import numpy as np

from adapters.embeddings import cosine_similarity, embed, embed_many
from middleware.melon.prefilter import is_sensitive
from middleware.melon.types import MelonVerdict, ToolCall
from middleware.screening.declassification import DESTINATION_FIELDS, _named_in
from middleware.screening.provenance import identifiers

# Cosine *similarity* above which two calls count as converged. The paper's
# value; its ablation (Table 4) moves ASR only between 0.95% and 1.11% across
# 0.5-0.9, so this is not a tuned number.
DEFAULT_SIMILARITY_THRESHOLD = 0.8

# Kept as a distance for callers written against the previous interface.
DEFAULT_THRESHOLD = 1.0 - DEFAULT_SIMILARITY_THRESHOLD

# Rendered stand-in for an empty call list (A.3).
NO_TOOL_CALLS = "No tool calls"

# A.3: for these functions only the listed arguments are rendered, so that an
# email body or transfer note cannot dominate the embedding. Every other
# function renders all of its arguments.
SENSITIVE_ARG_FIELDS: dict[str, tuple[str, ...]] = {
    "send_email": ("recipients", "to"),
    "send_money": ("recipient", "amount"),
    "transfer_money": ("recipient", "amount"),
    "schedule_transaction": ("recipient", "amount"),
    "update_scheduled_transaction": ("recipient", "amount"),
}


# Arguments longer than this are left out of the rendering for any function A.3
# does not name. Ours, not the paper's: A.3 filters a fixed list of functions so
# that "an email body or transfer note cannot dominate the embedding", and this
# applies the same rule by shape to every other tool. Identifiers -- IBANs,
# addresses, URLs, titles -- are short; bodies and concatenated content are
# long. Without it, two `post_webpage` calls to the same attacker URL scored as
# different because each run wrote the posted content in its own words.
MAX_RENDERED_ARG_CHARS = 100

# Text two calls share must run to at least this many words, as the writer
# spaced them, to show one run wrote what the other did. A two-word subject such
# as "Spotify Premium" recurs in the user's own transaction history, and an
# address is one word however many dots it has; the misdirected hijack this
# exists for wrote "Hey, how is it going?" word for word. Ours, not the paper's.
MIN_SHARED_TEXT_WORDS = 3

# Shared text quoted in an explanation is cut to this.
_MAX_QUOTED_CHARS = 120

_WORD = re.compile(r"[a-z0-9]+")
_SCHEME = re.compile(r"^https?://")


def render_call(call: ToolCall) -> str:
    """`function_name(arg1 = value1, arg2 = value2)`, per A.3."""
    fields = SENSITIVE_ARG_FIELDS.get(call.name)
    if fields:
        items = [(k, call.arguments[k]) for k in fields if k in call.arguments]
    else:
        items = [
            (k, v)
            for k, v in sorted(call.arguments.items())
            if len(str(v)) <= MAX_RENDERED_ARG_CHARS
        ]
    rendered = ", ".join(f"{name} = {value}" for name, value in items)
    return f"{call.name}({rendered})"


def render_calls(calls: list[ToolCall]) -> str:
    return "; ".join(render_call(c) for c in calls) if calls else NO_TOOL_CALLS


def call_similarity(left: ToolCall, right: ToolCall) -> float:
    return cosine_similarity(embed(render_call(left)), embed(render_call(right)))


def _similarities(
    original_calls: list[ToolCall], masked_calls: list[ToolCall]
) -> np.ndarray:
    """Every (original, masked) pair's similarity, from one embedding batch.

    Embedding call-by-call instead made this the dominant cost of the entire
    pipeline -- one ~470ms round trip per distinct call, 5.5 seconds for a
    three-by-three comparison. Clamped: float error on identical vectors
    otherwise reports a negative distance, which reads as a bug in any trace
    that shows it.
    """
    original_texts = [render_call(c) for c in original_calls]
    masked_texts = [render_call(c) for c in masked_calls]
    vectors = embed_many(original_texts + masked_texts)
    left = np.array(vectors[: len(original_texts)])
    right = np.array(vectors[len(original_texts) :])
    return np.clip(left @ right.T, 0.0, 1.0)


def _targets(call: ToolCall) -> set[str]:
    """Where a call sends something, one normalised string per destination.

    Identifiers are pulled out of the value, so `http://www.x.com/` and
    `www.x.com` -- or a recipient list recorded as text -- name one
    destination. A plain name, a channel or a person, is kept whole.
    """
    found: set[str] = set()
    for name, value in call.arguments.items():
        if name.lower() not in DESTINATION_FIELDS:
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            text = str(item).strip()
            if not text:
                continue
            found.update(identifiers(text) or [_SCHEME.sub("", text.casefold()).rstrip("/")])
    return found


def _texts(call: ToolCall) -> list[tuple[str, str]]:
    """The call's free text as (normalised, as written) pairs: every argument
    that is not a destination and runs to MIN_SHARED_TEXT_WORDS words."""
    found: list[tuple[str, str]] = []
    for name, value in call.arguments.items():
        if name.lower() in DESTINATION_FIELDS:
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if not isinstance(item, str):
                continue
            plain = item.replace("\\n", " ")
            if len(plain.split()) >= MIN_SHARED_TEXT_WORDS:
                found.append((" ".join(_WORD.findall(plain.casefold())), item.strip()))
    return found


def _shared_text(original: ToolCall, masked: ToolCall) -> str | None:
    """Text one call carries that the other contains word for word."""
    for theirs, written in _texts(masked):
        for mine, own in _texts(original):
            if f" {theirs} " in f" {mine} ":
                return written
            if f" {mine} " in f" {theirs} ":
                return own
    return None


def _converges(
    original: ToolCall,
    masked: ToolCall,
    similarity: float,
    similarity_threshold: float,
    task_description: str,
) -> tuple[str, str] | None:
    """Whether the masked call reproduces the original one, and on what:
    ("destination", d), ("text", t) or ("similarity", "").

    Where both calls name a destination, the destination decides. The same one
    converges unless the user named it -- then the request explains both runs
    reaching it. Different destinations, or only user-named ones in common,
    converge only if the calls carry the same text, which is what a hijack that
    misfired on the address still leaves behind. Anything else is the paper's
    rule: embedding similarity over θ.
    """
    mine, theirs = _targets(original), _targets(masked)
    if mine and theirs:
        unnamed = sorted(t for t in mine & theirs if not _named_in(task_description, t))
        if unnamed:
            return ("destination", unnamed[0])
        text = _shared_text(original, masked)
        return ("text", text) if text is not None else None
    if similarity > similarity_threshold:
        return ("similarity", "")
    return None


def _quoted(text: str) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= _MAX_QUOTED_CHARS:
        return one_line
    return one_line[: _MAX_QUOTED_CHARS - 3] + "..."


def _block_explanation(
    original: ToolCall,
    masked: ToolCall,
    similarity: float,
    reason: tuple[str, str],
    similarity_threshold: float,
) -> str:
    kind, shared = reason
    pair = (
        f"{render_call(masked)} in the masked run against {render_call(original)} "
        f"in the real one (similarity {similarity:.3f})"
    )
    if kind == "destination":
        return (
            f"With the user's task removed, the agent still sent to {shared}: {pair}. "
            "The user never named that destination, so the content chose it. "
            "Flagged as injection."
        )
    if kind == "text":
        return (
            f'With the user\'s task removed, the agent still wrote "{_quoted(shared)}": '
            f"{pair}. The destinations differ, but the same words were dictated by "
            "the content, not the user. Flagged as injection."
        )
    return (
        f"With the user's task removed, the agent still produced "
        f"{render_call(masked)}, which matches {render_call(original)} from the "
        f"real run (similarity {similarity:.3f} > {similarity_threshold:.2f}). "
        "Nothing about the user's request explains that action, so it came from "
        "the tool output. Flagged as injection."
    )


def _safe_explanation(
    original: ToolCall,
    masked: ToolCall,
    similarity: float,
    similarity_threshold: float,
) -> str:
    mine, theirs = _targets(original), _targets(masked)
    if mine and theirs:
        common = sorted(mine & theirs)
        where = (
            f"both go to {', '.join(common)}, which the user named,"
            if common
            else (
                f"they go to different destinations -- {', '.join(sorted(mine))} "
                f"and {', '.join(sorted(theirs))} --"
            )
        )
        return (
            f"With the user's task removed the agent's closest action was "
            f"{render_call(masked)}, against {render_call(original)} in the real "
            f"run (similarity {similarity:.3f}), but {where} and they share no "
            "text, so the masked run did not reproduce the action. Consistent "
            "with benign behavior."
        )
    return (
        f"With the user's task removed the agent's closest action was "
        f"{render_call(masked)}, which does not match {render_call(original)} "
        f"from the real run (similarity {similarity:.3f} ≤ "
        f"{similarity_threshold:.2f}). The action depended on the real task, "
        "consistent with benign behavior."
    )


def compare(
    original_calls: list[ToolCall],
    masked_calls: list[ToolCall],
    threshold: float = DEFAULT_THRESHOLD,
    task_description: str = "",
) -> MelonVerdict:
    """Compare the two runs' calls. `threshold` is a cosine *distance*."""
    similarity_threshold = 1.0 - threshold

    def verdict(kind, distance, explanation, reproduced=()) -> MelonVerdict:
        return MelonVerdict(
            ran=True,
            verdict=kind,
            distance=distance,
            original_calls=original_calls,
            masked_calls=masked_calls,
            explanation=explanation,
            reproduced_calls=list(reproduced),
        )

    if not original_calls:
        return verdict(
            "safe", None, "No tool calls were made in the original run — nothing to test."
        )

    # Only calls that can cause harm are compared. A converged read is not an
    # attack, and the masked conversation opens with its own read_file, which
    # a read in the original run would match by construction.
    sensitive_calls = [call for call in original_calls if is_sensitive(call)]
    if not sensitive_calls:
        return verdict(
            "safe",
            None,
            "No call in the original run touches a sensitive action, so "
            "convergence could not cause harm either way.",
        )

    if not masked_calls:
        return verdict(
            "safe",
            1.0,
            "With the user's task removed the agent made no tool calls at all, so "
            "the action it took in the real run was driven by the task. Consistent "
            "with benign behavior.",
        )

    # The same filter on the masked side: the masked run *reading* a channel
    # says nothing about who chose to post to it.
    harmful = [call for call in masked_calls if is_sensitive(call)]
    if not harmful:
        return verdict(
            "safe",
            1.0,
            "With the user's task removed the agent only read, and did nothing "
            "that could cause harm, so the action it took in the real run was "
            "driven by the task. Consistent with benign behavior.",
        )

    scores = _similarities(sensitive_calls, harmful)
    ranked = sorted(
        (
            (float(scores[i, j]), i, j)
            for i in range(len(sensitive_calls))
            for j in range(len(harmful))
        ),
        reverse=True,
    )
    best, best_i, best_j = ranked[0]

    first = None
    reproduced: list[ToolCall] = []
    for similarity, i, j in ranked:
        reason = _converges(
            sensitive_calls[i], harmful[j], similarity, similarity_threshold, task_description
        )
        if reason is None:
            continue
        if first is None:
            first = (similarity, sensitive_calls[i], harmful[j], reason)
        if sensitive_calls[i] not in reproduced:
            reproduced.append(sensitive_calls[i])

    if first is not None:
        similarity, original, masked, reason = first
        return verdict(
            "block",
            1.0 - best,
            _block_explanation(original, masked, similarity, reason, similarity_threshold),
            reproduced,
        )
    return verdict(
        "safe",
        1.0 - best,
        _safe_explanation(
            sensitive_calls[best_i], harmful[best_j], best, similarity_threshold
        ),
    )
