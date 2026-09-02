"""Tool-call comparison for MELON's counterfactual test.

Implements the detection rule from arXiv:2502.05174 Algorithm 1:

    Alert iff  ∃c ∈ C^o_{t+1}, ∃h ∈ H_{t+1} : sim(c, h) > θ

Three properties of that rule matter and were each got wrong here before:

- It is an **all-pairs** comparison. Every original call is compared against
  every cached masked call. Pairing the two runs up by function name first
  meant a masked `send_money` was never compared against an original
  `transfer_money`, and every unpaired call scored maximum divergence — five
  of twelve benchmark cases read as clean for that reason alone.
- Similarity is **always** the embedding cosine (§3.3). Short-circuiting to an
  exact match on the identifying arguments makes the threshold inert: on the
  banking suite every comparison landed on exactly 0.0 or 1.0.
- Comparison runs on the **rendered, argument-filtered** string (A.3), not on
  the raw call, so that free-text arguments cannot dominate the vector.
"""

from __future__ import annotations

import numpy as np

from adapters.embeddings import cosine_similarity, embed, embed_many
from middleware.melon.prefilter import is_sensitive
from middleware.melon.types import MelonVerdict, ToolCall

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


def render_call(call: ToolCall) -> str:
    """`function_name(arg1 = value1, arg2 = value2)`, per A.3."""
    fields = SENSITIVE_ARG_FIELDS.get(call.name)
    if fields:
        items = [(k, call.arguments[k]) for k in fields if k in call.arguments]
    else:
        items = sorted(call.arguments.items())
    rendered = ", ".join(f"{name} = {value}" for name, value in items)
    return f"{call.name}({rendered})"


def render_calls(calls: list[ToolCall]) -> str:
    return "; ".join(render_call(c) for c in calls) if calls else NO_TOOL_CALLS


def call_similarity(left: ToolCall, right: ToolCall) -> float:
    return cosine_similarity(embed(render_call(left)), embed(render_call(right)))


def most_similar_pair(
    original_calls: list[ToolCall], masked_calls: list[ToolCall]
) -> tuple[float, ToolCall | None, ToolCall | None]:
    """Highest similarity over all (original, masked) pairs, and the pair.

    All pairs, not name-matched pairs: the masked run frequently reaches the
    same effect through a differently-named tool, and the whole point of an
    embedding comparison is to catch that.

    Every text is embedded in one batch and the pairwise scores come from a
    single matrix product. Embedding call-by-call instead made this the
    dominant cost of the entire pipeline -- one ~470ms round trip per distinct
    call, 5.5 seconds for a three-by-three comparison -- which is why the
    always-on screener looked cheap next to it and the real bottleneck was
    invisible.
    """
    if not original_calls or not masked_calls:
        return (0.0, None, None)

    original_texts = [render_call(c) for c in original_calls]
    masked_texts = [render_call(c) for c in masked_calls]
    vectors = embed_many(original_texts + masked_texts)

    left = np.array(vectors[: len(original_texts)])
    right = np.array(vectors[len(original_texts) :])
    scores = left @ right.T

    flat = int(np.argmax(scores))
    row, column = divmod(flat, scores.shape[1])
    return (float(scores[row, column]), original_calls[row], masked_calls[column])


def compare(
    original_calls: list[ToolCall],
    masked_calls: list[ToolCall],
    threshold: float = DEFAULT_THRESHOLD,
) -> MelonVerdict:
    """Compare the two runs' calls. `threshold` is a cosine *distance*."""
    similarity_threshold = 1.0 - threshold

    if not original_calls:
        return MelonVerdict(
            ran=True, verdict="safe", distance=None,
            original_calls=original_calls, masked_calls=masked_calls,
            explanation="No tool calls were made in the original run — nothing to test.",
        )

    # Only calls that can cause harm are compared. A converged read is not an
    # attack, and the masked conversation opens with its own read_file, which
    # a read in the original run would match by construction.
    sensitive_calls = [call for call in original_calls if is_sensitive(call)]
    if not sensitive_calls:
        return MelonVerdict(
            ran=True, verdict="safe", distance=None,
            original_calls=original_calls, masked_calls=masked_calls,
            explanation=(
                "No call in the original run touches a sensitive action, so "
                "convergence could not cause harm either way."
            ),
        )

    if not masked_calls:
        return MelonVerdict(
            ran=True, verdict="safe", distance=1.0,
            original_calls=original_calls, masked_calls=masked_calls,
            explanation=(
                "With the user's task removed the agent made no tool calls at "
                "all, so the action it took in the real run was driven by the "
                "task. Consistent with benign behavior."
            ),
        )

    similarity, original_match, masked_match = most_similar_pair(sensitive_calls, masked_calls)
    # Clamped: float error on identical vectors otherwise reports a
    # negative distance, which reads as a bug in any trace that shows it.
    similarity = min(1.0, max(0.0, similarity))
    distance = 1.0 - similarity
    verdict = "block" if similarity > similarity_threshold else "safe"

    if verdict == "block":
        explanation = (
            f"With the user's task removed, the agent still produced "
            f"{render_call(masked_match)}, which matches "
            f"{render_call(original_match)} from the real run "
            f"(similarity {similarity:.3f} > {similarity_threshold:.2f}). "
            "Nothing about the user's request explains that action, so it came "
            "from the tool output. Flagged as injection."
        )
    else:
        explanation = (
            f"With the user's task removed the agent's closest action was "
            f"{render_call(masked_match)}, which does not match "
            f"{render_call(original_match)} from the real run "
            f"(similarity {similarity:.3f} ≤ {similarity_threshold:.2f}). "
            "The action depended on the real task, consistent with benign behavior."
        )

    return MelonVerdict(
        ran=True, verdict=verdict, distance=distance,
        original_calls=original_calls, masked_calls=masked_calls,
        explanation=explanation,
    )
