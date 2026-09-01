"""Embedding distance on tool calls only — not full text+calls.

Fixes MELON challenge (3): the natural-language response text can differ
between runs even when the actual dangerous tool call is identical,
inflating embedding distance and causing false negatives. We embed and
compare only the tool call itself (name + arguments), never response text.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from middleware.melon.cache import align
from middleware.melon.prefilter import is_sensitive
from middleware.melon.types import MelonVerdict, ToolCall

# Cosine distance threshold for "converged" — the reference implementation
# (github.com/kaijiezhu11/MELON, pi_detector.py) flags injection at cosine
# similarity > 0.8, i.e. distance < 0.2. We use <= 0.2 as the equivalent.
DEFAULT_THRESHOLD = 0.2

# Cosine distance assigned to an original call with no content-matching
# counterpart in the masked run — the strongest possible divergence signal.
NO_MATCH_DISTANCE = 1.0

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# For a call to one of these functions, these argument fields identify
# *who/how much* the action actually affects. They're compared exactly,
# not embedded — checked empirically (see cache.py's docstring for the
# related alignment finding): a small local embedding model doesn't
# discriminate two different short email addresses reliably enough to
# trust a distance threshold on them, and fuzzing them at all is the
# wrong idea regardless of model quality — we already isolated these
# fields specifically because they're the precise, structured part of the
# call. Embedding fuzziness belongs to genuine free text (e.g. an email
# body), where we have no structured handle on what "close enough" means.
SENSITIVE_ARG_FIELDS: dict[str, tuple[str, ...]] = {
    "send_email": ("to", "recipients"),
    "send_money": ("recipient", "amount"),
    "transfer_money": ("recipient", "amount"),
}


def _identifying_fields(call: ToolCall) -> dict | None:
    fields = SENSITIVE_ARG_FIELDS.get(call.name)
    if not fields:
        return None
    selected = {k: call.arguments[k] for k in fields if k in call.arguments}
    return selected or None


def _embedding_text(call: ToolCall) -> str:
    return json.dumps({"name": call.name, "arguments": call.arguments}, sort_keys=True)


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


def embed_call(call: ToolCall) -> np.ndarray:
    model = _get_model()
    return model.encode(_embedding_text(call), normalize_embeddings=True)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def _pair_distance(original_call: ToolCall, masked_call: ToolCall | None) -> float:
    if masked_call is None:
        return NO_MATCH_DISTANCE

    original_id = _identifying_fields(original_call)
    masked_id = _identifying_fields(masked_call)
    if original_id is not None and masked_id is not None:
        return 0.0 if original_id == masked_id else NO_MATCH_DISTANCE

    return cosine_distance(embed_call(original_call), embed_call(masked_call))


def compare(
    original_calls: list[ToolCall],
    masked_calls: list[ToolCall],
    threshold: float = DEFAULT_THRESHOLD,
) -> MelonVerdict:
    if not original_calls:
        return MelonVerdict(
            ran=True,
            verdict="safe",
            distance=None,
            original_calls=original_calls,
            masked_calls=masked_calls,
            explanation="No tool calls were made in the original run — nothing to test.",
        )

    # Only calls that can actually cause harm are compared. A converged read
    # is not an attack, and the masked conversation itself opens with a
    # read_file, so including reads would match that by construction.
    sensitive_calls = [call for call in original_calls if is_sensitive(call)]
    if not sensitive_calls:
        return MelonVerdict(
            ran=True,
            verdict="safe",
            distance=None,
            original_calls=original_calls,
            masked_calls=masked_calls,
            explanation=(
                "No call in the original run touches a sensitive action, so "
                "convergence could not cause harm either way."
            ),
        )

    aligned = align(sensitive_calls, masked_calls)
    distances = [_pair_distance(original_call, masked_call) for original_call, masked_call in aligned]
    # Closest (min) match across the step's sensitive calls. MELON alerts when
    # *any* original call converges with *any* masked call, not when all of
    # them do: an agent that completes the user's real task and the injected
    # one in the same step must still be caught, and taking the worst-case
    # distance would let the benign call mask the malicious one.
    distance = min(distances)
    verdict: str = "block" if distance <= threshold else "safe"

    if verdict == "block":
        explanation = (
            "The masked run (neutral placeholder task) produced the same tool "
            f"call(s) as the original run (distance={distance:.3f} <= "
            f"threshold={threshold}). The action wasn't driven by the user's "
            "task — it converged on the tool output alone. Flagged as injection."
        )
    else:
        explanation = (
            "The masked run (neutral placeholder task) diverged from the "
            f"original run's tool call(s) (distance={distance:.3f} > "
            f"threshold={threshold}). The action depended on the real task, "
            "consistent with benign behavior."
        )

    return MelonVerdict(
        ran=True,
        verdict=verdict,
        distance=distance,
        original_calls=original_calls,
        masked_calls=masked_calls,
        explanation=explanation,
    )
