"""Embedding distance on tool calls only — not full text+calls.

Fixes MELON challenge (3): the natural-language response text can differ
between runs even when the actual dangerous tool call is identical,
inflating embedding distance and causing false negatives. We embed and
compare only the tool call itself (name + arguments), never response text.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from middleware.melon.cache import align, canonical_key
from middleware.melon.types import MelonVerdict, ToolCall

# Placeholder — needs real tuning against benchmark data (see plan §7).
# Not a final value.
DEFAULT_THRESHOLD = 0.15

# Cosine distance assigned to an original call with no content-matching
# counterpart in the masked run — the strongest possible divergence signal.
NO_MATCH_DISTANCE = 1.0

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_EMBEDDING_MODEL_NAME)


def embed_call(call: ToolCall) -> np.ndarray:
    model = _get_model()
    return model.encode(canonical_key(call), normalize_embeddings=True)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


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

    aligned = align(original_calls, masked_calls)
    distances = [
        NO_MATCH_DISTANCE if masked_call is None else cosine_distance(embed_call(original_call), embed_call(masked_call))
        for original_call, masked_call in aligned
    ]
    # Worst-case (max) distance across the step's calls: any strongly
    # divergent call is evidence the real task drove the action, so we
    # only call it convergence if every call matched closely.
    distance = max(distances)
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
