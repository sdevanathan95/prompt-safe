"""Embedding backends for the tool-call comparison.

MELON (arXiv:2502.05174 §3.3.1) uses OpenAI's text-embedding-v3 for this
comparison, and the choice is load-bearing rather than incidental: the whole
detector is a threshold on cosine similarity between two rendered tool calls,
so a model that collapses two genuinely different calls onto nearby vectors
produces false positives directly. A small local sentence model was measured
doing exactly that here — a benign banking step scored 0.973 similarity
against a different masked call and was blocked.

The local backend stays as a no-key fallback, but it is not the paper's
configuration and should not be used to produce reported numbers.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

# The paper's model. "v3" in the paper's prose refers to this generation;
# -small is the cheap tier and is what we default to.
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# No-key fallback only. Materially weaker at separating similar-looking tool
# calls; see module docstring.
DEFAULT_LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _openai_client():
    import openai

    return openai.OpenAI()


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)


@lru_cache(maxsize=4096)
def _embed_openai(text: str, model: str) -> tuple[float, ...]:
    response = _openai_client().embeddings.create(model=model, input=text)
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    vector /= np.linalg.norm(vector) or 1.0
    return tuple(float(x) for x in vector)


@lru_cache(maxsize=4096)
def _embed_local(text: str) -> tuple[float, ...]:
    vector = _local_model().encode(text, normalize_embeddings=True)
    return tuple(float(x) for x in vector)


def openai_available() -> bool:
    """Whether to use the paper's embedding model.

    PROMPT_SAFE_EMBEDDINGS=local forces the offline backend regardless of key.
    The test suite sets it: eval.harness calls load_dotenv() at import, so
    without an override a unit test run would quietly bill embedding calls.
    """
    if os.environ.get("PROMPT_SAFE_EMBEDDINGS", "").strip().lower() == "local":
        return False
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def embed(text: str, model: str | None = None) -> np.ndarray:
    """Unit-norm embedding of one rendered tool call.

    Uses the paper's OpenAI model when a key is present, the local model
    otherwise. Cached: the same call text recurs constantly across a run's
    all-pairs comparison, and each miss is a network round trip.
    """
    if openai_available():
        return np.array(_embed_openai(text, model or DEFAULT_OPENAI_EMBEDDING_MODEL))
    return np.array(_embed_local(text))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
