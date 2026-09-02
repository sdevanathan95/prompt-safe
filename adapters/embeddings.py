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

from adapters.retry import with_retry

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


def _normalize(values) -> tuple[float, ...]:
    vector = np.array(values, dtype=np.float32)
    vector /= np.linalg.norm(vector) or 1.0
    return tuple(float(x) for x in vector)


# Process-lifetime memo shared by the single and batch paths, so a text
# embedded as part of a batch is free on a later single lookup and vice versa.
_CACHE: dict[tuple[str, str], tuple[float, ...]] = {}


def _embed_openai_many(texts: list[str], model: str) -> list[tuple[float, ...]]:
    """One request for many texts.

    The embeddings endpoint takes a list, and a round trip costs ~470ms here
    regardless of how many inputs it carries. Embedding one string at a time
    made the comparison's cost linear in the number of distinct tool calls: a
    three-by-three comparison spent 5.5 seconds, essentially all of it waiting
    on six sequential requests. Batching makes it one request.
    """
    missing = [t for t in dict.fromkeys(texts) if (model, t) not in _CACHE]
    if missing:
        response = with_retry(
            lambda: _openai_client().embeddings.create(model=model, input=missing)
        )
        for text, item in zip(missing, response.data):
            _CACHE[(model, text)] = _normalize(item.embedding)
    return [_CACHE[(model, t)] for t in texts]


def _embed_openai(text: str, model: str) -> tuple[float, ...]:
    return _embed_openai_many([text], model)[0]


@lru_cache(maxsize=4096)
def _embed_local(text: str) -> tuple[float, ...]:
    vector = _local_model().encode(text, normalize_embeddings=True)
    return tuple(float(x) for x in vector)


def _embed_local_many(texts: list[str]) -> list[tuple[float, ...]]:
    return [_embed_local(t) for t in texts]


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


def embed_many(texts: list[str], model: str | None = None) -> list[np.ndarray]:
    """Embed several texts in as few round trips as possible.

    Prefer this over repeated `embed` calls anywhere the full set of texts is
    known up front, which is every caller in this codebase.
    """
    if not texts:
        return []
    if openai_available():
        vectors = _embed_openai_many(texts, model or DEFAULT_OPENAI_EMBEDDING_MODEL)
    else:
        vectors = _embed_local_many(texts)
    return [np.array(v) for v in vectors]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
