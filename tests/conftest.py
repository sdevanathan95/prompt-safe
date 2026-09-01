"""Test-wide guards against accidental network use.

eval.harness calls load_dotenv() at import time, so a real API key is present
in the environment during a unit test run. Without this the embedding
comparison in middleware.melon.compare would silently make paid OpenAI calls
from the test suite.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _force_local_embeddings():
    os.environ["PROMPT_SAFE_EMBEDDINGS"] = "local"
    yield
