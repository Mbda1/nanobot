"""Shared pytest fixtures and configuration."""

import asyncio

import httpx
import pytest

from nanobot.config.constants import JUDGE_MODEL_DEFAULT


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "llm: marks tests that require a running local Ollama instance (deselect with -m 'not llm')",
    )


@pytest.fixture(scope="session")
def ollama_available():
    """Skip the test if Ollama is not reachable or the judge model isn't pulled.

    Run the LLM eval suite explicitly:
        pytest -m llm
    Or skip it in normal runs:
        pytest -m 'not llm'   (default behaviour — fast unit tests only)
    """
    # Strip "ollama/" prefix to get the bare model name for comparison
    judge_model = JUDGE_MODEL_DEFAULT.split("/")[-1]

    async def _check() -> tuple[bool, bool]:
        """Returns (reachable, model_available)."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get("http://host.docker.internal:11434/api/tags")
                if r.status_code != 200:
                    return False, False
                models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
                return True, judge_model in models
        except Exception:
            return False, False

    reachable, model_available = asyncio.run(_check())
    if not reachable:
        pytest.skip("Ollama not reachable — skipping LLM eval tests")
    if not model_available:
        pytest.skip(
            f"Judge model '{judge_model}' not pulled — run: ollama pull {judge_model}"
        )
