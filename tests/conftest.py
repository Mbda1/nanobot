"""Shared pytest fixtures and configuration."""

import asyncio

import httpx
import pytest


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "llm: marks tests that require a running local Ollama instance (deselect with -m 'not llm')",
    )


@pytest.fixture(scope="session")
def ollama_available():
    """Skip the test if Ollama is not reachable.

    Run the LLM eval suite explicitly:
        pytest -m llm
    Or skip it in normal runs:
        pytest -m 'not llm'   (default behaviour — fast unit tests only)
    """
    async def _ping() -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get("http://host.docker.internal:11434/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    reachable = asyncio.run(_ping())
    if not reachable:
        pytest.skip("Ollama not reachable — skipping LLM eval tests")
