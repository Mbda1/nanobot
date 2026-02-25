"""Direct Ollama API calls — bypasses LiteLLM provider routing.

LiteLLM's provider always injects the cloud api_base into every call, so
`provider.chat(model="ollama/mistral")` ends up hitting OpenRouter instead of
the local Ollama instance. This module talks to Ollama's /api/chat endpoint
directly via httpx, avoiding that routing entirely.
"""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from nanobot.config.constants import LOCAL_API_BASE


async def ollama_chat(
    model_name: str,
    messages: list[dict],
    *,
    max_tokens: int = 300,
    temperature: float = 0.2,
    timeout: float = 20.0,
) -> tuple[str, dict]:
    """Call Ollama /api/chat and return (content, usage_dict).

    model_name: bare model name, e.g. "mistral" (not "ollama/mistral").
    Returns ("", {}) on failure — callers fall back gracefully.
    """
    url = LOCAL_API_BASE.rstrip("/") + "/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "").strip()
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return content, usage

    except asyncio.TimeoutError:
        logger.debug("ollama_chat timed out (model={}, timeout={}s)", model_name, timeout)
        return "", {}
    except Exception as exc:
        logger.debug("ollama_chat failed (model={}) [{}]: {}", model_name, type(exc).__name__, exc)
        return "", {}
