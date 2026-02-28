"""Direct local-LLM API calls — bypasses LiteLLM provider routing.

Supports:
- Ollama-native API (`/api/chat`)
- OpenAI-compatible chat API (`/v1/chat/completions`), e.g. llama.cpp server
"""

from __future__ import annotations

import asyncio

import httpx
from loguru import logger

from nanobot.config.constants import LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_LLM_BACKEND


async def ollama_chat(
    model_name: str,
    messages: list[dict],
    *,
    max_tokens: int = 300,
    temperature: float = 0.2,
    timeout: float = 20.0,
) -> tuple[str, dict]:
    """Call local LLM API and return (content, usage_dict).

    model_name: bare model name, e.g. "mistral" (not "ollama/mistral").
    Returns ("", {}) on failure — callers fall back gracefully.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if LOCAL_LLM_BACKEND in {"openai", "llamacpp", "llama.cpp"}:
                url = LOCAL_API_BASE.rstrip("/") + "/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }
                headers = {"Content-Type": "application/json"}
                if LOCAL_API_KEY:
                    headers["Authorization"] = f"Bearer {LOCAL_API_KEY}"
                resp = await client.post(url, json=payload, headers=headers)
            else:
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
                resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if LOCAL_LLM_BACKEND in {"openai", "llamacpp", "llama.cpp"}:
            content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
            usage_data = data.get("usage", {}) or {}
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
        else:
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
        logger.debug(
            "local_llm_chat timed out (backend={}, model={}, timeout={}s)",
            LOCAL_LLM_BACKEND, model_name, timeout,
        )
        return "", {}
    except Exception as exc:
        logger.debug(
            "local_llm_chat failed (backend={}, model={}) [{}]: {}",
            LOCAL_LLM_BACKEND, model_name, type(exc).__name__, exc,
        )
        return "", {}
