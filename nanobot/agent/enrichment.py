"""Query enrichment: rewrite user messages with a local model before sending to cloud."""

from __future__ import annotations

from loguru import logger

from nanobot.agent.local_llm import ollama_chat
from nanobot.agent.usage import record as _usage_record
from nanobot.config.constants import TIMEOUT_ENRICHMENT


async def enrich_query(provider, model: str, message: str) -> str:
    """Rewrite user message with local Mistral before sending to cloud.

    Only triggers for substantial messages (>= 6 words). Falls back to
    the original on timeout or error. Cap at TIMEOUT_ENRICHMENT seconds.
    """
    if not model:
        return message

    # Skip commands and trivial messages
    stripped = message.strip()
    if stripped.startswith("/") or len(stripped.split()) < 6:
        return message

    system = (
        "You are a query-refinement assistant. "
        "Rewrite the user's message to be clearer and more specific "
        "so an AI assistant can give a better answer. "
        "Preserve the original intent exactly. "
        "Return ONLY the rewritten message — no explanation, no preamble."
    )

    # Strip "ollama/" prefix — ollama_chat takes the bare model name
    model_name = model.split("/")[-1]

    enriched, usage = await ollama_chat(
        model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": stripped},
        ],
        max_tokens=300,
        temperature=0.2,
        timeout=TIMEOUT_ENRICHMENT,
    )

    if enriched:
        _usage_record(model, usage, source="enrichment")
        logger.info("Query enriched: {} → {}", stripped[:60], enriched[:60])
        return enriched

    return message
