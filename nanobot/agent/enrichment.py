"""Query enrichment: rewrite user messages with a local model before sending to cloud."""

from __future__ import annotations

import asyncio

from loguru import logger

from nanobot.config.constants import LOCAL_API_BASE, TIMEOUT_ENRICHMENT


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
    try:
        import litellm
        resp = await asyncio.wait_for(
            litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": stripped},
                ],
                api_base=LOCAL_API_BASE,
                max_tokens=300,
                temperature=0.2,
            ),
            timeout=TIMEOUT_ENRICHMENT,
        )
        enriched = resp.choices[0].message.content.strip()
        if enriched:
            logger.info("Query enriched: {} → {}", stripped[:60], enriched[:60])
            return enriched
    except asyncio.TimeoutError:
        logger.debug("Query enrichment timed out, using original")
    except Exception as exc:
        logger.debug("Query enrichment failed: {}", exc)
    return message
