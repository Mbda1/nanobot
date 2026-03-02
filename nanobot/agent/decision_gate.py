"""Decision gate: keyword classifier that short-circuits eligible messages to the local model.

No LLM call for classification — pure Python keyword matching, zero overhead.
Conservative by design: any ambiguity routes to cloud. False-negatives (cloud when
local would work) are acceptable; false-positives (local when cloud needed) must be avoided.
"""

from __future__ import annotations

from loguru import logger

from nanobot.agent.local_llm import ollama_chat
from nanobot.agent.usage import record as _usage_record
from nanobot.config.constants import LOCAL_GATE_MAX_TOKENS, TIMEOUT_DECISION_GATE


# Keywords that signal a cloud-requiring task (tools, web, complex operations).
# If any of these appear in the lowercased message the gate passes the message to cloud.
_CLOUD_KEYWORDS: frozenset[str] = frozenset({
    # Web / search
    "search", "google", "look up", "find online", "recent", "today", "current", "latest",
    "news", "price", "stock", "weather",
    # Tool-requiring tasks
    "web", "browse", "fetch", "url", "link", "website",
    "code", "script", "write a", "implement", "program", "debug",
    "file", "read", "save", "create", "edit", "delete",
    "email", "send", "draft", "message",
    "schedule", "cron", "remind",
    # Research
    "research", "summarize", "analyze", "compare",
    # Garage skill needs tools
    "garage", "swap", "car", "camaro",
    # Obsidian/note operations
    "note", "obsidian", "vault", "tag",
    # Delegation
    "delegate", "subagent",
})


def should_use_local(message: str, history: list[dict], research_mode: str) -> bool:
    """Return True if this message is safe to handle locally (no cloud call).

    Conservative: local only when no cloud-requiring keyword is detected AND
    there is sufficient prior context (history) to answer the question.
    """
    msg = message.strip().lower()

    # Always cloud: slash commands
    if msg.startswith("/"):
        return False

    # Always cloud: research modes that expect deep/cheap web calls
    if research_mode in ("deep", "cheap"):
        return False

    # Too little signal to classify safely
    if len(msg.split()) < 2:
        return False

    # Cloud if any keyword requiring tools, web, or complex tasks appears
    if any(kw in msg for kw in _CLOUD_KEYWORDS):
        return False

    # No history → first message of session; route cloud (full context needed)
    if not history:
        return False

    return True


async def run_local(message: str, history: list[dict], local_model: str) -> str | None:
    """Run message through local model.

    Returns the response string, or None on timeout/error (caller falls through to cloud).
    """
    model_name = local_model.split("/")[-1]
    recent = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-6:]
        if m["role"] in ("user", "assistant")
    ]
    recent.append({"role": "user", "content": message})

    logger.debug("Decision gate: routing to local model ({}) for: {}", model_name, message[:80])
    content, usage = await ollama_chat(
        model_name=model_name,
        messages=recent,
        max_tokens=LOCAL_GATE_MAX_TOKENS,
        temperature=0.1,
        timeout=TIMEOUT_DECISION_GATE,
    )
    if content:
        _usage_record(local_model, usage, source="local_gate")
        return content
    return None  # fallback to cloud
