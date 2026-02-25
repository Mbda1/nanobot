"""Memory system for persistent agent memory.

Three-tier architecture:
  Hot  — MEMORY.md: always loaded, capped at MEMORY_HOT_MAX_LINES.
  Warm — memory/topics/*.md: overflow from hot tier; keyword-matched and loaded on demand.
  Cold — HISTORY.md: grep-searchable event log; never auto-loaded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.agent.usage import record as _usage_record
from nanobot.config.constants import (
    MEMORY_CHUNK_SIZE,
    MEMORY_CHUNK_MAX_TOKENS,
    MEMORY_HOT_MAX_LINES,
    MEMORY_MERGE_MAX_TOKENS,
    TIMEOUT_CHUNK_SUMMARY,
)
from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph (2-5 sentences) summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "Full updated long-term memory as markdown. Include all existing "
                        "facts plus new ones. Return unchanged if nothing new.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]

_CHUNK_SIZE = MEMORY_CHUNK_SIZE
_CHUNK_MAX_TOKENS = MEMORY_CHUNK_MAX_TOKENS
_MERGE_MAX_TOKENS = MEMORY_MERGE_MAX_TOKENS


async def _summarize_chunk(
    provider: LLMProvider,
    model: str,
    chunk_text: str,
) -> str:
    """Summarize one chunk of messages as plain text. No tool call required."""
    try:
        import asyncio
        response = await asyncio.wait_for(
            provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a memory summarizer. "
                            "Summarize the key events, decisions, and facts from this "
                            "conversation segment in 2-4 sentences. Be specific and include "
                            "dates, names, and outcomes. Return ONLY the summary."
                        ),
                    },
                    {"role": "user", "content": chunk_text},
                ],
                model=model,
                max_tokens=_CHUNK_MAX_TOKENS,
            ),
            timeout=TIMEOUT_CHUNK_SUMMARY,
        )
        _usage_record(model, getattr(response, "usage", {}), source="memory_chunk")
        return (response.content or "").strip()
    except Exception:
        logger.warning("Chunk summary failed, using raw fallback")
        return chunk_text[:300]


class MemoryStore:
    """Three-tier memory: hot (MEMORY.md) + warm (topics/*.md) + cold (HISTORY.md)."""

    # Words too common to be useful topic keywords.
    _STOP_WORDS = {
        "about", "also", "been", "does", "from", "have", "help",
        "here", "just", "like", "make", "more", "need", "some",
        "that", "their", "them", "then", "there", "this", "what",
        "when", "where", "which", "with", "would", "your",
    }

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    @property
    def topics_dir(self) -> Path:
        """Warm-tier directory (created on first overflow)."""
        return self.memory_dir / "topics"

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")
        self.trim_hot()

    def trim_hot(self) -> None:
        """Overflow oldest MEMORY.md sections to topics/ when line cap is exceeded."""
        content = self.read_long_term()
        if not content or len(content.splitlines()) <= MEMORY_HOT_MAX_LINES:
            return

        # Split into sections on ## headers, preserving the header with each section.
        raw = re.split(r"\n(?=## )", content.strip())

        if len(raw) <= 1:
            # No section headers — trim from the top, keep newest lines.
            lines = content.splitlines()
            self.memory_file.write_text("\n".join(lines[-MEMORY_HOT_MAX_LINES:]), encoding="utf-8")
            logger.info("Memory hot tier: trimmed to {} lines (no headers)", MEMORY_HOT_MAX_LINES)
            return

        topics = ensure_dir(self.topics_dir)
        overflowed = False

        while len(raw) > 1 and len("\n\n".join(raw).splitlines()) > MEMORY_HOT_MAX_LINES:
            oldest = raw.pop(0)
            overflowed = True

            # Derive a stable slug from the section header.
            m = re.match(r"## (.+)", oldest.strip())
            if m:
                # Strip trailing date stamps like "(2026-02-24 10:40)"
                base = re.sub(r"\s*\([\d\-: ]+\)", "", m.group(1)).strip()
                slug = re.sub(r"[^\w\s]", "", base.lower())
                slug = re.sub(r"\s+", "-", slug.strip())[:50] or "overflow"
            else:
                slug = "overflow"

            topic_file = topics / f"{slug}.md"
            existing = topic_file.read_text(encoding="utf-8").strip() if topic_file.exists() else ""
            sep = "\n\n" if existing else ""
            topic_file.write_text(f"{existing}{sep}{oldest.strip()}\n", encoding="utf-8")
            logger.info("Memory hot→warm: '{}' → topics/{}.md", slug, slug)

        if overflowed:
            self.memory_file.write_text("\n\n".join(raw).strip() + "\n", encoding="utf-8")

    def _load_warm_topics(self, user_message: str) -> str:
        """Return content of topic files whose stems keyword-match the user message."""
        if not user_message or not self.topics_dir.exists():
            return ""

        words = set(re.findall(r"\b[a-z]{4,}\b", user_message.lower())) - self._STOP_WORDS
        if not words:
            return ""

        loaded: list[str] = []
        for topic_file in sorted(self.topics_dir.glob("*.md")):
            # Topic keywords come from the file stem (e.g. "car-garage" → {"garage"})
            stem_words = set(re.findall(r"\b[a-z]{4,}\b", topic_file.stem.lower()))
            if words & stem_words:
                content = topic_file.read_text(encoding="utf-8").strip()
                if content:
                    loaded.append(f"### {topic_file.stem}\n{content}")
                    logger.debug("Warm memory loaded: {}", topic_file.stem)

        return "\n\n".join(loaded)

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self, user_message: str = "") -> str:
        """Return memory for this turn: hot tier always + warm tier on keyword match."""
        parts: list[str] = []

        hot = self.read_long_term()
        if hot:
            parts.append(f"## Long-term Memory\n{hot}")

        warm = self._load_warm_topics(user_message)
        if warm:
            parts.append(f"## Recalled Memory (topic match)\n\n{warm}")
            logger.info("Warm memory injected for turn")

        return "\n\n".join(parts)

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """Consolidate old messages into MEMORY.md + HISTORY.md via chunked pipeline.

        Pipeline: CHUNK (Python) → COLLECT (cloud×N) → ASSEMBLE (Python) → MERGE (cloud×1)
        Returns True on success (including no-op), False on failure.
        """
        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info("Memory consolidation (archive_all): {} messages", len(session.messages))
        else:
            keep_count = memory_window // 2
            if len(session.messages) <= keep_count:
                return True
            if len(session.messages) - session.last_consolidated <= 0:
                return True
            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return True
            logger.info("Memory consolidation: {} to consolidate, {} keep", len(old_messages), keep_count)

        # --- Build message lines (same as before) ---
        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        # --- PHASE 1: CHUNK (Python, zero tokens) ---
        chunks = [lines[i:i + _CHUNK_SIZE] for i in range(0, len(lines), _CHUNK_SIZE)]
        logger.info("Consolidation: {} messages → {} chunks", len(lines), len(chunks))

        # --- PHASE 2: COLLECT — summarize each chunk via cloud model ---
        partial_summaries: list[str] = []
        for idx, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            summary = await _summarize_chunk(provider, model, chunk_text)
            partial_summaries.append(f"[Segment {idx + 1}/{len(chunks)}] {summary}")

        # --- PHASE 3: ASSEMBLE (Python, zero tokens) ---
        assembled = "\n\n".join(partial_summaries)

        # --- PHASE 4: MERGE — one final call with save_memory tool ---
        current_memory = self.read_long_term()
        merge_prompt = f"""Update the long-term memory based on these conversation summaries.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation Summaries (assembled from {len(chunks)} segments)
{assembled}

Call save_memory with:
- history_entry: a 2-5 sentence log entry starting with [YYYY-MM-DD HH:MM]
- memory_update: the full updated MEMORY.md incorporating all new facts"""

        for attempt in range(2):
            try:
                response = await provider.chat(
                    messages=[
                        {"role": "system", "content": "You are a memory consolidation agent. You MUST call the save_memory tool — do not reply with text."},
                        {"role": "user", "content": merge_prompt},
                    ],
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    max_tokens=_MERGE_MAX_TOKENS,
                )
            except Exception:
                logger.exception("Memory consolidation failed")
                return False

            _usage_record(model, response.usage, source="memory_merge")
            if not response.has_tool_calls:
                if attempt == 0:
                    logger.warning("Memory consolidation: LLM did not call save_memory, retrying")
                    continue
                logger.warning("Memory consolidation: LLM did not call save_memory after retry, skipping")
                return False

            args = response.tool_calls[0].arguments
            # Some providers return arguments as a JSON string instead of dict
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                logger.warning("Memory consolidation: unexpected arguments type {}", type(args).__name__)
                return False

            if entry := args.get("history_entry"):
                if not isinstance(entry, str):
                    entry = json.dumps(entry, ensure_ascii=False)
                self.append_history(entry)
            if update := args.get("memory_update"):
                if not isinstance(update, str):
                    update = json.dumps(update, ensure_ascii=False)
                if update != current_memory:
                    self.write_long_term(update)

            session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
            logger.info("Memory consolidation done: {} messages, last_consolidated={}", len(session.messages), session.last_consolidated)
            return True
        return False
