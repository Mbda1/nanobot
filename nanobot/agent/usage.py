"""Lightweight token-usage logger. Appends one JSON line per LLM call to USAGE.jsonl."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_log_path: Path | None = None


def init(workspace: Path) -> None:
    """Call once at startup with the workspace path."""
    global _log_path
    _log_path = workspace / "memory" / "USAGE.jsonl"
    _log_path.parent.mkdir(parents=True, exist_ok=True)


def record(model: str, usage: dict, source: str = "agent", latency_ms: int = 0) -> None:
    """Append one usage record. No-ops silently if usage is empty or logger not init'd."""
    if not _log_path or not usage or not usage.get("total_tokens"):
        return
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "source": source,          # agent | memory_chunk | memory_merge | enrichment
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0),
        "latency_ms": latency_ms,
    }
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
