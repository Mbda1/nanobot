"""Embedding utilities for semantic warm-tier search.

ollama_embed()   — call Ollama /api/embed (nomic-embed-text by default)
cosine_similarity() — pure-Python cosine distance (no numpy required)
EmbeddingCache   — JSON cache at topics/.embeddings.json, keyed by filename+mtime
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
from loguru import logger

from nanobot.config.constants import EMBED_MODEL_DEFAULT, LOCAL_API_BASE, OLLAMA_KEEP_ALIVE, TIMEOUT_EMBED


async def ollama_embed(
    text: str,
    *,
    model: str = EMBED_MODEL_DEFAULT,
    timeout: float = TIMEOUT_EMBED,
) -> list[float]:
    """Call Ollama /api/embed and return the embedding vector.

    model: bare model name, e.g. "nomic-embed-text".
    Returns [] on any failure (Ollama down, timeout, model not pulled, etc.).
    """
    if not text or not text.strip():
        return []

    url = LOCAL_API_BASE.rstrip("/") + "/api/embed"
    payload = {"model": model, "input": text, "keep_alive": OLLAMA_KEEP_ALIVE}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Ollama /api/embed returns {"embeddings": [[...]], ...}
        embeddings = data.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        return []

    except Exception as exc:
        logger.debug("ollama_embed failed (model={}) [{}]: {}", model, type(exc).__name__, exc)
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns 0.0 if either vector is empty, zero-norm, or lengths differ.
    Pure Python — no numpy dependency.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingCache:
    """Persistent JSON cache for topic-file embeddings.

    Cache file: topics/.embeddings.json
    Key format: "{filename}:{mtime}"  (mtime as float string)
    Value: list[float] embedding vector

    Embeddings are reused as long as the file's mtime hasn't changed.
    Stale entries for the same filename (different mtime) are evicted on write.
    """

    def __init__(self, topics_dir: Path) -> None:
        self._path = topics_dir / ".embeddings.json"
        self._data: dict[str, list[float]] = self._load()

    def _load(self) -> dict[str, list[float]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._data), encoding="utf-8")
        except Exception as exc:
            logger.debug("EmbeddingCache save failed: {}", exc)

    def _key(self, topic_file: Path) -> str:
        try:
            mtime = topic_file.stat().st_mtime
        except OSError:
            mtime = 0.0
        return f"{topic_file.name}:{mtime}"

    def get(self, topic_file: Path) -> list[float] | None:
        """Return cached embedding if the file hasn't changed, else None."""
        return self._data.get(self._key(topic_file))

    def set(self, topic_file: Path, embedding: list[float]) -> None:
        """Store embedding for topic_file and persist to disk.

        Evicts any stale entries for the same filename with a different mtime.
        """
        key = self._key(topic_file)
        self._data[key] = embedding

        # Evict stale entries for this filename (different mtime)
        stale = [
            k for k in list(self._data)
            if k.split(":")[0] == topic_file.name and k != key
        ]
        for k in stale:
            del self._data[k]

        self._save()
