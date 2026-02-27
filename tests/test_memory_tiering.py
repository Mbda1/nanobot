"""Tests for the three-tier memory system.

Hot  — MEMORY.md, always loaded, capped at MEMORY_HOT_MAX_LINES.
Warm — memory/topics/*.md, keyword-triggered.
Cold — HISTORY.md, never auto-loaded (existing behaviour, unchanged).
"""

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.config.constants import MEMORY_HOT_MAX_LINES


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path)


def _make_memory(n_sections: int, lines_per_section: int = 5) -> str:
    """Build a MEMORY.md string with n_sections × lines_per_section lines."""
    sections = []
    for i in range(n_sections):
        header = f"## Section {i}"
        body = "\n".join(f"- Fact {i}.{j}" for j in range(lines_per_section - 1))
        sections.append(f"{header}\n{body}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# trim_hot — overflow tests
# ---------------------------------------------------------------------------

def test_trim_hot_no_op_under_limit(tmp_path):
    """MEMORY.md under the cap is not modified."""
    store = _store(tmp_path)
    content = _make_memory(n_sections=3, lines_per_section=5)
    assert len(content.splitlines()) < MEMORY_HOT_MAX_LINES

    store.memory_file.write_text(content, encoding="utf-8")
    store.trim_hot()

    assert store.memory_file.read_text(encoding="utf-8") == content
    assert not store.topics_dir.exists()


def test_trim_hot_overflows_oldest_section(tmp_path):
    """When MEMORY.md exceeds the cap the oldest ## section moves to topics/."""
    store = _store(tmp_path)

    # Build content big enough to exceed the cap
    lines_per = max(10, MEMORY_HOT_MAX_LINES // 5 + 1)
    content = _make_memory(n_sections=6, lines_per_section=lines_per)
    assert len(content.splitlines()) > MEMORY_HOT_MAX_LINES

    store.write_long_term(content)  # triggers trim_hot internally

    remaining = store.read_long_term()
    assert len(remaining.splitlines()) <= MEMORY_HOT_MAX_LINES
    # At least one topic file should have been created
    assert store.topics_dir.exists()
    topic_files = list(store.topics_dir.glob("*.md"))
    assert len(topic_files) >= 1


def test_trim_hot_oldest_section_first(tmp_path):
    """The FIRST section (oldest) is the one overflowed, not the newest."""
    store = _store(tmp_path)

    lines_per = max(10, MEMORY_HOT_MAX_LINES // 5 + 1)
    content = _make_memory(n_sections=6, lines_per_section=lines_per)
    store.write_long_term(content)

    # "Section 0" is the oldest (first); it should appear in topics/
    topic_contents = "".join(
        f.read_text(encoding="utf-8") for f in store.topics_dir.glob("*.md")
    )
    assert "Section 0" in topic_contents

    # "Section 5" is the newest; it should still be in MEMORY.md
    hot = store.read_long_term()
    assert "Section 5" in hot


def test_trim_hot_no_headers_trims_from_top(tmp_path):
    """MEMORY.md without ## headers is truncated to the last N lines."""
    store = _store(tmp_path)

    lines = [f"fact {i}" for i in range(MEMORY_HOT_MAX_LINES + 50)]
    store.memory_file.write_text("\n".join(lines), encoding="utf-8")
    store.trim_hot()

    remaining = store.read_long_term().splitlines()
    assert len(remaining) <= MEMORY_HOT_MAX_LINES
    # Newest lines are kept (those at the end)
    assert remaining[-1] == lines[-1]


def test_trim_hot_idempotent(tmp_path):
    """Calling trim_hot() multiple times doesn't keep shrinking a file that's already under cap."""
    store = _store(tmp_path)
    lines_per = max(10, MEMORY_HOT_MAX_LINES // 5 + 1)
    content = _make_memory(n_sections=6, lines_per_section=lines_per)
    store.write_long_term(content)

    after_first = store.read_long_term()
    store.trim_hot()
    after_second = store.read_long_term()

    assert after_first == after_second


# ---------------------------------------------------------------------------
# get_memory_context — warm-tier loading
# ---------------------------------------------------------------------------

async def test_warm_loading_keyword_match(tmp_path):
    """A topic file is injected when the user message contains a matching keyword."""
    store = _store(tmp_path)
    store.memory_file.write_text("## Recent\n- Some recent fact\n", encoding="utf-8")

    store.topics_dir.mkdir(parents=True, exist_ok=True)
    (store.topics_dir / "garage.md").write_text("## Garage\n- LS swap in progress\n", encoding="utf-8")

    ctx = await store.get_memory_context("what parts are in my garage")
    assert "LS swap" in ctx
    assert "Recalled Memory" in ctx


async def test_warm_loading_no_false_positive(tmp_path):
    """An unrelated user message does not load topic files."""
    store = _store(tmp_path)
    store.memory_file.write_text("## Recent\n- fact\n", encoding="utf-8")

    store.topics_dir.mkdir(parents=True, exist_ok=True)
    (store.topics_dir / "garage.md").write_text("## Garage\n- LS swap\n", encoding="utf-8")

    ctx = await store.get_memory_context("hello how are you today")
    assert "LS swap" not in ctx
    assert "Recalled Memory" not in ctx


async def test_warm_loading_empty_message(tmp_path):
    """Empty user message never triggers warm loading."""
    store = _store(tmp_path)
    store.topics_dir.mkdir(parents=True, exist_ok=True)
    (store.topics_dir / "garage.md").write_text("## Garage\n- LS swap\n", encoding="utf-8")

    ctx = await store.get_memory_context("")
    assert "LS swap" not in ctx


async def test_warm_loading_no_topics_dir(tmp_path):
    """get_memory_context works fine when topics/ directory doesn't exist."""
    store = _store(tmp_path)
    store.memory_file.write_text("## Recent\n- fact\n", encoding="utf-8")

    ctx = await store.get_memory_context("garage parts")
    assert "Recent" in ctx  # hot tier still loads
    assert "Recalled Memory" not in ctx


async def test_hot_tier_always_present(tmp_path):
    """Hot tier content is always included regardless of user message."""
    store = _store(tmp_path)
    store.memory_file.write_text("## Core\n- user is Merim\n", encoding="utf-8")

    ctx = await store.get_memory_context("completely unrelated topic xyz")
    assert "user is Merim" in ctx


async def test_get_memory_context_empty_when_no_files(tmp_path):
    """Returns empty string when MEMORY.md doesn't exist and no topics."""
    store = _store(tmp_path)
    ctx = await store.get_memory_context("any message")
    assert ctx == ""


async def test_warm_loading_content_match(tmp_path):
    """A topic file is loaded when the user message matches words in file content (not stem)."""
    store = _store(tmp_path)
    store.memory_file.write_text("## Recent\n- fact\n", encoding="utf-8")

    store.topics_dir.mkdir(parents=True, exist_ok=True)
    # Stem "builds" has no overlap with "xylophone"; content does
    (store.topics_dir / "builds.md").write_text(
        "## Build Notes\n- xylophone project ongoing\n", encoding="utf-8"
    )

    ctx = await store.get_memory_context("what about xylophone")
    assert "xylophone" in ctx
    assert "Recalled Memory" in ctx


async def test_warm_loading_top_k_limit(tmp_path):
    """Only top_k=3 (default) topic files are returned even when more match."""
    store = _store(tmp_path)
    store.topics_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        (store.topics_dir / f"topic-{i}.md").write_text(
            f"## Topic {i}\n- rust programming notes {i}\n", encoding="utf-8"
        )

    ctx = await store.get_memory_context("rust programming")
    # At most 3 topic blocks should appear
    assert ctx.count("### topic-") <= 3


@pytest.mark.llm
async def test_warm_loading_semantic_synonym(tmp_path, embed_model_available):
    """Semantic search matches synonym queries that keyword search would miss.

    'automobile project' has no keyword overlap with 'LS engine swap' or 'vehicle',
    but the embeddings should place them close enough (sim >= 0.25) to match.
    """
    store = _store(tmp_path)
    store.memory_file.write_text("## Recent\n- some fact\n", encoding="utf-8")

    store.topics_dir.mkdir(parents=True, exist_ok=True)
    (store.topics_dir / "vehicle.md").write_text(
        "## Vehicle\n- LS engine swap in progress\n", encoding="utf-8"
    )

    ctx = await store.get_memory_context("tell me about the automobile project")
    assert "LS engine" in ctx
    assert "Recalled Memory" in ctx
