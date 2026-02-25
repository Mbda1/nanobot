"""Circuit breaker tests for AgentLoop._run_agent_loop.

Verifies that:
1. A tool called more than CIRCUIT_BREAKER_PER_TOOL times in one turn is blocked.
2. The same tool+arg repeated CIRCUIT_BREAKER_CONSECUTIVE+1 times consecutively is blocked.
3. Different tools / different args do NOT trip the consecutive breaker.
4. Tools under the limit execute normally (no false positives).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.config.constants import CIRCUIT_BREAKER_CONSECUTIVE, CIRCUIT_BREAKER_PER_TOOL
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _tool_response(name: str, arg: str, call_id: str = "c1") -> LLMResponse:
    """LLM response containing a single tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments={"query": arg})],
    )


def _text_response(text: str = "done") -> LLMResponse:
    """LLM response with plain text (no tool call)."""
    return LLMResponse(content=text, tool_calls=[])


def _make_loop(tmp_path) -> AgentLoop:
    """Construct a minimal AgentLoop with mocked bus and provider."""
    bus = MagicMock()
    bus.publish_outbound = AsyncMock()

    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock()  # caller sets side_effect per test

    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path)


@pytest.mark.asyncio
async def test_per_tool_breaker_trips(tmp_path):
    """After CIRCUIT_BREAKER_PER_TOOL calls, further calls are blocked.

    Uses distinct args each time so the consecutive breaker doesn't fire first.
    """
    loop = _make_loop(tmp_path)

    # PER_TOOL calls with unique args (no consecutive trigger), then one more (trips), then text.
    responses = (
        [_tool_response("web_search", f"query-{i}", f"c{i}") for i in range(CIRCUIT_BREAKER_PER_TOOL + 1)]
        + [_text_response("all done")]
    )
    loop.provider.chat.side_effect = responses

    execute_mock = AsyncMock(return_value="search result")
    loop.tools._tools["web_search"] = MagicMock(
        name="web_search",
        validate_params=MagicMock(return_value=[]),
        execute=execute_mock,
    )

    messages = [{"role": "user", "content": "search for stuff"}]
    final_content, tools_used, _ = await loop._run_agent_loop(messages)

    # Tool should have been called exactly PER_TOOL times (the extra call is blocked).
    assert execute_mock.call_count == CIRCUIT_BREAKER_PER_TOOL
    assert final_content == "all done"


@pytest.mark.asyncio
async def test_consecutive_breaker_trips(tmp_path):
    """Identical tool+arg repeated CONSECUTIVE+1 times triggers the breaker."""
    loop = _make_loop(tmp_path)

    # CONSECUTIVE calls succeed, the next one is blocked.
    responses = (
        [_tool_response("web_fetch", "https://example.com") for _ in range(CIRCUIT_BREAKER_CONSECUTIVE + 1)]
        + [_text_response("stopped")]
    )
    loop.provider.chat.side_effect = responses

    execute_mock = AsyncMock(return_value="page content")
    loop.tools._tools["web_fetch"] = MagicMock(
        name="web_fetch",
        validate_params=MagicMock(return_value=[]),
        execute=execute_mock,
    )

    messages = [{"role": "user", "content": "fetch this"}]
    final_content, _, _ = await loop._run_agent_loop(messages)

    # Only the first CONSECUTIVE calls should execute; the (CONSECUTIVE+1)th is blocked.
    assert execute_mock.call_count == CIRCUIT_BREAKER_CONSECUTIVE
    assert final_content == "stopped"


@pytest.mark.asyncio
async def test_different_args_do_not_trip_consecutive(tmp_path):
    """Different args for the same tool reset the consecutive counter — no false trip."""
    loop = _make_loop(tmp_path)

    # Alternate between two different URLs — should never trip consecutive breaker.
    responses = [
        _tool_response("web_fetch", "https://site-a.com", "c1"),
        _tool_response("web_fetch", "https://site-b.com", "c2"),
        _tool_response("web_fetch", "https://site-a.com", "c3"),
        _text_response("done"),
    ]
    loop.provider.chat.side_effect = responses

    execute_mock = AsyncMock(return_value="content")
    loop.tools._tools["web_fetch"] = MagicMock(
        name="web_fetch",
        validate_params=MagicMock(return_value=[]),
        execute=execute_mock,
    )

    messages = [{"role": "user", "content": "fetch"}]
    _, tools_used, _ = await loop._run_agent_loop(messages)

    # All 3 calls executed — no false trip.
    assert execute_mock.call_count == 3


@pytest.mark.asyncio
async def test_no_trip_under_limit(tmp_path):
    """Tools called fewer than the limit execute without interference.

    Uses distinct args each time so neither breaker fires.
    """
    loop = _make_loop(tmp_path)

    calls = CIRCUIT_BREAKER_PER_TOOL - 1
    responses = (
        [_tool_response("web_search", f"q-{i}", f"c{i}") for i in range(calls)]
        + [_text_response("ok")]
    )
    loop.provider.chat.side_effect = responses

    execute_mock = AsyncMock(return_value="result")
    loop.tools._tools["web_search"] = MagicMock(
        name="web_search",
        validate_params=MagicMock(return_value=[]),
        execute=execute_mock,
    )

    messages = [{"role": "user", "content": "search"}]
    final_content, _, _ = await loop._run_agent_loop(messages)

    assert execute_mock.call_count == calls
    assert final_content == "ok"
