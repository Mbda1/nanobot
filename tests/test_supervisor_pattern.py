"""Tests for the supervisor pattern: DelegateTool + parallel batch execution.

Verifies:
1. DelegateTool calls SubagentManager.run_direct() and returns its result.
2. Multiple delegate calls in one LLM response execute in parallel (asyncio.gather).
3. Parallel execution does not break circuit breakers (sequential bookkeeping preserved).
4. Worker role presets exist for all documented roles.
5. DelegateTool is registered in AgentLoop by default.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.delegate import DelegateTool
from nanobot.agent.subagent import SubagentManager, _ROLE_PROMPTS
from nanobot.providers.base import LLMResponse, ToolCallRequest


# ---------------------------------------------------------------------------
# DelegateTool unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delegate_tool_calls_run_direct():
    """DelegateTool.execute() delegates to SubagentManager.run_direct()."""
    manager = MagicMock()
    manager.run_direct = AsyncMock(return_value="research findings")

    tool = DelegateTool(manager=manager)
    result = await tool.execute(task="find LS swap parts", role="researcher")

    manager.run_direct.assert_awaited_once_with(task="find LS swap parts", role="researcher")
    assert result == "research findings"


@pytest.mark.asyncio
async def test_delegate_tool_default_role():
    """DelegateTool defaults to role='general' when not specified."""
    manager = MagicMock()
    manager.run_direct = AsyncMock(return_value="done")

    tool = DelegateTool(manager=manager)
    await tool.execute(task="some task")

    _, kwargs = manager.run_direct.call_args
    assert kwargs.get("role") == "general" or manager.run_direct.call_args[1].get("role") == "general"


def test_delegate_tool_schema():
    """DelegateTool exposes correct name and required parameters."""
    tool = DelegateTool(manager=MagicMock())
    assert tool.name == "delegate"
    assert "task" in tool.parameters["required"]
    roles = tool.parameters["properties"]["role"]["enum"]
    assert set(roles) == {"researcher", "writer", "analyst", "coder", "general"}


# ---------------------------------------------------------------------------
# Worker role presets
# ---------------------------------------------------------------------------

def test_all_roles_have_prompts():
    """Every documented role has a non-empty system prompt."""
    for role in ("researcher", "writer", "analyst", "coder", "general"):
        assert role in _ROLE_PROMPTS
        assert len(_ROLE_PROMPTS[role]) > 20


# ---------------------------------------------------------------------------
# Parallel batch execution in AgentLoop
# ---------------------------------------------------------------------------

def _make_loop(tmp_path):
    from nanobot.agent.loop import AgentLoop
    bus = MagicMock()
    bus.publish_outbound = AsyncMock()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock()
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path)


def _tool_call(name: str, arg: str, call_id: str) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments={"task": arg})


def _response_with_tools(*tool_calls: ToolCallRequest) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=list(tool_calls))


def _text_response(text: str = "synthesized") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[])


@pytest.mark.asyncio
async def test_parallel_execution_of_delegate_batch(tmp_path):
    """Two delegate calls in one LLM response run concurrently, not sequentially."""
    loop = _make_loop(tmp_path)

    # Track start times of each tool execution
    start_times: list[float] = []
    SLEEP = 0.15  # each worker sleeps this long

    async def slow_tool(**kwargs):
        start_times.append(time.monotonic())
        await asyncio.sleep(SLEEP)
        return "result"

    # Register a fake "delegate" tool that records timing
    fake_delegate = MagicMock()
    fake_delegate.name = "delegate"
    fake_delegate.validate_params = MagicMock(return_value=[])
    fake_delegate.execute = slow_tool
    loop.tools._tools["delegate"] = fake_delegate

    responses = [
        # One LLM response with TWO delegate calls → should run in parallel
        _response_with_tools(
            _tool_call("delegate", "task A", "c1"),
            _tool_call("delegate", "task B", "c2"),
        ),
        _text_response("done"),
    ]
    loop.provider.chat.side_effect = responses

    t0 = time.monotonic()
    await loop._run_agent_loop([{"role": "user", "content": "go"}])
    elapsed = time.monotonic() - t0

    # Both started within SLEEP of each other (parallel), not SLEEP*2 apart (sequential)
    assert len(start_times) == 2
    assert abs(start_times[1] - start_times[0]) < SLEEP, (
        f"Tools ran sequentially (gap={start_times[1]-start_times[0]:.3f}s), expected parallel"
    )
    # Total time should be ~SLEEP, not ~2×SLEEP
    assert elapsed < SLEEP * 1.8, f"Total elapsed {elapsed:.3f}s suggests sequential execution"


@pytest.mark.asyncio
async def test_circuit_breaker_preserved_with_parallel(tmp_path):
    """Circuit breaker bookkeeping is still correct when batch execution runs in parallel."""
    loop = _make_loop(tmp_path)

    execute_mock = AsyncMock(return_value="result")
    loop.tools._tools["web_search"] = MagicMock(
        name="web_search",
        validate_params=MagicMock(return_value=[]),
        execute=execute_mock,
    )

    from nanobot.config.constants import CIRCUIT_BREAKER_PER_TOOL
    # Drive tool calls with unique args to avoid consecutive breaker
    responses = (
        [
            LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id=f"c{i}", name="web_search", arguments={"query": f"q{i}"})
            ])
            for i in range(CIRCUIT_BREAKER_PER_TOOL + 1)
        ]
        + [_text_response("all done")]
    )
    loop.provider.chat.side_effect = responses

    await loop._run_agent_loop([{"role": "user", "content": "search"}])
    # Only PER_TOOL executions should have happened
    assert execute_mock.call_count == CIRCUIT_BREAKER_PER_TOOL


# ---------------------------------------------------------------------------
# Integration: DelegateTool registered in AgentLoop
# ---------------------------------------------------------------------------

def test_delegate_registered_in_agent_loop(tmp_path):
    """AgentLoop registers 'delegate' tool by default."""
    loop = _make_loop(tmp_path)
    assert "delegate" in loop.tools
