"""Delegate tool — synchronous worker dispatch (supervisor pattern).

Unlike `spawn` (fire-and-forget background), `delegate` runs a worker agent
inline and returns its result as the tool result. The calling (supervisor)
agent receives the result in the same turn and can synthesize across multiple
worker outputs.

When the LLM emits several `delegate` calls in a single response, the agent
loop executes them in parallel via asyncio.gather — the supervisor gets all
results at once before synthesizing, matching the intent of the tool call batch.
"""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager

_ROLES = ["researcher", "writer", "analyst", "coder", "general"]


class DelegateTool(Tool):
    """Delegate a subtask to a specialized worker agent (synchronous — result returned inline).

    Use this to decompose complex requests:
      1. Call delegate() once per subtask (supervisor breaks the work down).
      2. Multiple delegate() calls in one response run in PARALLEL.
      3. Synthesize all results in the next LLM turn.

    Prefer delegate over spawn when you need the result before responding.
    Use spawn when the task can run in the background while you respond now.
    """

    name = "delegate"
    description = (
        "Delegate a subtask to a specialized worker agent and get the result inline. "
        "Multiple delegate calls in one response run in parallel. "
        "Choose a role: researcher (web research), writer (prose/drafting), "
        "analyst (data/comparisons), coder (code generation), general (default)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Clear, self-contained task description for the worker.",
            },
            "role": {
                "type": "string",
                "enum": _ROLES,
                "description": "Worker specialization. Default: general.",
            },
        },
        "required": ["task"],
    }

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    async def execute(self, task: str, role: str = "general", **kwargs: Any) -> str:
        """Run a worker agent synchronously and return its result."""
        return await self._manager.run_direct(task=task, role=role)
