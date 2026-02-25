"""Subagent manager for background and synchronous task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.constants import DELEGATE_MAX_ITERATIONS
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.browser import WebBrowseTool


# ---------------------------------------------------------------------------
# Worker role presets — specialized system-prompt fragments injected per role.
# ---------------------------------------------------------------------------
_ROLE_PROMPTS: dict[str, str] = {
    "researcher": (
        "You are a research specialist. Search the web thoroughly, cross-reference sources, "
        "and return structured findings with key facts, URLs, and dates. Be specific and cite "
        "evidence."
    ),
    "writer": (
        "You are a writing specialist. Produce clear, well-structured prose. Focus on clarity "
        "and conciseness. Avoid filler. Deliver the final text ready to use."
    ),
    "analyst": (
        "You are an analytical specialist. Break down data and information into structured "
        "insights. Identify patterns, trade-offs, and key takeaways. Use lists and tables "
        "where helpful."
    ),
    "coder": (
        "You are a coding specialist. Write clean, working code with minimal explanation. "
        "Prefer idiomatic style. Include error handling. Return runnable output."
    ),
    "general": (
        "You are a focused task-completion agent. Complete the assigned task thoroughly "
        "and concisely. Return a clear, actionable result."
    ),
}


class SubagentManager:
    """
    Manages background and synchronous subagent execution.

    - spawn()     — fire-and-forget background task (result announced later via bus)
    - run_direct()— synchronous delegate; waits for result and returns it inline.
                    Used by DelegateTool to implement the supervisor pattern.
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """Fire-and-forget: spawn a background subagent, announce result later."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def run_direct(self, task: str, role: str = "general") -> str:
        """Synchronous delegate: run a worker agent and return its result inline.

        Called by DelegateTool. The calling (supervisor) agent waits here until
        the worker finishes, then uses the result in the same turn.
        """
        task_id = str(uuid.uuid4())[:8]
        logger.info("Delegate [{}] role={}: {}", task_id, role, task[:80])
        result = await self._execute_subagent(task_id, task, role)
        logger.info("Delegate [{}] done ({} chars)", task_id, len(result))
        return result

    def get_running_count(self) -> int:
        return len(self._running_tasks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build the tool registry available to subagents."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        tools.register(WebSearchTool(api_key=self.brave_api_key))
        tools.register(WebFetchTool())
        tools.register(WebBrowseTool())
        return tools

    def _build_system_prompt(self, role: str = "general") -> str:
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        role_desc = _ROLE_PROMPTS.get(role, _ROLE_PROMPTS["general"])

        return f"""# Worker Agent ({role})

## Current Time
{now} ({tz})

## Role
{role_desc}

## Rules
1. Stay focused — complete only the assigned task, nothing else.
2. Your final response is returned directly to the supervisor agent.
3. Do not address the user directly; address the supervisor.
4. Be concise but complete. No filler.

## Capabilities
- Read/write/edit files in the workspace
- Execute shell commands
- Search the web, fetch pages, browse JS-heavy sites (web_browse)

## Workspace
{self.workspace}
Skills: {self.workspace}/skills/ (read SKILL.md files as needed)

When done, provide a clear, structured summary of your findings or actions."""

    async def _execute_subagent(self, task_id: str, task: str, role: str = "general") -> str:
        """Run the agent loop and return the final result string."""
        tools = self._build_tools()
        system_prompt = self._build_system_prompt(role)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        iteration = 0
        final_result: str | None = None

        while iteration < DELEGATE_MAX_ITERATIONS:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": tool_call_dicts,
                })

                for tool_call in response.tool_calls:
                    logger.debug(
                        "Worker [{}] tool: {}({})",
                        task_id, tool_call.name,
                        json.dumps(tool_call.arguments, ensure_ascii=False)[:120],
                    )
                    result = await tools.execute(tool_call.name, tool_call.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
            else:
                final_result = response.content
                break

        if final_result is None:
            final_result = f"Worker reached max iterations ({DELEGATE_MAX_ITERATIONS}) without a final answer."

        return final_result or ""

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Background wrapper: execute then announce result via bus."""
        logger.info("Subagent [{}] starting: {}", task_id, label)
        try:
            result = await self._execute_subagent(task_id, task, role="general")
            logger.info("Subagent [{}] completed", task_id)
            await self._announce_result(task_id, label, task, result, origin, "ok")
        except Exception as e:
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, f"Error: {e}", origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        status_text = "completed successfully" if status == "ok" else "failed"
        announce_content = (
            f"[Subagent '{label}' {status_text}]\n\n"
            f"Task: {task}\n\n"
            f"Result:\n{result}\n\n"
            "Summarize this naturally for the user. Keep it brief (1-2 sentences). "
            "Do not mention technical details like 'subagent' or task IDs."
        )
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )
        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced to {}:{}", task_id, origin["channel"], origin["chat_id"])
