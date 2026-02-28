"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.enrichment import enrich_query
from nanobot.agent.memory import MemoryStore
from nanobot.agent.usage import init as _usage_init, record as _usage_record
from nanobot.config.constants import (
    CIRCUIT_BREAKER_CONSECUTIVE,
    CIRCUIT_BREAKER_PER_TOOL,
    TOOL_RESULT_MAX_CHARS,
)
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.browser import WebBrowseTool
from nanobot.agent.tools.delegate import DelegateTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        local_model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.local_model = local_model  # Lightweight local LLM for memory consolidation etc.
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        _usage_init(workspace)
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(WebBrowseTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(DelegateTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _phase_from_tool_hint(hint: str) -> str | None:
        """Map a tool hint string to a coarse progress phase."""
        m = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\(", hint or "")
        if not m:
            return None
        tool = m.group(1).lower()
        if tool in {"delegate", "spawn"}:
            return "researching in parallel"
        if tool in {"web_search", "web_fetch", "web_browse"}:
            return "gathering sources"
        if tool in {"read_file", "list_dir"}:
            return "reviewing files"
        if tool in {"write_file", "edit_file"}:
            return "writing draft"
        if tool == "exec":
            return "running checks"
        return None

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        # Circuit breaker state (reset each turn)
        _tool_counts: dict[str, int] = {}
        _last_sig: str | None = None   # "tool_name:primary_arg" of last call
        _consecutive: int = 0          # consecutive identical-sig calls

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            _usage_record(self.model, response.usage, source="agent", latency_ms=response.latency_ms)

            if response.finish_reason == "error":
                raise RuntimeError(response.content)

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                # --- Phase 1: sequential — apply circuit breakers, decide what to run ---
                # Results list: (tool_call, pre_computed_result | None)
                # None means "needs actual execution" (not blocked).
                _call_plan: list[tuple[Any, str | None]] = []
                for tool_call in response.tool_calls:
                    name = tool_call.name
                    _tool_counts[name] = _tool_counts.get(name, 0) + 1

                    primary = next(iter(tool_call.arguments.values()), "") if tool_call.arguments else ""
                    sig = f"{name}:{str(primary)[:60]}"
                    if sig == _last_sig:
                        _consecutive += 1
                    else:
                        _consecutive = 1
                        _last_sig = sig

                    breaker_reason: str | None = None
                    if _consecutive > CIRCUIT_BREAKER_CONSECUTIVE:
                        breaker_reason = (
                            f"identical call repeated {_consecutive}× in a row — "
                            "this is a loop"
                        )
                    elif _tool_counts[name] > CIRCUIT_BREAKER_PER_TOOL:
                        breaker_reason = (
                            f"called {_tool_counts[name]}× this turn "
                            f"(limit: {CIRCUIT_BREAKER_PER_TOOL})"
                        )

                    if breaker_reason:
                        logger.warning("Circuit breaker tripped: {} — {}", name, breaker_reason)
                        blocked = (
                            f"Error: Circuit breaker tripped for '{name}' — {breaker_reason}. "
                            "Stop calling this tool and give a final answer with what you have."
                        )
                        _call_plan.append((tool_call, blocked))
                    else:
                        tools_used.append(name)
                        logger.info("Tool call: {}({})", name,
                                    json.dumps(tool_call.arguments, ensure_ascii=False)[:200])
                        _call_plan.append((tool_call, None))

                # --- Phase 2: parallel — execute non-blocked calls concurrently ---
                _pending_idx = [i for i, (_, r) in enumerate(_call_plan) if r is None]
                if _pending_idx:
                    _results = await asyncio.gather(*[
                        self.tools.execute(_call_plan[i][0].name, _call_plan[i][0].arguments)
                        for i in _pending_idx
                    ])
                    for idx, res in zip(_pending_idx, _results):
                        _call_plan[idx] = (_call_plan[idx][0], res)

                # --- Phase 3: add all results to messages in original order ---
                for tool_call, result in _call_plan:
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = self._strip_think(response.content)
                # Persist assistant text-only replies into session history.
                messages = self.context.add_assistant_message(
                    messages,
                    final_content,
                    reasoning_content=response.reasoning_content,
                )
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        if self.local_model:
            await self._warmup_local_model()  # await: ensure model is in RAM before first message
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id, content="", metadata=msg.metadata or {},
                        ))
                except Exception as e:
                    logger.error("Error processing message: {}", e)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="I ran into a temporary issue. Please try again in a moment."
                    ))
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = await self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        enriched_content = await enrich_query(self.provider, self.local_model, msg.content)
        initial_messages = await self.context.build_messages(
            history=history,
            current_message=enriched_content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        progress_state = {
            "started_at": asyncio.get_running_loop().time(),
            "last_sent_at": 0.0,
            "sent_count": 0,
            "phases": set(),
        }
        progress_lock = asyncio.Lock()
        progress_task: asyncio.Task | None = None
        progress_enabled = msg.channel != "cli"
        progress_start_delay_s = 20.0
        progress_interval_s = 45.0
        progress_max_messages = 5

        async def _emit_progress(text: str, *, tool_hint: bool = False) -> None:
            if not progress_enabled:
                return
            async with progress_lock:
                if progress_state["sent_count"] >= progress_max_messages:
                    return
                progress_state["last_sent_at"] = asyncio.get_running_loop().time()
                progress_state["sent_count"] += 1
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=text, metadata=meta,
            ))

        async def _heartbeat_loop() -> None:
            try:
                while True:
                    await asyncio.sleep(progress_interval_s)
                    now = asyncio.get_running_loop().time()
                    elapsed = now - progress_state["started_at"]
                    if elapsed < progress_start_delay_s:
                        continue
                    if (now - progress_state["last_sent_at"]) < progress_interval_s:
                        continue
                    await _emit_progress(
                        "Still working on this. I will send the final answer as soon as it's ready."
                    )
            except asyncio.CancelledError:
                return

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            if not progress_enabled:
                return
            if not tool_hint:
                return
            now = asyncio.get_running_loop().time()
            elapsed = now - progress_state["started_at"]
            phase = self._phase_from_tool_hint(content)
            if phase and phase not in progress_state["phases"] and elapsed >= progress_start_delay_s:
                progress_state["phases"].add(phase)
                await _emit_progress(f"Progress: {phase}.")
                return
            if elapsed >= progress_start_delay_s and (now - progress_state["last_sent_at"]) >= progress_interval_s:
                await _emit_progress(
                    "Still working on this. I will send the final answer as soon as it's ready."
                )

        if progress_enabled:
            await _emit_progress("Working on it now. This can take a few minutes for deep research.")
            progress_task = asyncio.create_task(_heartbeat_loop())

        try:
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages, on_progress=on_progress or _bus_progress,
            )
        finally:
            if progress_task:
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    _TOOL_RESULT_MAX_CHARS = TOOL_RESULT_MAX_CHARS

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        from nanobot.config.constants import TIMEOUT_MEMORY_CONSOLIDATION
        try:
            return await asyncio.wait_for(
                MemoryStore(self.workspace).consolidate(
                    session, self.provider, self.model,
                    archive_all=archive_all, memory_window=self.memory_window,
                ),
                timeout=TIMEOUT_MEMORY_CONSOLIDATION,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Memory consolidation timed out after {}s, skipping this run",
                TIMEOUT_MEMORY_CONSOLIDATION,
            )
            return False

    async def _warmup_local_model(self) -> None:
        """Warm up local model on supported backends.

        For Ollama we use keep_alive=-1 to pin model in RAM.
        For OpenAI-compatible backends (e.g. llama.cpp), issue a tiny prompt
        to prime model load without Ollama-only params.
        """
        if not self.local_model:
            return
        # Extract bare model name (e.g. "ollama/mistral" → "mistral")
        model_name = self.local_model.split("/")[-1]
        from nanobot.config.constants import LOCAL_API_BASE, LOCAL_API_KEY, LOCAL_LLM_BACKEND
        local_base = LOCAL_API_BASE.rstrip("/")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                if LOCAL_LLM_BACKEND in {"openai", "llamacpp", "llama.cpp"}:
                    headers = {"Content-Type": "application/json"}
                    if LOCAL_API_KEY:
                        headers["Authorization"] = f"Bearer {LOCAL_API_KEY}"
                    await client.post(
                        f"{local_base}/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 1,
                            "temperature": 0.0,
                            "stream": False,
                        },
                    )
                    logger.info("Local model warmup complete: {} (backend={})", model_name, LOCAL_LLM_BACKEND)
                else:
                    await client.post(
                        f"{local_base}/api/chat",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": False,
                            "keep_alive": -1,
                            "options": {"num_predict": 1},
                        },
                    )
                    logger.info("Local model pinned in RAM: {} (keep_alive=-1)", model_name)
        except Exception as exc:
            logger.debug("Local model warmup skipped (backend={}): {}", LOCAL_LLM_BACKEND, exc)

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
