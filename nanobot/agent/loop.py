"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.decision_gate import run_local, should_use_local
from nanobot.agent.enrichment import enrich_query
from nanobot.agent.memory import MemoryStore
from nanobot.agent.usage import init as _usage_init, record as _usage_record, metric as _usage_metric
from nanobot.config.constants import (
    CIRCUIT_BREAKER_CONSECUTIVE,
    CIRCUIT_BREAKER_PER_TOOL,
    GOVERNOR_PER_TURN_TOKENS,
    MAX_CLOUD_INPUT_EST_TOKENS,
    TARGET_CLOUD_INPUT_EST_TOKENS,
    TOOL_RESULT_MAX_CHARS,
    WEB_SEARCH_MAX_COUNT,
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
            local_model=self.local_model,
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

    @staticmethod
    def _detect_research_mode(text: str) -> str:
        msg = (text or "").lower()
        if any(k in msg for k in ("deep research", "exhaustive", "full research", "thorough research")):
            return "deep"
        if any(k in msg for k in ("quick", "cheap", "low cost", "minimal")):
            return "cheap"
        return "balanced"

    @staticmethod
    def _extract_text_for_estimate(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif item.get("type") == "image_url":
                        parts.append("[image]")
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    @classmethod
    def _estimate_messages_tokens(cls, messages: list[dict[str, Any]]) -> int:
        # Lightweight estimate: roughly 1 token per 4 chars + per-message overhead.
        chars = 0
        for m in messages:
            chars += len(cls._extract_text_for_estimate(m.get("content", "")))
            if m.get("tool_calls"):
                chars += len(json.dumps(m.get("tool_calls"), ensure_ascii=False))
            chars += 32
        return int(chars / 4)

    @classmethod
    def _compact_messages_for_budget(
        cls,
        messages: list[dict[str, Any]],
        budget_tokens: int,
        target_tokens: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        est = cls._estimate_messages_tokens(messages)
        if est <= budget_tokens:
            return messages, False

        compact = list(messages)

        # First pass: aggressively trim oversized tool messages.
        for m in compact:
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if not isinstance(content, str):
                continue
            if len(content) > 700:
                m["content"] = content[:700] + "\n... (compressed for cloud budget)"

        # Do not drop whole history messages here. It can break provider-specific
        # tool-use/tool-result chain invariants in persisted sessions.
        return compact, True

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        research_mode: str = "balanced",
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
        _web_search_calls = 0
        _web_search_limit = 2 if research_mode == "cheap" else 6 if research_mode == "deep" else 3

        # Token governor — hard cap on total tokens consumed in one user turn
        _turn_tokens = 0

        while iteration < self.max_iterations:
            iteration += 1

            est_before = self._estimate_messages_tokens(messages)
            messages, was_compacted = self._compact_messages_for_budget(
                messages,
                budget_tokens=MAX_CLOUD_INPUT_EST_TOKENS,
                target_tokens=TARGET_CLOUD_INPUT_EST_TOKENS,
            )
            est_after = self._estimate_messages_tokens(messages)
            _usage_metric(
                "cloud_preflight",
                mode=research_mode,
                estimated_tokens_before=est_before,
                estimated_tokens_after=est_after,
                compacted=was_compacted,
                message_count=len(messages),
            )
            if was_compacted:
                logger.warning(
                    "Cloud preflight compaction applied (mode={}, est {} -> {} tokens)",
                    research_mode, est_before, est_after,
                )

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            _usage_record(self.model, response.usage, source="agent", latency_ms=response.latency_ms)
            _usage_metric("turn_latency", latency_ms=response.latency_ms)

            _turn_tokens += response.usage.get("total_tokens", 0)
            if _turn_tokens > GOVERNOR_PER_TURN_TOKENS:
                logger.warning(
                    "Token governor: per-turn cap {} hit ({} tokens)",
                    GOVERNOR_PER_TURN_TOKENS, _turn_tokens,
                )
                _usage_record(self.model, {"total_tokens": _turn_tokens}, source="governor_abort")
                return (
                    f"⚠️ I stopped mid-task: this turn used {_turn_tokens:,} tokens "
                    f"(cap: {GOVERNOR_PER_TURN_TOKENS:,}). Please break the request into smaller steps.",
                    False, messages,
                )

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
                # Results list: (tool_call, exec_args, pre_computed_result | None)
                # None means "needs actual execution" (not blocked).
                _call_plan: list[tuple[Any, dict[str, Any], str | None]] = []
                for tool_call in response.tool_calls:
                    name = tool_call.name
                    _tool_counts[name] = _tool_counts.get(name, 0) + 1
                    exec_args = dict(tool_call.arguments or {})

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
                    elif name == "web_search":
                        _web_search_calls += 1
                        if _web_search_calls > _web_search_limit:
                            breaker_reason = (
                                f"web_search called {_web_search_calls}× this turn "
                                f"(mode={research_mode}, limit={_web_search_limit})"
                            )
                        else:
                            cap = WEB_SEARCH_MAX_COUNT if research_mode == "deep" else 3 if research_mode == "cheap" else 4
                            try:
                                req_count = int(exec_args.get("count", cap) or cap)
                            except Exception:
                                req_count = cap
                            exec_args["count"] = min(max(req_count, 1), cap)
                            if research_mode != "deep" and "provider" not in exec_args:
                                exec_args["provider"] = "auto"

                    if breaker_reason:
                        logger.warning("Circuit breaker tripped: {} — {}", name, breaker_reason)
                        blocked = (
                            f"Error: Circuit breaker tripped for '{name}' — {breaker_reason}. "
                            "Stop calling this tool and give a final answer with what you have."
                        )
                        _call_plan.append((tool_call, exec_args, blocked))
                    else:
                        tools_used.append(name)
                        logger.info("Tool call: {}({})", name,
                                    json.dumps(exec_args, ensure_ascii=False)[:200])
                        _call_plan.append((tool_call, exec_args, None))

                # --- Phase 2: parallel — execute non-blocked calls concurrently ---
                _pending_idx = [i for i, (_, _, r) in enumerate(_call_plan) if r is None]
                if _pending_idx:
                    _results = await asyncio.gather(*[
                        self.tools.execute(_call_plan[i][0].name, _call_plan[i][1])
                        for i in _pending_idx
                    ])
                    for idx, res in zip(_pending_idx, _results):
                        _call_plan[idx] = (_call_plan[idx][0], _call_plan[idx][1], res)

                # --- Phase 3: add all results to messages in original order ---
                for tool_call, _, result in _call_plan:
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result, research_mode=research_mode
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

    @staticmethod
    def _extract_obi_directive(text: str) -> str | None:
        m = re.match(r"^\s*obi\s*[-:]\s*(.+)$", text or "", flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        return m.group(1).strip()

    @staticmethod
    def _slugify(text: str, max_len: int = 64) -> str:
        s = re.sub(r"[^a-zA-Z0-9\s_-]", "", text or "").strip().lower()
        s = re.sub(r"\s+", "-", s)
        s = s.strip("-_")
        return (s[:max_len] or "context-note")

    @staticmethod
    def _obi_notes_dir() -> Path:
        return Path("/mnt/c/Users/Merim/Desktop/Merim_Personal/Bot Notes")

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        tags: list[str] = []
        for t in re.findall(r"#([A-Za-z0-9_-]+)", text or ""):
            v = t.strip().lower()
            if v and v not in tags:
                tags.append(v)
        return tags

    def _build_obi_note(self, session: Session, directive: str, user_text: str, tags: list[str] | None = None) -> str:
        recent = session.get_history(max_messages=14)
        lines: list[str] = []
        for m in recent:
            role = str(m.get("role", "")).lower()
            if role not in {"user", "assistant"}:
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            tag = "User" if role == "user" else "Assistant"
            lines.append(f"- **{tag}:** {content[:500]}")
        excerpt = "\n".join(lines[-10:]) if lines else "- (no recent text context)"
        tag_list = ["obi", "context"] + list(tags or [])
        tag_list = [t for i, t in enumerate(tag_list) if t and t not in tag_list[:i]]
        tags_yaml = ", ".join(tag_list)

        ts = datetime.now()
        return (
            "---\n"
            f"created: {ts.isoformat(timespec='seconds')}\n"
            "agent: Obi\n"
            f"tags: [{tags_yaml}]\n"
            "---\n\n"
            f"# Obi Note - {ts.strftime('%Y-%m-%d %H:%M')}\n\n"
            f"## Request\n{directive}\n\n"
            "## Source Message\n"
            f"{user_text}\n\n"
            "## Recent Context\n"
            f"{excerpt}\n"
        )

    def _save_obi_note(self, note_markdown: str, directive: str) -> Path:
        notes_dir = self._obi_notes_dir()
        notes_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now()
        stem = self._slugify(directive.replace("save this context as a note", "").strip())
        fname = f"{ts.strftime('%Y-%m-%d_%H%M%S')}_obi_{stem}.md"
        path = notes_dir / fname
        path.write_text(note_markdown, encoding="utf-8")
        return path

    def _find_obi_note(self, query: str) -> Path | None:
        notes_dir = self._obi_notes_dir()
        if not notes_dir.exists():
            return None
        q = self._slugify(query, max_len=120)
        candidates = sorted(notes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not q:
            return candidates[0] if candidates else None
        for p in candidates:
            if q in self._slugify(p.stem, max_len=120):
                return p
        for p in candidates:
            if query.lower().strip() in p.stem.lower():
                return p
        return candidates[0] if candidates else None

    def _extract_obi_review_target(self, directive: str) -> str:
        text = (directive or "").strip()
        m = re.search(r"review\s+(.+?)\s+and\s+(?:optimi[sz]e links|improve linking)\s*$", text, flags=re.IGNORECASE)
        if m:
            target = m.group(1).strip().strip("\"'")
            if target.lower() in {"this folder", "my folder", "vault", "my vault"}:
                return str(self._obi_notes_dir())
            return target
        if "review this folder" in text.lower() or "improve linking" in text.lower():
            return str(self._obi_notes_dir())
        return str(self._obi_notes_dir())

    def _append_to_obi_note(self, note_path: Path, text: str, title: str = "Obi Append") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = f"\n\n## {title} ({ts})\n{text.strip()}\n"
        note_path.write_text(note_path.read_text(encoding="utf-8") + block, encoding="utf-8")

    def _build_daily_summary(self, session: Session) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        recent = session.get_history(max_messages=24)
        items: list[str] = []
        for m in recent:
            role = str(m.get("role", "")).lower()
            if role not in {"user", "assistant"}:
                continue
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                items.append(f"- **{role}**: {c.strip()[:220]}")
        if not items:
            items = ["- No recent context captured."]
        return (
            f"# Daily Note - {today}\n\n"
            "## Summary\n"
            "Auto-captured by Obi from recent conversation context.\n\n"
            "## Timeline\n"
            + "\n".join(items[-16:])
            + "\n"
        )

    @staticmethod
    def _normalize_wikilink_target(target: str) -> str:
        t = target.strip()
        if "|" in t:
            t = t.split("|", 1)[0].strip()
        if "#" in t:
            t = t.split("#", 1)[0].strip()
        return t

    def _obi_review_and_optimize(self, target: str) -> str:
        root = Path(target).expanduser()
        if not root.exists() or not root.is_dir():
            return f"Obi review failed: folder not found: {root}"

        md_files = sorted(root.rglob("*.md"))
        if not md_files:
            return f"Obi review complete: no markdown files found in {root}"

        # Index note names for deterministic link repair.
        name_index: dict[str, list[str]] = {}
        for p in md_files:
            stem = p.stem.strip()
            key = re.sub(r"[^a-z0-9]+", "", stem.lower())
            name_index.setdefault(key, []).append(stem)

        total_links = 0
        broken_links = 0
        fixed_links = 0
        changed_files: list[str] = []
        suggestions: list[str] = []

        link_re = re.compile(r"\[\[([^\]]+)\]\]")

        for p in md_files[:200]:
            text = p.read_text(encoding="utf-8", errors="replace")
            original = text

            def _replace(m: re.Match[str]) -> str:
                nonlocal total_links, broken_links, fixed_links
                raw = m.group(1)
                target_raw = self._normalize_wikilink_target(raw)
                total_links += 1
                if not target_raw:
                    return m.group(0)

                candidate = root / f"{target_raw}.md"
                if candidate.exists():
                    return m.group(0)

                key = re.sub(r"[^a-z0-9]+", "", target_raw.lower())
                matches = name_index.get(key, [])
                if len(matches) == 1:
                    fixed_links += 1
                    repaired = raw.replace(target_raw, matches[0], 1)
                    return f"[[{repaired}]]"

                broken_links += 1
                return m.group(0)

            text = link_re.sub(_replace, text)

            # Add a small Related section if missing and there are clear parent notes.
            if "## Related" not in text and p.parent != root:
                parent_note = p.parent.name.replace("_", " ").replace("-", " ").strip()
                parent_key = re.sub(r"[^a-z0-9]+", "", parent_note.lower())
                parent_matches = name_index.get(parent_key, [])
                if parent_matches:
                    text += f"\n\n## Related\n- [[{parent_matches[0]}]]\n"

            if text != original:
                p.write_text(text, encoding="utf-8")
                changed_files.append(str(p))

        if len(md_files) > 200:
            suggestions.append(f"Scanned first 200/{len(md_files)} markdown files; run again for full pass.")
        if broken_links > 0:
            suggestions.append(f"{broken_links} unresolved links remain (ambiguous or missing targets).")
        if not suggestions:
            suggestions.append("No further action needed in this pass.")

        report = (
            f"Obi local review complete for {root}\n"
            f"- markdown files scanned: {min(len(md_files), 200)}\n"
            f"- links analyzed: {total_links}\n"
            f"- links auto-fixed: {fixed_links}\n"
            f"- unresolved links: {broken_links}\n"
            f"- files changed: {len(changed_files)}\n"
            f"- suggestions: {' '.join(suggestions)}"
        )

        report_path = root / f"OBI_AUDIT_{datetime.now().strftime('%Y-%m-%d')}.md"
        changed_preview = "\n".join(f"- {Path(f).name}" for f in changed_files[:30]) or "- none"
        report_body = (
            f"# Obi Audit {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"{report}\n\n"
            "## Changed Files\n"
            f"{changed_preview}\n"
        )
        report_path.write_text(report_body, encoding="utf-8")
        return f"{report}\n- audit report: {report_path}"

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
            research_mode = self._detect_research_mode(msg.content)
            response_style = "default" if research_mode == "deep" else "simple_first"
            messages = await self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                response_style=response_style,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages, research_mode=research_mode)
            self._save_turn(session, all_msgs, len(messages))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        obi_directive = self._extract_obi_directive(msg.content)
        if obi_directive:
            lower = obi_directive.lower()
            tags = self._extract_tags(obi_directive)

            if "review" in lower and ("optimize links" in lower or "improve linking" in lower):
                target = self._extract_obi_review_target(obi_directive)
                result = self._obi_review_and_optimize(target)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Obi review complete for `{target}`:\n\n{result}",
                    metadata=msg.metadata or {},
                )

            if "append to note" in lower:
                target = re.split(r"append to note", obi_directive, flags=re.IGNORECASE, maxsplit=1)[-1].strip(" :\"'")
                note = self._find_obi_note(target)
                if not note:
                    return OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Obi could not find a matching note to append to.",
                        metadata=msg.metadata or {},
                    )
                note_md = self._build_obi_note(session, obi_directive, msg.content, tags=tags)
                self._append_to_obi_note(note, note_md, title="Obi Appended Context")
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Done. Obi appended context to:\n{note}",
                    metadata=msg.metadata or {},
                )

            if "summarize today" in lower and "daily note" in lower:
                notes_dir = self._obi_notes_dir() / "Daily"
                notes_dir.mkdir(parents=True, exist_ok=True)
                day_file = notes_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
                summary = self._build_daily_summary(session)
                if day_file.exists():
                    self._append_to_obi_note(day_file, summary, title="Obi Daily Summary")
                else:
                    day_file.write_text(summary, encoding="utf-8")
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Saved. Obi updated daily note:\n{day_file}",
                    metadata=msg.metadata or {},
                )

            if "tag this" in lower:
                note_md = self._build_obi_note(session, obi_directive, msg.content, tags=tags)
                note_path = self._save_obi_note(note_md, obi_directive)
                tag_msg = ", ".join(f"#{t}" for t in tags) if tags else "(no tags detected)"
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Saved and tagged {tag_msg}:\n{note_path}",
                    metadata=msg.metadata or {},
                )

            if "save" in lower and "note" in lower:
                note_md = self._build_obi_note(session, obi_directive, msg.content, tags=tags)
                note_path = self._save_obi_note(note_md, obi_directive)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=(
                        f"Saved. Obi wrote a note:\n{note_path}\n\n"
                        "Want more? 1) Deep analysis 2) Structured summary 3) Sources only"
                    ),
                    metadata=msg.metadata or {},
                )

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
        research_mode = self._detect_research_mode(msg.content)
        response_style = "default" if research_mode == "deep" else "simple_first"
        enriched_content = await enrich_query(self.provider, self.local_model, msg.content)

        # Decision gate — short-circuit to local model if eligible (no cloud call)
        if self.local_model and should_use_local(enriched_content, history, research_mode):
            local_resp = await run_local(enriched_content, history, self.local_model)
            if local_resp:
                session.add_message("user", enriched_content)
                session.add_message("assistant", local_resp)
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=local_resp, metadata=msg.metadata or {},
                )
            # else: fall through to cloud silently

        initial_messages = await self.context.build_messages(
            history=history,
            current_message=enriched_content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            response_style=response_style,
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
            await _emit_progress("I'm working on it...")
            progress_task = asyncio.create_task(_heartbeat_loop())

        try:
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages, research_mode=research_mode, on_progress=on_progress or _bus_progress,
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

        self._save_turn(session, all_msgs, len(initial_messages))
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
