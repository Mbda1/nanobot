"""Context builder for assembling agent prompts."""

import base64
import inspect
import json
import mimetypes
import platform
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.config.constants import TOOL_RESULT_CONTEXT_MAX_CHARS


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
    
    async def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        user_message: str = "",
    ) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.

        Args:
            skill_names: Optional list of skills to include.
            user_message: Current user message — used for warm-tier keyword matching.

        Returns:
            Complete system prompt.
        """
        parts = []

        # Core identity
        parts.append(self._get_identity())

        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # Memory context (hot tier always + warm tier on keyword match)
        memory = self.memory.get_memory_context(user_message)
        if inspect.isawaitable(memory):
            memory = await memory
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (hot tier, always loaded)
- Topic memory: {workspace_path}/memory/topics/ (warm tier, auto-loaded when relevant)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel.

## Tool Call Guidelines
- Before calling tools, you may briefly state your intent (e.g. "Let me check that"), but NEVER predict or describe the expected result before receiving it.
- Before modifying a file, read it first to confirm its current content.
- Do not assume a file or directory exists — use list_dir or read_file to verify.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.

## Memory
- Remember important facts: write to {workspace_path}/memory/MEMORY.md
- Recall past events: grep {workspace_path}/memory/HISTORY.md"""

    @staticmethod
    def _inject_runtime_context(
        user_content: str | list[dict[str, Any]],
        channel: str | None,
        chat_id: str | None,
    ) -> str | list[dict[str, Any]]:
        """Append dynamic runtime context to the tail of the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        block = "[Runtime Context]\n" + "\n".join(lines)
        if isinstance(user_content, str):
            return f"{user_content}\n\n{block}"
        return [*user_content, {"type": "text", "text": block}]
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    async def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        response_style: str = "default",
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt — pass current_message for warm-tier topic matching.
        msg_text = current_message if isinstance(current_message, str) else ""
        system_prompt = await self.build_system_prompt(skill_names, user_message=msg_text)
        messages.append({"role": "system", "content": system_prompt})
        if response_style == "simple_first":
            messages.append({
                "role": "system",
                "content": (
                    "Response style policy: Start with a concise direct answer first. "
                    "Do not begin with a long structured brief unless the user explicitly asks for deep research. "
                    "After the direct answer, append exactly one line:\n"
                    "\"Want more? 1) Deep analysis 2) Structured summary 3) Sources only\""
                ),
            })

        # History (sanitize stale/orphaned tool chain artifacts)
        messages.extend(self._sanitize_history_for_tool_chains(history))

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        user_content = self._inject_runtime_context(user_content, channel, chat_id)
        messages.append({"role": "user", "content": user_content})

        return messages

    @staticmethod
    def _sanitize_history_for_tool_chains(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop orphan tool results that no longer have a matching prior tool call envelope."""
        out: list[dict[str, Any]] = []
        pending_tool_ids: set[str] = set()
        for m in history:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                ids = set()
                for tc in (m.get("tool_calls") or []):
                    try:
                        ids.add(str(tc.get("id", "")))
                    except Exception:
                        continue
                pending_tool_ids = {x for x in ids if x}
                out.append(m)
                continue
            if role == "tool":
                tcid = str(m.get("tool_call_id", "")).strip()
                if tcid and tcid in pending_tool_ids:
                    out.append(m)
                    pending_tool_ids.discard(tcid)
                # else: orphan tool result; drop it
                continue
            pending_tool_ids.clear()
            out.append(m)
        return out

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
        research_mode: str = "balanced",
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        compact = self._compact_tool_result(tool_name, result, research_mode=research_mode)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": compact
        })
        return messages

    @staticmethod
    def _compact_tool_result(tool_name: str, content: str, research_mode: str = "balanced") -> str:
        """Keep tool outputs compact before they are sent back to the cloud model."""
        if not isinstance(content, str):
            return str(content)
        name = (tool_name or "").strip().lower()
        raw = content.strip()
        if not raw:
            return raw

        if name == "web_search":
            compact = ContextBuilder._compact_web_search(raw, research_mode=research_mode)
        elif name == "web_fetch":
            compact = ContextBuilder._compact_web_fetch(raw, research_mode=research_mode)
        else:
            compact = raw

        if len(compact) > TOOL_RESULT_CONTEXT_MAX_CHARS:
            compact = compact[:TOOL_RESULT_CONTEXT_MAX_CHARS] + "\n... (truncated for context budget)"
        return compact

    @staticmethod
    def _compact_web_search(text: str, research_mode: str = "balanced") -> str:
        m = re.search(r"Results for:\s*(.+?)(?:\s+\(provider=(\w+)\))?\s*$", text.splitlines()[0] if text else "")
        query = m.group(1) if m else ""
        provider = (m.group(2) if m else None) or "unknown"

        max_items = 2 if research_mode == "cheap" else 4 if research_mode == "deep" else 3
        lines = text.splitlines()
        items: list[tuple[str, str, str]] = []
        cur_title = ""
        cur_url = ""
        cur_snip = ""
        for ln in lines[1:]:
            s = ln.strip()
            if re.match(r"^\d+\.\s+", s):
                if cur_title or cur_url:
                    items.append((cur_title, cur_url, cur_snip))
                cur_title = re.sub(r"^\d+\.\s+", "", s).strip()
                cur_url = ""
                cur_snip = ""
            elif s.startswith("http://") or s.startswith("https://"):
                cur_url = s
            elif s:
                cur_snip = (cur_snip + " " + s).strip()
        if cur_title or cur_url:
            items.append((cur_title, cur_url, cur_snip))

        out = [f"[EVIDENCE:web_search provider={provider}]"]
        if query:
            out.append(f"query: {query}")
        for i, (title, url, snip) in enumerate(items[:max_items], 1):
            out.append(f"{i}. {title}")
            if url:
                out.append(f"   url: {url}")
            if snip:
                out.append(f"   fact: {snip[:220]}")
        if len(items) > max_items:
            out.append(f"... {len(items) - max_items} additional results omitted")
        return "\n".join(out)

    @staticmethod
    def _compact_web_fetch(text: str, research_mode: str = "balanced") -> str:
        try:
            data = json.loads(text)
        except Exception:
            data = None
        if not isinstance(data, dict):
            return text

        max_excerpt = 600 if research_mode == "cheap" else 1400 if research_mode == "deep" else 900
        src = data.get("finalUrl") or data.get("url") or ""
        body = str(data.get("text", "")).strip()
        excerpt = body[:max_excerpt]
        return "\n".join([
            "[EVIDENCE:web_fetch]",
            f"url: {src}",
            f"status: {data.get('status', '?')}",
            f"extractor: {data.get('extractor', '?')}",
            f"excerpt:\n{excerpt}",
        ])
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant"}

        # Always include content — some providers (e.g. StepFun) reject
        # assistant messages that omit the key entirely.
        msg["content"] = content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Include reasoning content when provided (required by some thinking models)
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content

        messages.append(msg)
        return messages
