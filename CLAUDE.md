# nanobot — CLAUDE.md

## What it is
Lightweight personal AI assistant framework (~4k lines). Runs a 24/7 Telegram bot
backed by LiteLLM (OpenRouter/Anthropic). This is a fork of HKUDS/nanobot with
local customisations (see below).

## Dev workflow — ALWAYS follow this order
```
nanobot stop                          # stop gateway + supervisor first
# ... make code changes ...
pip install -e ~/nanobot --break-system-packages
nanobot start                         # restart all
```
Never edit Python files while gateway/supervisor are running.

## Key file locations

| Path | Purpose |
|------|---------|
| `nanobot/agent/loop.py` | AgentLoop — main LLM ↔ tool cycle |
| `nanobot/agent/context.py` | Builds system prompt (memory + skills) |
| `nanobot/agent/memory.py` | MEMORY.md + HISTORY.md persistence |
| `nanobot/agent/tools/web.py` | WebSearch (DuckDuckGo via ddgs) — customised |
| `nanobot/agent/tools/*.py` | All built-in tools |
| `nanobot/channels/telegram.py` | Telegram bot channel |
| `nanobot/providers/litellm_provider.py` | LiteLLM wrapper (default provider) |
| `nanobot/supervisor/daemon.py` | Supervisor watchdog — customised |
| `nanobot/cli/commands.py` | CLI: agent/gateway/ps/stop/start/restart |
| `nanobot/session/manager.py` | Conversation history (JSONL, append-only) |
| `nanobot/config/schema.py` | Pydantic config schema |

## Config (outside repo — never commit)
- `~/.nanobot/config.json` — API keys, model, telegram token
  - `config['channels']['telegram']['token']` — bot token
  - `config['agents']['defaults']['model']` — cloud model
  - `config['agents']['defaults']['local_model']` — ollama model
- `~/.nanobot/workspace/` — memory, sessions, skills, cron jobs
- `~/.nanobot/workspace/skills/` — user skills (override built-ins)
- `~/.nanobot/workspace/memory/MEMORY.md` — long-term memory
- `~/.nanobot/workspace/memory/SUPERVISOR_LOG.md` — supervisor audit log

## Local customisations in this fork
1. **Query enrichment** (`loop.py` `_process_message`): prompts ≥6 words rewritten
   by local Ollama/Mistral before sending to cloud. 10s timeout, silent fallback.
2. **Supervisor enhancements** (`supervisor/daemon.py`):
   - Uses Ollama/Mistral for log analysis (zero cloud tokens)
   - Only `restart_gateway` auto-applied; all other issues become `suggest`
   - Sends Telegram alerts on startup, crash, recovery
3. **Process control CLI** (`cli/commands.py`):
   `nanobot ps` / `stop [agent]` / `start [agent]` / `restart`
4. **Web search** (`tools/web.py`): DuckDuckGo via `ddgs` package (not built-in)

## Tool pattern
```python
class MyTool(Tool):
    @property
    def name(self): return "tool_name"
    @property
    def description(self): return "..."
    @property
    def parameters(self): return {"type": "object", "properties": {...}, "required": [...]}
    async def execute(self, param: str, **kwargs) -> str: ...
```

## Skill pattern (Markdown files)
```yaml
---
name: skill-name
description: "What it does"
metadata:
  nanobot:
    always: true   # load in every context window
---
# Instructions for the agent in plain Markdown
```
Built-in skills: `nanobot/skills/*/SKILL.md`
User skills: `~/.nanobot/workspace/skills/*/SKILL.md` (these override built-ins)

## Editing JSON files — try Python first
Use `python3 -c "import json; ..."` to read/modify/write JSON config files.
The Edit tool is unreliable on large JSON files (string matching fails on long blocks).
```bash
python3 -c "
import json
with open('file.json') as f: d = json.load(f)
d['key'] = 'value'
with open('file.json', 'w') as f: json.dump(d, f, indent=2)
"
```

## Gotchas
- `write_file` has no append mode — read → concatenate → write
- Session messages are append-only (never modify past messages)
- `HEARTBEAT.md` must contain only headers/comments — any instructions
  trigger the LLM every 30 min and drain tokens
- `allow_from` in telegram config: empty list = allow all
- Supervisor must NOT run alongside systemd nanobot-gateway.service
  (causes duplicate Telegram messages)
- Local model: Ollama on Windows at `http://host.docker.internal:11434`
  model `mistral:latest` (CPU only, ~0.4s warm, 30s+ cold)

## Tests
```bash
pytest                          # all
pytest tests/test_commands.py   # CLI
pytest tests/test_cron_*.py     # cron
```
Framework: pytest + pytest-asyncio

## Provider / model
- Cloud: `openrouter/anthropic/claude-haiku-4-5` (user-facing, via OpenRouter)
- Local: `ollama/mistral` (background tasks: memory consolidation, log analysis, query enrichment)
- LiteLLM routes both; local needs no API key

## Architecture Principles

### Core Pattern: Chunk → Collect → Assemble
Any operation on large or unbounded data MUST follow this pipeline:
1. **CHUNK** — split input into small, predictable pieces (Python, zero tokens)
2. **COLLECT** — process each piece with a small, focused LLM call
3. **ASSEMBLE** — combine partial results in Python (zero tokens)
4. **MERGE** — one final LLM call on the assembled output

Never send unbounded data to an LLM in one call. Always anticipate overflow.

### Memory Consolidation
- Consolidation uses local Mistral (zero cloud cost)
- Chunks: 8 messages → one short text summary per chunk (~100 tokens out)
- Assembly: Python string join (no LLM)
- Merge: one final call with `save_memory` tool (bounded input)
- Fallback: if chunk summary times out, use raw truncated text (never skip)

### Write File Pattern
- Large content = multiple small focused files, not one monolith
- Use outline-first: write INDEX/README with section stubs → fill each section separately
- max_tokens=4096 is the correct ceiling; 2000 caused silent truncation loops

### Configuration Boundaries
| Parameter        | Value | Rationale |
|-----------------|-------|-----------|
| max_tokens      | 4096  | Headroom for complex tool-call JSON without truncation |
| memory_window   | 25    | 12 messages in live context; consolidation fires every 25 new |
| chunk_size      | 8     | ~800 token input per chunk; Mistral 7B handles reliably |
| chunk_max_tokens| 300   | Short summary only; no tool call needed |
| merge_max_tokens| 2048  | Full MEMORY.md update via save_memory tool |
