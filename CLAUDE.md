# nanobot — CLAUDE.md

## What it is
Lightweight personal AI assistant framework (~5k lines). Runs a 24/7 Telegram bot
backed by LiteLLM (OpenRouter/Anthropic). Fork of HKUDS/nanobot with extensive
local customisations — local LLM inference, three-tier memory, token governance,
supervisor hardening, browser automation, and a decision gate.

## Dev workflow — ALWAYS follow this order
```
nanobot stop                          # stop gateway + supervisor + dashboard first
# ... make code changes ...
pip install -e ~/nanobot --break-system-packages
nanobot start                         # restart all three agents
```
Never edit Python files while gateway/supervisor are running.

---

## Key file locations

| Path | Purpose |
|------|---------|
| `nanobot/agent/loop.py` | AgentLoop — main LLM ↔ tool cycle; decision gate; token governor |
| `nanobot/agent/context.py` | Builds system prompt (memory + skills + tool result compaction) |
| `nanobot/agent/memory.py` | Three-tier memory persistence (hot/warm/cold) |
| `nanobot/agent/subagent.py` | SubagentManager — delegate/spawn; curator local routing; per-job governor |
| `nanobot/agent/decision_gate.py` | Keyword classifier — short-circuits eligible messages to local model |
| `nanobot/agent/local_llm.py` | Direct httpx wrapper for llama.cpp (OpenAI-compat) and Ollama APIs |
| `nanobot/agent/enrichment.py` | Query enrichment via local qwen2.5-3b before cloud call |
| `nanobot/agent/embeddings.py` | nomic-embed-text cosine similarity for warm-tier topic matching |
| `nanobot/agent/eval.py` | LLM-as-judge eval suite (qwen2.5-3b, zero cloud tokens) |
| `nanobot/agent/usage.py` | Token usage logger → USAGE.jsonl |
| `nanobot/agent/tools/web.py` | WebSearch (DuckDuckGo via ddgs) + WebFetch |
| `nanobot/agent/tools/browser.py` | Playwright/Chromium headless browser (anti-detection) |
| `nanobot/agent/tools/delegate.py` | DelegateTool — synchronous worker dispatch (supervisor pattern) |
| `nanobot/agent/tools/filesystem.py` | File I/O tools with Obsidian backup safety |
| `nanobot/channels/telegram.py` | Telegram bot channel |
| `nanobot/providers/litellm_provider.py` | LiteLLM wrapper (cloud provider) |
| `nanobot/supervisor/daemon.py` | Supervisor watchdog — hardened, local-LLM log analysis |
| `nanobot/cli/commands.py` | CLI: ps / stop [agent] / start [agent] / restart |
| `nanobot/session/manager.py` | Conversation history (JSONL, append-only) |
| `nanobot/config/constants.py` | Central variable table — change values here, propagates everywhere |
| `nanobot/config/schema.py` | Pydantic config schema |

---

## Config (outside repo — never commit)
- `~/.nanobot/config.json` — API keys, model, telegram token
  - `config['channels']['telegram']['token']` — bot token
  - `config['agents']['defaults']['model']` — cloud model
  - `config['agents']['defaults']['local_model']` — local model alias
- `~/.nanobot/workspace/` — memory, sessions, skills, cron jobs
- `~/.nanobot/workspace/skills/` — user skills (override built-ins)
- `~/.nanobot/workspace/memory/MEMORY.md` — hot-tier long-term memory (cap: 200 lines)
- `~/.nanobot/workspace/memory/topics/` — warm-tier overflow topics (semantic search)
- `~/.nanobot/workspace/memory/HISTORY.md` — cold-tier event log (grep-only)
- `~/.nanobot/workspace/memory/USAGE.jsonl` — token usage per call
- `~/.nanobot/workspace/memory/SUPERVISOR_LOG.md` — supervisor audit log
- `~/.nanobot/workspace/memory/METRICS.jsonl` — instrumentation events

**config.json takes precedence over constants.py for model selection — always check both.**

---

## Provider / model stack

| Tier | Model | Runtime | Used for |
|------|-------|---------|---------|
| Cloud | `openrouter/anthropic/claude-haiku-4-5` | OpenRouter | User-facing responses |
| Local inference | `qwen2.5-3b` (qwen2.5-3b-q4.gguf, ~2 GB) | llama.cpp in WSL2 | Enrichment, memory consolidation, supervisor log analysis, decision gate, curator delegates |
| Embeddings | `nomic-embed-text` | Ollama on Windows | Warm-tier cosine similarity only |

**CRITICAL — LiteLLM always injects the cloud api_base into every call. Never pass a local
model through `provider.chat()`. Always use `local_llm.ollama_chat()` / `local_llm.py` directly.**

Local endpoint: `http://127.0.0.1:8080` (llama.cpp server, OpenAI-compat `/v1/chat/completions`)
Embeddings endpoint: `http://host.docker.internal:11434` (Ollama on Windows, `/api/embed`)

---

## Local customisations in this fork

### 1. Query enrichment (`loop.py` → `enrichment.py`)
Messages ≥6 words are rewritten by local qwen2.5-3b before the cloud call.
10s timeout, silent fallback to original. Skips `/commands` and short messages.

### 2. Decision gate (`loop.py` → `decision_gate.py`)
Pure-Python keyword classifier — zero LLM overhead. Messages with no cloud-requiring
keywords, with prior history, and not in research mode are short-circuited to qwen2.5-3b.
Falls through silently to cloud on empty response or timeout.
Recorded as `source="local_gate"` in USAGE.jsonl.

### 3. Token governor (P1)
- **Per-turn cap**: 50k tokens across all iterations in one user turn. Aborts with ⚠️
  user-visible message, records `source="governor_abort"`.
- **Per-job cap**: 30k tokens per delegate job. Returns partial result with cap message.
- Both additive with existing circuit breakers (CIRCUIT_BREAKER_PER_TOOL, CIRCUIT_BREAKER_CONSECUTIVE).

### 4. Curator local routing / Obi fix (P1)
Curator delegates bypass the cloud entirely. `SubagentManager._execute_local_subagent()`
calls `ollama_chat()` directly. Recorded as `source="obi_local"`. Eliminates the
323k-token event caused by SubagentManager never receiving `local_model`.

### 5. Three-tier memory (`memory.py`, `embeddings.py`)
- **Hot**: MEMORY.md — always loaded, capped at 200 lines. Evicts oldest `##` section to warm tier.
- **Warm**: `memory/topics/*.md` — semantic cosine search (nomic-embed-text, threshold 0.50), keyword fallback if Ollama down.
- **Cold**: HISTORY.md — append-only grep log, never auto-loaded.
- Consolidation uses local qwen2.5-3b. Fires AFTER response is sent (never blocks inference).
- Guardrails: MEMORY_CHUNK_FAIL_FAST_THRESHOLD=2 raw fallback; 60s hard budget.

### 6. Supervisor hardening (`supervisor/daemon.py`)
- Polls every 30s; uses local qwen2.5-3b for log analysis (zero cloud tokens).
- Only `restart_gateway` is auto-applied; all other issues are `suggest`.
- Restart storm protection: 3 events in 10 min required before auto-restart.
- Cooldown: 3 min between any two restart attempts.
- Deduplication: SIGTERMs extra gateway PIDs every poll.
- Telegram alerts: critical severity or auto-fix only (no warning spam).
- **Do NOT run alongside systemd nanobot-gateway.service** (duplicate messages).

### 7. Browser automation (`tools/browser.py`)
Playwright/Chromium headless with anti-detection (navigator.webdriver override, real UA).
`--disable-http2` required for WSL2. Cloudflare/advanced bot detection still blocks;
basic sites and Craigslist work. Graceful ImportError fallback — never crashes agent.

### 8. Process control CLI (`cli/commands.py`)
`nanobot ps` / `stop [agent]` / `start [agent]` / `restart`
Dashboard server runs on port 8765 — `http://localhost:8765`.

### 9. Obsidian backup safety (`tools/filesystem.py`)
Any write/edit targeting an Obsidian vault path auto-backs up the file first.
Retention: 30 days. Configurable via `OBSIDIAN_BACKUP_DIR` / `OBSIDIAN_BACKUP_RETENTION_DAYS`.

### 10. Token usage tracking (`usage.py`)
USAGE.jsonl: one JSON line per LLM call — ts, model, source, prompt, completion, total, latency_ms.
Sources: `agent`, `memory_chunk`, `memory_merge`, `enrichment`, `local_gate`, `obi_local`, `governor_abort`.
Dashboard at port 8765 visualises this data.

---

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
User skills: `~/.nanobot/workspace/skills/*/SKILL.md` (override built-ins)

---

## Editing JSON files — use Python
```bash
python3 -c "
import json
with open('file.json') as f: d = json.load(f)
d['key'] = 'value'
with open('file.json', 'w') as f: json.dump(d, f, indent=2)
"
```
The Edit tool is unreliable on large JSON files (string matching fails on long blocks).

---

## Tests
```bash
cd ~/nanobot
pytest -m 'not llm' -q      # fast suite (~3s, 100+ tests) — run after every change
pytest -m llm                # LLM-dependent tests (~95s) — requires llama.cpp running
pytest tests/test_commands.py
pytest tests/test_cron_*.py
```
- Circuit breaker tests use `exec` tool (not `web_search`) — web_search has its own
  per-turn limit (3 balanced / 6 deep) which fires below CIRCUIT_BREAKER_PER_TOOL=5.
- `@pytest.mark.llm` tests are skipped automatically if llama.cpp is unreachable.

---

## Gotchas

| Issue | Fix |
|-------|-----|
| LiteLLM injects cloud api_base into ALL calls | Use `local_llm.ollama_chat()` directly for local models — never `provider.chat()` |
| `write_file` has no append mode | read → concatenate → write |
| Session messages are append-only | Never modify past messages |
| `HEARTBEAT.md` with instructions triggers LLM every 30 min | Keep it headers/comments only |
| `allow_from` empty list in telegram config | Empty = allow all (not deny all) |
| Supervisor + systemd nanobot-gateway.service running together | Duplicate Telegram messages — systemd service is DISABLED |
| llama.cpp cold start is slow | Warmup fires on gateway startup (120s timeout, max_tokens=1) |
| config.json overrides constants.py | Always check both when debugging model selection |
| max_tokens=2000 causes silent truncation | Use 4096 — that's the correct ceiling |
| qwen3.5 has no sub-27B local variant | Attempted 2026-02-27, thrashed RAM, rolled back to qwen2.5-3b |

---

## Architecture Principles

### 1. Fix the Architecture, Not the Symptoms

When a component requires 3+ defensive layers (timeouts, retries, fail-fast thresholds,
cooldowns, circuit breakers) to function reliably — that is a signal the **component is wrong**,
not that the defenses are right.

Before adding another layer, ask: *"Why does this keep breaking?"* not *"How do I make it less broken?"*

### 2. Infrastructure Placement (WSL2 runtime)

| Layer | Where it should run | Rationale |
|-------|--------------------|----|
| Cloud LLM | External (OpenRouter) | Unavoidable |
| Local inference | WSL2-native (llama.cpp) | No Windows hop, full process control, fits in RAM |
| Embeddings | Ollama on Windows | Acceptable — read-only, has keyword fallback if down |
| Network hops (WSL2 → host.docker.internal → Windows) | Avoid | Latency + instability + loss of process control |

**Lesson from Ollama migration (2026-02-27):** Ollama was architecturally wrong — Windows-native
process, 4.7 GB model competing for ~5.7 GB free RAM, cross-OS network hop, no Linux-side control.
llama.cpp in WSL2 eliminated all failure modes. Should have been the first choice given the RAM analysis.

### 3. Core Data Pattern: Chunk → Collect → Assemble → Merge

Any operation on large or unbounded data MUST follow this pipeline:
1. **CHUNK** — split into small, predictable pieces (Python, zero tokens)
2. **COLLECT** — process each piece with a small, focused LLM call
3. **ASSEMBLE** — combine partial results in Python (zero tokens)
4. **MERGE** — one final LLM call on the assembled output

Never send unbounded data to an LLM in one call. Always anticipate overflow.

### 4. Token Cost Hierarchy (cheapest → most expensive)

```
Python computation (free)
  → local qwen2.5-3b on llama.cpp (~0 cost, ~1-5s)
    → cloud haiku-4-5 on OpenRouter (paid, ~1-3s warm)
```

Design new features to use the cheapest tier that can handle the task.
- Classification, routing, simple Q&A → local model or pure Python
- Tool use, web search, complex reasoning → cloud
- Memory, enrichment, log analysis, eval judge → always local

### 5. Reactive vs. Proactive Engineering

**Reactive (avoid):** Gateway crashed → fix the crash. Timeout → add timeout handling.

**Proactive (prefer):** Pattern of timeouts across sessions → question the infrastructure.
Multiple defensive layers around same component → consider replacement.
Same class of bug recurs → root cause has not been fixed.

### 6. Anchoring Bias — Treat Existing Architecture as a Variable

Shipping with Component X does not mean Component X is correct. Before adding a third
defensive layer:
1. What problem was this component chosen to solve?
2. Is there a simpler component without the failure modes?
3. What would a fresh design look like?

### 7. When to Patch vs. When to Replace

**Patch:** Bug is in application logic (not infrastructure). Failure is rare / edge-case.
Component is correctly placed, known quirk.

**Replace:** Same component fails across multiple sessions. Requires 3+ defensive layers.
Component is in the wrong place (wrong OS plane, wrong size, wrong protocol).
A clearly better alternative exists.

---

## Design Requirements (active constraints)

These are non-negotiable design requirements derived from production incidents.

### Token governance
- No turn may consume >50k tokens (GOVERNOR_PER_TURN_TOKENS). Abort with ⚠️ user message.
- No delegate job may consume >30k tokens (GOVERNOR_PER_JOB_TOKENS). Return partial result.
- Both caps are additive with circuit breakers — do not remove existing breakers.

### Local-first routing
- Curator delegates MUST use local model — never cloud. Obi reads/writes files; it needs
  no cloud intelligence. Cloud routing here was the source of the 323k-token event.
- Decision gate MUST be conservative: false-negatives (cloud when local works) are fine;
  false-positives (local when cloud needed) break the user experience.
- Any new background/enrichment task defaults to local model unless it genuinely requires
  tool use or web access.

### Consolidation timing
- Memory consolidation fires AFTER the response is sent to the user. Never block inference.
- Local model calls during consolidation have a 60s hard budget total.
- Chunk failures fall back to raw truncated text — consolidation never skips silently.

### Supervisor autonomy
- Only `restart_gateway` is auto-applied. All other supervisor actions are `suggest`.
- Restart requires 3 events in 10 min (storm protection) + 3 min cooldown.
- Telegram alerts: critical/auto-fix only. No spam.

### Write file discipline
- Large content = multiple small focused files, not one monolith.
- Outline-first: write INDEX with stubs → fill each section separately.
- max_tokens=4096 is the ceiling. 2000 causes silent truncation loops.

---

## Configuration reference

| Constant | Value | File | Notes |
|----------|-------|------|-------|
| `CLOUD_MODEL_DEFAULT` | `openrouter/anthropic/claude-haiku-4-5` | constants.py | User-facing |
| `LOCAL_MODEL_DEFAULT` | `qwen2.5-3b` | constants.py | Enrichment, memory, gate |
| `JUDGE_MODEL_DEFAULT` | `qwen2.5-3b` | constants.py | Supervisor + eval |
| `MAX_TOKENS_DEFAULT` | 4096 | constants.py | Never lower — truncation |
| `MEMORY_WINDOW_DEFAULT` | 25 | constants.py | Consolidation trigger |
| `MEMORY_CHUNK_SIZE` | 8 | constants.py | Messages per chunk |
| `GOVERNOR_PER_TURN_TOKENS` | 50,000 | constants.py | Hard per-turn cap |
| `GOVERNOR_PER_JOB_TOKENS` | 30,000 | constants.py | Hard per-delegate cap |
| `TIMEOUT_DECISION_GATE` | 30s | constants.py | Local gate + curator |
| `LOCAL_GATE_MAX_TOKENS` | 800 | constants.py | Gate/curator response budget |
| `CIRCUIT_BREAKER_PER_TOOL` | 5 | constants.py | Max tool calls per turn |
| `CIRCUIT_BREAKER_CONSECUTIVE` | 2 | constants.py | Max identical consecutive calls |
| `MEMORY_HOT_MAX_LINES` | 200 | constants.py | Hot tier cap (MEMORY.md) |
| `SIMILARITY_THRESHOLD` | 0.50 | constants.py | Warm-tier cosine threshold |
| `DELEGATE_MAX_ITERATIONS` | 15 | constants.py | Max delegate loop iterations |
