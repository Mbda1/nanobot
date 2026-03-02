"""Central constants for nanobot. Change a value here — it propagates everywhere.

Quick reference:
  Change cloud model   → CLOUD_MODEL_DEFAULT
  Change local model   → LOCAL_MODEL_DEFAULT (enrichment/memory) / JUDGE_MODEL_DEFAULT (eval/supervisor)
  Change API base      → LOCAL_API_BASE
  Change local backend → LOCAL_LLM_BACKEND
  Change token budget  → MAX_TOKENS_DEFAULT / MEMORY_MERGE_MAX_TOKENS
  Change memory tuning → MEMORY_CHUNK_SIZE / MEMORY_WINDOW_DEFAULT
  Change timeouts      → TIMEOUT_* values
"""
import os

# --- Models ---
CLOUD_MODEL_DEFAULT = "openrouter/anthropic/claude-haiku-4-5"
LOCAL_MODEL_DEFAULT = "qwen2.5-3b"      # llama.cpp local model alias (enrichment + memory)
JUDGE_MODEL_DEFAULT = "qwen2.5-3b"      # supervisor/eval judge on local llama.cpp
EMBED_MODEL_DEFAULT = "nomic-embed-text"  # semantic warm-tier search model (Ollama /api/embed)
LOCAL_API_BASE      = os.getenv("NB_LOCAL_API_BASE", "http://127.0.0.1:8080").strip()
LOCAL_LLM_BACKEND   = os.getenv("NB_LOCAL_LLM_BACKEND", "openai").strip().lower()
LOCAL_API_KEY       = os.getenv("NB_LOCAL_API_KEY", "").strip()

# --- Obsidian backup safety ---
# Any write/edit targeting an Obsidian vault path is auto-backed up first.
OBSIDIAN_BACKUP_DIR = os.path.expanduser(
    os.getenv("NB_OBSIDIAN_BACKUP_DIR", "~/.nanobot/workspace/obsidian_backups")
).strip()
OBSIDIAN_BACKUP_RETENTION_DAYS = int(os.getenv("NB_OBSIDIAN_BACKUP_RETENTION_DAYS", "30"))

# --- Agent limits ---
MAX_TOKENS_DEFAULT    = 4096
MAX_ITERATIONS        = 40
TEMPERATURE_DEFAULT   = 0.1
MEMORY_WINDOW_DEFAULT = 25

# --- Memory pipeline (Chunk → Collect → Assemble → Merge) ---
MEMORY_CHUNK_SIZE       = 8     # messages per chunk
MEMORY_CHUNK_MAX_TOKENS = 300   # summary tokens per chunk (plain text)
MEMORY_MERGE_MAX_TOKENS = 2048  # final save_memory call
MEMORY_CHUNK_FAIL_FAST_THRESHOLD = 2  # after N chunk summary failures, skip LLM for remaining chunks

# --- Timeouts (seconds) ---
TIMEOUT_ENRICHMENT    = 10.0   # query enrichment          (enrichment.py)
TIMEOUT_EMBED         =  5.0   # semantic embedding        (embeddings.py) — fast model, fail-fast
TIMEOUT_CHUNK_SUMMARY = 20.0   # chunk summarization       (memory.py) — cloud, fast
TIMEOUT_MEMORY_CONSOLIDATION = 60.0   # total consolidation budget per run
TIMEOUT_WEB_FETCH     = 30.0   # HTTP fetch                (tools/web.py)
TIMEOUT_JUDGE         = 60.0   # LLM-as-judge eval         (eval.py) — keep headroom for local cold starts
TIMEOUT_FLUSH         = 30.0   # local-LLM memory/todo flush timeout

# --- Ollama keep-alive ---
# -1 = keep model loaded in RAM indefinitely (never unload on idle)
# Ollama default is 5 minutes, which causes cold-start timeouts
OLLAMA_KEEP_ALIVE     = -1

# --- Limits ---
TOOL_RESULT_MAX_CHARS = 500    # tool output truncation    (loop.py)
TOOL_RESULT_CONTEXT_MAX_CHARS = 1600  # max tool text injected back into model context
MAX_CLOUD_INPUT_EST_TOKENS = 8000     # estimated prompt-token budget before cloud offload
TARGET_CLOUD_INPUT_EST_TOKENS = 6500  # compaction target when budget is exceeded

# --- Search behavior ---
WEB_SEARCH_DEFAULT_COUNT = 3   # default results per query for web_search
WEB_SEARCH_MAX_COUNT = 5       # hard cap unless deep-research mode is explicit

# --- Memory tiering ---
# MEMORY.md is capped at this many lines (hot tier).
# When exceeded, the oldest ## section is moved to memory/topics/<slug>.md (warm tier).
# Warm-tier files are matched by semantic similarity then keyword fallback.
MEMORY_HOT_MAX_LINES   = 200
SIMILARITY_THRESHOLD   = 0.50  # minimum cosine similarity to load warm-tier topic
MEMORY_FLUSH_THRESHOLD = 18    # soft flush trigger before consolidation window slides

# --- Supervisor / delegation ---
DELEGATE_MAX_ITERATIONS = 15   # max LLM iterations per delegated worker

# --- Circuit breakers (loop.py) ---
# Trip when the same tool is called more than this many times in one turn.
CIRCUIT_BREAKER_PER_TOOL   = 5
# Trip immediately when the same tool + same primary argument is repeated
# this many times consecutively (classic stuck-loop signature).
CIRCUIT_BREAKER_CONSECUTIVE = 2

# --- Token governor (hard caps) ---
GOVERNOR_PER_TURN_TOKENS = 50_000   # max total tokens for one user turn (all iterations)
GOVERNOR_PER_JOB_TOKENS  = 30_000   # max total tokens for one delegate job

# --- Decision gate + local routing ---
TIMEOUT_DECISION_GATE = 30.0   # local response timeout (curator jobs need more headroom)
LOCAL_GATE_MAX_TOKENS = 800    # max local response tokens

# --- Context builder ---
BOOTSTRAP_FILES  = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
TIMESTAMP_FORMAT = "%Y-%m-%d (%A)"   # date-only → system prompt stable 24h (cache hits)
