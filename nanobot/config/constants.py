"""Central constants for nanobot. Change a value here — it propagates everywhere.

Quick reference:
  Change cloud model   → CLOUD_MODEL_DEFAULT
  Change local model   → LOCAL_MODEL_DEFAULT / LOCAL_API_BASE
  Change token budget  → MAX_TOKENS_DEFAULT / MEMORY_MERGE_MAX_TOKENS
  Change memory tuning → MEMORY_CHUNK_SIZE / MEMORY_WINDOW_DEFAULT
  Change timeouts      → TIMEOUT_* values
"""

# --- Models ---
CLOUD_MODEL_DEFAULT = "openrouter/anthropic/claude-haiku-4-5"
LOCAL_MODEL_DEFAULT = "ollama/mistral"
LOCAL_API_BASE      = "http://host.docker.internal:11434"

# --- Agent limits ---
MAX_TOKENS_DEFAULT    = 4096
MAX_ITERATIONS        = 40
TEMPERATURE_DEFAULT   = 0.1
MEMORY_WINDOW_DEFAULT = 25

# --- Memory pipeline (Chunk → Collect → Assemble → Merge) ---
MEMORY_CHUNK_SIZE       = 8     # messages per chunk
MEMORY_CHUNK_MAX_TOKENS = 300   # summary tokens per chunk (plain text)
MEMORY_MERGE_MAX_TOKENS = 2048  # final save_memory call

# --- Timeouts (seconds) ---
TIMEOUT_ENRICHMENT    = 10.0   # query enrichment          (loop.py)
TIMEOUT_CHUNK_SUMMARY = 20.0   # chunk summarization       (memory.py)
TIMEOUT_WEB_FETCH     = 30.0   # HTTP fetch                (tools/web.py)

# --- Ollama keep-alive ---
# -1 = keep model loaded in RAM indefinitely (never unload on idle)
# Ollama default is 5 minutes, which causes cold-start timeouts
OLLAMA_KEEP_ALIVE     = -1

# --- Limits ---
TOOL_RESULT_MAX_CHARS = 500    # tool output truncation    (loop.py)

# --- Context builder ---
BOOTSTRAP_FILES  = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
TIMESTAMP_FORMAT = "%Y-%m-%d (%A)"   # date-only → system prompt stable 24h (cache hits)
