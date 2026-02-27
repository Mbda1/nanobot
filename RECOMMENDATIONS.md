# nanobot — Architecture & Stability Recommendations

## 1. Tracing & Observability
- **[DONE] Trace ID Integration**: Implemented `trace_id` in `loguru` using `contextvars`. Every inbound message, cron job, and background task (memory consolidation, flushing) now has a unique ID in the logs.
- **[DONE] Supervisor Suggestion UX**: Created `SupervisorTool` and a corresponding Skill to allow the agent to read and resolve issues detected by the supervisor daemon via `SUPERVISOR_SUGGESTIONS.json`.

## 2. Local LLM Stability (Ollama/Alternative)
- **Pin Model in RAM**: Use `keep_alive: -1` to eliminate cold-start latency. (Current setting in `constants.py` is `-1`).
- **Increase Background Timeouts**: Background tasks (Memory Flush, Consolidation) currently have aggressive timeouts (15-20s). These should be increased to 60-120s to allow completion on local hardware without killing healthy inference.
- **Harden API Resolution**: Use IP addresses (e.g., `127.0.0.1` or `172.17.0.1`) instead of hostnames (`host.docker.internal`) to bypass DNS resolution flickers that cause "Temporary failure in name resolution."
- **Inference Concurrency**: Transition to a backend that supports parallel request processing (unlike Ollama's default sequential queue) to prevent "heavy" background tasks from blocking real-time user requests.

## 3. High-Performance Local Backend (GPU/NPU)
- **Transition from Ollama**: Ollama's architecture (specifically in your setup) is limited to CPU-only.
- **vLLM**: Recommended for Linux/GPU environments. Uses PagedAttention for high throughput and memory efficiency. Supports OpenAI-compatible API.
- **llama.cpp (Server Mode)**: Best for "bare-metal" control. Can be compiled with CUDA, Vulkan, or OpenCL to target specific GPU/NPU hardware.
- **MLX (macOS)**: If the hardware is Apple Silicon (NPS might refer to the Neural Engine), MLX is the most performant option.
- **TensorRT-LLM (NVIDIA)**: If "NPS" refers to a specific NVIDIA architecture, this is the gold standard for performance.

## 4. Feature Enhancements
- **Human-in-the-Loop Fixes**: Extend Supervisor to allow interactive Telegram approvals for non-critical fixes (Suggestion #3 from review).
- **Vector DB for Warm Memory**: Replace raw Markdown topic matching with a lightweight vector store (e.g., ChromaDB) for better scaling.
