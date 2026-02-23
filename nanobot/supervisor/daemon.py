"""NanoSupervisor daemon — tails nanobot logs, detects issues with Claude, and applies fixes."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from loguru import logger


# ---------------------------------------------------------------------------
# Log Watcher — async tail of nanobot.log
# ---------------------------------------------------------------------------


async def tail_file(path: Path) -> AsyncIterator[str]:
    """Async generator that yields new lines appended to *path*.

    Starts from the end of the file so only future lines are returned.
    Polls every 100 ms when idle.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        fh.seek(0, os.SEEK_END)  # skip existing content on startup
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip("\n")
            else:
                await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Issue Detector — send log batches to Claude and parse JSON response
# ---------------------------------------------------------------------------


async def detect_issues(lines: list[str]) -> list[dict]:
    """Send a batch of log lines to the local LLM and return a list of issue dicts.

    Each issue dict has keys:
      description, severity, fix_type, fix_details
    fix_type ∈ {restart_gateway, edit_config, patch_file, edit_workspace, none}
    severity ∈ {critical, warning, info}
    """
    try:
        import litellm
    except ImportError:
        logger.warning("litellm not installed; skipping issue detection")
        return []

    batch_text = "\n".join(lines)
    system_prompt = (
        "You are a nanobot supervisor. Analyze the following log lines for errors or issues.\n"
        "Return a JSON array of issues with fields:\n"
        "  description (str), severity (critical|warning|info),\n"
        "  fix_type (restart_gateway|edit_config|patch_file|edit_workspace|none),\n"
        "  fix_details (dict with keys depending on fix_type).\n"
        "\n"
        "fix_details schema per fix_type:\n"
        "  restart_gateway: {}\n"
        "  edit_config: {\"patch\": {<key>: <value>, ...}}\n"
        "  patch_file: {\"file\": \"<abs_path>\", \"old_string\": \"...\", \"new_string\": \"...\"}\n"
        "  edit_workspace: {\"file\": \"<relative_path_in_workspace>\", \"content\": \"...\"}\n"
        "  none: {}\n"
        "\n"
        "Return [] if no actionable issues found.\n"
        "Return ONLY valid JSON, no markdown fences, no commentary."
    )

    try:
        response = await litellm.acompletion(
            model="ollama/mistral",
            api_base="http://host.docker.internal:11434",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_text},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        issues = json.loads(raw)
        if not isinstance(issues, list):
            return []
        return issues
    except json.JSONDecodeError as exc:
        logger.warning(f"Local LLM returned non-JSON: {exc}")
        return []
    except Exception as exc:
        logger.error(f"Issue detection failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Fix Engine — execute fixes
# ---------------------------------------------------------------------------


async def apply_fix(issue: dict, config, workspace: Path) -> str:
    """Execute the fix described in *issue* and return a human-readable result string."""
    fix_type = issue.get("fix_type", "none")
    details = issue.get("fix_details", {}) or {}

    try:
        if fix_type == "restart_gateway":
            return await _restart_gateway()

        if fix_type == "edit_config":
            return _edit_config(details.get("patch", {}))

        if fix_type == "patch_file":
            return _patch_file(
                details.get("file", ""),
                details.get("old_string", ""),
                details.get("new_string", ""),
            )

        if fix_type == "edit_workspace":
            return _edit_workspace(
                workspace,
                details.get("file", ""),
                details.get("content", ""),
            )

        return "no fix applied"

    except Exception as exc:
        msg = f"fix failed: {exc}"
        logger.error(msg)
        return msg


async def _restart_gateway() -> str:
    """Kill the running gateway process and start a fresh one."""
    # Send SIGTERM to any process with 'nanobot gateway' in its command line
    killed = False
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nanobot gateway"],
            capture_output=True,
            text=True,
        )
        pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed = True
            except (ProcessLookupError, ValueError):
                pass
    except FileNotFoundError:
        pass  # pgrep not available

    await asyncio.sleep(2)  # give old process time to exit

    # Spawn a new gateway (detached)
    try:
        subprocess.Popen(
            ["nanobot", "gateway"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"gateway {'restarted' if killed else 'started'}"
    except FileNotFoundError:
        return "restart failed: nanobot not found in PATH"


def _edit_config(patch: dict) -> str:
    """Apply *patch* dict to ~/.nanobot/config.json."""
    config_path = Path.home() / ".nanobot" / "config.json"
    if not config_path.exists():
        return "edit_config failed: config.json not found"

    with open(config_path, encoding="utf-8") as fh:
        data = json.load(fh)

    def _deep_update(base: dict, updates: dict) -> None:
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _deep_update(base[k], v)
            else:
                base[k] = v

    _deep_update(data, patch)

    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    return f"config updated: {list(patch.keys())}"


def _patch_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace *old_string* with *new_string* in the given file."""
    if not file_path:
        return "patch_file failed: no file path provided"

    path = Path(file_path).expanduser()
    if not path.exists():
        return f"patch_file failed: {path} not found"

    content = path.read_text(encoding="utf-8")
    if old_string not in content:
        return f"patch_file failed: old_string not found in {path}"

    path.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    return f"patched {path}"


def _edit_workspace(workspace: Path, relative_file: str, content: str) -> str:
    """Write *content* to a file inside the workspace directory."""
    if not relative_file:
        return "edit_workspace failed: no file path provided"

    # Sandbox: ensure path stays within workspace
    target = (workspace / relative_file).resolve()
    if not str(target).startswith(str(workspace.resolve())):
        return f"edit_workspace rejected: {relative_file} is outside workspace"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {target}"


# ---------------------------------------------------------------------------
# Audit Logger — append entries to SUPERVISOR_LOG.md
# ---------------------------------------------------------------------------


async def write_audit(issue: dict, result: str, workspace: Path) -> None:
    """Append a structured entry to SUPERVISOR_LOG.md in the workspace memory dir."""
    log_path = workspace / "memory" / "SUPERVISOR_LOG.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    severity = issue.get("severity", "info")
    description = issue.get("description", "(no description)")
    fix_type = issue.get("fix_type", "none")

    ok = result.startswith(("gateway", "config", "patched", "wrote", "no fix"))
    result_icon = "✅" if ok else "❌"

    entry = (
        f"\n## {now}\n"
        f"**Issue**: {description}\n"
        f"**Severity**: {severity}\n"
        f"**Fix**: {fix_type}\n"
        f"**Result**: {result_icon} {result}\n"
    )

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(entry)


# ---------------------------------------------------------------------------
# Telegram Notifier — send alert via HTTP (no library dep)
# ---------------------------------------------------------------------------


async def send_event(text: str, config) -> None:
    """Send a plain-text lifecycle event message to Telegram."""
    try:
        tg = config.channels.telegram
        token = tg.token
        allow_from = tg.allow_from
    except AttributeError:
        return

    if not token or not allow_from:
        return

    chat_id = allow_from[0]
    try:
        import httpx

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            )
    except Exception as exc:
        logger.warning(f"Telegram event failed: {exc}")


async def notify_telegram(issue: dict, result: str, config) -> None:
    """Push a Telegram message for critical/warning issues."""
    severity = issue.get("severity", "info")
    description = issue.get("description", "(no description)")
    fix_type = issue.get("fix_type", "none")
    text = (
        f"🤖 *NanoSupervisor Alert*\n"
        f"*Severity*: {severity}\n"
        f"*Issue*: {description}\n"
        f"*Fix*: {fix_type}\n"
        f"*Result*: {result}"
    )
    await send_event(text, config)


# ---------------------------------------------------------------------------
# Gateway Watchdog — poll process, restart if gone, notify on events
# ---------------------------------------------------------------------------


def _gateway_running() -> bool:
    """Return True if a 'nanobot gateway' process is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nanobot gateway"],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        return False


async def watch_gateway(config, poll_interval: int = 30) -> None:
    """Periodically check the gateway process; restart and notify if it's gone."""
    was_running = True  # assume healthy at startup

    while True:
        await asyncio.sleep(poll_interval)
        running = _gateway_running()

        if was_running and not running:
            logger.warning("[watchdog] gateway process gone — restarting")
            result = await _restart_gateway()
            await send_event(f"⚠️ *Gateway crashed* — {result}", config)
        elif not was_running and running:
            logger.info("[watchdog] gateway is back up")
            await send_event("✅ *Gateway recovered*", config)

        was_running = running


# ---------------------------------------------------------------------------
# Main supervisor loop
# ---------------------------------------------------------------------------


async def run_supervisor(config, verbose: bool = False) -> None:
    """Main entry point for the supervisor daemon."""
    from nanobot.config.loader import get_data_dir

    log_file = get_data_dir() / "logs" / "nanobot.log"
    workspace = config.workspace_path

    if verbose:
        logger.enable("nanobot.supervisor")

    logger.info(f"NanoSupervisor waiting for log file: {log_file}")

    # Wait for gateway to create the log file
    while not log_file.exists():
        await asyncio.sleep(5)

    logger.info(f"NanoSupervisor watching {log_file}")
    await send_event("🟢 *NanoSupervisor started* — watching gateway", config)

    # Start gateway watchdog as a background task
    asyncio.create_task(watch_gateway(config))

    buffer: list[str] = []
    last_flush = time.monotonic()

    async for line in tail_file(log_file):
        if verbose:
            print(f"  [log] {line}")

        buffer.append(line)
        elapsed = time.monotonic() - last_flush

        if len(buffer) >= 20 or elapsed >= 30:
            if buffer:
                issues = await detect_issues(buffer)
                for issue in issues:
                    if issue.get("fix_type", "none") == "none":
                        continue
                    result = await apply_fix(issue, config, workspace)
                    await write_audit(issue, result, workspace)
                    if issue.get("severity") in ("critical", "warning"):
                        await notify_telegram(issue, result, config)
                        logger.info(
                            f"[supervisor] {issue.get('severity').upper()} — "
                            f"{issue.get('description')} — fix: {result}"
                        )
            buffer = []
            last_flush = time.monotonic()
