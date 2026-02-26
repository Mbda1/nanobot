---
name: supervisor
description: "Resolve issues detected by the NanoSupervisor daemon."
metadata:
  nanobot:
    always: true
---

# Supervisor Skill 🤖

The NanoSupervisor monitors the bot's logs and detects issues (e.g., config errors, gateway crashes, repetitive failures).
When the supervisor identifies a problem that requires human intervention, it creates a "suggestion".

## How to resolve suggestions
1. **Check for suggestions**: If the user mentions a supervisor alert or asks "what's wrong?", read the suggestions file at `~/.nanobot/workspace/memory/SUPERVISOR_SUGGESTIONS.json`.
2. **Understand the recommendation**: Each suggestion includes a `description` and `fix_details.suggestion`.
3. **Execute the fix**: Use your existing tools (e.g., `edit_file`, `write_file`, `exec`) to perform the recommended action.
4. **Confirm resolution**: After fixing the issue, inform the user.
5. **Supervisor Logs**: For more context, you can also search `~/.nanobot/workspace/memory/SUPERVISOR_LOG.md`.

Note: Gateway restarts are handled automatically by the supervisor. Focus on configuration and code-level fixes.
