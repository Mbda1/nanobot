from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class SupervisorTool(Tool):
    """Tool for managing the NanoSupervisor daemon and its suggestions."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.suggestions_path = workspace / "memory" / "SUPERVISOR_SUGGESTIONS.json"
        self.log_path = workspace / "memory" / "SUPERVISOR_LOG.md"

    @property
    def name(self) -> str:
        return "supervisor"

    @property
    def description(self) -> str:
        return "Manage the NanoSupervisor: list suggestions, clear resolved issues, or check status."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_suggestions", "clear_suggestion", "status"],
                    "description": "The action to perform."
                },
                "suggestion_id": {
                    "type": "string",
                    "description": "Required if action is 'clear_suggestion'."
                }
            },
            "required": ["action"]
        }

    async def execute(self, action: str, suggestion_id: str | None = None, **kwargs: Any) -> str:
        if action == "list_suggestions":
            return await self._list_suggestions()
        elif action == "clear_suggestion":
            if not suggestion_id:
                return "Error: suggestion_id is required for clear_suggestion."
            return await self._clear_suggestion(suggestion_id)
        elif action == "status":
            return await self._status()
        else:
            return f"Error: unknown action '{action}'"

    async def _list_suggestions(self) -> str:
        if not self.suggestions_path.exists():
            return "No pending supervisor suggestions found."
        
        try:
            with open(self.suggestions_path, encoding="utf-8") as fh:
                suggestions = json.load(fh)
            
            if not suggestions:
                return "No pending supervisor suggestions found."
            
            output = ["### Pending Supervisor Suggestions"]
            for s in suggestions:
                output.append(f"- **ID**: `{s.get('id')}`")
                output.append(f"  **Issue**: {s.get('description')}")
                output.append(f"  **Severity**: {s.get('severity')}")
                output.append(f"  **Recommendation**: {s.get('fix_details', {}).get('suggestion', 'no details')}")
                output.append(f"  **Time**: {s.get('timestamp')}\n")
            
            return "\n".join(output)
        except Exception as e:
            return f"Error reading suggestions: {e}"

    async def _clear_suggestion(self, suggestion_id: str) -> str:
        # Import clear_suggestion here to avoid circular dependencies
        from nanobot.supervisor.daemon import clear_suggestion
        if await clear_suggestion(suggestion_id, self.workspace):
            return f"Suggestion '{suggestion_id}' cleared successfully."
        else:
            return f"Error: Suggestion '{suggestion_id}' not found or could not be cleared."

    async def _status(self) -> str:
        from nanobot.supervisor.daemon import _gateway_running
        status = "running" if _gateway_running() else "NOT running"
        
        output = [
            f"### NanoSupervisor Status",
            f"- **Gateway**: {status}",
            f"- **Workspace**: {self.workspace}",
            f"- **Log File**: {self.workspace}/memory/SUPERVISOR_LOG.md"
        ]
        
        if self.suggestions_path.exists():
            try:
                with open(self.suggestions_path, encoding="utf-8") as fh:
                    count = len(json.load(fh))
                output.append(f"- **Pending Suggestions**: {count}")
            except Exception:
                pass
        
        return "\n".join(output)
