from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Optional

# Global context variable for trace_id
trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)

def set_trace_id(tid: Optional[str] = None) -> str:
    """Set the trace_id in the current context."""
    new_id = tid or str(uuid.uuid4())[:8]
    trace_id.set(new_id)
    return new_id

def get_trace_id() -> Optional[str]:
    """Get the trace_id from the current context."""
    return trace_id.get()
