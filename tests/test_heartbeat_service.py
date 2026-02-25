import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.heartbeat.service import HeartbeatService


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=MagicMock(has_tool_calls=False, content="ok"))

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="test-model",
        interval_s=9999,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()

    assert service._task is first_task

    service.stop()
    await asyncio.sleep(0)
