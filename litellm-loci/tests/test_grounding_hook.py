from __future__ import annotations

import pytest

from loci_litellm.grounding_hook import GroundingOutcome, LociGroundingHook


class _Resolver:
    @staticmethod
    def is_disabled() -> bool:
        return False


class _HookWithOutcome(LociGroundingHook):
    def __init__(self, outcome: GroundingOutcome | None = None, exc: Exception | None = None) -> None:
        super().__init__(resolver=_Resolver())
        self._outcome = outcome
        self._exc = exc

    async def _ground(self, intent: str) -> GroundingOutcome:
        if self._exc is not None:
            raise self._exc
        assert self._outcome is not None
        return self._outcome


@pytest.mark.asyncio
async def test_fail_open_loudly_on_grounding_miss() -> None:
    hook = _HookWithOutcome(GroundingOutcome(status="miss", reason="no_matching_context"))
    data = {"messages": [{"role": "user", "content": "Explain the current callback stack"}]}

    result = await hook.async_pre_call_hook(None, None, data, "chat_completion")

    assert result is data
    metadata = result["metadata"]
    assert metadata["loci_grounding_status"] == "miss"
    assert metadata["loci_grounding_reason"] == "no_matching_context"
    assert result["messages"][0]["role"] == "system"
    assert "LOCI_GROUNDING_STATUS: miss" in result["messages"][0]["content"]


@pytest.mark.asyncio
async def test_fail_open_loudly_on_grounding_exception() -> None:
    hook = _HookWithOutcome(exc=TimeoutError("qdrant timed out"))
    data = {"messages": [{"role": "user", "content": "Summarize the latest routing state"}]}

    result = await hook.async_pre_call_hook(None, None, data, "chat_completion")

    assert result is data
    metadata = result["metadata"]
    assert metadata["loci_grounding_status"] == "error"
    assert metadata["loci_grounding_reason"] == "TimeoutError"
    assert result["messages"][0]["role"] == "system"
    assert "LOCI_GROUNDING_STATUS: error" in result["messages"][0]["content"]
