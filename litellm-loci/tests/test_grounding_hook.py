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


class _DepositResolver:
    def __init__(self, deposit_disabled: bool) -> None:
        self._deposit_disabled = deposit_disabled

    @staticmethod
    def is_disabled() -> bool:
        return False

    @staticmethod
    def qdrant_url() -> str:
        return "http://qdrant.invalid"

    def pheromone_deposit_disabled(self) -> bool:
        return self._deposit_disabled


class _GroundedHook(LociGroundingHook):
    def __init__(self, resolver: _DepositResolver) -> None:
        super().__init__(resolver=resolver)
        self.deposited: list[str] = []

    async def _embed(self, text: str) -> list[float]:
        return [0.1]

    async def _search_collection(self, client, col, vector, cf, impf, named, top_k):
        if col != "loci_memory":
            return []
        return [{
            "collection": "loci_memory",
            "point_id": "p1",
            "score": 0.9,
            "importance": 0.8,
            "fused": 0.72,
            "content": "alpha beta gamma delta",
            "payload": {},
        }]

    async def _pheromone_deposit(self, hit: dict) -> None:
        self.deposited.append(hit["point_id"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("deposit_disabled", "expected"), [(False, ["p1"]), (True, [])])
async def test_pheromone_deposit_respects_disable_flag(
    monkeypatch: pytest.MonkeyPatch, deposit_disabled: bool, expected: list[str]
) -> None:
    monkeypatch.delenv("LOCI_INVESTIGATION_ID", raising=False)
    hook = _GroundedHook(_DepositResolver(deposit_disabled))

    outcome = await hook._ground("alpha beta gamma delta lookup")

    assert outcome.status == "grounded"
    assert hook.deposited == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", False), ("0", False), ("1", True), ("true", True), ("ON", True)],
)
def test_resolver_pheromone_deposit_disabled_env(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    from loci_litellm.backend_resolver import BackendResolver

    monkeypatch.setenv("LOCI_PHERO_DEPOSIT_DISABLED", value)

    assert BackendResolver.pheromone_deposit_disabled() is expected
