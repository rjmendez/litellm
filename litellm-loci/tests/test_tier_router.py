from __future__ import annotations

from types import SimpleNamespace

import pytest

from loci_litellm.tier_router import LociTierRouter


@pytest.mark.asyncio
async def test_ignores_unknown_provider_failures() -> None:
    router = LociTierRouter()

    await router.async_log_failure_event(
        {"exception": TimeoutError("timed out")},
        None,
        0.0,
        0.0,
    )

    assert router.get_status() == {}


@pytest.mark.asyncio
async def test_uses_response_model_when_request_metadata_is_missing() -> None:
    router = LociTierRouter()
    response = SimpleNamespace(model="openrouter/openai/gpt-4.1-nano")

    await router.async_log_success_event({}, response, 0.0, 0.0)

    assert router.get_status()["openrouter"]["state"] == "closed"
