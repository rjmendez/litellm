from __future__ import annotations

import pytest

from loci_litellm.tier_router import _FAILURE_THRESHOLD, LociTierRouter


def _event(group: str, deployment_model: str, exception: Exception | None = None) -> dict:
    payload: dict = {
        "model": deployment_model,
        "litellm_params": {"metadata": {"model_group": group}},
    }
    if exception is not None:
        payload["exception"] = exception
    return payload


async def _fail(router: LociTierRouter, group: str, deployment_model: str, times: int, exc: Exception) -> None:
    for _ in range(times):
        await router.async_log_failure_event(_event(group, deployment_model, exc), None, 0.0, 0.0)


async def _pre_call_metadata(router: LociTierRouter, requested_model: str) -> dict:
    data: dict = {"model": requested_model, "messages": [{"role": "user", "content": "hi"}]}
    await router.async_pre_call_hook(None, None, data, "acompletion")
    return data.get("metadata") or {}


@pytest.mark.asyncio
async def test_ignores_failures_with_no_model_identity() -> None:
    router = LociTierRouter()

    await router.async_log_failure_event({"exception": TimeoutError("timed out")}, None, 0.0, 0.0)

    assert router.get_status() == {}


@pytest.mark.asyncio
async def test_failures_recorded_by_events_open_the_circuit_seen_by_pre_call() -> None:
    router = LociTierRouter()

    await _fail(router, "simple-tier", "workers-ai/@cf/model", _FAILURE_THRESHOLD, TimeoutError("timed out"))

    assert (await _pre_call_metadata(router, "simple-tier"))["loci_circuit_open"] == "simple-tier"


@pytest.mark.asyncio
async def test_circuit_is_tracked_per_model_group() -> None:
    router = LociTierRouter()

    await _fail(router, "simple-tier", "workers-ai/@cf/model", _FAILURE_THRESHOLD, TimeoutError("timed out"))

    assert "loci_circuit_open" not in await _pre_call_metadata(router, "complex-tier")


@pytest.mark.asyncio
async def test_deployments_of_one_group_share_a_circuit_regardless_of_provider_prefix() -> None:
    router = LociTierRouter()
    deployments = ("anthropic/claude-sonnet-5", "workers-ai/@cf/model", "resident")

    for index in range(_FAILURE_THRESHOLD):
        await _fail(router, "complex-tier", deployments[index % len(deployments)], 1, TimeoutError("timed out"))

    assert list(router.get_status()) == ["complex-tier"]
    assert (await _pre_call_metadata(router, "complex-tier"))["loci_circuit_open"] == "complex-tier"


@pytest.mark.asyncio
async def test_success_from_a_fallback_deployment_closes_the_group_circuit() -> None:
    router = LociTierRouter()
    await _fail(router, "simple-tier", "resident", _FAILURE_THRESHOLD, TimeoutError("timed out"))

    await router.async_log_success_event(_event("simple-tier", "workers-ai/@cf/model"), object(), 0.0, 0.0)

    assert "loci_circuit_open" not in await _pre_call_metadata(router, "simple-tier")
    assert router.get_status()["simple-tier"]["state"] == "closed"


@pytest.mark.asyncio
async def test_permanent_errors_do_not_count_toward_the_circuit() -> None:
    router = LociTierRouter()

    await _fail(router, "medium-tier", "anthropic/claude-haiku-5.5", _FAILURE_THRESHOLD * 2, ValueError("model not found"))

    assert "loci_circuit_open" not in await _pre_call_metadata(router, "medium-tier")
