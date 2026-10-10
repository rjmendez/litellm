from __future__ import annotations

import pytest

import litellm.integrations.loci.dedup_gate as dedup_gate


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class FakeAsyncClient:
    def __init__(
        self,
        *,
        embed_payload: object | None = None,
        qdrant_payload: object | None = None,
        embed_exception: Exception | None = None,
        qdrant_exception: Exception | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._embed_payload = embed_payload or {"data": [{"embedding": [0.1] * 768}]}
        self._qdrant_payload = qdrant_payload or {
            "result": [{"score": 0.99, "payload": {"response_preview": "cached answer"}}]
        }
        self._embed_exception = embed_exception
        self._qdrant_exception = qdrant_exception

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        if "/embeddings" in url:
            self.calls.append(url)
            if self._embed_exception is not None:
                raise self._embed_exception
            return FakeResponse(self._embed_payload)
        if "/points/search" in url:
            self.calls.append(url)
            if self._qdrant_exception is not None:
                raise self._qdrant_exception
            return FakeResponse(self._qdrant_payload)
        raise AssertionError(f"Unexpected POST URL: {url}")


@pytest.mark.asyncio
async def test_returns_cached_response_on_high_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_DEDUP_DISABLED", raising=False)
    monkeypatch.setenv("LOCI_DEDUP_MIN_LEN", "20")

    fake_client = FakeAsyncClient()
    monkeypatch.setattr(dedup_gate.httpx, "AsyncClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_base", staticmethod(lambda: "http://example.local/v1"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_key", staticmethod(lambda: ""))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_model", staticmethod(lambda: "nomic-embed-text"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_url", staticmethod(lambda: "http://qdrant.local"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_api_key", staticmethod(lambda: ""))

    hook = dedup_gate.LociDedupGate()
    result = await hook.async_pre_call_hook(
        None,
        None,
        {"messages": [{"role": "user", "content": "This is a long enough prompt to check for exact semantic dedup."}]},
        "chat_completion",
    )

    assert result is not None
    assert result["model"] == "loci-dedup-cache"
    assert result["choices"][0]["message"]["content"] == "cached answer"
    assert result["usage"]["total_tokens"] == 0
    assert result["_loci_cache_hit"] is True


@pytest.mark.asyncio
async def test_returns_none_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_DEDUP_DISABLED", raising=False)
    monkeypatch.setenv("LOCI_DEDUP_MIN_LEN", "20")
    monkeypatch.setenv("LOCI_DEDUP_THRESHOLD", "0.99")

    fake_client = FakeAsyncClient(qdrant_payload={"result": [{"score": 0.5, "payload": {"response_preview": "cached answer"}}]})
    monkeypatch.setattr(dedup_gate.httpx, "AsyncClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_base", staticmethod(lambda: "http://example.local/v1"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_key", staticmethod(lambda: ""))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_model", staticmethod(lambda: "nomic-embed-text"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_url", staticmethod(lambda: "http://qdrant.local"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_api_key", staticmethod(lambda: ""))

    result = await dedup_gate.LociDedupGate().async_pre_call_hook(
        None,
        None,
        {"messages": [{"role": "user", "content": "This is a long enough prompt to check the threshold gate."}]},
        "chat_completion",
    )

    assert result is None


@pytest.mark.asyncio
async def test_fail_open_on_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_DEDUP_DISABLED", raising=False)
    monkeypatch.setenv("LOCI_DEDUP_MIN_LEN", "20")

    fake_client = FakeAsyncClient(embed_exception=ConnectionError("embed failed"))
    monkeypatch.setattr(dedup_gate.httpx, "AsyncClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_base", staticmethod(lambda: "http://example.local/v1"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_key", staticmethod(lambda: ""))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_model", staticmethod(lambda: "nomic-embed-text"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_url", staticmethod(lambda: "http://qdrant.local"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_api_key", staticmethod(lambda: ""))

    result = await dedup_gate.LociDedupGate().async_pre_call_hook(
        None,
        None,
        {"messages": [{"role": "user", "content": "This prompt is long enough that embedding should be attempted."}]},
        "chat_completion",
    )

    assert result is None
    assert fake_client.calls == ["http://example.local/v1/embeddings"]


@pytest.mark.asyncio
async def test_disabled_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_DEDUP_DISABLED", "1")
    monkeypatch.setenv("LOCI_DEDUP_MIN_LEN", "20")

    def _raise_if_used(*args: object, **kwargs: object) -> object:
        raise AssertionError("HTTP client should not be used when dedup is disabled")

    monkeypatch.setattr(dedup_gate.httpx, "AsyncClient", _raise_if_used)

    result = await dedup_gate.LociDedupGate().async_pre_call_hook(
        None,
        None,
        {"messages": [{"role": "user", "content": "This is a long enough prompt that should be skipped."}]},
        "chat_completion",
    )

    assert result is None


@pytest.mark.asyncio
async def test_skips_short_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_DEDUP_DISABLED", raising=False)
    monkeypatch.setenv("LOCI_DEDUP_MIN_LEN", "20")

    def _raise_if_used(*args: object, **kwargs: object) -> object:
        raise AssertionError("HTTP client should not be used for short messages")

    monkeypatch.setattr(dedup_gate.httpx, "AsyncClient", _raise_if_used)
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_base", staticmethod(lambda: "http://example.local/v1"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_model", staticmethod(lambda: "nomic-embed-text"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_url", staticmethod(lambda: "http://qdrant.local"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_api_key", staticmethod(lambda: ""))

    result = await dedup_gate.LociDedupGate().async_pre_call_hook(
        None,
        None,
        {"messages": [{"role": "user", "content": "short"}]},
        "chat_completion",
    )

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_response_preview_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_DEDUP_DISABLED", raising=False)
    monkeypatch.setenv("LOCI_DEDUP_MIN_LEN", "20")

    fake_client = FakeAsyncClient(qdrant_payload={"result": [{"score": 0.99, "payload": {"other": "field"}}]})
    monkeypatch.setattr(dedup_gate.httpx, "AsyncClient", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_base", staticmethod(lambda: "http://example.local/v1"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_api_key", staticmethod(lambda: ""))
    monkeypatch.setattr(dedup_gate.BackendResolver, "embed_model", staticmethod(lambda: "nomic-embed-text"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_url", staticmethod(lambda: "http://qdrant.local"))
    monkeypatch.setattr(dedup_gate.BackendResolver, "qdrant_api_key", staticmethod(lambda: ""))

    result = await dedup_gate.LociDedupGate().async_pre_call_hook(
        None,
        None,
        {"messages": [{"role": "user", "content": "This is a long enough prompt to check missing preview handling."}]},
        "chat_completion",
    )

    assert result is None
