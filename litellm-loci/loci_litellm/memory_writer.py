from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Callable, Optional

import httpx
from litellm.integrations.custom_logger import CustomLogger
from loci_litellm.backend_resolver import BackendResolver

logger = logging.getLogger("loci_litellm.memory_writer")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class LociMemoryWriter(CustomLogger):
    def __init__(
        self,
        _http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ) -> None:
        super().__init__()
        self._resolver = BackendResolver
        self._http_client_factory: Callable[[], httpx.AsyncClient] = (
            _http_client_factory or (lambda: httpx.AsyncClient())
        )
        self._stream_buffers: dict[str, list[str]] = {}  # mutable-ok: stream accumulation buffer, cleared on finish

    def _client_for_timeout(self, timeout: float) -> httpx.AsyncClient:
        client = self._http_client_factory()
        if hasattr(client, "timeout"):
            try:
                client.timeout = timeout  # pyright: ignore[reportAttributeAccessIssue]
            except Exception:
                pass
        return client

    @staticmethod
    def _extract_user_message(kwargs: dict[str, Any]) -> str:
        messages: Any = kwargs.get("messages") or []
        if not isinstance(messages, list):
            return ""

        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content: Any = message.get("content") or ""
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                bits: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text: Any = part.get("text") or ""
                    if isinstance(text, str) and text.strip():
                        bits.append(text.strip())
                return " ".join(bits).strip()
        return ""

    @staticmethod
    def _extract_response_text(response_obj: Any) -> str:
        try:
            choices: Any = getattr(response_obj, "choices", None) or []
            if not choices:
                return ""
            message: Any = choices[0].message
            content: Any = getattr(message, "content", "") or ""
            if isinstance(content, str):
                stripped = content.strip()
                if stripped:
                    return stripped
            if isinstance(content, list):
                bits: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text: Any = part.get("text") or ""
                    if isinstance(text, str) and text.strip():
                        bits.append(text.strip())
                joined = " ".join(bits).strip()
                if joined:
                    return joined
            reasoning_content: Any = getattr(message, "reasoning_content", "") or ""
            if isinstance(reasoning_content, str):
                stripped_reasoning = reasoning_content.strip()
                if stripped_reasoning:
                    return stripped_reasoning
            provider_specific_fields: Any = getattr(message, "provider_specific_fields", None) or {}
            if isinstance(provider_specific_fields, dict):
                reasoning: Any = provider_specific_fields.get("reasoning") or ""
                if isinstance(reasoning, str):
                    return reasoning.strip()
        except Exception:
            return ""
        return ""

    @staticmethod
    def _point_id_for(prompt: str, response: str) -> str:
        seed: str = f"{prompt[:200]}{response[:100]}"
        return hashlib.sha256(seed.encode()).hexdigest()[:32]

    async def _embed_text(self, text: str) -> Optional[list[float]]:
        base: Optional[str] = self._resolver.embed_api_base()
        if not base:
            return None

        url: str = f"{base.rstrip('/')}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key: str = self._resolver.embed_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {"model": self._resolver.embed_model(), "input": text}
        timeout: float = float(os.environ.get("LOCI_MEM_EMBED_TIMEOUT", "3.0"))
        try:
            async with self._client_for_timeout(timeout) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data: Any = response.json()
            return data["data"][0]["embedding"]
        except Exception as exc:
            logger.debug("loci memory writer: embed fail-open: %r", exc)
            return None

    async def _upsert_point(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        qdrant_url: str = self._resolver.qdrant_url()
        if not qdrant_url:
            return

        url: str = f"{qdrant_url.rstrip('/')}/collections/{collection}/points?wait=false"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key: str = self._resolver.qdrant_api_key()
        if api_key:
            headers["api-key"] = api_key

        body: dict[str, Any] = {
            "points": [
                {
                    "id": point_id,
                    "vector": {os.environ.get("LOCI_MEMORY_VECTOR_NAME", "dense"): vector},
                    "payload": payload,
                },
            ]
        }
        timeout: float = float(os.environ.get("LOCI_MEM_QDRANT_TIMEOUT", "2.0"))
        async with self._client_for_timeout(timeout) as client:
            response = await client.put(url, json=body, headers=headers)
            response.raise_for_status()

    async def _write_qa_point(self, kwargs: dict[str, Any], prompt: str, response: str) -> None:
        min_prompt_len: int = int(os.environ.get("LOCI_MEM_MIN_PROMPT_LEN", "20"))
        if len(prompt) < min_prompt_len or len(response) < min_prompt_len:
            return

        embed_text: str = f"{prompt}\n\n{response[:200]}"
        vector: Optional[list[float]] = await self._embed_text(embed_text)
        if vector is None:
            return

        collection: str = os.environ.get("LOCI_MEMORY_WRITE_COLLECTION", "hermes_memory")
        prompt_text: str = prompt[:300]
        response_limit: int = int(os.environ.get("LOCI_MEM_MAX_CHARS", "500"))
        response_preview: str = response[:response_limit]

        payload: dict[str, Any] = {
            "text": f"Q: {prompt_text}\nA: {response_preview}",
            "prompt": prompt_text,
            "response_preview": response_preview,
            "model": kwargs.get("model") or "",
            "created_at_ts": time.time(),
            "source": "litellm",
            "record_type": "observed",
            "confidence": "medium",
            "importance": 0.6,
            "pheromone": 0.0,
            "investigation_id": os.environ.get("LOCI_INVESTIGATION_ID", ""),
        }
        point_id: str = self._point_id_for(prompt, response)
        await self._upsert_point(collection, point_id, vector, payload)

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        try:
            if _env_flag("LOCI_MEMORY_WRITER_DISABLED"):
                return

            prompt: str = self._extract_user_message(kwargs)
            response: str = self._extract_response_text(response_obj)
            if not prompt or not response:
                return

            await self._write_qa_point(kwargs, prompt, response)
        except Exception as exc:
            logger.debug("loci memory writer: fail-open on error: %r", exc)
            return

    @staticmethod
    def _stream_request_key(kwargs: dict[str, Any]) -> str:
        call_id: Any = kwargs.get("litellm_call_id") or kwargs.get("request_id")
        if call_id is not None:
            return str(call_id)
        prompt: str = LociMemoryWriter._extract_user_message(kwargs)
        if prompt:
            return hashlib.sha256(prompt[:200].encode()).hexdigest()[:32]
        return f"stream-{hashlib.sha256(repr(sorted(kwargs.items())).encode()).hexdigest()[:16]}"

    async def async_log_stream_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        try:
            if _env_flag("LOCI_MEMORY_WRITER_DISABLED"):
                return

            choices: Any = getattr(response_obj, "choices", None) or []
            if not choices:
                return
            choice: Any = choices[0]
            delta: Any = getattr(choice, "delta", None)
            if delta is None:
                return

            content: Any = getattr(delta, "content", None) or ""
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict)
                )
            if not isinstance(content, str):
                content = str(content)

            request_key: str = self._stream_request_key(kwargs)
            if content:
                self._stream_buffers.setdefault(request_key, [])
                self._stream_buffers[request_key].append(content)

            finish_reason: Any = getattr(choice, "finish_reason", None)
            if finish_reason is None:
                return

            prompt: str = self._extract_user_message(kwargs)
            if not prompt:
                self._stream_buffers.pop(request_key, None)
                return

            accumulated: str = "".join(self._stream_buffers.pop(request_key, []))
            if not accumulated:
                return

            await self._write_qa_point(kwargs, prompt, accumulated)
        except Exception as exc:
            logger.debug("loci memory writer: fail-open on stream error: %r", exc)
            try:
                request_key: str = self._stream_request_key(kwargs)
                self._stream_buffers.pop(request_key, None)
            except Exception:
                pass
            return


loci_memory_writer = LociMemoryWriter()
