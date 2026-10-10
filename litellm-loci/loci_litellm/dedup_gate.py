from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx
from litellm.integrations.custom_logger import CustomLogger
from loci_litellm.backend_resolver import BackendResolver

logger = logging.getLogger("loci_litellm.dedup_gate")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class LociDedupGate(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._resolver = BackendResolver

    @staticmethod
    def _extract_user_message(data: dict[str, Any]) -> str:
        messages: Any = data.get("messages") or []
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

    async def _embed_message(self, text: str) -> Optional[list[float]]:
        base: Optional[str] = self._resolver.embed_api_base()
        if not base:
            return None

        url = f"{base.rstrip('/')}/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key: str = self._resolver.embed_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {"model": self._resolver.embed_model(), "input": text}
        timeout: float = float(os.environ.get("LOCI_DEDUP_EMBED_TIMEOUT", "2.0"))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data: Any = response.json()
            return data["data"][0]["embedding"]
        except Exception as exc:
            logger.debug("loci dedup gate: embed fail-open: %r", exc)
            return None

    async def _search_similar(self, vector: list[float]) -> Optional[dict[str, Any]]:
        qdrant_url: str = self._resolver.qdrant_url()
        if not qdrant_url:
            return None

        threshold: float = float(os.environ.get("LOCI_DEDUP_THRESHOLD", "0.93"))
        window_s: int = int(os.environ.get("LOCI_DEDUP_WINDOW_S", "86400"))
        collection: str = os.environ.get("LOCI_DEDUP_COLLECTION", "hermes_memory")
        cutoff: float = time.time() - window_s

        url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/search"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key: str = self._resolver.qdrant_api_key()
        if api_key:
            headers["api-key"] = api_key

        body: dict[str, Any] = {
            "vector": vector,
            "limit": 1,
            "with_payload": True,
            "score_threshold": threshold,
            "filter": {"must": [{"key": "created_at_ts", "range": {"gt": cutoff}}]},
        }
        timeout: float = float(os.environ.get("LOCI_DEDUP_QDRANT_TIMEOUT", "1.5"))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data: Any = response.json()
        except Exception as exc:
            logger.debug("loci dedup gate: qdrant fail-open: %r", exc)
            return None

        hits: Any = data.get("result") or []
        if not hits:
            return None
        first_hit: dict[str, Any] = hits[0]
        score: float = float(first_hit.get("score", 0.0) or 0.0)
        if score < threshold:
            return None
        return first_hit

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: Any,
    ) -> dict[str, Any] | None:
        if _env_flag("LOCI_DEDUP_DISABLED"):
            return None

        user_message = self._extract_user_message(data)
        if not user_message:
            return None

        min_len: int = int(os.environ.get("LOCI_DEDUP_MIN_LEN", "20"))
        if len(user_message) < min_len:
            return None

        try:
            vector: Optional[list[float]] = await self._embed_message(user_message)
            if vector is None:
                return None

            match: Optional[dict[str, Any]] = await self._search_similar(vector)
            if match is None:
                return None

            payload: dict[str, Any] = match.get("payload") or {}
            preview: Any = payload.get("response_preview")
            if preview is None or str(preview).strip() == "":
                return None

            cached_response = str(preview).strip()
            return {
                "id": "loci-dedup-cache",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "loci-dedup-cache",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": cached_response},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "_loci_cache_hit": True,
            }
        except Exception as exc:
            logger.debug("loci dedup gate: fail-open on error: %r", exc)
            return None


loci_dedup_gate = LociDedupGate()
