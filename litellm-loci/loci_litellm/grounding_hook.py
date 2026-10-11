"""
LociGroundingHook — pre-call grounding hook for LiteLLM proxy.

Ported from Loci's scripts/hooks/pre_llm_grounding.py. Runs as a LiteLLM
CustomLogger callback. Before each chat/completion call is forwarded to the
upstream LLM, the hook:

  1. Extracts intent from the user message
  2. Embeds via nomic-embed-text on Ollama (or any OpenAI-compatible endpoint)
  3. Fans out parallel Qdrant searches across all 768-dim collections
  4. Merges + ranks by a four-term weighted score (relevance / recency /
     trust / type) with MMR diversification and stigmergic pheromone
     reinforcement
  5. Injects a MEMORY MATCH context block into the request messages

Fail-open: if Qdrant, Ollama, or any collection is unreachable, the hook
returns the request unchanged. Grounding is best-effort, never blocking.

Register in litellm config.yaml:
    litellm_settings:
      callbacks:
        - litellm.integrations.loci.grounding_hook.loci_grounding_hook
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from loci_litellm.backend_resolver import BackendResolver

logger = logging.getLogger("loci_litellm.grounding")

# ── Tunables (all env-overridable) ────────────────────────────────────────────

_RECALL_TOP_K = int(os.environ.get("LOCI_RECALL_TOP_K", "3"))
_MIN_IMPORTANCE = float(os.environ.get("LOCI_RECALL_MIN_IMPORTANCE", "0.2"))
_MIN_SCORE = float(os.environ.get("LOCI_RECALL_MIN_SCORE", "0.55"))
_VECTOR_NAME = os.environ.get("LOCI_RECALL_VECTOR_NAME", "dense")
_MAX_CONTENT_CHARS = int(os.environ.get("LOCI_RECALL_MAX_CHARS", "200"))
_EMBED_TIMEOUT = float(os.environ.get("LOCI_EMBED_TIMEOUT", "3.0"))
_QDRANT_TIMEOUT = float(os.environ.get("LOCI_QDRANT_TIMEOUT", "2.0"))
_INVESTIGATION_BOOST = float(os.environ.get("LOCI_INVESTIGATION_BOOST", "1.5"))
_WORKERS = int(os.environ.get("LOCI_QDRANT_WORKERS", "8"))

# Must sum to 1.0
_RANKER_W_RELEVANCE = float(os.environ.get("LOCI_RANKER_W_RELEVANCE", "0.50"))
_RANKER_W_RECENCY = float(os.environ.get("LOCI_RANKER_W_RECENCY", "0.20"))
_RANKER_W_TRUST = float(os.environ.get("LOCI_RANKER_W_TRUST", "0.15"))
_RANKER_W_TYPE = float(os.environ.get("LOCI_RANKER_W_TYPE", "0.15"))
_RECENCY_HALFLIFE_DAYS = float(os.environ.get("LOCI_RECENCY_HALFLIFE_DAYS", "7"))
_MMR_LAMBDA = float(os.environ.get("LOCI_MMR_LAMBDA", "0.75"))

# Stigmergic pheromone
_PHERO_BETA = float(os.environ.get("LOCI_PHERO_BETA", "0.08"))
_PHERO_HALFLIFE_H = float(os.environ.get("LOCI_PHERO_HALFLIFE_H", "24"))
_PHERO_DEPOSIT = float(os.environ.get("LOCI_PHERO_DEPOSIT", "1.0"))
_PHERO_EPSILON = float(os.environ.get("LOCI_PHERO_EPSILON", "0.05"))

_LN2 = math.log(2)

# ── Collections ──────────────────────────────────────────────────────────────
# fmt: (collection, content_field, importance_field_or_None, use_named_vector)
_BASE_COLLECTIONS = [
    ("mnemosyne", "content", "importance", True),
    ("loci_sessions", "content_preview", None, True),
    ("loci_memory", "text", "numeric_confidence", True),
]

_EXTRA_FIELD_MAP: dict[str, tuple] = {
    "ecc_skills": ("content_preview", None, True),
    "agent_core_chunks": ("text", None, False),
    "dama_gotchi_code": ("text", None, True),
}
_DEFAULT_EXTRA_FIELDS = ("text", None, True)

_extra_names = [
    c.strip() for c in os.environ.get("LOCI_GROUNDING_EXTRA_COLLECTIONS", "").split(",")
    if c.strip()
]
COLLECTIONS = _BASE_COLLECTIONS + [
    (name,) + _EXTRA_FIELD_MAP.get(name, _DEFAULT_EXTRA_FIELDS)
    for name in _extra_names
]

# ── Trust / type lookup tables ───────────────────────────────────────────────
_CONF_TRUST = {"high": 1.0, "medium": 0.7, "low": 0.4}
_TYPE_WEIGHT = {"observed": 1.0, "inferred": 0.8, "assumed": 0.6, "gap": 0.4}

# Slash commands that don't need grounding
_NAVIGATION_SLASH_COMMANDS = frozenset({
    "/new", "/clear", "/compact", "/stop", "/cancel", "/exit", "/quit",
    "/help", "/models", "/status", "/reset", "/undo", "/redo",
})


@dataclass(frozen=True, slots=True)
class GroundingOutcome:
    status: str
    context: str = ""
    reason: str = ""


class LociGroundingHook(CustomLogger):
    """
    LiteLLM CustomLogger that grounds each chat/completion request against
    Loci's Qdrant vector store before forwarding to the upstream LLM.

    Fail-open: any error in embedding or search returns the original request
    unchanged. Grounding is best-effort and never blocks the proxy.
    """

    def __init__(self, resolver: Any = BackendResolver) -> None:
        super().__init__()
        self._resolver = resolver

    # ── Main hook: async_pre_call_hook ──────────────────────────────────────

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Exception | str | dict | None:
        """Inject grounded memory context before the LLM call is dispatched."""
        if self._resolver.is_disabled():
            return None

        # Only ground chat completions and text completions
        if call_type not in ("completion", "chat_completion", "chat.completions", "completion.completions"):
            return None

        user_message = self._extract_user_message(data)
        if not user_message or not user_message.strip():
            return None

        # Skip slash commands
        stripped = user_message.strip()
        if stripped.startswith("/"):
            cmd = stripped.split()[0].lower()
            if cmd in _NAVIGATION_SLASH_COMMANDS:
                return None

        intent = self._extract_intent(stripped)
        if not intent:
            return None

        try:
            outcome = await self._ground(intent)
        except Exception as exc:
            logger.warning("loci grounding: fail-open error, request continues without retrieved context, error=%r", exc)
            self._mark_grounding_failure(data, "error", type(exc).__name__)
            return data

        self._record_grounding_metadata(data, outcome.status, outcome.reason)
        if outcome.status != "grounded":
            self._inject_failure_notice(data, outcome.status, outcome.reason)
            return data

        self._inject_context(data, outcome.context)
        return data

    # ── Intent extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_user_message(data: dict) -> str:
        """Extract the latest user message from the request payload."""
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            return ""

        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # OpenAI multimodal format: [{"type": "text", "text": "..."}]
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                return " ".join(parts)
        return ""

    @staticmethod
    def _extract_intent(prompt: str) -> str:
        """Strip leading politeness/meta phrases to get the core intent."""
        meta = re.compile(
            r"^(can you|please|help me|could you|I need you to|I want you to"
            r"|I'd like you to|would you)\s+",
            re.IGNORECASE,
        )
        return meta.sub("", prompt.strip())[:200].strip()

    # ── Embedding ───────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> Optional[list[float]]:
        """Embed text via the OpenAI-compatible /v1/embeddings endpoint."""
        base = self._resolver.embed_api_base()
        if not base:
            return None

        url = f"{base.rstrip('/')}/embeddings"
        headers = {"Content-Type": "application/json"}
        api_key = self._resolver.embed_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {"model": self._resolver.embed_model(), "input": text}

        try:
            async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                d = resp.json()
            return d["data"][0]["embedding"]
        except Exception as exc:
            logger.debug("loci grounding: embed fail-open: %r", exc)
            return None

    # ── Qdrant search ───────────────────────────────────────────────────────

    async def _search_collection(
        self,
        client: httpx.AsyncClient,
        collection: str,
        vector: list[float],
        content_field: str,
        importance_field: Optional[str],
        named_vec: bool,
        top_k: int = 5,
    ) -> list[dict]:
        """Search one Qdrant collection, return normalised hit dicts."""
        qdrant_url = self._resolver.qdrant_url()
        if not qdrant_url:
            return []

        url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/search"
        headers = {"Content-Type": "application/json"}
        api_key = self._resolver.qdrant_api_key()
        if api_key:
            headers["api-key"] = api_key

        vec_payload = {"name": _VECTOR_NAME, "vector": vector} if named_vec else vector
        body = {
            "vector": vec_payload,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }

        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        d = resp.json()

        hits = []
        for pt in d.get("result", []):
            score = pt.get("score", 0.0)
            if score < _MIN_SCORE:
                continue
            payload = pt.get("payload") or {}
            content = payload.get(content_field, "")
            if not content:
                for f in ("text", "content", "content_preview", "description"):
                    content = payload.get(f, "")
                    if content:
                        break
            if not content:
                continue
            importance = float(payload.get(importance_field, 0.5) or 0.5) \
                if importance_field else 0.5
            hits.append({
                "collection": collection,
                "point_id": pt.get("id"),
                "score": score,
                "importance": importance,
                "fused": score * importance,
                "content": str(content),
                "payload": payload,
            })
        return hits

    async def _search_investigation(
        self,
        client: httpx.AsyncClient,
        vector: list[float],
        investigation_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        qdrant_url = self._resolver.qdrant_url()
        if not qdrant_url:
            return []

        url = f"{qdrant_url.rstrip('/')}/collections/loci_memory/points/search"
        headers = {"Content-Type": "application/json"}
        api_key = self._resolver.qdrant_api_key()
        if api_key:
            headers["api-key"] = api_key

        body = {
            "vector": {"name": _VECTOR_NAME, "vector": vector},
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
            "filter": {
                "must": [{"key": "investigation_id", "match": {"value": investigation_id}}],
            },
        }

        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        d = resp.json()

        hits = []
        for pt in d.get("result", []):
            score = pt.get("score", 0.0)
            if score < _MIN_SCORE:
                continue
            payload = pt.get("payload") or {}
            content = payload.get("text") or payload.get("content") or payload.get("content_preview") or ""
            if not content:
                continue
            importance = float(payload.get("importance", 0.5) or 0.5)
            hits.append({
                "collection": "loci_memory",
                "point_id": pt.get("id"),
                "score": score,
                "importance": importance,
                "fused": score * importance,
                "content": str(content),
                "payload": payload,
            })
        return hits

    # ── Core grounding flow ─────────────────────────────────────────────────

    async def _ground(self, intent: str) -> GroundingOutcome:
        """
        Embed the intent, fan out to Qdrant, rank, and return the
        MEMORY MATCH context block. Returns empty string if nothing found.
        """
        qdrant_url = self._resolver.qdrant_url()
        if not qdrant_url:
            logger.warning("loci grounding: fail-open, request continues without retrieved context, reason=qdrant_unconfigured")
            return GroundingOutcome(status="unavailable", reason="qdrant_unconfigured")

        vector = await self._embed(intent)
        if vector is None:
            logger.warning("loci grounding: fail-open, request continues without retrieved context, reason=embedding_failed")
            return GroundingOutcome(status="unavailable", reason="embedding_failed")

        # Parallel fan-out across all collections
        all_hits: list[dict] = []
        failed = 0

        async with httpx.AsyncClient(timeout=_QDRANT_TIMEOUT) as client:
            tasks = []
            investigation_id = os.environ.get("LOCI_INVESTIGATION_ID", "").strip()
            investigation_task_index = None
            if investigation_id:
                investigation_task_index = len(tasks)
                tasks.append(self._search_investigation(client, vector, investigation_id, _RECALL_TOP_K))
            for col, cf, impf, named in COLLECTIONS:
                tasks.append(
                    self._search_collection(
                        client, col, vector, cf, impf, named, _RECALL_TOP_K
                    )
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)

        searched = len(results)
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                continue
            if not isinstance(result, list):
                continue
            if index == investigation_task_index:
                for hit in result:
                    hit["fused"] = hit.get("fused", 0.0) * _INVESTIGATION_BOOST
                all_hits.extend(result)
                continue
            all_hits.extend(result)

        if not all_hits:
            logger.warning("loci grounding: no matching context found for request")
            return GroundingOutcome(status="miss", reason="no_matching_context")

        # Keyword rerank
        all_hits = self._keyword_rerank(all_hits, intent)

        # Filter by importance
        all_hits = [h for h in all_hits if h["importance"] >= _MIN_IMPORTANCE]

        # Multi-signal score
        now_ts = time.time()
        for h in all_hits:
            h["_ms_score"] = self._multi_signal_score(h, now_ts)

        # Sort and MMR-select
        all_hits.sort(key=lambda h: h["_ms_score"], reverse=True)
        hits = self._mmr_select(all_hits, _RECALL_TOP_K)

        # Pheromone deposit (fire-and-forget, best-effort)
        for h in hits:
            if h.get("collection") == "loci_memory" and h.get("point_id"):
                try:
                    await self._pheromone_deposit(h)
                except Exception:
                    pass

        context = self._format_results(hits, intent, failed, searched)
        if not context:
            logger.warning("loci grounding: no matching context found for request")
            return GroundingOutcome(status="miss", reason="no_matching_context")
        return GroundingOutcome(status="grounded", context=context)

    # ── Ranking helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _keyword_rerank(hits: list[dict], query: str) -> list[dict]:
        query_tokens = set(re.sub(r"[^a-z0-9]", " ", query.lower()).split())
        if len(query_tokens) < 2:
            return hits
        for h in hits:
            content_lower = (h.get("content", "") or "").lower()
            overlap = sum(1 for t in query_tokens if len(t) > 3 and t in content_lower)
            h["fused"] = h.get("fused", 0.0) + min(0.25, overlap * 0.05)
        return sorted(hits, key=lambda h: h["fused"], reverse=True)

    @staticmethod
    def _multi_signal_score(hit: dict, now_ts: float) -> float:
        """Score a hit on four axes plus stigmergic pheromone boost."""
        payload = hit.get("payload") or {}

        relevance = hit.get("fused", hit["score"] * hit.get("importance", 0.5))

        created = payload.get("created_at_ts") or payload.get("ts_epoch")
        if created:
            age_days = max(0.0, (now_ts - float(created)) / 86400.0)
            recency = math.exp(-_LN2 * age_days / _RECENCY_HALFLIFE_DAYS)
        else:
            recency = 0.5

        conf = str(payload.get("confidence", "") or "").lower()
        trust = _CONF_TRUST.get(conf, 0.6)

        rtype = str(payload.get("record_type") or payload.get("type", "") or "").lower()
        type_w = _TYPE_WEIGHT.get(rtype, 0.7)

        base = (
            _RANKER_W_RELEVANCE * relevance
            + _RANKER_W_RECENCY * recency
            + _RANKER_W_TRUST * trust
            + _RANKER_W_TYPE * type_w
        )

        phero = LociGroundingHook._effective_pheromone(payload, now_ts)
        return base + _PHERO_BETA * math.log1p(phero)

    @staticmethod
    def _effective_pheromone(payload: dict, now_ts: float) -> float:
        stored = float(payload.get("pheromone", 0.0) or 0.0)
        if stored <= 0.0:
            return 0.0
        reinforced = payload.get("pheromone_reinforced_ts")
        if not reinforced:
            return stored
        age_h = max(0.0, (now_ts - float(reinforced)) / 3600.0)
        return stored * math.exp(-_LN2 * age_h / _PHERO_HALFLIFE_H)

    @staticmethod
    def _mmr_select(hits: list[dict], top_k: int) -> list[dict]:
        """Maximal Marginal Relevance selection with ε-exploration."""
        import random as _random

        if len(hits) <= top_k:
            return hits

        def _tok_sim(a: dict, b: dict) -> float:
            ta = set((a.get("content") or "").lower().split())
            tb = set((b.get("content") or "").lower().split())
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / max(len(ta), len(tb))

        selected: list[dict] = []
        remaining = list(hits)
        while remaining and len(selected) < top_k:
            if (len(selected) == top_k - 1
                    and len(remaining) > 1
                    and _random.random() < _PHERO_EPSILON):
                best = _random.choice(remaining)
            elif not selected:
                best = max(remaining, key=lambda h: h["_ms_score"])
            else:
                def _mmr(h):
                    rel = h["_ms_score"]
                    max_sim = max(_tok_sim(h, s) for s in selected)
                    return _MMR_LAMBDA * rel - (1.0 - _MMR_LAMBDA) * max_sim
                best = max(remaining, key=_mmr)
            selected.append(best)
            remaining.remove(best)
        return selected

    # ── Pheromone deposit ────────────────────────────────────────────────────

    async def _pheromone_deposit(self, hit: dict) -> None:
        """Bump pheromone on a retrieved point via Qdrant set_payload."""
        point_id = hit.get("point_id")
        if not point_id:
            return

        qdrant_url = self._resolver.qdrant_url()
        collection = hit["collection"]
        url = f"{qdrant_url.rstrip('/')}/collections/{collection}/points/payload"

        now_ts = time.time()
        current = self._effective_pheromone(hit.get("payload") or {}, now_ts)
        new_phero = current + _PHERO_DEPOSIT

        headers = {"Content-Type": "application/json"}
        api_key = self._resolver.qdrant_api_key()
        if api_key:
            headers["api-key"] = api_key

        body = {
            "points": [point_id],
            "payload": {
                "pheromone": new_phero,
                "pheromone_reinforced_ts": now_ts,
            },
        }

        async with httpx.AsyncClient(timeout=0.5) as client:
            await client.post(url, json=body, headers=headers)

    # ── Formatting ───────────────────────────────────────────────────────────

    @staticmethod
    def _clean_content(content: str) -> str:
        """Extract readable text from content that might be a JSON chat log."""
        stripped = content.strip()
        if not (stripped.startswith("[{") or stripped.startswith('["')):
            return stripped
        try:
            obj = json.loads(stripped)
            if isinstance(obj, list):
                parts = []
                for msg in obj:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role", "")
                    c = msg.get("content") or ""
                    if role == "user" and isinstance(c, str) and c.strip():
                        parts.append(f"user: {c.strip()[:150]}")
                    elif role == "assistant" and isinstance(c, str) and c.strip():
                        parts.append(f"assistant: {c.strip()[:150]}")
                return " | ".join(parts[:2]) if parts else ""
        except (json.JSONDecodeError, TypeError):
            pass
        m = re.search(r'"role"\s*:\s*"user"[^}]*"content"\s*:\s*"([^"]{10,})"', stripped)
        return f"user: {m.group(1)[:200]}" if m else ""

    def _format_results(
        self, hits: list[dict], query: str, failed: int, searched: int
    ) -> str:
        """Render the MEMORY MATCH context block."""
        lines = []
        seen = set()
        for h in hits:
            content = self._clean_content(h["content"])
            if not content:
                continue
            key = content[:80]
            if key in seen:
                continue
            seen.add(key)
            if len(content) > _MAX_CONTENT_CHARS:
                content = content[:_MAX_CONTENT_CHARS].rstrip() + "..."
            col = h["collection"].replace("_", "-")
            score = h["score"]
            lines.append(f"[{col}|{score:.2f}] {content}")
            if len(lines) >= _RECALL_TOP_K:
                break

        if not lines:
            return ""

        total = len(COLLECTIONS)
        if failed:
            reached = max(total - failed, 0)
            src = f"{reached} of {total} Qdrant collections ({failed} unreachable)"
        else:
            src = f"{total} Qdrant collection" + ("" if total == 1 else "s")

        header = f'MEMORY MATCH ({len(lines)} results from {src}) for "{query}":'
        body = "\n".join(lines)
        return f"{header}\n{body}"

    # ── Context injection ────────────────────────────────────────────────────

    @staticmethod
    def _inject_context(data: dict, context: str) -> None:
        """
        Inject the grounding context into the request as a system message
        prepended to the messages list. This makes it visible to the LLM
        without modifying the user's message.
        """
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            messages = []
            data["messages"] = messages

        # Prepend a system message with the grounding context
        grounding_msg = {
            "role": "system",
            "content": context,
        }
        messages.insert(0, grounding_msg)
        data["messages"] = messages

    @staticmethod
    def _record_grounding_metadata(data: dict, status: str, reason: str = "") -> None:
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["loci_grounding_status"] = status
        if reason:
            metadata["loci_grounding_reason"] = reason
        data["metadata"] = metadata

    def _mark_grounding_failure(self, data: dict, status: str, reason: str) -> None:
        self._record_grounding_metadata(data, status, reason)
        self._inject_failure_notice(data, status, reason)

    @staticmethod
    def _inject_failure_notice(data: dict, status: str, reason: str) -> None:
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            messages = []
            data["messages"] = messages

        reason_text = reason.replace("_", " ").strip() if reason else "unknown reason"
        notice = {
            "role": "system",
            "content": (
                f"LOCI_GROUNDING_STATUS: {status}. "
                f"Grounding did not add retrieved context for this request because {reason_text}. "
                "Proceed without retrieved memory context."
            ),
        }
        messages.insert(0, notice)
        data["messages"] = messages


# ── Singleton instance for LiteLLM callback registration ────────────────────
# LiteLLM's callback system expects either a callable or a CustomLogger instance.
# A module-level singleton lets the proxy load it via dotted path:
#   callbacks: ["litellm.integrations.loci.grounding_hook.loci_grounding_hook"]

loci_grounding_hook = LociGroundingHook()
