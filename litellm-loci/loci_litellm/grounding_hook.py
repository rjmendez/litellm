"""
LociGroundingHook: pre-call grounding hook for the LiteLLM proxy.

Runs as a LiteLLM CustomLogger callback. Before a chat, text-completion, Responses or Anthropic-messages call
is forwarded to the upstream LLM, the hook:

  1. Takes the latest user message (first 200 characters, one leading politeness phrase removed) as the query
  2. Embeds it with the resolver's embed model through an OpenAI-compatible /embeddings endpoint
  3. Searches, in parallel, a fixed list of Qdrant collections: mnemosyne, loci_sessions and loci_memory, plus
     any named in LOCI_GROUNDING_EXTRA_COLLECTIONS. It does not discover collections and does not check vector
     dimensions (a mismatched collection fails and is counted as a failed search). The memory writer's
     [memory] write_collection is NOT added to this list: a private write collection is read back only if it
     is also listed in LOCI_GROUNDING_EXTRA_COLLECTIONS.
  4. Scores each hit as a weighted sum of relevance (score x importance, plus a keyword-overlap bonus of up to
     0.25 and, for LOCI_INVESTIGATION_ID matches, a LOCI_INVESTIGATION_BOOST multiplier), recency, trust and
     record type, plus a logarithmic pheromone term. The relevance term is not normalised, so the weights are
     relative importances, not a convex blend. Hits are de-duplicated on the first 80 characters of their
     cleaned content, then chosen by MMR; with probability LOCI_RECALL_EXPLORE_EPSILON (default 0.05, old name
     LOCI_PHERO_EPSILON) the last slot is a random remaining hit, so retrieval is not deterministic by default.
  5. Injects a MEMORY MATCH block into the request's instruction slot, as a system-role message or the
     surface's equivalent. The block carries the first 200 characters of the user's own text and retrieved
     records written by any earlier caller: collections are shared and not filtered per tenant, and the
     block is labelled as reference data.

Side effect: retrieval WRITES to the memory store. Each returned loci_memory point gets its ``pheromone`` payload
field bumped (a read-modify-write from the search payload, so concurrent requests overwrite rather than add; the
value is not capped). Set LOCI_PHERO_DEPOSIT_DISABLED=1 to stop it. Requests marked ``loci_internal_call`` are
not grounded and therefore do not deposit.

Failure behavior (fail-open): the request is never blocked. When retrieval finds nothing, cannot run or raises,
the request continues WITHOUT retrieved context, but it is not unchanged: a LOCI_GROUNDING_STATUS notice is
injected into the same instruction slot (the model sees it), the status is recorded in the request metadata and
returned in the x-loci-grounding response header, and a WARNING is logged with the reason.

The tunables listed here are read from the environment once, at import (a change needs a proxy restart).
The switches and endpoints are read per request through the resolver: LOCI_GROUNDING_DISABLED,
LOCI_PHERO_DEPOSIT_DISABLED, LOCI_TRUST_CLIENT_RAG, LOCI_CLIENT_RAG_OK_MARKERS, LOCI_HONOR_INTERNAL_CALL,
LOCI_GROUNDING_AUDIT_FILE, LOCI_INVESTIGATION_ID and the Qdrant and embedding endpoints. The tunables are LOCI_RECALL_TOP_K, LOCI_RECALL_MIN_IMPORTANCE,
LOCI_RECALL_MIN_SCORE, LOCI_RECALL_VECTOR_NAME, LOCI_RECALL_MAX_CHARS, LOCI_EMBED_TIMEOUT, LOCI_QDRANT_TIMEOUT,
LOCI_INVESTIGATION_BOOST, LOCI_RANKER_W_{RELEVANCE,RECENCY,TRUST,TYPE}, LOCI_RECENCY_HALFLIFE_DAYS,
LOCI_MMR_LAMBDA, LOCI_PHERO_BETA, LOCI_PHERO_HALFLIFE_H, LOCI_PHERO_DEPOSIT and LOCI_RECALL_EXPLORE_EPSILON.

Register in the proxy config.yaml with either dotted path; the legacy shim path re-exports the same singleton:
    litellm_settings:
      callbacks:
        - loci_litellm.grounding_hook.loci_grounding_hook
        - litellm.integrations.loci.grounding_hook.loci_grounding_hook   # shim; use one of the two
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

# Relative weights. They are meant to sum to 1.0 (a WARNING is logged at import if they do not), but the
# relevance term itself is not bounded by 1, so the result is not a normalised blend.
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
# Probability that the last selected slot is a random remaining hit instead of the MMR pick. Not a pheromone
# parameter; LOCI_PHERO_EPSILON is the old name of this setting and is still read when the new one is unset.
_EXPLORE_EPSILON = float(
    os.environ.get("LOCI_RECALL_EXPLORE_EPSILON") or os.environ.get("LOCI_PHERO_EPSILON") or "0.05"
)

_LN2 = math.log(2)

_WEIGHT_SUM = _RANKER_W_RELEVANCE + _RANKER_W_RECENCY + _RANKER_W_TRUST + _RANKER_W_TYPE
if abs(_WEIGHT_SUM - 1.0) > 1e-6:
    logger.warning("loci grounding: LOCI_RANKER_W_* weights sum to %.3f, not 1.0; scores are scaled accordingly",
                   _WEIGHT_SUM)

_MS_EPOCH_THRESHOLD = 1e11  # epoch values above this (year 5138 in seconds) are read as milliseconds


def _num(value: Any, default: float, what: str = "", point_id: Any = None) -> float:
    """``value`` as a finite float, or ``default`` when it is missing, non-numeric or not finite. An explicit
    0 stays 0. A bad value is logged at DEBUG with the point id and never aborts ranking."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        logger.debug("loci grounding: ignoring non-numeric %s=%r on point %s", what or "value", value, point_id)
        return default
    return number

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

_OWN_CONTEXT_PREFIX = "MEMORY MATCH ("

_CHAT_CALL_TYPES = frozenset({
    "completion", "acompletion", "chat_completion", "chat.completions", "completion.completions",
})
_TEXT_COMPLETION_CALL_TYPES = frozenset({"text_completion", "atext_completion"})
_RESPONSES_CALL_TYPES = frozenset({"responses", "aresponses"})
_ANTHROPIC_CALL_TYPES = frozenset({"anthropic_messages"})
_GROUNDED_CALL_TYPES = (
    _CHAT_CALL_TYPES | _TEXT_COMPLETION_CALL_TYPES | _RESPONSES_CALL_TYPES | _ANTHROPIC_CALL_TYPES
)

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


class _EmbedUnavailable(Exception):
    """Embedding could not run; ``str(exc)`` is the reason slug reported in the outcome."""


class LociGroundingHook(CustomLogger):
    """
    LiteLLM CustomLogger that grounds each chat/completion request against
    Loci's Qdrant vector store before forwarding to the upstream LLM.

    Fail-open: an error anywhere in the hook never fails the request. The request continues without
    retrieved context, but it is changed: a LOCI_GROUNDING_STATUS notice is injected, the status is recorded in
    the request metadata and the x-loci-grounding header, and a WARNING is logged. See the module docstring.
    """

    def __init__(self, resolver: Any = BackendResolver) -> None:
        super().__init__()
        self._resolver = resolver
        self._audit_errors_seen: set[str] = set()

    # ── Main hook: async_pre_call_hook ──────────────────────────────────────

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Exception | str | dict | None:
        """Inject grounded memory context before the LLM call is dispatched. Never raises: an unexpected error
        is logged at WARNING, recorded as status ``error`` and the request continues without context."""
        try:
            return await self._pre_call(data, call_type)
        except Exception as exc:
            logger.warning("loci grounding: fail-open error, request continues without retrieved context, error=%r", exc)
            try:
                self._mark_grounding_failure(data, "error", type(exc).__name__, self._surface(call_type))
            except Exception as mark_exc:
                logger.warning("loci grounding: could not record the failure on the request: %r", mark_exc)
            return data

    async def _pre_call(self, data: dict, call_type: Any) -> Exception | str | dict | None:
        if self._resolver.is_disabled():
            self._record_grounding_metadata(data, "skipped", "disabled")
            return None

        # A resolver without is_internal_call (a partial custom resolver) falls back to the default marker check.
        is_internal = getattr(self._resolver, "is_internal_call", BackendResolver.is_internal_call)
        if is_internal(data.get("metadata"), data.get("litellm_metadata")):
            self._record_grounding_metadata(data, "skipped", "internal_call")
            return None

        # Only the call types in _GROUNDED_CALL_TYPES: chat, text completion, Responses, Anthropic messages
        if call_type not in _GROUNDED_CALL_TYPES:
            self._record_grounding_metadata(data, "skipped", "call_type")
            return None

        user_message = self._extract_user_message(data)
        if not user_message or not user_message.strip():
            self._record_grounding_metadata(data, "skipped", "no_user_message")
            return None

        # Skip slash commands
        stripped = user_message.strip()
        if stripped.startswith("/"):
            cmd = stripped.split()[0].lower()
            if cmd in _NAVIGATION_SLASH_COMMANDS:
                self._record_grounding_metadata(data, "skipped", "navigation_command")
                return None

        intent = self._extract_intent(stripped)
        if not intent:
            self._record_grounding_metadata(data, "skipped", "no_intent")
            return None

        surface = self._surface(call_type)
        markers = self._resolver.client_rag_ok_markers() if self._resolver.client_rag_trusted() else ()
        existing = self._existing_grounding(self._role_texts(data, surface), markers)
        if existing:
            self._record_grounding_metadata(data, existing)
            return data

        outcome = await self._ground(intent)

        self._record_grounding_metadata(data, outcome.status, outcome.reason)
        if outcome.status != "grounded":
            self._inject_failure_notice(data, outcome.status, outcome.reason, surface)
            return data

        self._inject_context(data, outcome.context, surface)
        return data

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: object,
        request_headers: dict | None = None,
        litellm_call_info: dict | None = None,
    ) -> dict | None:
        metadata = None
        for key in ("metadata", "litellm_metadata"):
            bucket = data.get(key) if isinstance(data, dict) else None
            if isinstance(bucket, dict) and isinstance(bucket.get("loci_grounding_status"), str):
                metadata = bucket
                break
        status = metadata.get("loci_grounding_status") if metadata else None
        if not status:
            return None
        headers = {"x-loci-grounding": status}
        reason = metadata.get("loci_grounding_reason")
        if isinstance(reason, str) and re.fullmatch(r"[A-Za-z0-9_]{1,64}", reason):
            headers["x-loci-grounding-reason"] = reason
        return headers

    # ── Delivery audit ───────────────────────────────────────────────────────

    def log_pre_api_call(self, model: str, messages: Any, kwargs: dict) -> None:
        """Check the payload that is about to leave for the provider. The pre-call hook decides and injects, but a
        bridge between the proxy and the provider (for example Responses -> chat) can drop the slot the context
        went into; this is the one place the final body is visible.

        Delivery is judged only on the instruction slots of the body (system/developer messages, ``instructions``,
        ``system``, ``system_instruction``, and the start of a ``prompt``), so the marker appearing in a user
        message does not count. A provider body that carries instructions somewhere else is reported as lost.
        The audit row holds metadata only (no prompt or context text). It is appended to the file named by
        LOCI_GROUNDING_AUDIT_FILE with no rotation or size bound; when that is unset only the log line exists."""
        try:
            params = kwargs.get("litellm_params") or {}
            metadata = params.get("metadata") or params.get("litellm_metadata") or {}
            status = metadata.get("loci_grounding_status")
            if status not in ("grounded", "miss", "unavailable", "error"):
                return
            body = (kwargs.get("additional_args") or {}).get("complete_input_dict")
            payload = body if isinstance(body, (dict, list)) else {"messages": messages}
            marker = _OWN_CONTEXT_PREFIX if status == "grounded" else "LOCI_GROUNDING_STATUS"
            delivered = any(marker in text for text in self._instruction_texts(payload))
            row = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "call_id": str(kwargs.get("litellm_call_id") or ""),
                "model": str(model or ""),
                "status": status,
                "delivered": delivered,
            }
            if delivered:
                logger.info("loci grounding: delivered status=%s model=%s", status, model)
            else:
                logger.error("loci grounding: context LOST before the provider call status=%s model=%s", status, model)
            path = self._resolver.grounding_audit_file()
            if path:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            key = f"{type(exc).__name__}:{exc}"
            if key not in self._audit_errors_seen and len(self._audit_errors_seen) < 20:
                self._audit_errors_seen.add(key)
                logger.warning("loci grounding: delivery audit failed, delivery of context is NOT being verified: %r", exc)

    @classmethod
    def _instruction_texts(cls, payload: Any) -> list[str]:
        """Texts that sit in an instruction slot of a provider-bound body, whichever API shape it has."""
        texts: list[str] = []
        if isinstance(payload, list):
            payload = {"messages": payload}
        if not isinstance(payload, dict):
            return texts

        def collect(value: Any) -> None:
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict):
                collect(value.get("text"))
                collect(value.get("parts"))
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        for key in ("instructions", "system", "system_instruction", "systemInstruction"):
            collect(payload.get(key))
        messages = payload.get("messages")
        for message in messages if isinstance(messages, list) else []:
            if isinstance(message, dict) and message.get("role") in ("system", "developer"):
                collect(message.get("content"))
        prompt = payload.get("prompt")
        for item in prompt if isinstance(prompt, list) else [prompt]:
            if isinstance(item, str):
                texts.append(item[:4096])  # a completion has no system role: the block is prepended
        return texts

    # ── Existing-grounding detection ─────────────────────────────────────────

    @staticmethod
    def _surface(call_type: Any) -> str:
        """Which request shape this call type carries: chat, completion, responses or anthropic."""
        if call_type in _TEXT_COMPLETION_CALL_TYPES:
            return "completion"
        if call_type in _RESPONSES_CALL_TYPES:
            return "responses"
        if call_type in _ANTHROPIC_CALL_TYPES:
            return "anthropic"
        return "chat"

    @staticmethod
    def _text_parts(content: Any) -> list[str]:
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            return [p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") in ("text", "input_text")
                    and isinstance(p.get("text"), str)]
        return []

    @classmethod
    def _role_texts(cls, data: dict, surface: str) -> list[tuple[str, str]]:
        """Every text the request carries as (role, text), whichever API shape delivered it. The instruction
        slot of a surface (Responses ``instructions``, Anthropic ``system``) reads as role ``system``."""
        pairs: list[tuple[str, str]] = []
        if surface == "completion":
            prompt = data.get("prompt")
            prompts = prompt if isinstance(prompt, list) else [prompt]
            return [("user", p) for p in prompts if isinstance(p, str)]
        if surface == "responses":
            pairs += [("system", t) for t in cls._text_parts(data.get("instructions"))]
            items = data.get("input")
            if isinstance(items, str):
                return pairs + [("user", items)]
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict):
                    role = item.get("role") or "user"
                    pairs += [(role, t) for t in cls._text_parts(item.get("content"))]
            return pairs
        if surface == "anthropic":
            pairs += [("system", t) for t in cls._text_parts(data.get("system"))]
        for message in data.get("messages") or []:
            if isinstance(message, dict):
                pairs += [(message.get("role"), t) for t in cls._text_parts(message.get("content"))]
        return pairs

    @staticmethod
    def _existing_grounding(role_texts: list[tuple[str, str]], client_ok_markers: tuple[str, ...]) -> str:
        """Why this request needs no proxy-side grounding, or '' when it does.

        ``already_grounded``: an earlier hop injected our own MEMORY MATCH system message.
        ``client_grounded``: some text of the request, of ANY role (a client usually puts its block in the user
        message), contains a configured client marker. The marker is taken as proof the client's own retrieval
        succeeded; nothing verifies that, and anyone who can send text can trigger it. Turn the shortcut off
        with LOCI_TRUST_CLIENT_RAG=0 or leave the markers unconfigured (the default).
        A client block that says UNAVAILABLE is not a reason to skip: this is the fallback that makes
        retrieval happen when the client's could not."""
        for role, text in role_texts:
            if role == "system" and text.lstrip().startswith(_OWN_CONTEXT_PREFIX):
                return "already_grounded"
            if any(marker in text for marker in client_ok_markers):
                return "client_grounded"
        return ""

    # ── Intent extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_user_message(data: dict) -> str:
        """Extract the latest user message from the request payload."""
        messages = data.get("messages")
        if messages is None:
            prompt = data.get("prompt")
            if isinstance(prompt, str):
                return prompt
            if isinstance(prompt, list) and prompt and isinstance(prompt[-1], str):
                return prompt[-1]
            items = data.get("input")
            if isinstance(items, str):
                return items
            for item in reversed(items) if isinstance(items, list) else []:
                if isinstance(item, dict) and (item.get("role") or "user") == "user":
                    text = " ".join(LociGroundingHook._text_parts(item.get("content")))
                    if text.strip():
                        return text
            return ""
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
                # Multimodal parts: only text / input_text parts count. A latest user message with no text
                # (image only) returns "" and is skipped as no_user_message; earlier turns are not used.
                return " ".join(LociGroundingHook._text_parts(content))
        return ""

    @staticmethod
    def _extract_intent(prompt: str) -> str:
        """Strip ONE leading politeness/meta phrase and keep the first 200 characters as the query."""
        meta = re.compile(
            r"^(can you|please|help me|could you|I need you to|I want you to"
            r"|I'd like you to|would you)\s+",
            re.IGNORECASE,
        )
        return meta.sub("", prompt.strip())[:200].strip()

    # ── Embedding ───────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> Optional[list[float]]:
        """Embed text via the OpenAI-compatible /embeddings endpoint. Raises _EmbedUnavailable when no endpoint is
        configured or the request fails."""
        base = self._resolver.embed_api_base()
        if not base:
            raise _EmbedUnavailable("embed_unconfigured")

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
            logger.warning("loci grounding: embedding request failed: %s: %s", type(exc).__name__, exc)
            raise _EmbedUnavailable("embedding_failed") from exc

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
            importance = _num(payload.get(importance_field), 0.5, importance_field, pt.get("id")) \
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
            importance = _num(payload.get("importance"), 0.5, "importance", pt.get("id"))
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
        Embed the intent, fan out to Qdrant, rank, and return an outcome: ``grounded`` with the MEMORY MATCH
        block, ``miss`` when nothing matched, or ``unavailable`` when retrieval could not run (not configured,
        embedding failed, or every search failed).
        """
        qdrant_url = self._resolver.qdrant_url()
        if not qdrant_url:
            logger.warning("loci grounding: fail-open, request continues without retrieved context, reason=qdrant_unconfigured")
            return GroundingOutcome(status="unavailable", reason="qdrant_unconfigured")

        try:
            vector = await self._embed(intent)
            reason = "embedding_failed"
        except _EmbedUnavailable as exc:
            vector, reason = None, str(exc)
        if vector is None:
            logger.warning("loci grounding: fail-open, request continues without retrieved context, reason=%s", reason)
            return GroundingOutcome(status="unavailable", reason=reason)

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
        offset = 1 if investigation_task_index is not None else 0
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                failed += 1
                name = "investigation" if index == investigation_task_index else COLLECTIONS[index - offset][0]
                logger.warning("loci grounding: search of %s failed: %s: %s", name, type(result).__name__, result)
                continue
            if not isinstance(result, list):
                continue
            if index == investigation_task_index:
                for hit in result:
                    hit["fused"] = hit.get("fused", 0.0) * _INVESTIGATION_BOOST
                all_hits.extend(result)
                continue
            all_hits.extend(result)

        if searched and failed == searched:
            logger.warning("loci grounding: fail-open, request continues without retrieved context, "
                           "reason=all_collections_failed (%d searches)", searched)
            return GroundingOutcome(status="unavailable", reason="all_collections_failed")

        if not all_hits:
            logger.warning("loci grounding: no matching context found for request (%d of %d searches failed)",
                           failed, searched)
            # A miss with failed searches is not "nothing there": part of the store was never searched.
            return GroundingOutcome(status="miss", reason=(
                "no_matching_context_some_searches_failed" if failed else "no_matching_context"))

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
        # De-duplicate before selecting so a duplicate cannot take a slot and leave the block short of top_k.
        all_hits = self._dedup_hits(all_hits)
        hits = self._mmr_select(all_hits, _RECALL_TOP_K)

        if not self._resolver.pheromone_deposit_disabled():
            targets = [h for h in hits if h.get("collection") == "loci_memory" and h.get("point_id")]
            outcomes = await asyncio.gather(*(self._pheromone_deposit(h) for h in targets), return_exceptions=True)
            errors = [o for o in outcomes if isinstance(o, BaseException)]
            if errors:
                logger.warning("loci grounding: pheromone deposit failed for %d of %d points (first: %s: %s); "
                               "set LOCI_PHERO_DEPOSIT_DISABLED=1 to stop deposits",
                               len(errors), len(targets), type(errors[0]).__name__, errors[0])

        context = self._format_results(hits, intent, failed)
        if not context:
            logger.warning("loci grounding: no matching context found for request (%d of %d searches failed)",
                           failed, searched)
            return GroundingOutcome(status="miss", reason=(
                "no_matching_context_some_searches_failed" if failed else "no_matching_context"))
        return GroundingOutcome(status="grounded", context=context)

    # ── Ranking helpers ──────────────────────────────────────────────────────

    @classmethod
    def _dedup_hits(cls, hits: list[dict]) -> list[dict]:
        """Keep the first (best-ranked) hit per first-80-characters of cleaned content; drop hits whose content
        cleans to nothing."""
        seen: set[str] = set()
        kept: list[dict] = []
        for hit in hits:
            content = cls._clean_content(hit["content"])
            if not content or content[:80] in seen:
                continue
            seen.add(content[:80])
            kept.append(hit)
        return kept

    @staticmethod
    def _keyword_rerank(hits: list[dict], query: str) -> list[dict]:
        """Add a keyword bonus (0.05 per query token longer than 3 characters found as a substring of the hit's
        content, capped at 0.25) to ``fused``. It is additive on the same scale as score x importance and
        applied before the importance filter."""
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
        """Score a hit: weighted relevance, recency, trust and record type, plus a logarithmic pheromone term."""
        payload = hit.get("payload") or {}

        relevance = hit.get("fused", hit["score"] * hit.get("importance", 0.5))

        # Epoch seconds, or milliseconds (read as such above 1e11). A missing or non-numeric value is unknown
        # recency (0.5), not an error.
        created = _num(payload.get("created_at_ts") or payload.get("ts_epoch"), 0.0,
                       "created_at_ts", hit.get("point_id"))
        if created > 0:
            if created > _MS_EPOCH_THRESHOLD:
                created /= 1000.0
            age_days = max(0.0, (now_ts - created) / 86400.0)
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
        stored = _num(payload.get("pheromone"), 0.0, "pheromone")
        if stored <= 0.0:
            return 0.0
        reinforced = _num(payload.get("pheromone_reinforced_ts"), 0.0, "pheromone_reinforced_ts")
        if reinforced <= 0.0:
            return stored
        age_h = max(0.0, (now_ts - reinforced) / 3600.0)
        return stored * math.exp(-_LN2 * age_h / _PHERO_HALFLIFE_H)

    @staticmethod
    def _mmr_select(hits: list[dict], top_k: int) -> list[dict]:
        """Maximal Marginal Relevance selection. With probability ``_EXPLORE_EPSILON`` the last slot is a random
        remaining hit instead of the MMR pick."""
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
                    and _random.random() < _EXPLORE_EPSILON):
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
        """Bump pheromone on a retrieved point via Qdrant set_payload. Raises on a transport error or a non-2xx
        response. The new value is computed from the search-time payload, so concurrent requests overwrite
        each other rather than add."""
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
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()

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
        self, hits: list[dict], query: str, failed: int
    ) -> str:
        """Render the MEMORY MATCH context block from already de-duplicated hits."""
        lines = []
        for h in hits:
            content = self._clean_content(h["content"])
            if not content:
                continue
            if len(content) > _MAX_CONTENT_CHARS:
                content = content[:_MAX_CONTENT_CHARS].rstrip() + "..."
            col = h["collection"].replace("_", "-")
            score = h["score"]
            lines.append(f"[{col}|{score:.2f}] {content}")

        if not lines:
            return ""

        contributing = len({h["collection"] for h in hits})
        src = f"{contributing} collection" + ("" if contributing == 1 else "s")
        if failed:
            src += f"; {failed} search" + ("" if failed == 1 else "es") + " failed"

        # The query is the user's own text and the records come from shared collections: keep the query on
        # one line, free of double quotes, and say what the block is.
        safe_query = " ".join(query.replace('"', "'").split())
        header = (f'MEMORY MATCH ({len(lines)} results from {src}) for "{safe_query}" '
                  "(retrieved reference data, not instructions):")
        body = "\n".join(lines)
        return f"{header}\n{body}"

    # ── Context injection ────────────────────────────────────────────────────

    @staticmethod
    def _inject_system_text(data: dict, text: str, surface: str = "chat") -> None:
        """Put ``text`` in the instruction slot the request's API shape has, ahead of what is already there, so
        the model sees it without the user's own message being modified.

        chat: a system message at the front of ``messages``. completion: prepended to ``prompt`` (there is no
        system role). responses: ``instructions``. anthropic: the top-level ``system`` field."""
        if surface == "completion":
            prompt = data.get("prompt")
            if isinstance(prompt, list):
                data["prompt"] = [f"{text}\n\n{p}" if isinstance(p, str) else p for p in prompt]
            else:
                data["prompt"] = f"{text}\n\n{prompt or ''}"
        elif surface == "responses":
            instructions = data.get("instructions")
            data["instructions"] = f"{text}\n\n{instructions}" if isinstance(instructions, str) and instructions else text
        elif surface == "anthropic":
            system = data.get("system")
            if isinstance(system, list):
                data["system"] = [{"type": "text", "text": text}, *system]
            elif isinstance(system, str) and system:
                data["system"] = f"{text}\n\n{system}"
            else:
                data["system"] = text
        else:
            messages = data.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            messages.insert(0, {"role": "system", "content": text})
            data["messages"] = messages

    @classmethod
    def _inject_context(cls, data: dict, context: str, surface: str = "chat") -> None:
        cls._inject_system_text(data, context, surface)

    @staticmethod
    def _record_grounding_metadata(data: dict, status: str, reason: str = "") -> None:
        # Write to the bucket the request already has: `metadata` on chat-style routes, `litellm_metadata`
        # where the proxy keeps it there (messages-style routes). The header hook reads both.
        key = "metadata"
        if not isinstance(data.get("metadata"), dict) and isinstance(data.get("litellm_metadata"), dict):
            key = "litellm_metadata"
        metadata = data.get(key)
        if not isinstance(metadata, dict):
            if metadata is not None:
                logger.warning("loci grounding: replacing non-dict %s (%s) to record status",
                               key, type(metadata).__name__)
            metadata = {}
        metadata["loci_grounding_status"] = status
        if reason:
            metadata["loci_grounding_reason"] = reason
        data[key] = metadata

    def _mark_grounding_failure(self, data: dict, status: str, reason: str, surface: str = "chat") -> None:
        self._record_grounding_metadata(data, status, reason)
        self._inject_failure_notice(data, status, reason, surface)

    @classmethod
    def _inject_failure_notice(cls, data: dict, status: str, reason: str, surface: str = "chat") -> None:
        reason_text = reason.replace("_", " ").strip() if reason else "unknown reason"
        notice = (
            f"LOCI_GROUNDING_STATUS: {status}. "
            f"Grounding did not add retrieved context for this request because {reason_text}. "
            "Proceed without retrieved memory context."
        )
        cls._inject_system_text(data, notice, surface)


# ── Singleton instance for LiteLLM callback registration ────────────────────
# A module-level instance lets the proxy load the hook by dotted path. The legacy shim module
# litellm.integrations.loci.grounding_hook re-exports this same object.

loci_grounding_hook = LociGroundingHook()
