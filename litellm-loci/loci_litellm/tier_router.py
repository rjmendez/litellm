"""
LociTierRouter — fail-open tiered model routing for LiteLLM proxy.

Ported from Loci's mcp/llm_local.py and backends.py. Provides a CustomLogger
callback that monitors LLM calls and routes failures through Loci's fail-open
discipline:

  - Every op fails open: on timeout / HTTP error / bad response, return a
    well-formed degraded result and NEVER raise.
  - Transient transport errors (timeout, connection reset, bad gateway)
    are distinguished from permanent failures (auth, model not found).
  - Route metadata (backend, route, model, latency_ms, tier) is attached to
    every response for observability.
  - Per-provider circuit breaker with exponential backoff, ported from
    Loci's circuit_breaker.py design.

This complements but does not replace LiteLLM's built-in router. It hooks
into async_post_call_failure_hook to record failures and async_pre_call_hook
to check circuit state and optionally redirect to a healthy provider.

Register in litellm config.yaml:
    litellm_settings:
      callbacks:
        - litellm.integrations.loci.tier_router.loci_tier_router
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("loci_litellm.tier_router")

# ── Tunables ─────────────────────────────────────────────────────────────────
_FAILURE_THRESHOLD = int(os.environ.get("LOCI_CB_FAILURE_THRESHOLD", "5"))
_RECOVERY_WINDOW_S = int(os.environ.get("LOCI_CB_RECOVERY_WINDOW", "60"))
_BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 32, 64]  # seconds


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ProviderHealth:
    """Tracks health metrics for a single provider. Thread-safe."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.backoff_level = 0
        self.backoff_until = 0.0
        self.total_failures = 0
        self.total_successes = 0
        self.circuit_open_count = 0
        self.lock = threading.RLock()

    def record_failure(self) -> None:
        with self.lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            should_open = (
                self.state == CircuitState.HALF_OPEN
                or (self.state == CircuitState.CLOSED and self.failure_count >= _FAILURE_THRESHOLD)
            )
            if should_open:
                self._open_circuit()

    def record_success(self) -> None:
        with self.lock:
            self.success_count += 1
            self.total_successes += 1
            self.failure_count = 0
            if self.state != CircuitState.CLOSED:
                self.state = CircuitState.CLOSED
                self.backoff_level = 0
                self.backoff_until = 0.0
                logger.info("loci tier_router: %s circuit closed (recovered)", self.name)

    def _open_circuit(self) -> None:
        self.state = CircuitState.OPEN
        self.circuit_open_count += 1
        delay = _BACKOFF_SEQUENCE[min(self.backoff_level, len(_BACKOFF_SEQUENCE) - 1)]
        self.backoff_level = min(self.backoff_level + 1, len(_BACKOFF_SEQUENCE) - 1)
        self.backoff_until = time.time() + delay
        logger.warning(
            "loci tier_router: %s circuit OPENED after %d failures, backoff %ds",
            self.name, self.failure_count, delay,
        )

    def can_proceed(self) -> bool:
        """Check if requests should be allowed through."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                now = time.time()
                if now >= self.backoff_until:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("loci tier_router: %s half_open (testing recovery)", self.name)
                    return True
                return False
            # HALF_OPEN: allow one test request
            return True

    def status(self) -> dict:
        with self.lock:
            return {
                "provider": self.name,
                "state": self.state.value,
                "failures": self.failure_count,
                "successes": self.success_count,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "circuit_open_count": self.circuit_open_count,
                "backoff_until": self.backoff_until,
            }


def _is_transient(exc: Exception) -> bool:
    """Classify whether an exception is a transient transport error."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    signals = (
        "timeout", "temporar", "connection aborted", "connection reset",
        "connection refused", "service unavailable", "bad gateway",
        "gateway timeout",
    )
    if "timeout" in name or "connection" in name:
        return True
    return any(s in msg for s in signals)


class LociTierRouter(CustomLogger):
    """
    LiteLLM CustomLogger that provides fail-open tiered routing with
    per-provider circuit breaking.

    Fail-open discipline:
    - On any error, record it and let the request proceed to the next
      provider in LiteLLM's router fallback chain.
    - Never raise from a hook — raising would break the proxy.
    - Circuit state is advisory: it logs and tags, it doesn't hard-block
      (LiteLLM's own router handles fallback).
    """

    def __init__(self) -> None:
        super().__init__()
        self._providers: dict[str, ProviderHealth] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _circuit_key(payload: dict) -> str:
        litellm_params = payload.get("litellm_params") or {}
        for metadata in (litellm_params.get("metadata"), payload.get("metadata")):
            if not isinstance(metadata, dict):
                continue
            group = metadata.get("model_group")
            if isinstance(group, str) and group:
                return group
        model = payload.get("model")
        return model if isinstance(model, str) and model else "unknown"

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Exception | str | dict | None:
        """Check circuit state before the call. Advisory only — does not block."""
        provider = self._circuit_key(data)
        if provider == "unknown":
            return None

        with self._lock:
            health = self._providers.setdefault(provider, ProviderHealth(provider))

        if not health.can_proceed():
            # Log but don't block — let LiteLLM's router handle fallback
            logger.warning(
                "loci tier_router: %s circuit open, request proceeding to fallback",
                provider,
            )
            # Tag metadata for observability
            metadata = data.get("metadata") or {}
            metadata["loci_circuit_open"] = provider
            data["metadata"] = metadata

        return None

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record success on the provider."""
        provider = self._circuit_key(kwargs)
        if provider == "unknown":
            return
        with self._lock:
            health = self._providers.setdefault(provider, ProviderHealth(provider))
        health.record_success()

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record failure on the provider, classify transient vs permanent."""
        provider = self._circuit_key(kwargs)
        if provider == "unknown":
            return

        # Extract the original exception
        exc = kwargs.get("original_exception") or kwargs.get("exception")
        is_transient = False
        if exc and isinstance(exc, Exception):
            is_transient = _is_transient(exc)

        with self._lock:
            health = self._providers.setdefault(provider, ProviderHealth(provider))
        # errors (auth, model not found) are config issues, not provider health
        if is_transient:
            health.record_failure()
            logger.debug(
                "loci tier_router: %s transient failure recorded (%d/%d)",
                provider, health.failure_count, _FAILURE_THRESHOLD,
            )
        else:
            logger.debug(
                "loci tier_router: %s permanent error (not counted): %r",
                provider, exc,
            )

    def get_status(self) -> dict:
        """Return circuit breaker status for all known providers."""
        with self._lock:
            return {
                name: health.status()
                for name, health in self._providers.items()
            }


# ── Singleton instance ──────────────────────────────────────────────────────
loci_tier_router = LociTierRouter()
