"""Cost and reliability ledger: one metadata-only JSON line per call that LiteLLM reports through its
async success or failure callbacks.

Records what each call cost, how long it took, whether it succeeded and where it ran, so spend can be
reported per model group and the local-versus-cloud routing can be judged on measured reliability.

Coverage: only calls logged through LiteLLM's async callbacks are recorded (sync log paths are not), and
one row is written per ATTEMPT, not per client request: a router retry or fallback is its own LiteLLM call
with its own logging object, so a request that fails twice and then succeeds is three rows (verified with a
Router at num_retries=3 against an always-failing mock: 4 failure rows for 1 request). Call counts and the
local success rate in cost_report are therefore per attempt. The callback records only once it is registered in the LiteLLM config
(`litellm.integrations.loci.cost_ledger.loci_cost_ledger` or `loci_litellm.cost_ledger.loci_cost_ledger`);
registered, it records unless LOCI_COST_LEDGER_DISABLED=1.

Row fields (the complete set, enforced by a test): ts, call_id, trace_id, group, deployment, model, host,
local, ok, call_type, stream, prompt_tokens, completion_tokens, cost_usd, latency_s, cache_hit,
error_class, internal, grounding, agent (first 60 chars of the User-Agent), key_alias. No prompt or
response text is written. `agent` and `key_alias` identify callers, so treat the file as sensitive.

`local` is an address-class heuristic on the deployment's api_base host, not a statement about where the
model runs: loopback, link-local, private (RFC 1918 / ULA) and CGNAT 100.64.0.0/10 (Tailscale) addresses,
`localhost`, and any host named in LOCI_LOCAL_HOSTS (comma separated) count as local. Any other name is
`local: false`, so name a local GPU host in LOCI_LOCAL_HOSTS or the report counts its calls as cloud.
`latency_s` is null when the latency could not be determined (the report skips nulls).

File: env LOCI_COST_LEDGER_FILE, else `[cost] ledger_file` in the private config, else
~/.loci/litellm_cost_ledger.jsonl. The file is append-only with no rotation, size cap or fsync: growth
and retention are the operator's job. Rows are written by independent processes using O_APPEND; very
large rows could interleave. A write failure is logged at WARNING (the first, then every
100th with a count) and never fails the request. A call with neither model group nor api_base cannot be
attributed and is dropped with the same rate-limited WARNING.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from litellm.integrations.custom_logger import CustomLogger
from loci_litellm.backend_resolver import BackendResolver

logger = logging.getLogger("loci_litellm.cost_ledger")

_WRITE_LOCK = threading.Lock()


def ledger_path() -> str:
    explicit = os.environ.get("LOCI_COST_LEDGER_FILE") or BackendResolver.cost_ledger_file()
    return str(explicit or (Path.home() / ".loci" / "litellm_cost_ledger.jsonl"))


def _disabled() -> bool:
    return os.environ.get("LOCI_COST_LEDGER_DISABLED", "").strip().lower() in ("1", "true", "yes", "on")


_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _local_hosts() -> set:
    raw = os.environ.get("LOCI_LOCAL_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _is_local(api_base: str) -> bool:
    """Address-class heuristic: loopback, link-local, private, CGNAT (Tailscale) or a host named in
    LOCI_LOCAL_HOSTS. Any other hostname is treated as not local."""
    host = (urlparse(api_base or "").hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host in _local_hosts():
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip in _CGNAT


_WARN_EVERY = 100
_warn_counts: dict = {}


def _warn_throttled(kind: str, message: str, *args: Any) -> None:
    """Log the first occurrence of `kind` and then every _WARN_EVERY-th, with the running count."""
    count = _warn_counts.get(kind, 0) + 1
    _warn_counts[kind] = count
    if count == 1 or count % _WARN_EVERY == 0:
        logger.warning(message + " (occurrence %d)", *args, count)


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


class LociCostLedger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def build_row(kwargs: dict[str, Any], ok: bool, start_time: Any, end_time: Any) -> Optional[dict[str, Any]]:
        slo: Any = kwargs.get("standard_logging_object")
        slo = slo if isinstance(slo, dict) else {}
        params: Any = kwargs.get("litellm_params")
        params = params if isinstance(params, dict) else {}
        meta_slo: Any = slo.get("metadata")
        meta_slo = meta_slo if isinstance(meta_slo, dict) else {}
        meta_params: Any = params.get("metadata")
        meta_params = meta_params if isinstance(meta_params, dict) else {}
        requester: Any = meta_slo.get("requester_metadata")
        api_base = str(slo.get("api_base") or params.get("api_base") or kwargs.get("api_base") or "")
        group = slo.get("model_group") or meta_params.get("model_group") or kwargs.get("model") or ""
        latency: Optional[float]
        try:
            latency = float(slo.get("response_time") or (end_time - start_time).total_seconds())
        except Exception:
            latency = None
        error = slo.get("error_information") if isinstance(slo.get("error_information"), dict) else {}
        if not group and not api_base:
            return None
        return {
            "ts": round(time.time(), 3),
            "call_id": str(slo.get("litellm_call_id") or kwargs.get("litellm_call_id") or ""),
            "trace_id": str(slo.get("trace_id") or ""),
            "group": str(group),
            "deployment": str(slo.get("model_id") or meta_params.get("model_id") or ""),
            "model": str(slo.get("model") or kwargs.get("model") or ""),
            "host": urlparse(api_base).hostname or "",
            "local": _is_local(api_base),
            "ok": bool(ok),
            "call_type": str(slo.get("call_type") or kwargs.get("call_type") or ""),
            "stream": bool(slo.get("stream") or kwargs.get("stream")),
            "prompt_tokens": int(_num(slo.get("prompt_tokens"))),
            "completion_tokens": int(_num(slo.get("completion_tokens"))),
            "cost_usd": _num(slo.get("response_cost") or kwargs.get("response_cost")),
            "latency_s": None if latency is None else round(latency, 3),
            "cache_hit": bool(slo.get("cache_hit")),
            "error_class": str(error.get("error_class") or error.get("error_code") or "") if not ok else "",
            "internal": BackendResolver.is_internal_call(kwargs.get("metadata"), meta_params, requester),
            "grounding": str(meta_params.get("loci_grounding_status") or ""),
            "agent": str(slo.get("user_agent") or "")[:60],
            "key_alias": str(meta_slo.get("user_api_key_alias") or ""),
        }

    @staticmethod
    def _append(row: dict[str, Any]) -> None:
        path = Path(ledger_path())
        line = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        with _WRITE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)

    async def _record(self, kwargs: dict[str, Any], ok: bool, start_time: Any, end_time: Any) -> None:
        if _disabled():
            return
        try:
            row = self.build_row(kwargs, ok, start_time, end_time)
            if row is None:
                _warn_throttled("dropped", "loci cost ledger: dropped a call with no model group and no api_base")
                return
            await asyncio.to_thread(self._append, row)
        except Exception as exc:
            _warn_throttled("write", "loci cost ledger: could not record a call: %r", exc)

    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any,
                                      start_time: Any, end_time: Any) -> None:
        await self._record(kwargs, True, start_time, end_time)

    async def async_log_failure_event(self, kwargs: dict[str, Any], response_obj: Any,
                                      start_time: Any, end_time: Any) -> None:
        await self._record(kwargs, False, start_time, end_time)


loci_cost_ledger = LociCostLedger()
