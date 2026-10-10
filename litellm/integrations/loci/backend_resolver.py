"""
BackendResolver — portable backend resolution for the Loci integration.

Ported from Loci's mcp/backends.py. Lets a LiteLLM install work UNCHANGED on
any machine without hardcoding infra — a laptop uses its own local GPU, a
headless box falls back to shared infra over tailscale — and keeps every
machine-specific endpoint/key OUT of the codebase.

Resolution chain per backend:
  1. explicit env var — power-user override
  2. a LOCAL probe (e.g. localhost:11434 for Ollama) — laptop auto-uses its own hardware
  3. a gitignored config file — remote infra (GPU host, Qdrant key)
  4. a safe default / empty — the tiers already fail-open when a backend is empty

Config file: $LOCI_CONFIG, else ~/.loci/backends.toml. TOML (stdlib tomllib).
"""

from __future__ import annotations

import functools
import logging
import os
import socket
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("litellm.integrations.loci.backends")

_CONFIG_PATH = os.environ.get("LOCI_CONFIG") or str(Path.home() / ".loci" / "backends.toml")

_LOCAL_OLLAMA = os.environ.get("LOCI_LOCAL_OLLAMA", "http://localhost:11434")
_LOCAL_VLLM = os.environ.get("LOCI_LOCAL_VLLM", "http://localhost:8000")
_OLLAMA_LIST_TIMEOUT = float(os.environ.get("LOCI_OLLAMA_LIST_TIMEOUT", "2.0"))


@functools.lru_cache(maxsize=1)
def _config() -> dict:
    """Parse the gitignored TOML config. Fail-open: missing/broken file -> {}."""
    try:
        import tomllib
        p = Path(_CONFIG_PATH)
        if p.exists():
            return tomllib.loads(p.read_text())
    except Exception as exc:
        logger.debug("loci backends: _config fail-open: %r", exc)
    return {}


def _cfg(section: str, key: str, default=None):
    return (_config().get(section) or {}).get(key, default)


def _alive(url: str, timeout: float = 1.0) -> bool:
    """Cheap TCP reachability probe of a URL's host:port. Never raises."""
    if not url:
        return False
    try:
        u = urlparse(url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


class BackendResolver:
    """
    Portable backend resolution ported from Loci's backends.py.

    All methods are memoized — the probe runs once per process lifetime.
    Fail-open: every method returns '' or a safe default, never raises.
    """

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def ollama_url(probe_timeout: float = 1.0) -> str:
        """Ollama base URL: env -> local probe -> config -> ''.

        This is the EMBEDDING endpoint. Generation may live on a different host.
        """
        env = os.environ.get("LOCI_OLLAMA_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
        if env:
            return env
        if _alive(_LOCAL_OLLAMA, timeout=probe_timeout):
            return _LOCAL_OLLAMA
        return _cfg("ollama", "url", "") or ""

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def ollama_gen_url(probe_timeout: float = 1.0) -> str:
        """Ollama GENERATION endpoint — resolved separately from the embedding one.

        A shared/cluster Ollama may serve embeddings only, while your own GPU
        box serves generation with a richer model library.
        """
        env = (os.environ.get("LOCI_OLLAMA_GEN_URL")
               or os.environ.get("OLLAMA_GEN_URL") or "")
        if env:
            return env
        configured = _cfg("ollama", "gen_url", "") or ""
        if configured:
            return configured
        return BackendResolver.ollama_url(probe_timeout)

    @staticmethod
    def ollama_gen_model() -> str:
        """Generation model tag. Env -> config -> safe default."""
        env_or_cfg = (os.environ.get("LOCI_OLLAMA_GEN_MODEL")
                      or _cfg("ollama", "gen_model", ""))
        return env_or_cfg or "qwen2.5:3b"

    @staticmethod
    def embed_model() -> str:
        return (os.environ.get("LOCI_EMBED_MODEL")
                or os.environ.get("MNEMOSYNE_EMBEDDING_MODEL")
                or _cfg("embed", "model", "nomic-embed-text"))

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def vllm_url(probe_timeout: float = 1.0) -> str:
        """vLLM/OpenAI base URL: env -> config -> local probe -> ''."""
        env = os.environ.get("LOCI_VLLM_BASE_URL") or os.environ.get("VLLM_BASE_URL")
        if env:
            return env
        configured = _cfg("vllm", "url", "") or ""
        if configured:
            return configured
        if _alive(_LOCAL_VLLM, timeout=probe_timeout):
            return _LOCAL_VLLM
        return ""

    @staticmethod
    def vllm_model() -> str:
        return (os.environ.get("LOCI_VLLM_MODEL")
                or os.environ.get("VLLM_MODEL")
                or _cfg("vllm", "model", "Qwen2.5-3B-Instruct"))

    @staticmethod
    def rerank_model() -> str:
        return (os.environ.get("LOCI_RERANK_MODEL")
                or os.environ.get("RERANK_MODEL")
                or _cfg("rerank", "model", "BAAI/bge-reranker-v2-m3"))

    @staticmethod
    def qdrant() -> Tuple[str, str]:
        """(url, api_key) for Qdrant. Env -> config -> ('', '')."""
        url = (os.environ.get("LOCI_QDRANT_URL")
               or os.environ.get("QDRANT_URL")
               or _cfg("qdrant", "url", "") or "")
        key = (os.environ.get("LOCI_QDRANT_API_KEY")
               or os.environ.get("QDRANT_API_KEY")
               or _cfg("qdrant", "api_key", "") or "")
        return url, key

    @staticmethod
    def qdrant_url() -> str:
        return BackendResolver.qdrant()[0]

    @staticmethod
    def qdrant_api_key() -> str:
        return BackendResolver.qdrant()[1]

    @staticmethod
    def embed_api_base() -> Optional[str]:
        """OpenAI-compatible embeddings base URL.

        OLLAMA_BASE_URL is a bare host and needs /v1 appended;
        MNEMOSYNE_EMBEDDING_API_URL is already a full /v1 endpoint.
        """
        base = BackendResolver.ollama_url()
        if base:
            return f"{base.rstrip('/')}/v1"
        api = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL")
        return api.rstrip("/") if api else None

    @staticmethod
    def embed_api_key() -> str:
        return (os.environ.get("LOCI_EMBED_API_KEY")
                or os.environ.get("EMBED_API_KEY")
                or "")

    @staticmethod
    def is_disabled() -> bool:
        """Check if the Loci integration is explicitly disabled."""
        return os.environ.get("LOCI_GROUNDING_DISABLED", "").lower() in ("1", "true", "yes", "on")
