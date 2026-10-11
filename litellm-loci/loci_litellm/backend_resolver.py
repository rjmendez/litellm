"""
BackendResolver - backend resolution for the Loci integration, so a LiteLLM install carries no
machine-specific endpoint or key in code.

Resolution order, per backend (this is the actual order; it differs between backends):
  - Ollama embedding URL (ollama_url): env LOCI_OLLAMA_BASE_URL / OLLAMA_BASE_URL, then a TCP probe of
    LOCI_LOCAL_OLLAMA (default http://localhost:11434), then `[ollama] url` in the config file, else ''.
  - Embeddings API base (embed_api_base): ollama_url() + "/v1"; only when that is empty,
    MNEMOSYNE_EMBEDDING_API_URL (already a full /v1 endpoint). So a reachable local Ollama wins over
    MNEMOSYNE_EMBEDDING_API_URL.
  - Embedding model: LOCI_EMBED_MODEL, MNEMOSYNE_EMBEDDING_MODEL, `[embed] model`, "nomic-embed-text".
  - Qdrant: LOCI_QDRANT_URL / QDRANT_URL then `[qdrant] url`; key LOCI_QDRANT_API_KEY / QDRANT_API_KEY
    then `[qdrant] api_key`; else ''.
  - Other settings: env, then config file, then default.

Config file: $LOCI_CONFIG, else ~/.loci/backends.toml (TOML). The path, LOCI_LOCAL_OLLAMA and the file
contents are read once per process: changing them later has no effect until restart (tests clear
_config.cache_clear()).

Fail-open, loudly: an unreadable or unparseable config file yields an empty config for the life of the
process and a WARNING naming the path and the error (also when LOCI_CONFIG points at a missing file).
config_load_error() returns that reason (or ''), so callers can surface it. A config section that is
not a table is ignored with a WARNING. Most methods read the environment on every call; only ollama_url
is memoized (per probe_timeout), so its first answer, including "no Ollama reachable", is kept until
restart and its env var is frozen at first call.
"""

from __future__ import annotations

import functools
import logging
import os
import socket
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("loci_litellm.backends")

_CONFIG_PATH = os.environ.get("LOCI_CONFIG") or str(Path.home() / ".loci" / "backends.toml")

_LOCAL_OLLAMA = os.environ.get("LOCI_LOCAL_OLLAMA", "http://localhost:11434")

_config_error = ""


def config_load_error() -> str:
    """Why the private config is not in effect ('' when it loaded, or none exists at the default path)."""
    return _config_error


@functools.lru_cache(maxsize=1)
def _config() -> dict:
    """Parse the private TOML config. Fail-open to {} - with a WARNING and config_load_error() set."""
    global _config_error
    _config_error = ""
    p = Path(_CONFIG_PATH)
    try:
        import tomllib
        if p.exists():
            return tomllib.loads(p.read_text(encoding="utf-8"))
        if os.environ.get("LOCI_CONFIG"):
            _config_error = f"LOCI_CONFIG points at a missing file: {p}"
    except Exception as exc:
        _config_error = f"could not read/parse {p}: {exc!r}"
    if _config_error:
        logger.warning("loci backends: running with an EMPTY config (defaults only): %s", _config_error)
    return {}


_warned_sections: set = set()


def _cfg(section: str, key: str, default=None):
    value = _config().get(section)
    if value is None:
        return default
    if not isinstance(value, dict):
        if section not in _warned_sections:
            _warned_sections.add(section)
            logger.warning("loci backends: config section [%s] is not a table; ignored", section)
        return default
    return value.get(key, default)


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
    Backend resolution (see the module docstring for the per-backend order).

    Methods return '' or a default rather than raising. Only ollama_url is memoized.
    """

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def ollama_url(probe_timeout: float = 1.0) -> str:
        """Ollama embedding base URL: env -> TCP probe of LOCI_LOCAL_OLLAMA -> config -> ''.

        Memoized per probe_timeout: the first result, including '' (nothing reachable), is kept for
        the process lifetime, so an Ollama started later is not picked up until restart.
        """
        env = os.environ.get("LOCI_OLLAMA_BASE_URL") or os.environ.get("OLLAMA_BASE_URL")
        if env:
            return env
        if _alive(_LOCAL_OLLAMA, timeout=probe_timeout):
            return _LOCAL_OLLAMA
        return _cfg("ollama", "url", "") or ""

    @staticmethod
    def embed_model() -> str:
        return (os.environ.get("LOCI_EMBED_MODEL")
                or os.environ.get("MNEMOSYNE_EMBEDDING_MODEL")
                or _cfg("embed", "model", "nomic-embed-text"))

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

        ollama_url() (env, local probe, config) + /v1 wins; only when it is empty is
        MNEMOSYNE_EMBEDDING_API_URL (already a full /v1 endpoint) used.
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

    @staticmethod
    def cost_ledger_file() -> str:
        """`[cost] ledger_file` from the private config ('' when unset). LOCI_COST_LEDGER_FILE is
        handled by cost_ledger.ledger_path()."""
        return str(_cfg("cost", "ledger_file", "") or "")

    @staticmethod
    def grounding_audit_file() -> str:
        return os.environ.get("LOCI_GROUNDING_AUDIT_FILE", "")

    @staticmethod
    def memory_write_collection() -> str:
        """Collection the memory writer upserts into. Env LOCI_MEMORY_WRITE_COLLECTION, else `[memory]
        write_collection` in the private config, else the generic `loci_memory`."""
        return (os.environ.get("LOCI_MEMORY_WRITE_COLLECTION")
                or _cfg("memory", "write_collection", "")
                or "loci_memory")

    @staticmethod
    def client_rag_ok_markers() -> tuple[str, ...]:
        """Strings a client puts in a request when its own retrieval succeeded. Machine-private: set them in the
        gitignored config (`[grounding] client_rag_ok_markers = [...]`) or in LOCI_CLIENT_RAG_OK_MARKERS ('||'
        separated). Empty by default, which turns the client shortcut off."""
        raw = os.environ.get("LOCI_CLIENT_RAG_OK_MARKERS")
        if raw is not None:
            values: list = raw.split("||")
        else:
            configured = _cfg("grounding", "client_rag_ok_markers", [])
            values = configured if isinstance(configured, list) else [configured]
        return tuple(v.strip() for v in values if isinstance(v, str) and v.strip())

    @staticmethod
    def client_rag_trusted() -> bool:
        return os.environ.get("LOCI_TRUST_CLIENT_RAG", "1").lower() in ("1", "true", "yes", "on")

    @staticmethod
    def is_internal_call(*metadata_sources) -> bool:
        """True when a request is marked as the memory system's own plumbing (judges, classifiers, embeddings),
        which must be neither grounded nor written back to memory. The client sets
        ``metadata.loci_internal_call``; LOCI_HONOR_INTERNAL_CALL=0 ignores the marker so every request is
        treated as user traffic. Note the marker is an opt-out any authenticated caller can use."""
        if os.environ.get("LOCI_HONOR_INTERNAL_CALL", "1").lower() not in ("1", "true", "yes", "on"):
            return False
        return any(isinstance(m, dict) and m.get("loci_internal_call") is True for m in metadata_sources)

    @staticmethod
    def pheromone_deposit_disabled() -> bool:
        return os.environ.get("LOCI_PHERO_DEPOSIT_DISABLED", "").lower() in ("1", "true", "yes", "on")

def ledger_path() -> str:
    """Where the cost ledger lives: LOCI_COST_LEDGER_FILE, else `[cost] ledger_file` in the private config,
    else ~/.loci/litellm_cost_ledger.jsonl. Here (not in cost_ledger) so the offline report can resolve it
    without importing LiteLLM."""
    explicit = os.environ.get("LOCI_COST_LEDGER_FILE") or BackendResolver.cost_ledger_file()
    return str(explicit or (Path.home() / ".loci" / "litellm_cost_ledger.jsonl"))
