"""
Loci integration for LiteLLM proxy.

Port of Loci's (github.com/rjmendez/loci) grounding pipeline, backend
resolution, and fail-open tiered routing into LiteLLM's native callback system.

Components:
  - LociGroundingHook: async_pre_call_hook that embeds the user message,
    fans out to Qdrant collections in parallel, and injects a MEMORY MATCH
    context block into the request before the upstream LLM is called.
  - BackendResolver: portable resolution chain (env → local probe → config →
    default) for Ollama/Qdrant/vLLM endpoints.  Machine-specific endpoints
    stay out of the codebase.
  - FailOpenMixin: the fail-open discipline Loci uses everywhere — on timeout,
    HTTP error, or bad JSON, return a well-formed degraded result and never
    raise.

Configuration via litellm_settings in config.yaml:
    callbacks:
      - litellm.integrations.loci.grounding_hook.loci_grounding_hook

Environment variables (all optional — sensible defaults, fail-open if unset):
    LOCI_QDRANT_URL, LOCI_QDRANT_API_KEY
    LOCI_OLLAMA_BASE_URL, LOCI_EMBED_MODEL
    LOCI_RECALL_TOP_K, LOCI_RECALL_MIN_SCORE, LOCI_RECALL_MIN_IMPORTANCE
    LOCI_GROUNDING_EXTRA_COLLECTIONS
    LOCI_GROUNDING_DISABLED (set to "1" to disable)
"""

from litellm.integrations.loci.grounding_hook import (
    LociGroundingHook,
    loci_grounding_hook,
)
from litellm.integrations.loci.backend_resolver import BackendResolver
from litellm.integrations.loci.memory_writer import LociMemoryWriter, loci_memory_writer

__all__ = [
    "LociGroundingHook",
    "loci_grounding_hook",
    "BackendResolver",
    "LociMemoryWriter",
    "loci_memory_writer",
]
