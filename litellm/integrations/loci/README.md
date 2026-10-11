# Loci Integration for LiteLLM Proxy

Forks [BerriAI/litellm](https://github.com/BerriAI/litellm) and integrates
[Loci](https://github.com/rjmendez/loci), a persistent memory MCP server,
into the LiteLLM proxy's callback system.

The implementation lives in the standalone `litellm-loci/loci_litellm` package.
The modules in this directory are thin shims that re-export it, so both the
`litellm.integrations.loci.*` and `loci_litellm.*` callback paths work.

## What it does

Three integration points, all implemented as LiteLLM `CustomLogger` callbacks:

### 1. Grounding Hook (`grounding_hook.py`)

Before each chat/completion call is forwarded to the upstream LLM, the hook:
1. Extracts intent from the user's latest message
2. Embeds it via `nomic-embed-text` on Ollama (or any OpenAI-compatible endpoint)
3. Fans out parallel Qdrant searches across all configured collections
4. Ranks results by a four-term weighted score (relevance / recency / trust /
   type) with MMR diversification and stigmergic pheromone reinforcement
5. Injects a `MEMORY MATCH` context block as a system message prepended to
   the request

**Fail-open:** if Qdrant, Ollama, or any collection is unreachable, the request
proceeds unchanged. Grounding is best-effort, never blocking.

### 2. Backend Resolver (`backend_resolver.py`)

Portable backend resolution ported from Loci's `mcp/backends.py`. Lets the
proxy work unchanged on any machine without hardcoding infra:

```
env var → local probe → ~/.loci/backends.toml → safe default/empty
```

A laptop auto-uses its own local Ollama; a headless box falls back to a GPU
host over tailscale. Machine-specific endpoints stay in the gitignored TOML,
never in the codebase.

### 3. Tier Router (`tier_router.py`)

Per-model-group circuit breaker with exponential backoff, ported from Loci's
fail-open discipline. The circuit key is the router's model group (for example
`simple-tier`), so the request-time check and the success/failure events agree.
Distinguishes transient transport errors (timeout, connection reset) from
permanent failures (auth, model not found). Only transient errors count toward
the circuit. A failed half-open probe moves to the next backoff step, and
a success closes the circuit and resets the sequence. The circuit is advisory.
It logs and tags metadata, but lets LiteLLM's own router handle fallback.

### 4. Memory Writer (`memory_writer.py`)

After each successful completion, writes the prompt and a response preview as a
point in the `LOCI_MEMORY_WRITE_COLLECTION` collection (default `hermes_memory`).
Grounding can retrieve these points later, so test traffic ends up in memory
unless the writer is disabled with `LOCI_MEMORY_WRITER_DISABLED=1`.

## Installation

### In your LiteLLM config.yaml:

```yaml
litellm_settings:
  callbacks:
    - litellm.integrations.loci.grounding_hook.loci_grounding_hook
    - litellm.integrations.loci.tier_router.loci_tier_router
```

### Environment variables

All optional — the integration fails open (no-op) if nothing is configured.

| Variable | Default | Description |
|---|---|---|
| `LOCI_QDRANT_URL` | (from config/empty) | Qdrant endpoint |
| `LOCI_QDRANT_API_KEY` | (from config/empty) | Qdrant API key |
| `LOCI_OLLAMA_BASE_URL` | (local probe) | Ollama host |
| `LOCI_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `LOCI_GROUNDING_EXTRA_COLLECTIONS` | (empty) | Extra Qdrant collections |
| `LOCI_RECALL_TOP_K` | `3` | Max grounding results |
| `LOCI_RECALL_MIN_SCORE` | `0.55` | Min Qdrant cosine score |
| `LOCI_RECALL_MIN_IMPORTANCE` | `0.2` | Min importance weight |
| `LOCI_GROUNDING_DISABLED` | (unset) | Set to `1` to disable |
| `LOCI_PHERO_DEPOSIT_DISABLED` | (unset) | Set to `1` to make grounding read-only (no pheromone write-back) |
| `LOCI_MEMORY_WRITER_DISABLED` | (unset) | Set to `1` to stop the memory writer |
| `LOCI_MEMORY_WRITE_COLLECTION` | `hermes_memory` | Collection the memory writer upserts into |
| `LOCI_CB_FAILURE_THRESHOLD` | `5` | Transient failures before a circuit opens |

### Backend config file

Machine-specific endpoints go in `~/.loci/backends.toml` (gitignored):

```toml
[ollama]
url = "http://gpu-host:11434"        # embedding endpoint
gen_url = "http://gpu-host:11434"    # generation (may differ)
gen_model = "qwen2.5:3b"

[qdrant]
url = "http://qdrant-host:6333"
api_key = "..."

[embed]
model = "nomic-embed-text"
```

## Architecture

```
Client request
      │
      ▼
┌─────────────────────────────────────────────┐
│  LociGroundingHook (async_pre_call_hook)     │
│  ┌─────────────────────────────────────┐    │
│  │ 1. Extract user message intent      │    │
│  │ 2. Embed via Ollama /v1/embeddings  │    │
│  │ 3. Parallel Qdrant fan-out          │    │
│  │ 4. Rank: relevance × recency ×      │    │
│  │    trust × type + pheromone boost   │    │
│  │ 5. MMR diversity selection          │    │
│  │ 6. Inject MEMORY MATCH as system msg│    │
│  └─────────────────────────────────────┘    │
│         (fail-open on any error)             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  LociTierRouter (async_pre_call_hook)        │
│  Checks circuit state → tags metadata        │
│  (advisory only, LiteLLM router handles      │
│   fallback)                                   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  LiteLLM Router → Upstream LLM               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  LociTierRouter (async_log_success/failure)  │
│  Records success → resets failure count     │
│  Records transient failure → circuit count   │
└─────────────────────────────────────────────┘
```

## Collections

Default collections searched per turn (all 768-dim Qdrant):

| Collection | Role |
|---|---|
| `mnemosyne` | Personal facts, preferences, project notes |
| `loci_sessions` | Past conversation history |
| `loci_memory` | Investigation notes, research findings |

Add project-specific collections via `LOCI_GROUNDING_EXTRA_COLLECTIONS`:

```
LOCI_GROUNDING_EXTRA_COLLECTIONS=ecc_skills,agent_core_chunks,dama_gotchi_code
```

## Origin

This is a fork of [BerriAI/litellm](https://github.com/BerriAI/litellm) with
Loci's patterns ported from:
- `scripts/hooks/pre_llm_grounding.py` → `grounding_hook.py`
- `mcp/backends.py` → `backend_resolver.py`
- `circuit_breaker.py` + `mcp/llm_local.py` fail-open → `tier_router.py`

See the upstream repos:
- [BerriAI/litellm](https://github.com/BerriAI/litellm) — the proxy we forked
- [rjmendez/loci](https://github.com/rjmendez/loci) — the memory system we ported from
