# loci-litellm

LiteLLM proxy callbacks that read from and write to a Qdrant-backed memory store:

| Callback | Module | What it does |
|---|---|---|
| grounding hook | `grounding_hook.loci_grounding_hook` | Before a call, retrieves matching records from Qdrant and puts them in the request's instruction slot. |
| memory writer | `memory_writer.loci_memory_writer` | After a successful call, writes a prompt/answer preview into a Qdrant collection. |
| tier router | `tier_router.loci_tier_router` | Counts transient failures per model group and logs when a circuit opens. Advisory: it does not change routing. |
| cost ledger | `cost_ledger.loci_cost_ledger` | Appends one metadata-only JSON line per call to a file. `python -m loci_litellm.cost_report` summarises it (offline: it does not need LiteLLM installed). |
| capability gate | `capability_gate.loci_capability_gate` | Tags a request that carries tool definitions or tool turns with `!<local tag>` so the router can keep it off local deployments. Inert unless router tag filtering is on and the local deployments are tagged (see below). |

`dedup_gate` was a semantic cache. It could not work (LiteLLM treats a dict returned from a pre-call hook as the
replacement request, not as a response) and was removed. The module remains as an inert placeholder that logs one
WARNING if it is still configured; delete it from `callbacks`.

Version: `loci_litellm.__version__` (from `pyproject.toml` in a source checkout, else installed metadata, else
`unknown`). It and the package path are logged once at INFO on import.

## Requirements and install

- Python 3.11+, `httpx`, `pyyaml`, and a LiteLLM that provides `CustomLogger`. The `x-loci-grounding` response header
  needs LiteLLM to call `async_post_call_response_headers_hook`; this checkout does. Elsewhere grounding still runs
  and the header is not sent. Nothing here has been verified against other LiteLLM versions.
- From a checkout, nothing is installed: the shim `litellm.integrations.loci` adds `litellm-loci/` to `sys.path`
  (see below). `pip install ./litellm-loci` also works. This README does not claim the package is published anywhere.
- Register callbacks in `config.yaml` with one dotted path per callback, not both:

```yaml
litellm_settings:
  callbacks:
    - litellm.integrations.loci.grounding_hook.loci_grounding_hook   # or loci_litellm.grounding_hook.loci_grounding_hook
    - litellm.integrations.loci.memory_writer.loci_memory_writer
    - litellm.integrations.loci.tier_router.loci_tier_router
    - litellm.integrations.loci.cost_ledger.loci_cost_ledger
    - litellm.integrations.loci.capability_gate.loci_capability_gate
```

Both paths give the same object. `litellm/integrations/loci/*.py` contains no logic; each file re-exports from
`loci_litellm`. If `loci_litellm` can be found neither in the checkout nor on `sys.path`, importing the shim raises an
`ImportError` that says so and the proxy does not start.

## What happens when something fails

Nothing in these callbacks raises into the request. Failures are visible as follows.

| Situation | Request | Where you see it |
|---|---|---|
| Grounding: no hit above the thresholds | continues, no context | status `miss`, reason `no_matching_context`; notice injected; WARNING |
| Grounding: no hit, and some collection searches failed | continues, no context | status `miss`, reason `no_matching_context_some_searches_failed` (part of the store was never searched); notice injected; WARNING with `N of M searches failed` |
| Grounding: Qdrant or embedder unconfigured, unreachable, or every collection search failed | continues, no context | status `unavailable` + reason (`qdrant_unconfigured`, `embed_unconfigured`, `embedding_failed`, `all_collections_failed`); notice injected; WARNING |
| Grounding: some collections failed, others returned hits | continues with the hits | WARNING per failed collection; block header says `N searches failed` |
| Grounding: unexpected exception | continues, no context | status `error` + exception class; notice injected; WARNING |
| Grounding skipped (disabled, internal call, unsupported call type, no user text, slash command, client or earlier hop already grounded) | unchanged | `x-loci-grounding: skipped` (or `client_grounded` / `already_grounded`) with `x-loci-grounding-reason` |
| Memory write fails (no Qdrant URL, no embed endpoint, embed or upsert error) | unaffected | WARNING with reason (first per reason, then every 100th); `loci_memory_writer.failure_counts` |
| Cost ledger cannot write | unaffected | WARNING (first, then every 100th) |

So with nothing configured the grounding hook is not a no-op: every request of a grounded call type gets a
`LOCI_GROUNDING_STATUS: unavailable` system notice and a WARNING. Set `LOCI_GROUNDING_DISABLED=1` to turn it off.

Other facts that follow from the code:

- Retrieval writes. Each returned `loci_memory` point gets its `pheromone` field bumped (read-modify-write, not
  atomic, not capped) unless `LOCI_PHERO_DEPOSIT_DISABLED=1`. The deposits run concurrently and add up to one 0.5 s
  timeout to the request.
- The memory writer stores the first 300 characters of the prompt and the first `LOCI_MEM_MAX_CHARS` of the answer
  with no redaction, labelled `record_type=inferred, confidence=low, provenance=model_output`. Writes use
  `wait=false`, so success is not confirmed. Grounding searches `loci_memory`, so what one caller sent can be
  injected into another caller's request; collections are not filtered per tenant or key.
- The `loci_internal_call` marker (`metadata.loci_internal_call: true`) exempts a request from grounding and from
  write-back. Any authenticated caller can set it.
- A client marker in `LOCI_CLIENT_RAG_OK_MARKERS`, in any message role, makes grounding skip the request. Nothing
  verifies the client's claim. Unset by default.
- `LOCI_INVESTIGATION_ID` is process-wide: it boosts matching records in retrieval and is stamped on every record
  the writer stores, whoever made the request.
- Streams are written once from the assembled response LiteLLM passes to the success event; a stream LiteLLM cannot
  assemble is not recorded.
- Circuit state, failure counts and the Ollama probe result are per process and in memory. The Ollama URL, including
  "nothing reachable", is resolved once and kept until restart.
- Numeric variables read at import: in `grounding_hook` and `tier_router` a non-numeric value makes the import fail
  (the proxy reports it at startup). `memory_writer` and `cost_report` fall back to the default with a WARNING.
- The delivery audit reports "context LOST" for provider body shapes it does not recognise.

## Environment variables

Read once at import (restart to change) unless marked "per call". Defaults are the values in the code.

| Variable | Default | Effect |
|---|---|---|
| `LOCI_CONFIG` | `~/.loci/backends.toml` | Path of the private TOML config (sections below). Read once. An unreadable file gives an empty config and a WARNING. |
| `LOCI_LOCAL_OLLAMA` | `http://localhost:11434` | URL probed with a TCP connect when no Ollama URL is set. |
| `LOCI_OLLAMA_BASE_URL`, `OLLAMA_BASE_URL` | unset | Ollama base URL for embeddings (first set wins). Else the probe, else `[ollama] url`. First answer is kept. |
| `MNEMOSYNE_EMBEDDING_API_URL` | unset | Full `/v1` embeddings endpoint, used only when no Ollama URL resolves. |
| `LOCI_EMBED_MODEL`, `MNEMOSYNE_EMBEDDING_MODEL` | `[embed] model`, else `nomic-embed-text` | Embedding model name. Per call. |
| `LOCI_EMBED_API_KEY`, `EMBED_API_KEY` | empty | Bearer key for the embeddings endpoint. Per call. |
| `LOCI_QDRANT_URL`, `QDRANT_URL` | `[qdrant] url`, else empty | Qdrant endpoint. Empty makes grounding `unavailable` and the writer warn. Per call. |
| `LOCI_QDRANT_API_KEY`, `QDRANT_API_KEY` | `[qdrant] api_key`, else empty | Qdrant API key. Per call. |
| `LOCI_GROUNDING_DISABLED` | unset | `1`/`true`/`yes`/`on` skips grounding (status `skipped`). Per call. |
| `LOCI_GROUNDING_EXTRA_COLLECTIONS` | empty | Comma-separated collections searched in addition to `mnemosyne`, `loci_sessions`, `loci_memory`. Read from payload field `text` with the named vector, except `ecc_skills` and `agent_core_chunks`, which have built-in mappings. |
| `LOCI_RECALL_TOP_K` | `3` | Records injected. |
| `LOCI_RECALL_MIN_SCORE` | `0.55` | Minimum Qdrant similarity score. |
| `LOCI_RECALL_MIN_IMPORTANCE` | `0.2` | Minimum importance (an explicit 0 stays 0 and is filtered). |
| `LOCI_RECALL_VECTOR_NAME` | `dense` | Named vector searched. |
| `LOCI_RECALL_MAX_CHARS` | `200` | Characters of each record shown. |
| `LOCI_EMBED_TIMEOUT` | `3.0` | Seconds for the embedding request. |
| `LOCI_QDRANT_TIMEOUT` | `2.0` | Seconds per Qdrant search. |
| `LOCI_INVESTIGATION_BOOST` | `1.5` | Relevance multiplier for records whose investigation matches `LOCI_INVESTIGATION_ID`. |
| `LOCI_INVESTIGATION_ID` | empty | Per call. See above; also stamped on written records. |
| `LOCI_RANKER_W_RELEVANCE`, `_RECENCY`, `_TRUST`, `_TYPE` | `0.50`, `0.20`, `0.15`, `0.15` | Weights of a weighted sum. A WARNING is logged at import if they do not sum to 1.0. |
| `LOCI_RECENCY_HALFLIFE_DAYS` | `7` | Recency decay half-life. |
| `LOCI_MMR_LAMBDA` | `0.75` | MMR relevance/diversity trade-off. |
| `LOCI_RECALL_EXPLORE_EPSILON` (old name `LOCI_PHERO_EPSILON`) | `0.05` | Probability that the last slot is a random remaining hit, so retrieval is not deterministic by default. `0` makes it deterministic. |
| `LOCI_PHERO_BETA` | `0.08` | Weight of the logarithmic pheromone term in the score. |
| `LOCI_PHERO_HALFLIFE_H` | `24` | Pheromone decay half-life, hours. |
| `LOCI_PHERO_DEPOSIT` | `1.0` | Amount added to a returned record's pheromone. |
| `LOCI_PHERO_DEPOSIT_DISABLED` | unset | `1` makes grounding read-only. Per call. |
| `LOCI_CLIENT_RAG_OK_MARKERS` | empty (or `[grounding] client_rag_ok_markers`) | `\|\|`-separated strings; a request containing one is not grounded. Per call. |
| `LOCI_TRUST_CLIENT_RAG` | `1` | `0` ignores those markers. Per call. |
| `LOCI_HONOR_INTERNAL_CALL` | `1` | `0` treats `loci_internal_call` requests as ordinary traffic. Per call. |
| `LOCI_GROUNDING_AUDIT_FILE` | unset | Append a metadata-only delivered/lost row per grounded request (no rotation). Unset: log line only. Per call. |
| `LOCI_MEMORY_WRITER_DISABLED` | unset | `1` stops the memory writer. Per call. |
| `LOCI_MEMORY_WRITE_COLLECTION` | `[memory] write_collection`, else `loci_memory` | Collection written to. Grounding does not search a non-default write collection unless it is also in `LOCI_GROUNDING_EXTRA_COLLECTIONS`. Per call. |
| `LOCI_MEMORY_VECTOR_NAME` | `dense` | Named vector the writer sets. Per call. |
| `LOCI_MEM_MIN_PROMPT_LEN` | `20` | Prompts shorter than this are not written. Per call. |
| `LOCI_MEM_MAX_CHARS` | `500` | Answer characters stored. Per call. |
| `LOCI_MEM_EMBED_TIMEOUT` | `3.0` | Seconds for the writer's embedding request. Per call. |
| `LOCI_MEM_QDRANT_TIMEOUT` | `2.0` | Seconds for the writer's upsert. Per call. |
| `LOCI_CB_FAILURE_THRESHOLD` | `5` | Consecutive transient failures before a model group's circuit opens (no time window). |
| `LOCI_COST_LEDGER_FILE` | `[cost] ledger_file`, else `~/.loci/litellm_cost_ledger.jsonl` | Ledger path. Append-only, no rotation. Rows hold `agent` and `key_alias`; treat as sensitive. Per call. |
| `LOCI_COST_LEDGER_DISABLED` | unset | `1` stops the ledger once registered. Per call. |
| `LOCI_CAPABILITY_GATE_DISABLED` | unset | `1` (or `true`/`yes`/`on`) stops the gate from tagging requests. Per call. |
| `LOCI_LOCAL_DEPLOYMENT_TAG` | `local` | The router tag the gate excludes (`!<tag>`); must match the `tags:` on your local deployments. Per call. |
| `LOCI_LOCAL_HOSTS` | empty | Comma-separated hostnames counted as `local` in the ledger. Without it only loopback, private, link-local and `100.64.0.0/10` addresses and `localhost` are local; any other hostname counts as cloud. Per call. |
| `LOCI_LOCAL_SLO_SUCCESS`, `_P95_S`, `_MIN_CALLS` | `0.98`, `20`, `50` | Thresholds used by `cost_report`. |

Config file sections (all optional): `[ollama] url`, `[embed] model`, `[qdrant] url` and `api_key`,
`[memory] write_collection`, `[cost] ledger_file`, `[grounding] client_rag_ok_markers`. The file lives outside any
repository and is not managed by git; it may hold an API key.

## Deployment invariants

`deploy/config.example.yaml` is a sanitized reference config. `deploy/check_deployment.py` checks a real
`config.yaml` (and optionally a `.env`) against the generic proxy rules below. It does not check that the Loci
callbacks are registered or that any `LOCI_*` setting exists. Exit status 0 means no violations. The tests in
`tests/test_deploy_check.py` run it against the example and a set of mutations.

```
python deploy/check_deployment.py --config config.yaml --env .env
```

Run it with the proxy's interpreter and import path (so LiteLLM's cost map resolves). Without `--env` the
environment half of `no-ip-literal` is not evaluated and the output says so.

| Rule | What is enforced |
|---|---|
| `embedding-group` | A model group named `embedding` (`--embedding-group`) exists; it is not in `router_settings.fallbacks`, `context_window_fallbacks` or `content_policy_fallbacks`; it is not a value in a deployment's `complexity_router_config.tiers`. Not checked: `litellm_settings.fallbacks`, `default_fallbacks`, aliases pointing at it. |
| `priced` | Every deployment has `input_cost_per_token` and `output_cost_per_token`, or LiteLLM's cost map knows the model. Exempt: the embedding group, auto-router deployments, and deployments whose `api_base` is `os.environ/NAME` with `NAME` beginning `OLLAMA`, `VLLM` or `LLAMA`. The exemption goes by that variable name only. |
| `no-ip-literal` | No literal `api_base` is an IP address. With `--env`: no variable named `*_URL`, `*_BASE`, `*_BASE_URL`, `*_ENDPOINT` or `*_HOST` (names containing `BIND` or `LISTEN` excepted), and no variable used as `api_base` through `os.environ/NAME`, holds an IP address. |
| `alias-targets` | Every `model_group_alias` target is a model group. |
| `trusted-proxies` | Warning only: `general_settings.trusted_proxy_ranges` is set. |

Notes:

- The plugin resolves its embedding endpoint from `LOCI_OLLAMA_BASE_URL` / `OLLAMA_BASE_URL`, then a TCP probe of
  `LOCI_LOCAL_OLLAMA`, then `[ollama] url`; only then `MNEMOSYNE_EMBEDDING_API_URL`.
- The auth-failure path treats a missing `prisma` package as "not a database error", so a missing key is a 401
  rather than a 500 on a proxy without a database.
- Invented model names are deliberately not aliased to `embedding`, so an unknown model fails loudly.

## Tests

From the repository root, with `HOME` and `LOCI_CONFIG` pointing at an empty temporary location so the tests do not
read a real config:

```
python -m pytest litellm-loci/tests -q -p no:cacheprovider
```

## Capability gate

`capability_gate` exists because, on the one local model it was measured on (`evals/medium_tier_fit.py`), a
tool result made the model call the tool again instead of answering, which loops an agent. That is a measurement
of one model, not a general claim.

What the code does: if the request has `tools`/`functions`, a `tool`/`function` role message, or `tool_calls`, it
adds `!<LOCI_LOCAL_DEPLOYMENT_TAG>` to `metadata.tags` and sets `metadata.loci_capability_gate = "tools"`. It
never changes the model or the request body, and an internal error is logged at WARNING and ignored.

What it needs from you, or it does nothing: `router_settings.enable_tag_filtering: true`, and `tags: ["local"]`
(the same tag name) on every local deployment. It writes only to `metadata`; whether routes that carry
`litellm_metadata` instead (for example Anthropic `/v1/messages`) honour the tag has not been verified.

## Behavior changes since 0.1.0

These change what existing deployments observe. Nothing here is applied until the code is deployed.

- **Memory writer.** New records embed the prompt only (they used to embed the prompt plus the first 200
  characters of the answer), so they match what recall queries embed; a collection written by 0.1.0 therefore
  mixes two embedding recipes. They no longer carry `importance: 0.6`; they are labelled `record_type=inferred`,
  `confidence=low`, `provenance=model_output`, so under the grounding ranker's defaults new memories rank below
  records written by 0.1.0. Reasoning-only responses are not stored. The point id hashes model, prompt and
  response, so only an identical triple overwrites. A streamed call is written once, from the assembled response.
- **Dedup gate removed.** `dedup_gate` is an inert placeholder that logs one WARNING when configured; remove it
  from the callbacks.
- **Grounding.** A search that fails is a WARNING, and if every search fails the status is `unavailable`
  (`all_collections_failed`), no longer `miss`. A miss with failed searches carries reason
  `no_matching_context_some_searches_failed`. An `importance` of 0 is kept as 0. The retrieved block is labelled
  as reference data, and the query line is flattened to one line.
- **Cost ledger.** Rows are per attempt, not per request; `cost_report` counts are per attempt. Rows gain
  `cache_read_tokens`, `cache_write_tokens` (what the provider reported, 0 when it reports none) and
  `empty_response` (a call that succeeded but returned no answer text and no tool call; a reasoning model that
  spent its budget thinking counts). `cost_report` counts an empty answer as a failure for the local SLO and adds
  `empty` and `cache%` columns; a ledger written before these fields reads as 0 cache and 0 empty.
- **Package import.** `import loci_litellm` no longer imports the callbacks (and so LiteLLM); they load on first
  attribute access. Code that did `from loci_litellm import loci_grounding_hook` is unaffected.
- **Removed settings (set-but-ignored now).** `LOCI_DEDUP_*`, `LOCI_CB_RECOVERY_WINDOW`, `LOCI_QDRANT_WORKERS`,
  `LOCI_LOCAL_VLLM`, `LOCI_OLLAMA_LIST_TIMEOUT`, `LOCI_OLLAMA_GEN_URL` / `LOCI_OLLAMA_GEN_MODEL`,
  `LOCI_VLLM_*`, `LOCI_RERANK_MODEL`, and the `BackendResolver` methods `ollama_gen_url`, `vllm_url`,
  `vllm_model`, `rerank_model`. Setting one has no effect and logs nothing.
- **Renamed.** `LOCI_PHERO_EPSILON` is now `LOCI_RECALL_EXPLORE_EPSILON` (the old name is still read).
- **Version.** `pyproject.toml` is 0.2.0 and `loci_litellm.__version__` reports it at runtime.
