loci-litellm adds a pre-call Qdrant grounding step, a write-back memory path, and a fail-open circuit breaker for LiteLLM callbacks. It keeps grounding best-effort so requests continue when the vector store or backend is unavailable

Install it with `uv add loci-litellm`. In your LiteLLM config, use `litellm.integrations.loci.grounding_hook.loci_grounding_hook` for the legacy shim path or `loci_litellm.grounding_hook.loci_grounding_hook` for the direct package path, and use `litellm.integrations.loci.memory_writer.loci_memory_writer` or `loci_litellm.memory_writer.loci_memory_writer` in the same way. The usual environment variables are `LOCI_QDRANT_URL`, `LOCI_QDRANT_API_KEY`, `LOCI_OLLAMA_BASE_URL`, `LOCI_EMBED_MODEL`, `LOCI_GROUNDING_DISABLED`, `LOCI_RECALL_TOP_K`, and `LOCI_CB_FAILURE_THRESHOLD`

## Deployment invariants

`deploy/config.example.yaml` is a sanitized reference config, and `deploy/check_deployment.py`
validates a real `config.yaml` (and `.env`) against the rules this integration depends on.
The checker runs against the example in `tests/test_deploy_check.py`, so the rules cannot
drift silently.

```
python deploy/check_deployment.py --config config.yaml --env .env
```

Run it with the proxy's own interpreter and import path (so LiteLLM's built-in cost map
resolves), e.g. `PYTHONPATH=<checkout> <venv>/python deploy/check_deployment.py ...`.

| Rule | Why it exists |
|---|---|
| `embedding-group` | A pinned embedding group exists and is in no fallback chain and no auto-router tier. Vectors from different embedders are not comparable, so a failure must surface instead of falling back. |
| `priced` | Every non-local deployment declares per-token prices or is in LiteLLM's cost map. Otherwise spend is logged as `$0` with only a warning. |
| `no-ip-literal` | No `api_base` and no `*_URL` / `*_BASE` variable points at an IP address. Consumers use DNS names over TLS; a loopback IP is acceptable only for a listener bind, which is not a URL. |
| `alias-targets` | Every `model_group_alias` target is a real group, so a renamed group cannot leave dangling aliases. |
| `trusted-proxies` | `general_settings.trusted_proxy_ranges` is set (warning): without it the admin-UI login throttle logs a warning on every start. |

Other deployment notes:

- The plugin resolves its embedding endpoint from `LOCI_OLLAMA_BASE_URL` / `OLLAMA_BASE_URL`
  before `MNEMOSYNE_EMBEDDING_API_URL`, so set the former, to a DNS name.
- A proxy without a database never installs `prisma`; the auth-failure path treats a missing
  `prisma` as "not a database error" (so a missing key is a 401, not a 500).
- Clients that send `ollama/nomic-embed-text` or `nomic-embed-text` reach the `embedding` group
  through `router_settings.model_group_alias`. Invented model names are deliberately **not**
  aliased: an unknown model must fail loudly.
