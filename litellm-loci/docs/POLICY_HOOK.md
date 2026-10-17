# LociPolicyHook — Policy Supervision & Enforcement

Policy enforcement layer for LiteLLM proxy using DRPG discovery/refinement and EVO-RAG 7D reward semantics.

## Quick Start

### 1. Register in litellm config.yaml

```yaml
litellm_settings:
  callbacks:
    - loci_litellm.policy_hook.loci_policy_hook
    - loci_litellm.grounding_hook.loci_grounding_hook
    - loci_litellm.tier_router.loci_tier_router
```

### 2. Set Environment Variables

```bash
# Enable policy supervision
export LOCI_POLICY_ENABLE=1

# Optional: Enforce model whitelist
export LOCI_POLICY_ALLOWED_MODELS=gpt-4,gpt-4-turbo,claude-3-opus

# Optional: Limit request size (chars)
export LOCI_POLICY_MAX_CHARS=10000

# Optional: Block violations (default: warn and allow)
export LOCI_POLICY_BLOCK_VIOLATIONS=0

# Optional: EVO-RAG reward tracking window
export LOCI_REWARD_HORIZON_HOURS=24

# Optional: DRPG refinement interval (samples)
export LOCI_DRPG_REFINEMENT_INTERVAL=100
```

### 3. Monitor Policy Status

Call the monitoring endpoint:

```python
from loci_litellm.policy_hook import get_policy_status

status = get_policy_status()
print(status)
# {
#   "enabled": true,
#   "total_violations": 3,
#   "recent_violations": {"model_not_allowed": 2, "max_chars_exceeded": 1},
#   "reward_stats": {
#     "count": 50,
#     "avg_r_ret": 0.78,
#     "avg_r_rem": 0.82,
#     "avg_r_step": 0.91,
#     ...
#   },
#   "drpg_suggestions": [
#     "Max char limit hit 6 times: consider raising LOCI_POLICY_MAX_CHARS from 5000"
#   ]
# }
```

## Policy Constraints

### Model Whitelist

`LOCI_POLICY_ALLOWED_MODELS` enforces a model allowlist. When set:

```bash
export LOCI_POLICY_ALLOWED_MODELS=gpt-4,claude-3-opus
```

Requests to other models are flagged as `model_not_allowed` violations. By default, violations are logged but requests proceed (`LOCI_POLICY_BLOCK_VIOLATIONS=0`). To block:

```bash
export LOCI_POLICY_BLOCK_VIOLATIONS=1
```

### Max Character Limit

`LOCI_POLICY_MAX_CHARS` limits request size (e.g., to prevent token explosion):

```bash
export LOCI_POLICY_MAX_CHARS=10000
```

Requests longer than this are flagged as `max_chars_exceeded` violations.

## EVO-RAG 7D Reward Semantics

After each successful completion, the hook records a **7-dimensional reward tensor** to track LLM behavior quality:

| Dimension | Name | Meaning | Source |
|-----------|------|---------|--------|
| `r_ret` | Retrieval Quality | Qdrant recall success (0–1) | `grounding_score` metadata |
| `r_rem` | Relevance to Memory | Evidence grounding score | `memory_match_count` |
| `r_ovp` | Overfitting Prevention | Diversity penalty (avoid repeated contexts) | Computed from response |
| `r_bkp` | Background Knowledge | Prefer retrieval over base model knowledge | `grounding_used` flag |
| `r_ref` | Refinement Success | DRPG cycle improvement signal | `refinement_delta` |
| `r_step` | Step Efficiency | Latency penalty (prefer <5s) | Measured latency |
| `r_cor` | Correction Signal | Feedback on policy catches | `correction_signal` |

**Weighted sum:** `0.20·r_ret + 0.20·r_rem + 0.15·r_ovp + 0.15·r_bkp + 0.10·r_ref + 0.10·r_step + 0.10·r_cor`

### Example: Tracking Reward Degradation

```python
from loci_litellm.policy_hook import get_policy_status

status = get_policy_status()
recent_stats = status["reward_stats"]

if recent_stats["avg_r_step"] < 0.5:
    print("⚠️ Step efficiency degraded — latency > 5s average")
    
if recent_stats["avg_r_rem"] < 0.6:
    print("⚠️ Memory relevance poor — grounding not helping")
```

## DRPG Discovery/Refinement Cycle

The hook implements **DRPG** (Discovery, Refinement, Propagation, Grounding):

- **Discovery**: Track policy violations and reward signals
- **Refinement**: Every `LOCI_DRPG_REFINEMENT_INTERVAL` samples (default 100), suggest policy adjustments
- **Propagation**: Violations and suggestions are logged and exposed via `get_policy_status()`

### Refinement Suggestions

After the interval, suggestions are auto-generated:

```
High violation rate (15 in last 100 samples): consider tightening policy constraints
Max char limit hit 6 times: consider raising LOCI_POLICY_MAX_CHARS from 5000
Model restrictions blocked 4 requests: review LOCI_POLICY_ALLOWED_MODELS configuration
```

Retrieve suggestions via `get_policy_status()["drpg_suggestions"]`.

## Integration with Grounding & Routing

The policy hook integrates with **grounding_hook** and **tier_router**:

```
Request
  │
  ├─→ [LociPolicyHook.async_pre_call_hook]
  │   • Check model whitelist
  │   • Check max char limit
  │
  ├─→ [LociGroundingHook.async_pre_call_hook]
  │   • Embed + Qdrant search
  │   • Inject MEMORY MATCH
  │
  ├─→ [LociTierRouter.async_pre_call_hook]
  │   • Check circuit breaker state
  │
  ├─→ [LLM Call]
  │
  └─→ [LociPolicyHook.async_post_call_success_hook]
      • Compute 7D reward tensor
      • Update DRPG state
      • Suggest refinements if needed
```

## Violation History

Query violations since a timestamp:

```python
from loci_litellm.policy_hook import get_violations_since
import time

now = time.time()
since = now - (6 * 3600)  # Last 6 hours

violations = get_violations_since(since)
for v in violations:
    print(f"{v['timestamp']}: {v['type']} — {v['message']}")
```

Output:
```
1697500000.5: model_not_allowed — Model 'gpt-5' not in allowlist: ['gpt-4', 'claude-3']
1697500030.2: max_chars_exceeded — Request exceeds max chars: 15000 > 10000
```

## Tool Call Enforcement

Set `LOCI_POLICY_ALLOWED_TOOLS` to a comma-separated list of allowed tool names.
If unset, all tools are permitted.

```bash
export LOCI_POLICY_ALLOWED_TOOLS=web_search,read_file,list_directory
```

Requests containing tools not in this list are flagged as `tool_not_allowed` violations.
Combined with `LOCI_POLICY_BLOCK_VIOLATIONS=1`, the request is rejected outright.

## Trajectory Persistence

Set `LOCI_TRAJECTORY_DIR` (default: `~/.loci/trajectories`) to control where
step-level reward trajectories are written. Each policy gets its own JSONL file:
`{LOCI_TRAJECTORY_DIR}/{policy_id}.jsonl`.

These files are consumed by `mcp__loci__policy_evolve` (companion loci PR) to
close the DRPG loop and promote policies based on accumulated evidence.

Each line is one JSON object with keys `timestamp`, `policy_id`, `model`,
`prompt_tokens`, `response_time_ms`, and `reward` (the full 7D tensor).

## Default Enablement

The recommended production configuration is opt-out (always-on):

```yaml
# config.yaml
callbacks:
  - path: ./loci_litellm/policy_hook.py
    class: LociPolicyHook
general_settings:
  environment_variables:
    LOCI_POLICY_ENABLE: "1"
```

To disable policy enforcement for a specific deployment, set `LOCI_POLICY_ENABLE=0`
in that deployment's environment without changing the shared config.

## Grounded References

This implementation is grounded in:

- **[6003c71b]** LiteLLM Loci Plugin Configuration (hermes_memory): grounding_hook, tier_router, backend_resolver patterns
- **[b5e01c4e]** LiteLLM Loci Plugin Overview (hermes_memory): integration architecture
- **[b64ebbf6]** Copilot CLI hooks precedent (hermes_memory): fail-open discipline and hook contract

## Testing

Run the test suite:

```bash
cd litellm-loci
python -m pytest tests/test_policy_hook.py -v

# Specific test class
python -m pytest tests/test_policy_hook.py::TestEvoRagRewardTensor -v

# With coverage
python -m pytest tests/test_policy_hook.py --cov=loci_litellm.policy_hook
```

Key tests:
- `TestEvoRagRewardTensor`: 7D reward tensor logic
- `TestPolicyViolation`: Violation recording
- `TestLociPolicyHook`: Hook invocation and policy enforcement
- `TestDRPGRefinement`: Discovery/refinement cycle
- `TestIntegration`: Full callback chain with grounding + routing

## Troubleshooting

### Policy violations not appearing

1. Check `LOCI_POLICY_ENABLE=1` is set
2. Verify hook is registered in config.yaml
3. Call `get_policy_status()` to check hook is active
4. Check logs for `loci_litellm.policy` logger

### DRPG suggestions not generated

- Suggestions only appear after `LOCI_DRPG_REFINEMENT_INTERVAL` samples (default 100)
- Check violation count: `get_policy_status()["recent_violations"]`
- If < 10 violations, no suggestion will trigger (threshold configurable in code)

### Reward stats empty

- Ensure completions are reaching `async_post_call_success_hook`
- Check response includes metadata fields (e.g., `grounding_score`)
- Review logs for post-call errors

## Example: Complete Config

```yaml
# litellm config.yaml
litellm_settings:
  max_tokens: 2048
  
  # Callbacks run in order
  callbacks:
    # 1. Policy supervision (pre-call)
    - loci_litellm.policy_hook.loci_policy_hook
    
    # 2. Memory grounding
    - loci_litellm.grounding_hook.loci_grounding_hook
    
    # 3. Tier routing & circuit breaker
    - loci_litellm.tier_router.loci_tier_router
    
    # 4. Memory write-back
    - loci_litellm.memory_writer.loci_memory_writer

model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: ${OPENAI_API_KEY}
  
  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: ${ANTHROPIC_API_KEY}
```

```bash
# .env
LOCI_POLICY_ENABLE=1
LOCI_POLICY_ALLOWED_MODELS=gpt-4,claude-3-opus
LOCI_POLICY_MAX_CHARS=10000
LOCI_POLICY_BLOCK_VIOLATIONS=0
LOCI_REWARD_HORIZON_HOURS=24
LOCI_DRPG_REFINEMENT_INTERVAL=50
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│         LiteLLM Proxy Request                   │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ LociPolicyHook      │
        │ pre_call_hook       │
        │ • Model whitelist   │
        │ • Max char check    │
        │ • Record access     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ LociGroundingHook   │
        │ pre_call_hook       │
        │ • Embed + search    │
        │ • Inject MEMORY     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ LociTierRouter      │
        │ pre_call_hook       │
        │ • Circuit breaker   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   LLM Call          │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ LociPolicyHook      │
        │ post_call_hook      │
        │ • 7D reward tensor  │
        │ • DRPG refinement   │
        │ • Update state      │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ LociTierRouter      │
        │ post_call_hook      │
        │ • Record success    │
        │ • Update circuit    │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Response + metadata │
        └─────────────────────┘
```

---

**Grounded in**: REA_HERMES_ENHANCEMENT investigation (key_uuids=405f56ac, 51f70bb5) and hermes_memory corpus (IDs: 6003c71b, b5e01c4e, b64ebbf6).
