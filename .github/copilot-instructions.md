# LiteLLM-Loci Copilot Instructions

## Policy Hook
- LociPolicyHook is a CustomLogger registered in config.yaml
- LOCI_POLICY_ENABLE=1 enables enforcement

## Environment Variables
The following are available after PR #6 (`feat/policy-tool-enforcement`) merges:

| Variable | Purpose |
|---|---|
| `LOCI_POLICY_ENABLE` | Enable policy enforcement (set to `1`) |
| `LOCI_POLICY_ALLOWED_TOOLS` | Comma-separated allowed tool names (PR #6) |
| `LOCI_TRAJECTORY_DIR` | DRPG step logging directory (PR #6) |

## Testing
- `cd litellm-loci && python -m pytest tests/ -v`

## Companion
- Policy MCP tools: https://github.com/rjmendez/loci/pull/482
- DRPG loop: https://github.com/rjmendez/loci/pull/483
