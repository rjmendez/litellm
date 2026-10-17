# LiteLLM-Loci Copilot Instructions

## Policy Hook
- LociPolicyHook is a CustomLogger registered in config.yaml
- LOCI_POLICY_ENABLE=1 enables enforcement
- LOCI_POLICY_ALLOWED_TOOLS controls tool call gating
- LOCI_TRAJECTORY_DIR for step data persistence

## Testing
- `cd litellm-loci && python -m pytest tests/ -v`

## Companion
- Policy MCP tools: https://github.com/rjmendez/loci/pull/482
- DRPG loop: https://github.com/rjmendez/loci/pull/483
