"""Capability gate: keep requests the local model cannot serve reliably off local deployments.

Measured on the resident local model: given a tool result it calls the tool again instead of answering
(4 of 5 to 5 of 5 times, `tool_choice` ignored), which loops any agent. Requests that carry tool
definitions or tool turns therefore get the request tag `!<local tag>`; with router tag filtering on and
local deployments tagged, the router excludes them and the cloud deployments serve the call.

Needs `router_settings.enable_tag_filtering: true` and `tags: ["local"]` on the local deployments (the tag
name is LOCI_LOCAL_DEPLOYMENT_TAG). Without those it is inert. LOCI_CAPABILITY_GATE_DISABLED=1 turns it off.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("loci_litellm.capability_gate")


def local_tag() -> str:
    return os.environ.get("LOCI_LOCAL_DEPLOYMENT_TAG", "local").strip() or "local"


def _disabled() -> bool:
    return os.environ.get("LOCI_CAPABILITY_GATE_DISABLED", "").strip().lower() in ("1", "true", "yes", "on")


def needs_tools(data: dict) -> bool:
    if data.get("tools") or data.get("functions"):
        return True
    for message in data.get("messages") or []:
        if isinstance(message, dict) and (message.get("role") in ("tool", "function") or message.get("tool_calls")):
            return True
    return False


class LociCapabilityGate(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Any:
        if _disabled() or not isinstance(data, dict):
            return None
        try:
            if not needs_tools(data):
                return None
            metadata = data.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            tags = [t for t in (metadata.get("tags") or []) if isinstance(t, str)]
            exclude = f"!{local_tag()}"
            if exclude not in tags:
                tags.append(exclude)
            metadata["tags"] = tags
            metadata["loci_capability_gate"] = "tools"
            data["metadata"] = metadata
        except Exception as exc:
            logger.warning("loci capability gate: fail-open: %r", exc)
        return None


loci_capability_gate = LociCapabilityGate()
