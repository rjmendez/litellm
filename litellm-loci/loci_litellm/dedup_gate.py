"""Removed feature: the semantic dedup gate.

The gate used to look up a stored answer for a near-identical prompt and return it as a
"cache hit" from ``async_pre_call_hook``. That never worked and was unsafe:

- litellm treats a dict returned from ``async_pre_call_hook`` as the replacement request
  body, not as a response, so a "hit" would have produced a broken request.
- It searched with an unnamed vector against a collection written with a named vector.
- It matched on the last user message alone (no model, tenant or system-prompt scoping)
  and would have replayed truncated previews, across callers.

It has been removed rather than repaired. This module remains only so existing imports
(``LociDedupGate``, ``loci_dedup_gate``) keep resolving; it performs no lookups and
never alters a request. If it is still configured as a callback it logs one WARNING.
"""
from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("loci_litellm.dedup_gate")


class LociDedupGate(CustomLogger):
    """Inert placeholder for the removed dedup gate. Never modifies or answers a request."""

    def __init__(self) -> None:
        super().__init__()
        self._warned: bool = False

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: Any,
    ) -> dict[str, Any] | None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "loci dedup gate is configured but the feature has been removed; it does "
                "nothing and no request is answered from a cache. Remove it from callbacks."
            )
        return None


loci_dedup_gate = LociDedupGate()
