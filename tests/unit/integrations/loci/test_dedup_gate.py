"""The dedup gate was removed: the shim path still imports, and the gate never answers or alters a request.

The full behavioural tests live in litellm-loci/tests (test_memory_honesty.py).
"""
from __future__ import annotations

import pytest

import litellm.integrations.loci.dedup_gate as dedup_gate


@pytest.mark.asyncio
async def test_gate_never_replaces_the_request() -> None:
    data = {"model": "m", "messages": [{"role": "user", "content": "a question long enough to be looked up"}]}
    before = {**data}
    result = await dedup_gate.loci_dedup_gate.async_pre_call_hook(None, None, data, "completion")
    assert result is None
    assert data == before
