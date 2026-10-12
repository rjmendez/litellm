from __future__ import annotations

import pytest

from loci_litellm.capability_gate import LociCapabilityGate, needs_tools

TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_CAPABILITY_GATE_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_LOCAL_DEPLOYMENT_TAG", raising=False)


def test_tool_definitions_and_tool_turns_are_detected() -> None:
    assert needs_tools({"tools": [TOOL], "messages": []})
    assert needs_tools({"functions": [{"name": "f"}]})
    assert needs_tools({"messages": [{"role": "tool", "content": "x"}]})
    assert needs_tools({"messages": [{"role": "assistant", "tool_calls": [{"id": "1"}]}]})
    assert not needs_tools({"messages": [{"role": "user", "content": "hi"}]})
    assert not needs_tools({"tools": [], "messages": []})


@pytest.mark.asyncio
async def test_a_tool_request_gets_the_exclusion_tag() -> None:
    data = {"model": "medium-tier", "tools": [TOOL], "messages": [{"role": "user", "content": "hi"}]}
    await LociCapabilityGate().async_pre_call_hook(None, None, data, "acompletion")
    assert data["metadata"]["tags"] == ["!local"]
    assert data["metadata"]["loci_capability_gate"] == "tools"


@pytest.mark.asyncio
async def test_existing_tags_are_kept_and_not_duplicated() -> None:
    data = {"tools": [TOOL], "metadata": {"tags": ["team-a", "!local"]}}
    await LociCapabilityGate().async_pre_call_hook(None, None, data, "acompletion")
    assert data["metadata"]["tags"] == ["team-a", "!local"]


@pytest.mark.asyncio
async def test_a_plain_request_is_untouched() -> None:
    data = {"model": "medium-tier", "messages": [{"role": "user", "content": "hi"}]}
    await LociCapabilityGate().async_pre_call_hook(None, None, data, "acompletion")
    assert "metadata" not in data


@pytest.mark.asyncio
async def test_the_tag_name_and_the_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_LOCAL_DEPLOYMENT_TAG", "onprem")
    data = {"tools": [TOOL]}
    await LociCapabilityGate().async_pre_call_hook(None, None, data, "acompletion")
    assert data["metadata"]["tags"] == ["!onprem"]

    monkeypatch.setenv("LOCI_CAPABILITY_GATE_DISABLED", "1")
    off = {"tools": [TOOL]}
    await LociCapabilityGate().async_pre_call_hook(None, None, off, "acompletion")
    assert "metadata" not in off
