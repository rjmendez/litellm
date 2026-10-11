"""The shim path exposes the package's memory writer, not a copy of it.

The full behavioural tests live in litellm-loci/tests (test_memory_writer.py, test_memory_honesty.py).
"""
from __future__ import annotations

import loci_litellm.memory_writer as impl
from litellm.integrations.loci.memory_writer import LociMemoryWriter, loci_memory_writer


def test_shim_is_the_package_object() -> None:
    assert LociMemoryWriter is impl.LociMemoryWriter
    assert loci_memory_writer is impl.loci_memory_writer


def test_streams_are_recorded_from_the_success_event_only() -> None:
    from litellm.integrations.custom_logger import CustomLogger

    assert LociMemoryWriter.async_log_stream_event is CustomLogger.async_log_stream_event
