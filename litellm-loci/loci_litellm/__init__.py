"""Loci grounding, memory write-back and routing callbacks for the LiteLLM proxy.

``__version__`` states which release this code is. It is read from the ``pyproject.toml`` next to
the package when running from a source checkout (what the proxy does through the
``litellm.integrations.loci`` shim), else from the installed distribution metadata, else it is
``"unknown"``. ``__file__`` of this package is logged once at INFO on import, so the log names the
code that is actually running.
"""
from __future__ import annotations

import logging
from pathlib import Path


def _read_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        import tomllib

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        if data.get("project", {}).get("name") == "loci-litellm":
            return str(data["project"]["version"])
    except Exception:  # not a source checkout (no pyproject.toml beside the package) or unreadable
        pass
    try:
        from importlib.metadata import version

        return version("loci-litellm")
    except Exception:
        return "unknown"


__version__ = _read_version()

logging.getLogger("loci_litellm").info(
    "loci_litellm %s loaded from %s", __version__, Path(__file__).resolve().parent
)

from loci_litellm.backend_resolver import BackendResolver  # noqa: E402
from loci_litellm.grounding_hook import LociGroundingHook, loci_grounding_hook  # noqa: E402
from loci_litellm.memory_writer import LociMemoryWriter, loci_memory_writer  # noqa: E402

__all__ = [
    "__version__",
    "BackendResolver",
    "LociGroundingHook",
    "loci_grounding_hook",
    "LociMemoryWriter",
    "loci_memory_writer",
]
