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

# The callbacks import LiteLLM. They are loaded on first attribute access so that the offline tools
# (``python -m loci_litellm.cost_report``) run without the proxy stack installed.
_LAZY = {
    "BackendResolver": "loci_litellm.backend_resolver",
    "LociGroundingHook": "loci_litellm.grounding_hook",
    "loci_grounding_hook": "loci_litellm.grounding_hook",
    "LociMemoryWriter": "loci_litellm.memory_writer",
    "loci_memory_writer": "loci_litellm.memory_writer",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'loci_litellm' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "BackendResolver",
    "LociGroundingHook",
    "loci_grounding_hook",
    "LociMemoryWriter",
    "loci_memory_writer",
]
