"""Compatibility path for the Loci callbacks: ``litellm.integrations.loci.<module>``.

There is no implementation here. Every module in this directory re-exports the same names from the
``loci_litellm`` package (``litellm-loci/loci_litellm`` in this checkout), so a callback registered
under either dotted path is the same object.

``loci_litellm`` is not on ``sys.path`` by default. Importing this package puts the checkout's
``litellm-loci`` directory on ``sys.path`` when it exists, and otherwise needs ``loci_litellm`` to be
installed. If neither holds, the import fails with an ImportError that says so; the proxy does not
start without its configured callbacks, rather than running without grounding.
"""
import logging
import sys
from pathlib import Path

_plugin_root = Path(__file__).resolve().parents[3] / "litellm-loci"
if (_plugin_root / "loci_litellm" / "__init__.py").is_file() and str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

try:
    import loci_litellm as _impl
except ImportError as exc:  # pragma: no cover - exercised by a test through a subprocess
    raise ImportError(
        "litellm.integrations.loci needs the loci_litellm package, which was found neither at "
        f"{_plugin_root} nor on sys.path (install loci-litellm, or run from a checkout that "
        f"contains litellm-loci/). Original error: {exc}"
    ) from exc

if (_plugin_root / "loci_litellm").is_dir() and Path(_impl.__file__).resolve().parent != (
    _plugin_root / "loci_litellm"
).resolve():
    logging.getLogger("loci_litellm").warning(
        "litellm.integrations.loci is running loci_litellm from %s, not the checkout copy at %s",
        Path(_impl.__file__).resolve().parent, _plugin_root / "loci_litellm",
    )

from loci_litellm import (  # noqa: E402
    BackendResolver,
    LociGroundingHook,
    LociMemoryWriter,
    __version__,
    loci_grounding_hook,
    loci_memory_writer,
)

__all__ = [
    "__version__",
    "BackendResolver",
    "LociGroundingHook",
    "loci_grounding_hook",
    "LociMemoryWriter",
    "loci_memory_writer",
]
