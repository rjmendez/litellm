import sys
from pathlib import Path

_plugin_root = Path(__file__).resolve().parents[3] / "litellm-loci"
if _plugin_root.exists() and str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

import sys
from pathlib import Path

_plugin_root = Path(__file__).resolve().parents[3] / "litellm-loci"
if _plugin_root.exists() and str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

from loci_litellm.backend_resolver import BackendResolver
from loci_litellm.dedup_gate import LociDedupGate, loci_dedup_gate
from loci_litellm.grounding_hook import LociGroundingHook, loci_grounding_hook
from loci_litellm.memory_writer import LociMemoryWriter, loci_memory_writer

__all__ = [
    "BackendResolver",
    "LociGroundingHook",
    "loci_grounding_hook",
    "LociMemoryWriter",
    "loci_memory_writer",
    "LociDedupGate",
    "loci_dedup_gate",
]