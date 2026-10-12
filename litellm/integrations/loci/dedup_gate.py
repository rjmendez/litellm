import sys
from pathlib import Path

_plugin_root = Path(__file__).resolve().parents[3] / "litellm-loci"
if _plugin_root.exists() and str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

import loci_litellm.dedup_gate as _dedup_gate_impl

LociDedupGate = _dedup_gate_impl.LociDedupGate
loci_dedup_gate = _dedup_gate_impl.loci_dedup_gate
httpx = _dedup_gate_impl.httpx
BackendResolver = _dedup_gate_impl.BackendResolver

__all__ = ["LociDedupGate", "loci_dedup_gate", "httpx", "BackendResolver"]
