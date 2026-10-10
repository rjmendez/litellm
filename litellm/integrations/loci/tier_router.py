import sys
from pathlib import Path

_plugin_root = Path(__file__).resolve().parents[3] / "litellm-loci"
if _plugin_root.exists() and str(_plugin_root) not in sys.path:
    sys.path.insert(0, str(_plugin_root))

from loci_litellm.tier_router import LociTierRouter, loci_tier_router

__all__ = ["LociTierRouter", "loci_tier_router"]