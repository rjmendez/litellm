"""ops_logs collector: ship hourly cost-ledger summaries into the Loci ops_logs index, and write the
health snapshot a router can read.

It plugs into ~/.loci/ops_logs_ext.py (the Loci ops_logs shipper). The shipper's aggregator collapses lines by
signature with digits removed, so this collector spells the hour out as a word; each (day, hour, group,
deployment) is then its own point, and re-running upserts the latest numbers for that hour.

What is shipped: aggregates only (counts, rates, latency percentiles, spend, token totals, error classes).
Never raw rows, prompts, answers, `agent` or `key_alias`.

    from ops_logs_ledger_collector import ship_ledger
    ship_ledger(agg, state, new_state, "win-litellm-ledger-prod", r"...\\litellm_cost_ledger.jsonl", full,
                health_paths=[...])
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from loci_litellm import ledger_summary as ls  # noqa: E402  (LiteLLM-free)


def ship_ledger(agg, state, new_state, source: str, ledger_path: str, full: bool = False,
                health_paths=(), window_hours: float = 48.0, host: str = "WINDOWS",
                write_health: bool = True) -> int:
    """Summarise the last `window_hours` of the ledger into `agg` (one line per hour/group/deployment) and
    write the health snapshot to `health_paths` unless `write_health` is false (the shipper's --dry-run must
    not touch the filesystem). Returns the number of summary lines."""
    now = time.time()
    try:
        rows = list(ls.read_rows(ledger_path, now - window_hours * 3600))
    except OSError as exc:
        # Loud, not silent: the shipper prints this in its run log and the point carries the reason.
        agg.add(source, host, ledger_path, time.strftime("%Y-%m-%d"), time.strftime("%Y-%m-%dT%H:%M:%S"),
                f"litellm ledger unreadable: {type(exc).__name__}: {exc}", "error")
        return 0
    summaries = ls.hourly_summaries(rows)
    for s in summaries:
        agg.add(source, host, ledger_path, s["day"], s["hour"] + ":00", ls.format_line(s), ls.severity(s))
    if health_paths and write_health:
        snapshot = ls.health(rows, now=now)
        written = ls.write_health_file(snapshot, *health_paths)
        if len(written) < len(health_paths):
            missed = sorted(set(map(str, health_paths)) - set(written))
            agg.add(source, host, ledger_path, time.strftime("%Y-%m-%d"), time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "litellm ledger health snapshot not written to: " + ", ".join(missed), "warning")
    return len(summaries)
