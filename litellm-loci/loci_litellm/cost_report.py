"""Report on the cost ledger: spend per model group, what the local tier saved, and whether local is
reliable enough to put first.

    python -m loci_litellm.cost_report --config <litellm config.yaml> [--ledger FILE] [--since-hours 24]

Savings are counterfactual: every token served locally is priced at the cheapest cloud deployment of the
same model group (its `input_cost_per_token` / `output_cost_per_token` in the config). The report also
states, per group, whether local meets the reliability bar (LOCI_LOCAL_SLO_SUCCESS, LOCI_LOCAL_SLO_P95_S,
LOCI_LOCAL_SLO_MIN_CALLS) and what order flip that would justify. It only reports; it never edits config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _pctl(values: list, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def cloud_prices(config_path: str) -> dict:
    """{group: (input_per_token, output_per_token)} of each group's cheapest priced deployment."""
    import yaml
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    best: dict = {}
    for entry in data.get("model_list") or []:
        group = entry.get("model_name")
        info = entry.get("model_info") or {}
        p_in, p_out = info.get("input_cost_per_token"), info.get("output_cost_per_token")
        if group and p_in is not None and p_out is not None:
            price = (float(p_in), float(p_out))
            if group not in best or sum(price) < sum(best[group]):
                best[group] = price
    return best


def read_rows(path: str, since_ts: float) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("ts", 0) >= since_ts:
                yield row


def summarise(rows: Iterable[dict], prices: dict, include_internal: bool = True) -> dict:
    groups: dict = defaultdict(lambda: {"local": defaultdict(list), "cloud": defaultdict(list)})
    for row in rows:
        if not include_internal and row.get("internal"):
            continue
        groups[row.get("group") or "?"]["local" if row.get("local") else "cloud"][row.get("deployment") or "?"].append(row)
    out: dict = {}
    slo_ok = float(os.environ.get("LOCI_LOCAL_SLO_SUCCESS", "0.98"))
    slo_p95 = float(os.environ.get("LOCI_LOCAL_SLO_P95_S", "20"))
    slo_min = int(os.environ.get("LOCI_LOCAL_SLO_MIN_CALLS", "50"))
    for group, sides in groups.items():
        local_rows = [r for rs in sides["local"].values() for r in rs]
        cloud_rows = [r for rs in sides["cloud"].values() for r in rs]
        p_in, p_out = prices.get(group, (0.0, 0.0))
        counterfactual = sum(r.get("prompt_tokens", 0) * p_in + r.get("completion_tokens", 0) * p_out
                             for r in local_rows if r.get("ok"))
        local_ok = [r for r in local_rows if r.get("ok")]
        success = (len(local_ok) / len(local_rows)) if local_rows else None
        lat = [r.get("latency_s", 0.0) for r in local_ok]
        p95 = _pctl(lat, 0.95)
        meets = bool(local_rows) and len(local_rows) >= slo_min and success >= slo_ok and p95 <= slo_p95
        out[group] = {
            "calls": len(local_rows) + len(cloud_rows),
            "local_calls": len(local_rows),
            "cloud_calls": len(cloud_rows),
            "local_success": None if success is None else round(success, 4),
            "local_p50_s": round(_pctl(lat, 0.5), 2),
            "local_p95_s": round(p95, 2),
            "cloud_spend_usd": round(sum(r.get("cost_usd", 0.0) for r in cloud_rows), 6),
            "local_savings_usd": round(counterfactual, 6),
            "local_failures": sorted({r.get("error_class") or "?" for r in local_rows if not r.get("ok")}),
            "meets_slo": meets,
            "recommendation": (
                "local is reliable: put it first (order below the cloud deployments)" if meets else
                "keep cloud first" + ("" if len(local_rows) >= slo_min else f" (only {len(local_rows)} local calls, need {slo_min})")
                if local_rows or cloud_rows else ""),
        }
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the LiteLLM config.yaml (for per-deployment prices)")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--since-hours", type=float, default=24.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from loci_litellm.cost_ledger import ledger_path
    path = args.ledger or ledger_path()
    if not os.path.exists(path):
        print(f"no ledger at {path}", file=sys.stderr)
        return 2
    summary = summarise(read_rows(path, time.time() - args.since_hours * 3600), cloud_prices(args.config))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"{'group':16s} {'calls':>6s} {'local':>6s} {'ok%':>6s} {'p95s':>6s} {'cloud$':>9s} {'saved$':>9s}  recommendation")
    for group, s in sorted(summary.items()):
        ok = "-" if s["local_success"] is None else f"{100 * s['local_success']:.1f}"
        print(f"{group:16s} {s['calls']:6d} {s['local_calls']:6d} {ok:>6s} {s['local_p95_s']:6.1f} "
              f"{s['cloud_spend_usd']:9.4f} {s['local_savings_usd']:9.4f}  {s['recommendation']}")
    total_saved = sum(s["local_savings_usd"] for s in summary.values())
    total_spend = sum(s["cloud_spend_usd"] for s in summary.values())
    print(f"\ncloud spend ${total_spend:.4f}, saved by local ${total_saved:.4f} over the last {args.since_hours:g} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
